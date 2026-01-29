"""Player agent with LLM integration and tool-based interaction."""

import asyncio
import json
from datetime import datetime
from typing import Optional, Callable, Awaitable

from .mafia import (
    Player, Role, Phase, ChatMessage, PublicMessage, ActionType, Action,
    GameEvent, EventType
)
from .llm_client import (
    CachedLLMClient, get_llm_client, LLMResponse,
    DAY_TOOLS, VOTING_TOOLS, NIGHT_TOOLS,
    MAFIA_DISCUSSION_TOOLS, MAFIA_VOTE_TOOLS
)
from .events import get_broadcaster


# Base instruction for reasoning summary
REASONING_INSTRUCTION = """
IMPORTANT: Before taking any action, briefly summarize your strategic thinking in 1-2 sentences. Example: "I suspect Bob because he was defensive. I'll push back on Alice's accusation." Then use tools to act."""

# System prompts for each role
ROLE_SYSTEM_PROMPTS = {
    Role.MAFIA: """You are playing a game of Mafia. You are a MAFIA member.

Your goal is to eliminate all town members without being discovered. During the night, you can kill one player. During the day, you must blend in with the town and avoid suspicion while subtly casting doubt on others.

Strategy tips:
- Act like a regular townsperson during discussions
- Don't be too aggressive in accusing others early on
- Support accusations against other players to seem helpful
- If accused, defend yourself calmly without being defensive
- Coordinate with your fellow mafia if possible (you know who they are)

Remember: You win when mafia equals or outnumbers town.
""" + REASONING_INSTRUCTION,

    Role.DOCTOR: """You are playing a game of Mafia. You are the DOCTOR.

Your goal is to help the town win by saving players from the mafia's night kills. Each night, you can choose one player to protect - if the mafia tries to kill them, they will survive.

IMPORTANT RULE: You CANNOT protect the same player two nights in a row. You must choose a different target each night.

CRITICAL STRATEGY:
- If someone claims to be Detective and reveals a Mafia member, PROTECT THEM! Mafia will try to kill the Detective.
- Players who are leading town discussion or making strong accusations are likely mafia targets.
- Don't reveal you're the Doctor unless absolutely necessary - you're a high-value target.
- If you're protecting yourself, you can't protect others - use this sparingly.

Remember: Town wins when all mafia are eliminated. Keeping the Detective alive is crucial!
""" + REASONING_INSTRUCTION,

    Role.DETECTIVE: """You are playing a game of Mafia. You are the DETECTIVE.

Your goal is to help the town win by investigating players. Each night, you can investigate one player to learn if they are MAFIA or not.

CRITICAL STRATEGY:
- When you find a Mafia member, SHARE THIS IMMEDIATELY and CLEARLY. Say something like "I am the Detective. I investigated [NAME] and they ARE MAFIA."
- Repeat your findings multiple times if needed to ensure town hears you.
- Push hard to get town to vote with you. Coordinate the lynch!
- Yes, mafia may target you after you reveal, but getting a confirmed mafia lynched is worth the risk.
- If you find someone is NOT mafia, that's also useful info - share it to build trust.

Investigation strategy:
- Investigate the most suspicious players first
- If accused, reveal your role and findings to defend yourself

Remember: Town wins when all mafia are eliminated. Your information is the town's best weapon!
""" + REASONING_INSTRUCTION,

    Role.TOWN: """You are playing a game of Mafia. You are a regular TOWN member.

Your goal is to identify and vote out the mafia members. You have no special abilities, but your vote and voice are powerful tools.

CRITICAL STRATEGY - How to evaluate claims:
1. DETECTIVE CLAIMS are extremely valuable. If someone claims to be Detective and says they found a Mafia member, this is STRONG EVIDENCE. Follow up on it!
2. Be suspicious of people who vote AGAINST someone the Detective accused - they might be protecting Mafia.
3. Mafia will try to discredit the Detective or lynch them. Don't fall for it.
4. Coordinate votes with other town members - scattered votes let Mafia win.

Other tips:
- Pay attention to who is being defensive or evasive
- Look for inconsistencies in people's stories
- If the Detective found someone, VOTE FOR THAT PERSON unless you have strong contrary evidence

Remember: Town wins when all mafia are eliminated. Coordinated voting is essential!
""" + REASONING_INSTRUCTION
}


def get_day_phase_prompt(
    player: Player, 
    living_players: list[str], 
    messages_so_far: list[PublicMessage],
    last_seen_count: int = 0,
    is_first_prompt: bool = True
) -> str:
    """Generate the prompt for the day discussion phase.
    
    Args:
        player: The player receiving this prompt
        living_players: List of living player names
        messages_so_far: All messages in the chat so far
        last_seen_count: How many messages the player has already seen
        is_first_prompt: Whether this is the first prompt of the day phase
    """
    other_players = [p for p in living_players if p != player.name]
    
    # Build XML-structured message view
    if is_first_prompt:
        # First prompt of the day - show context and any messages
        if messages_so_far:
            messages_xml = "\n<town_square>\n"
            for msg in messages_so_far:
                messages_xml += f'  <message sender="{msg.sender_name}" time="{msg.timestamp.strftime("%H:%M:%S")}">{msg.content}</message>\n'
            messages_xml += "</town_square>"
        else:
            messages_xml = "\n<town_square>\n  <!-- No messages yet. You may be the first to speak! -->\n</town_square>"
        
        return f"""It is now DAYTIME. You have 5 minutes to discuss with the other players.

<game_state>
  <living_players>{', '.join(living_players)}</living_players>
  <you>{player.name}</you>
</game_state>
{messages_xml}

Briefly summarize your current thinking (1-2 sentences), then use send_message to speak or wait_for_messages to listen."""
    
    else:
        # Subsequent prompt - only show new messages since last seen
        new_messages = messages_so_far[last_seen_count:]
        
        if new_messages:
            messages_xml = "\n<new_messages>\n"
            for msg in new_messages:
                messages_xml += f'  <message sender="{msg.sender_name}" time="{msg.timestamp.strftime("%H:%M:%S")}">{msg.content}</message>\n'
            messages_xml += "</new_messages>"
        else:
            messages_xml = "\n<new_messages>\n  <!-- No new messages -->\n</new_messages>"
        
        return f"""Discussion continues. {len(new_messages)} new message(s) since you last checked.
{messages_xml}

Briefly summarize your thinking, then use send_message to speak or wait_for_messages to listen."""


def get_voting_phase_prompt(player: Player, living_players: list[str], messages: Optional[list[PublicMessage]] = None) -> str:
    """Generate the prompt for the voting phase."""
    other_players = [p for p in living_players if p != player.name]
    
    # Summarize key claims from discussion
    claims_summary = ""
    if messages:
        claims_summary = "\n<discussion_summary>\n"
        for msg in messages[-10:]:  # Last 10 messages
            claims_summary += f"  {msg.sender_name}: {msg.content[:150]}{'...' if len(msg.content) > 150 else ''}\n"
        claims_summary += "</discussion_summary>\n"
    
    return f"""<phase>VOTING - Choose who to lynch</phase>

<living_players>{', '.join(living_players)}</living_players>
<vote_options>{', '.join(other_players)}</vote_options>
{claims_summary}
IMPORTANT: Review what was said during discussion. Did anyone claim to be Detective and identify a Mafia member? If so, strongly consider voting for the accused unless you have good reason not to. Scattered votes help Mafia win!

Summarize your reasoning (who made accusations? who was accused? what evidence?), then use cast_vote."""


def get_night_phase_prompt(player: Player, living_players: list[str], last_protected: Optional[str] = None) -> str:
    """Generate the prompt for the night phase based on role."""
    other_players = [p for p in living_players if p != player.name]
    
    if player.role == Role.MAFIA:
        return f"""<phase>NIGHT</phase>
<your_role>MAFIA</your_role>
<action>Choose someone to kill</action>
<targets>{', '.join(other_players)}</targets>

Briefly explain your target choice (1-2 sentences), then use night_action."""
    
    elif player.role == Role.DOCTOR:
        # Filter out last protected player (consecutive protection rule)
        valid_targets = living_players
        restriction_note = ""
        if last_protected and last_protected in living_players:
            valid_targets = [p for p in living_players if p != last_protected]
            restriction_note = f"\n<restriction>You protected {last_protected} last night. You CANNOT protect them again tonight.</restriction>"
        
        return f"""<phase>NIGHT</phase>
<your_role>DOCTOR</your_role>
<action>Choose someone to protect</action>
<targets>{', '.join(valid_targets)}</targets>{restriction_note}

Briefly explain who you'll protect and why (1-2 sentences), then use night_action."""
    
    elif player.role == Role.DETECTIVE:
        return f"""<phase>NIGHT</phase>
<your_role>DETECTIVE</your_role>
<action>Choose someone to investigate</action>
<targets>{', '.join(other_players)}</targets>

Briefly explain who you'll investigate and why (1-2 sentences), then use night_action."""
    
    else:
        # Town has no night action
        return "<phase>NIGHT</phase>\nAs a regular town member, you have no night action. Wait for morning."


class PlayerAgent:
    """An AI-controlled player in the Mafia game."""
    
    def __init__(
        self,
        player: Player,
        llm_client: CachedLLMClient,
        message_queue: asyncio.Queue,  # Queue to receive new messages
        send_message_callback: Callable[[str, str], Awaitable[None]],  # (player_name, content) -> send
        new_message_event: asyncio.Event,  # Event triggered when new message arrives
    ):
        self.player = player
        self.llm_client = llm_client
        self.message_queue = message_queue
        self.send_message_callback = send_message_callback
        self.new_message_event = new_message_event
        self._last_seen_message_id: Optional[str] = None
        self._running = False
        
        # Track last protected player (for doctor's consecutive protection rule)
        self._last_protected: Optional[str] = None
        
        # Mafia coordination callbacks (set during mafia phase)
        self._mafia_send_callback: Optional[Callable[[str, str], Awaitable[None]]] = None
        self._mafia_message_event: Optional[asyncio.Event] = None
        
        # Initialize chat history with system prompt
        self.player.chat_history = [
            ChatMessage(
                role="system",
                content=ROLE_SYSTEM_PROMPTS[player.role]
            )
        ]
    
    async def _call_llm(self, prompt: str, tools: list[dict]) -> LLMResponse:
        """Make an LLM call with the current chat history."""
        # Add the prompt as a user message
        self.player.chat_history.append(ChatMessage(role="user", content=prompt))
        
        print(f"  [{self.player.name}] Calling LLM...")
        response = await self.llm_client.chat_completion(
            model=self.player.model,
            messages=self.player.chat_history,
            tools=tools if tools else None,
            temperature=1,
        )
        
        # Add assistant response to history
        assistant_msg = ChatMessage(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls
        )
        self.player.chat_history.append(assistant_msg)
        
        # Broadcast thought event for web viewer
        broadcaster = get_broadcaster()
        await broadcaster.broadcast(
            GameEvent(
                event_type=EventType.PLAYER_THOUGHT,
                data={
                    "player_name": self.player.name,
                    "prompt": prompt,
                    "response": response.content,
                    "tool_calls": response.tool_calls
                }
            ),
            channels=[f"player_{self.player.name}"]
        )
        
        return response
    
    async def _handle_tool_call(self, tool_call: dict) -> str:
        """Execute a tool call and return the result."""
        func_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        
        if func_name == "send_message":
            content = args["content"]
            # Timestamp is NOW (after inference completed)
            await self.send_message_callback(self.player.name, content)
            return f"Message sent: {content}"
        
        elif func_name == "wait_for_messages":
            # Wait for the new message event
            print(f"  [{self.player.name}] Waiting for new messages...")
            await self.new_message_event.wait()
            return "New messages have arrived. Check the chat."
        
        elif func_name == "cast_vote":
            target = args["target"]
            return f"VOTE:{target}"  # Special return value for voting
        
        elif func_name == "night_action":
            target = args["target"]
            return f"NIGHT_ACTION:{target}"  # Special return value for night action
        
        elif func_name == "mafia_chat":
            content = args["content"]
            if self._mafia_send_callback:
                await self._mafia_send_callback(self.player.name, content)
            return f"Mafia message sent: {content}"
        
        elif func_name == "mafia_kill_vote":
            target = args["target"]
            return f"MAFIA_VOTE:{target}"  # Special return value for mafia kill vote
        
        else:
            return f"Unknown tool: {func_name}"
    
    async def run_day_phase(
        self,
        living_players: list[str],
        get_messages: Callable[[], list[PublicMessage]],
        phase_end_event: asyncio.Event,
    ) -> None:
        """Run the player's day phase loop."""
        self._running = True
        last_seen_count = 0
        is_first_prompt = True
        print(f"[{self.player.name}] Starting day phase")
        
        while self._running and not phase_end_event.is_set():
            # Snapshot current messages
            current_messages = get_messages()
            
            # Generate prompt with current state (only show new messages after first prompt)
            prompt = get_day_phase_prompt(
                self.player, 
                living_players, 
                current_messages,
                last_seen_count=last_seen_count,
                is_first_prompt=is_first_prompt
            )
            
            # Update last seen count BEFORE the LLM call (blind writing)
            last_seen_count = len(current_messages)
            is_first_prompt = False
            
            # Call LLM (player is "blind" during this call)
            try:
                response = await asyncio.wait_for(
                    self._call_llm(prompt, DAY_TOOLS),
                    timeout=30.0  # 30 second timeout for LLM call
                )
            except asyncio.TimeoutError:
                print(f"  [{self.player.name}] LLM call timed out, retrying...")
                continue
            
            if phase_end_event.is_set():
                break
            
            # Handle tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    result = await self._handle_tool_call(tool_call)
                    
                    # Add tool result to history
                    self.player.chat_history.append(ChatMessage(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call["id"]
                    ))
                    
                    if phase_end_event.is_set():
                        break
            else:
                # No tool calls - add a small delay and continue
                await asyncio.sleep(1.0)
        
        self._running = False
        print(f"[{self.player.name}] Day phase ended")
    
    async def run_voting_phase(self, living_players: list[str], messages: Optional[list[PublicMessage]] = None) -> Optional[str]:
        """Run the voting phase and return the player's vote."""
        print(f"[{self.player.name}] Starting voting phase")
        
        prompt = get_voting_phase_prompt(self.player, living_players, messages)
        response = await self._call_llm(prompt, VOTING_TOOLS)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = await self._handle_tool_call(tool_call)
                
                # Add tool result to history
                self.player.chat_history.append(ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
                
                if result.startswith("VOTE:"):
                    vote_target = result[5:]
                    print(f"[{self.player.name}] Voted for {vote_target}")
                    return vote_target
        
        print(f"[{self.player.name}] Failed to cast vote")
        return None
    
    async def run_night_phase(self, living_players: list[str]) -> Optional[str]:
        """Run the night phase and return the night action target (if any)."""
        print(f"[{self.player.name}] Starting night phase")
        
        # Town members have no night action
        if self.player.role == Role.TOWN:
            self.player.chat_history.append(ChatMessage(
                role="user",
                content="It is night. You have no special abilities, so you wait for morning."
            ))
            self.player.chat_history.append(ChatMessage(
                role="assistant", 
                content="I'll wait and see what happens tonight. Hopefully the doctor protects the right person."
            ))
            return None
        
        # Pass last_protected to doctor's prompt
        last_protected = self._last_protected if self.player.role == Role.DOCTOR else None
        prompt = get_night_phase_prompt(self.player, living_players, last_protected)
        response = await self._call_llm(prompt, NIGHT_TOOLS)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = await self._handle_tool_call(tool_call)
                
                # Add tool result to history
                self.player.chat_history.append(ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
                
                if result.startswith("NIGHT_ACTION:"):
                    target = result[13:]
                    print(f"[{self.player.name}] Night action target: {target}")
                    
                    # Update last protected for doctor
                    if self.player.role == Role.DOCTOR:
                        self._last_protected = target
                    
                    return target
        
        print(f"[{self.player.name}] No night action taken")
        return None
    
    def add_game_event(self, event_description: str) -> None:
        """Add a game event to the player's chat history."""
        self.player.chat_history.append(ChatMessage(
            role="user",
            content=f"[GAME EVENT] {event_description}"
        ))
    
    def set_mafia_callbacks(
        self,
        send_callback: Callable[[str, str], Awaitable[None]],
        message_event: asyncio.Event
    ) -> None:
        """Set the mafia coordination callbacks."""
        self._mafia_send_callback = send_callback
        self._mafia_message_event = message_event
    
    async def run_mafia_discussion(
        self,
        living_players: list[str],
        other_mafia: list[str],
        get_mafia_messages: Callable[[], list[dict]],
        phase_end_event: asyncio.Event,
    ) -> None:
        """Run the mafia discussion phase."""
        self._running = True
        last_seen_count = 0
        print(f"[{self.player.name}] Starting mafia discussion")
        
        targets = [p for p in living_players if p != self.player.name and p not in other_mafia]
        
        while self._running and not phase_end_event.is_set():
            # Get current mafia messages
            mafia_messages = get_mafia_messages()
            new_messages = mafia_messages[last_seen_count:]
            last_seen_count = len(mafia_messages)
            
            # Build prompt
            messages_xml = ""
            if new_messages:
                messages_xml = "\n<new_mafia_messages>\n"
                for msg in new_messages:
                    messages_xml += f"  <message sender='{msg['sender']}'>{msg['content']}</message>\n"
                messages_xml += "</new_mafia_messages>"
            
            prompt = f"""<phase>NIGHT - MAFIA COORDINATION</phase>
<your_team>MAFIA</your_team>
<teammates>{', '.join(other_mafia) if other_mafia else 'You are the only mafia'}</teammates>
<potential_targets>{', '.join(targets)}</potential_targets>
{messages_xml}

Discuss with your mafia partners who to kill tonight. Use mafia_chat to communicate privately, or wait_for_messages to see what your partners say."""
            
            try:
                response = await asyncio.wait_for(
                    self._call_llm(prompt, MAFIA_DISCUSSION_TOOLS),
                    timeout=30.0
                )
                
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        result = await self._handle_tool_call(tool_call)
                        self.player.chat_history.append(ChatMessage(
                            role="tool",
                            content=result,
                            tool_call_id=tool_call["id"]
                        ))
                        
                        if result == "New messages have arrived. Check the chat.":
                            # Continue discussion loop
                            continue
                
            except asyncio.TimeoutError:
                print(f"  [{self.player.name}] LLM call timed out, ending discussion")
                break
            except asyncio.CancelledError:
                break
        
        self._running = False
        print(f"[{self.player.name}] Mafia discussion ended")
    
    async def run_mafia_vote(self, living_players: list[str], other_mafia: list[str]) -> Optional[str]:
        """Vote on who the mafia should kill."""
        print(f"[{self.player.name}] Voting on mafia kill target")
        
        targets = [p for p in living_players if p != self.player.name and p not in other_mafia]
        
        prompt = f"""<phase>NIGHT - MAFIA KILL VOTE</phase>
<action>Vote for who to kill</action>
<targets>{', '.join(targets)}</targets>

The discussion is over. Now vote for who the mafia should kill using mafia_kill_vote."""
        
        response = await self._call_llm(prompt, MAFIA_VOTE_TOOLS)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = await self._handle_tool_call(tool_call)
                self.player.chat_history.append(ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
                
                if result.startswith("MAFIA_VOTE:"):
                    target = result[11:]
                    print(f"[{self.player.name}] Mafia kill vote: {target}")
                    return target
        
        print(f"[{self.player.name}] No mafia vote cast")
        return None
    
    def stop(self) -> None:
        """Stop the player's current phase loop."""
        self._running = False
