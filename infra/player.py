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


# Comprehensive game rules - shared by all players
GAME_RULES = """
=== MAFIA GAME RULES ===

OVERVIEW:
This is a social deduction game. There are two teams: Town and Mafia. Town wins by eliminating all Mafia. Mafia wins when they equal or outnumber the Town.

ROLES:
- MAFIA (2 players): Know each other. Can kill one player per night. Win by outnumbering Town.
- DETECTIVE (1 player): Town-aligned. Can investigate one player per night to learn if they are Mafia.
- DOCTOR (1 player): Town-aligned. Can protect one player per night from being killed. Cannot protect the same player two nights in a row.
- TOWN (remaining players): No special abilities. Win by eliminating all Mafia.

GAME PHASES:

1. NIGHT PHASE:
   - Order of actions: Detective investigates → Doctor protects → Mafia kills
   - Mafia members have private discussion time to coordinate their kill
   - If there's a tie in Mafia kill votes, the tie is broken randomly
   - Mafia can vote not to kill anybody
   - Town members with no special role simply sleep through the night

2. DAY PHASE (Discussion):
   - All living players discuss openly in the town square
   - Discussion has a time limit (announced at start)
   - No votes can be cast or locked in during the discussion phase

3. VOTING PHASE:
   - After discussion, each player votes simultaneously (blind voting)
   - Options: vote for any living player OR vote "no_lynch"
   - A player is lynched only if they receive a STRICT MAJORITY (more than half of living players)
   - Ties or no majority = nobody is lynched
   - When someone is lynched, only their alignment (Mafia or not Mafia) is revealed, not their exact role (Detective, Doctor, or Town)

INFORMATION:
   - You can see who is alive and who has died
   - Previous day's votes are shown at the start of each new day
   - When a player is killed, their alignment (Mafia or not Mafia) is revealed, but not their exact role (Detective, Doctor, or Town)
"""

# Brief instruction for action formatting
REASONING_INSTRUCTION = """
Before acting, briefly state your thinking (1-2 sentences). Then use the appropriate tool."""

# System prompts for each role - informative, not prescriptive
ROLE_SYSTEM_PROMPTS = {
    Role.MAFIA: GAME_RULES + """
=== YOUR ROLE: MAFIA ===

You are a Mafia member. Your teammates are revealed to you at night.

Your abilities:
- During night: Participate in private Mafia chat and vote on a kill target
- During day: Participate in town discussion like any other player

Your win condition: Mafia wins when Mafia players equal or outnumber Town players.

Things to consider:
- Town doesn't know who the Mafia members are
- The Detective can investigate you and discover you're Mafia
- The Doctor might protect your kill target
- Dead players only reveal alignment (Mafia/not Mafia), not exact role
- Coordinating with fellow Mafia during night is important

""" + REASONING_INSTRUCTION,

    Role.DOCTOR: GAME_RULES + """
=== YOUR ROLE: DOCTOR ===

You are the Doctor, aligned with Town.

Your abilities:
- During night: Choose one player to protect from being killed
- Restriction: You CANNOT protect the same player two consecutive nights

Your win condition: Town wins when all Mafia are eliminated.

Things to consider:
- If you protect the Mafia's target, that player survives the night
- You can protect yourself, but then you can't protect others that night
- The Mafia doesn't know who the Doctor is unless you reveal yourself
- If the Detective claims publicly, the Mafia may target them

""" + REASONING_INSTRUCTION,

    Role.DETECTIVE: GAME_RULES + """
=== YOUR ROLE: DETECTIVE ===

You are the Detective, aligned with Town.

Your abilities:
- During night: Investigate one player to learn if they are MAFIA or NOT MAFIA

Your win condition: Town wins when all Mafia are eliminated.

Things to consider:
- You are the only source of confirmed information about player alignments
- Revealing your findings publicly gives Town information but also reveals you to Mafia
- False Detective claims are possible - Mafia might claim to be Detective

""" + REASONING_INSTRUCTION,

    Role.TOWN: GAME_RULES + """
=== YOUR ROLE: TOWN ===

You are a regular Town member with no special abilities.

Your abilities:
- During day: Participate in discussion and share your observations
- During voting: Cast your vote for who should be lynched

Your win condition: Town wins when all Mafia are eliminated.

Things to consider:
- Your vote is your main power - coordinated voting is more effective than scattered votes
- Pay attention to claims made during discussion (role claims, Mafia accusations)

""" + REASONING_INSTRUCTION
}


def get_day_phase_prompt(
    player: Player, 
    living_players: list[str], 
    messages_so_far: list[PublicMessage],
    last_seen_count: int = 0,
    is_first_prompt: bool = True,
    previous_votes: Optional[dict[str, str]] = None,
    phase_start_time: Optional[datetime] = None,
    phase_duration_seconds: int = 300,
    day_number: int = 1,
    current_draft: Optional[str] = None
) -> str:
    """Generate the prompt for the day discussion phase.
    
    Args:
        player: The player receiving this prompt
        living_players: List of living player names
        messages_so_far: All messages in the chat so far
        last_seen_count: How many messages the player has already seen
        is_first_prompt: Whether this is the first prompt of the day phase
        previous_votes: Dict of voter -> target from previous day's vote (if any)
        phase_start_time: When this phase started (for relative timestamps)
        phase_duration_seconds: How long this phase lasts
        current_draft: The player's current unsent draft message, if any
    """
    other_players = [p for p in living_players if p != player.name]
    
    # Build previous votes summary if available
    previous_votes_xml = ""
    if previous_votes and is_first_prompt:
        previous_votes_xml = "\n<previous_day_votes>\n"
        for voter, target in sorted(previous_votes.items()):
            previous_votes_xml += f'  <vote voter="{voter}" target="{target}"/>\n'
        previous_votes_xml += "</previous_day_votes>\n"
    
    # Build draft indicator
    draft_xml = ""
    if current_draft:
        draft_xml = f'\n<your_draft>"{current_draft}"</your_draft>\n'
    
    # Build XML-structured message view
    if is_first_prompt:
        # First prompt of the day - show context and any messages
        if messages_so_far:
            messages_xml = "\n<town_square>\n"
            for msg in messages_so_far:
                messages_xml += f'  <message sender="{msg.sender_name}">{msg.content}</message>\n'
            messages_xml += "</town_square>"
        else:
            messages_xml = "\n<town_square>\n  <!-- No messages yet. You may be the first to speak! -->\n</town_square>"
        
        duration_mins = phase_duration_seconds // 60
        return f"""<phase>DAY {day_number} - Discussion ({duration_mins} minutes)</phase>{previous_votes_xml}{draft_xml}

<game_state>
  <living_players>{', '.join(living_players)}</living_players>
  <you>{player.name}</you>
</game_state>
{messages_xml}

Tools: draft_message to prepare a message, send_message to post your draft, wait_for_messages to listen."""
    
    else:
        # Subsequent prompt - only show new messages since last seen
        new_messages = messages_so_far[last_seen_count:]
        
        if new_messages:
            messages_xml = "\n<new_messages>\n"
            for msg in new_messages:
                messages_xml += f'  <message sender="{msg.sender_name}">{msg.content}</message>\n'
            messages_xml += "</new_messages>"
        else:
            messages_xml = "\n<new_messages>\n  <!-- No new messages -->\n</new_messages>"
        
        # Calculate time remaining
        time_remaining = ""
        if phase_start_time:
            elapsed = (datetime.now() - phase_start_time).total_seconds()
            remaining = max(0, phase_duration_seconds - elapsed)
            mins, secs = divmod(int(remaining), 60)
            time_remaining = f" (~{mins}:{secs:02d} remaining)"
        
        return f"""<phase>DAY {day_number} - Discussion continues{time_remaining}</phase>{draft_xml}

{len(new_messages)} new message(s) since you last checked.
{messages_xml}

Tools: draft_message to prepare, send_message to post your draft, wait_for_messages to listen."""


def get_voting_phase_prompt(player: Player, living_players: list[str], messages: Optional[list[PublicMessage]] = None, day_number: int = 1) -> str:
    """Generate the prompt for the voting phase."""
    other_players = [p for p in living_players if p != player.name]
    
    # Summarize key claims from discussion
    claims_summary = ""
    if messages:
        claims_summary = "\n<discussion_summary>\n"
        for msg in messages[-10:]:  # Last 10 messages
            claims_summary += f"  {msg.sender_name}: {msg.content[:150]}{'...' if len(msg.content) > 150 else ''}\n"
        claims_summary += "</discussion_summary>\n"
    
    vote_options = other_players + ["no_lynch"]
    
    return f"""<phase>DAY {day_number} - VOTING</phase>

<living_players>{', '.join(living_players)}</living_players>
<vote_options>{', '.join(vote_options)}</vote_options>
{claims_summary}
Reminder: A strict majority (more than half of living players) is needed to lynch someone. You can vote for any living player or "no_lynch".

Consider: What claims were made? What evidence was presented? Who seems suspicious or trustworthy?

Use cast_vote to submit your vote."""


def get_night_phase_prompt(player: Player, living_players: list[str], last_protected: Optional[str] = None, night_number: int = 1) -> str:
    """Generate the prompt for the night phase based on role."""
    other_players = [p for p in living_players if p != player.name]
    
    if player.role == Role.MAFIA:
        return f"""<phase>NIGHT {night_number}</phase>
<your_role>MAFIA</your_role>
<action>Choose a kill target</action>
<targets>{', '.join(other_players)}</targets>

Use night_action to select your target."""
    
    elif player.role == Role.DOCTOR:
        # Filter out last protected player (consecutive protection rule)
        valid_targets = living_players
        restriction_note = ""
        if last_protected and last_protected in living_players:
            valid_targets = [p for p in living_players if p != last_protected]
            restriction_note = f"\n<restriction>You protected {last_protected} last night and cannot protect them again tonight.</restriction>"
        
        return f"""<phase>NIGHT {night_number}</phase>
<your_role>DOCTOR</your_role>
<action>Choose someone to protect</action>
<valid_targets>{', '.join(valid_targets)}</valid_targets>{restriction_note}

Use night_action to select who to protect."""
    
    elif player.role == Role.DETECTIVE:
        return f"""<phase>NIGHT {night_number}</phase>
<your_role>DETECTIVE</your_role>
<action>Choose someone to investigate</action>
<targets>{', '.join(other_players)}</targets>

Use night_action to select who to investigate. You will learn if they are MAFIA or NOT MAFIA."""
    
    else:
        # Town has no night action
        return f"<phase>NIGHT {night_number}</phase>\nYou have no night action. Waiting for morning."


class PlayerAgent:
    """An AI-controlled player in the Mafia game."""
    
    def __init__(
        self,
        player: Player,
        llm_client: CachedLLMClient,
        message_queue: asyncio.Queue,  # Queue to receive new messages
        send_message_callback: Callable[[str, str], Awaitable[None]],  # (player_name, content) -> send
        new_message_event: asyncio.Event,  # Event triggered when new message arrives
        log_thought_callback: Optional[Callable[[dict], None]] = None,  # Callback to log thoughts
    ):
        self.player = player
        self.llm_client = llm_client
        self.message_queue = message_queue
        self.send_message_callback = send_message_callback
        self.new_message_event = new_message_event
        self._log_thought_callback = log_thought_callback
        self._last_seen_message_id: Optional[str] = None
        self._running = False
        
        # Track last protected player (for doctor's consecutive protection rule)
        self._last_protected: Optional[str] = None
        
        # Mafia coordination callbacks (set during mafia phase)
        self._mafia_send_callback: Optional[Callable[[str, str], Awaitable[None]]] = None
        self._mafia_message_event: Optional[asyncio.Event] = None
        self._mafia_intention_callback: Optional[Callable[[str, str], Awaitable[None]]] = None
        
        # Current phase tracking (for thought events)
        self._current_phase: str = "NIGHT"
        self._current_day: int = 1
        
        # Draft message (for two-step send workflow)
        self._current_draft: Optional[str] = None
        self._mafia_draft: Optional[str] = None
        
        # Initialize chat history with system prompt (insert player name before reasoning instruction)
        base_prompt = ROLE_SYSTEM_PROMPTS[player.role]
        # Insert player name before the reasoning instruction
        system_prompt = base_prompt.replace(
            REASONING_INSTRUCTION,
            f"\nYour name is {player.name}.\n" + REASONING_INSTRUCTION
        )
        self.player.chat_history = [
            ChatMessage(
                role="system",
                content=system_prompt
            )
        ]
    
    def _cleanup_dangling_tool_calls(self) -> None:
        """Remove any assistant messages with tool_calls that don't have corresponding tool responses."""
        if not self.player.chat_history:
            return
        
        # Find all tool_call_ids that have responses
        responded_ids = set()
        for msg in self.player.chat_history:
            if msg.role == "tool" and msg.tool_call_id:
                responded_ids.add(msg.tool_call_id)
        
        # Check the last assistant message
        cleaned_history = []
        for i, msg in enumerate(self.player.chat_history):
            if msg.role == "assistant" and msg.tool_calls:
                # Check if all tool calls have responses
                all_responded = all(
                    tc.get("id") in responded_ids 
                    for tc in msg.tool_calls
                )
                if not all_responded:
                    # Add dummy tool responses for missing ones
                    cleaned_history.append(msg)
                    for tc in msg.tool_calls:
                        if tc.get("id") not in responded_ids:
                            cleaned_history.append(ChatMessage(
                                role="tool",
                                content="[Task cancelled]",
                                tool_call_id=tc.get("id")
                            ))
                            responded_ids.add(tc.get("id"))
                else:
                    cleaned_history.append(msg)
            else:
                cleaned_history.append(msg)
        
        self.player.chat_history = cleaned_history
    
    async def _call_llm(self, prompt: str, tools: list[dict]) -> LLMResponse:
        """Make an LLM call with the current chat history."""
        # Clean up any dangling tool calls from cancelled tasks
        self._cleanup_dangling_tool_calls()
        
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
        thought_data = {
            "player_name": self.player.name,
            "prompt": prompt,
            "response": response.content,
            "tool_calls": response.tool_calls,
            "phase": self._current_phase,
            "day_number": self._current_day
        }
        
        broadcaster = get_broadcaster()
        await broadcaster.broadcast(
            GameEvent(
                event_type=EventType.PLAYER_THOUGHT,
                data=thought_data
            ),
            channels=[f"player_{self.player.name}"]
        )
        
        # Also log for replay
        if self._log_thought_callback:
            self._log_thought_callback(thought_data)
        
        return response
    
    async def _handle_tool_call(self, tool_call: dict) -> str:
        """Execute a tool call and return the result."""
        func_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        
        if func_name == "draft_message":
            content = args["content"]
            self._current_draft = content
            return f"Draft saved: \"{content}\"\nYou can now send_message to post it, or draft_message again to revise."
        
        elif func_name == "send_message":
            if self._current_draft is None:
                return "Error: No draft to send. Use draft_message first."
            content = self._current_draft
            self._current_draft = None  # Clear draft after sending
            await self.send_message_callback(self.player.name, content)
            return f"Message sent: {content}"
        
        elif func_name == "wait_for_messages":
            # Wait for the appropriate message event (mafia or public)
            print(f"  [{self.player.name}] Waiting for new messages...")
            if self._mafia_message_event is not None:
                # In mafia coordination phase - wait for mafia messages
                await self._mafia_message_event.wait()
            else:
                # In day phase - wait for public messages
                await self.new_message_event.wait()
            return "New messages have arrived. Check the chat."
        
        elif func_name == "cast_vote":
            target = args["target"]
            return f"VOTE:{target}"  # Special return value for voting
        
        elif func_name == "night_action":
            target = args["target"]
            return f"NIGHT_ACTION:{target}"  # Special return value for night action
        
        elif func_name == "mafia_draft_message":
            content = args["content"]
            self._mafia_draft = content
            return f"Mafia draft saved: \"{content}\"\nUse mafia_send_message to post it, or mafia_draft_message to revise."
        
        elif func_name == "mafia_send_message":
            if self._mafia_draft is None:
                return "Error: No mafia draft to send. Use mafia_draft_message first."
            content = self._mafia_draft
            self._mafia_draft = None  # Clear draft after sending
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
        previous_votes: Optional[dict[str, str]] = None,
        phase_start_time: Optional[datetime] = None,
        phase_duration_seconds: int = 300,
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
                is_first_prompt=is_first_prompt,
                previous_votes=previous_votes if is_first_prompt else None,
                phase_start_time=phase_start_time,
                phase_duration_seconds=phase_duration_seconds,
                day_number=self._current_day,
                current_draft=self._current_draft,
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
    
    async def run_voting_phase(self, living_players: list[str], messages: Optional[list[PublicMessage]] = None) -> tuple[Optional[str], Optional[str]]:
        """Run the voting phase and return (vote_target, reasoning) tuple."""
        print(f"[{self.player.name}] Starting voting phase")
        
        prompt = get_voting_phase_prompt(self.player, living_players, messages, day_number=self._current_day)
        response = await self._call_llm(prompt, VOTING_TOOLS)
        
        reasoning = response.content  # Capture the LLM's reasoning
        
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
                    return (vote_target, reasoning)
        
        print(f"[{self.player.name}] Failed to cast vote")
        return (None, reasoning)
    
    async def run_night_phase(self, living_players: list[str]) -> tuple[Optional[str], Optional[str]]:
        """Run the night phase and return (target, reasoning) tuple."""
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
            return (None, None)
        
        # Pass last_protected to doctor's prompt
        last_protected = self._last_protected if self.player.role == Role.DOCTOR else None
        prompt = get_night_phase_prompt(self.player, living_players, last_protected, night_number=self._current_day)
        response = await self._call_llm(prompt, NIGHT_TOOLS)
        
        reasoning = response.content  # Capture the LLM's reasoning
        
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
                    
                    return (target, reasoning)
        
        print(f"[{self.player.name}] No night action taken")
        return (None, reasoning)
    
    def add_game_event(self, event_description: str) -> None:
        """Add a game event to the player's chat history."""
        self.player.chat_history.append(ChatMessage(
            role="user",
            content=f"[GAME EVENT] {event_description}"
        ))
    
    def set_phase(self, phase: str, day: int) -> None:
        """Set the current game phase for thought tracking."""
        self._current_phase = phase
        self._current_day = day
        # Clear any pending drafts from previous phase
        self._current_draft = None
        self._mafia_draft = None
    
    def set_mafia_callbacks(
        self,
        send_callback: Callable[[str, str], Awaitable[None]],
        message_event: asyncio.Event,
        intention_callback: Optional[Callable[[str, str], Awaitable[None]]] = None
    ) -> None:
        """Set the mafia coordination callbacks."""
        self._mafia_send_callback = send_callback
        self._mafia_message_event = message_event
        self._mafia_intention_callback = intention_callback
    
    async def run_mafia_discussion(
        self,
        living_players: list[str],
        other_mafia: list[str],
        get_mafia_messages: Callable[[], list[dict]],
        phase_end_event: asyncio.Event,
    ) -> Optional[str]:
        """Run the mafia discussion phase. Returns the player's final kill intention."""
        self._running = True
        last_seen_count = 0
        self._current_kill_intention: Optional[str] = None
        print(f"[{self.player.name}] Starting mafia discussion")
        
        targets = [p for p in living_players if p != self.player.name and p not in other_mafia]
        
        while self._running and not phase_end_event.is_set():
            # Get current mafia messages
            mafia_messages = get_mafia_messages()
            new_messages = mafia_messages[last_seen_count:]
            last_seen_count = len(mafia_messages)
            
            # Build prompt with current intentions visible
            messages_xml = ""
            if new_messages:
                messages_xml = "\n<new_mafia_messages>\n"
                for msg in new_messages:
                    if msg.get("type") == "intention":
                        messages_xml += f"  <kill_intention player='{msg['sender']}' target='{msg['target']}'/>\n"
                    else:
                        messages_xml += f"  <message sender='{msg['sender']}'>{msg['content']}</message>\n"
                messages_xml += "</new_mafia_messages>"
            
            current_intention = f"\n<your_current_intention>{self._current_kill_intention or 'Not set'}</your_current_intention>"
            
            # Build draft indicator
            draft_xml = ""
            if self._mafia_draft:
                draft_xml = f'\n<your_draft>"{self._mafia_draft}"</your_draft>'
            
            night_num = self._current_day  # Night number matches day number
            prompt = f"""<phase>NIGHT {night_num} - MAFIA COORDINATION</phase>
<your_team>MAFIA</your_team>
<teammates>{', '.join(other_mafia) if other_mafia else 'You are the only mafia'}</teammates>
<potential_targets>{', '.join(targets)}</potential_targets>{current_intention}{draft_xml}
{messages_xml}

This is your private Mafia channel. Coordinate with your teammates on tonight's kill.

Tools: mafia_draft_message to prepare, mafia_send_message to post, mafia_kill_vote to set target, wait_for_messages to listen.

Reminder: If Mafia kill votes are tied, the tie will be broken randomly. If no Mafia kill votes are cast, nobody will be killed."""
            
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
                        
                        # Handle kill vote - update intention and broadcast
                        if result.startswith("MAFIA_VOTE:"):
                            target = result[11:]
                            self._current_kill_intention = target
                            print(f"  [{self.player.name}] Sets kill intention: {target}")
                            
                            # Broadcast intention to other mafia via callback
                            if self._mafia_intention_callback:
                                await self._mafia_intention_callback(self.player.name, target)
                            
                            # Continue discussion - don't break
                            continue
                        
                        if result == "New messages have arrived. Check the chat.":
                            # Continue discussion loop
                            continue
                
            except asyncio.TimeoutError:
                print(f"  [{self.player.name}] LLM call timed out, continuing...")
                continue
            except asyncio.CancelledError:
                break
        
        self._running = False
        print(f"[{self.player.name}] Mafia discussion ended, intention: {self._current_kill_intention}")
        return self._current_kill_intention
    
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
