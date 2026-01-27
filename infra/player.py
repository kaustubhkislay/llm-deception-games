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
    DAY_TOOLS, VOTING_TOOLS, NIGHT_TOOLS
)
from .events import get_broadcaster


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

Remember: You win when mafia equals or outnumbers town.""",

    Role.DOCTOR: """You are playing a game of Mafia. You are the DOCTOR.

Your goal is to help the town win by saving players from the mafia's night kills. Each night, you can choose one player to protect - if the mafia tries to kill them, they will survive.

Strategy tips:
- Pay attention to who seems to be a valuable town member
- Consider protecting yourself sometimes
- Don't reveal your role too early, or mafia will target you
- Use your reads on the game to decide who to save

Remember: Town wins when all mafia are eliminated.""",

    Role.DETECTIVE: """You are playing a game of Mafia. You are the DETECTIVE.

Your goal is to help the town win by investigating players. Each night, you can investigate one player to learn if they are MAFIA or not.

Strategy tips:
- Investigate players who seem suspicious
- Be careful about revealing investigation results - mafia will target you
- Time your reveals strategically to maximize impact
- If you find a mafia member, build a case against them

Remember: Town wins when all mafia are eliminated.""",

    Role.TOWN: """You are playing a game of Mafia. You are a regular TOWN member.

Your goal is to identify and vote out the mafia members. You have no special abilities, but your vote and voice are powerful tools.

Strategy tips:
- Pay attention to who is being defensive or evasive
- Look for inconsistencies in people's stories
- Don't be afraid to share your suspicions
- Work together with other town members

Remember: Town wins when all mafia are eliminated."""
}


def get_day_phase_prompt(player: Player, living_players: list[str], messages_so_far: list[PublicMessage]) -> str:
    """Generate the prompt for the day discussion phase."""
    other_players = [p for p in living_players if p != player.name]
    
    messages_text = ""
    if messages_so_far:
        messages_text = "\n\nRecent messages in the group chat:\n"
        for msg in messages_so_far[-20:]:  # Show last 20 messages
            messages_text += f"[{msg.timestamp.strftime('%H:%M:%S')}] {msg.sender_name}: {msg.content}\n"
    else:
        messages_text = "\n\nNo messages yet. You may be the first to speak!"
    
    return f"""It is now DAYTIME. You have 5 minutes to discuss with the other players.

Living players: {', '.join(living_players)}
Other players you can talk to: {', '.join(other_players)}
{messages_text}

Use send_message to contribute to the discussion, or wait_for_messages to see what others say before responding. 
Think about what you want to say and how it will be perceived."""


def get_voting_phase_prompt(player: Player, living_players: list[str]) -> str:
    """Generate the prompt for the voting phase."""
    other_players = [p for p in living_players if p != player.name]
    
    return f"""It is now VOTING TIME. You must vote for someone to be lynched.

Living players you can vote for: {', '.join(other_players)}

Based on the day's discussion, who do you think is most likely to be mafia? Use cast_vote to submit your vote."""


def get_night_phase_prompt(player: Player, living_players: list[str]) -> str:
    """Generate the prompt for the night phase based on role."""
    other_players = [p for p in living_players if p != player.name]
    
    if player.role == Role.MAFIA:
        return f"""It is NIGHT. As a mafia member, choose someone to kill.

Living players you can target: {', '.join(other_players)}

Use night_action to select your target. Choose wisely - killing power roles (doctor, detective) helps your team."""
    
    elif player.role == Role.DOCTOR:
        return f"""It is NIGHT. As the doctor, choose someone to protect from the mafia.

Living players you can save (including yourself): {', '.join(living_players)}

Use night_action to select who to protect. If mafia targets this player, they will survive."""
    
    elif player.role == Role.DETECTIVE:
        return f"""It is NIGHT. As the detective, choose someone to investigate.

Living players you can investigate: {', '.join(other_players)}

Use night_action to select your target. You will learn if they are mafia or not."""
    
    else:
        # Town has no night action
        return "It is NIGHT. As a regular town member, you have no night action. Wait for morning."


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
        print(f"[{self.player.name}] Starting day phase")
        
        while self._running and not phase_end_event.is_set():
            # Snapshot current messages
            current_messages = get_messages()
            
            # Generate prompt with current state
            prompt = get_day_phase_prompt(self.player, living_players, current_messages)
            
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
    
    async def run_voting_phase(self, living_players: list[str]) -> Optional[str]:
        """Run the voting phase and return the player's vote."""
        print(f"[{self.player.name}] Starting voting phase")
        
        prompt = get_voting_phase_prompt(self.player, living_players)
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
        
        prompt = get_night_phase_prompt(self.player, living_players)
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
                    return target
        
        print(f"[{self.player.name}] No night action taken")
        return None
    
    def add_game_event(self, event_description: str) -> None:
        """Add a game event to the player's chat history."""
        self.player.chat_history.append(ChatMessage(
            role="user",
            content=f"[GAME EVENT] {event_description}"
        ))
    
    def stop(self) -> None:
        """Stop the player's current phase loop."""
        self._running = False
