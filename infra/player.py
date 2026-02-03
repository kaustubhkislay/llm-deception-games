"""Player agent with LLM integration for One Night Ultimate Werewolf."""

import asyncio
import json
from datetime import datetime
from typing import Optional, Callable

from .onuw import (
    Player, Role, Phase, ChatMessage, PublicMessage, NightAction,
    GameEvent, EventType, get_team, PASSIVE_ROLES
)
from .llm_client import (
    CachedLLMClient, get_llm_client, LLMResponse,
    DAY_TOOLS, VOTING_TOOLS,
    SEER_TOOLS, ROBBER_TOOLS, TROUBLEMAKER_TOOLS, DRUNK_TOOLS,
    WEREWOLF_LONE_TOOLS, WEREWOLF_TEAM_TOOLS, MINION_TOOLS, INSOMNIAC_TOOLS,
    ACKNOWLEDGE_TOOL
)
from .events import get_broadcaster


# Role descriptions - can be filtered based on active roles
ROLE_DESCRIPTIONS = {
    # Village Team
    Role.VILLAGER: "VILLAGER: No special ability. Just deduce and vote.",
    Role.SEER: "SEER: Wake up and look at ONE player's card OR TWO center cards.",
    Role.ROBBER: "ROBBER: Wake up and swap your card with another player's card. You see your NEW card.",
    Role.TROUBLEMAKER: "TROUBLEMAKER: Wake up and swap TWO OTHER players' cards. You don't see the cards.",
    Role.DRUNK: "DRUNK: Wake up and swap your card with a center card. You DON'T see your new card.",
    Role.INSOMNIAC: "INSOMNIAC: Wake up LAST and look at your own card (to see if it was swapped).",
    Role.HUNTER: "HUNTER: No night action. If you die, the player you voted for also dies.",
    # Werewolf Team
    Role.WEREWOLF: "WEREWOLF: Wake up and see other werewolves. If you're the ONLY werewolf, you may look at one center card.",
    Role.MINION: "MINION: Wake up and see who the werewolves are. They don't know you exist. You win with werewolves.",
    # Neutral
    Role.TANNER: "TANNER: No night action. You WIN if you get killed. You LOSE if you survive.",
}

# Night action order for each role (only include if role is active)
NIGHT_ACTION_ORDER_DESCRIPTIONS = {
    Role.WEREWOLF: "Werewolf(s) - see each other, or if alone, may peek at center",
    Role.MINION: "Minion - sees the werewolves",
    Role.SEER: "Seer - looks at one player OR two center cards",
    Role.ROBBER: "Robber - swaps with a player and sees new card",
    Role.TROUBLEMAKER: "Troublemaker - swaps two other players' cards",
    Role.DRUNK: "Drunk - swaps with center (blind)",
    Role.INSOMNIAC: "Insomniac - sees own card",
}

# Canonical night action order
NIGHT_ACTION_ORDER_LIST = [
    Role.WEREWOLF, Role.MINION, Role.SEER, Role.ROBBER, 
    Role.TROUBLEMAKER, Role.DRUNK, Role.INSOMNIAC
]

def get_game_rules(active_roles: list[Role]) -> str:
    """Generate game rules showing only the roles active in this game."""
    active_set = set(active_roles)
    
    # Categorize active roles by team
    village_roles = [r for r in [Role.VILLAGER, Role.SEER, Role.ROBBER, Role.TROUBLEMAKER, 
                                  Role.DRUNK, Role.INSOMNIAC, Role.HUNTER] if r in active_set]
    werewolf_roles = [r for r in [Role.WEREWOLF, Role.MINION] if r in active_set]
    neutral_roles = [r for r in [Role.TANNER] if r in active_set]
    
    # Build role descriptions
    roles_section = "ROLES IN THIS GAME:\n\n"
    
    if village_roles:
        roles_section += "Village Team:\n"
        for r in village_roles:
            roles_section += f"- {ROLE_DESCRIPTIONS[r]}\n"
        roles_section += "\n"
    
    if werewolf_roles:
        roles_section += "Werewolf Team:\n"
        for r in werewolf_roles:
            roles_section += f"- {ROLE_DESCRIPTIONS[r]}\n"
        roles_section += "\n"
    
    if neutral_roles:
        roles_section += "Neutral:\n"
        for r in neutral_roles:
            roles_section += f"- {ROLE_DESCRIPTIONS[r]}\n"
        roles_section += "\n"
    
    # Build night action order (only for active roles with night actions)
    night_order_section = "NIGHT ACTION ORDER:\n"
    order_num = 1
    for role in NIGHT_ACTION_ORDER_LIST:
        if role in active_set and role in NIGHT_ACTION_ORDER_DESCRIPTIONS:
            night_order_section += f"{order_num}. {NIGHT_ACTION_ORDER_DESCRIPTIONS[role]}\n"
            order_num += 1
    
    # Build win conditions (adjust based on active roles)
    win_conditions = """WIN CONDITIONS:
- VILLAGE TEAM wins if at least one Werewolf player is killed
- WEREWOLF TEAM wins if no Werewolf player is killed
- If there are NO werewolves among players (all in center), Village wins only if NO ONE is killed"""
    
    if Role.TANNER in active_set:
        win_conditions += "\n- TANNER wins if the Tanner is killed (this overrides other conditions)"
    
    # Build strategy notes (adjust based on active roles)
    strategy_notes = "\nIMPORTANT STRATEGY NOTES:\n"
    strategy_notes += "- Your starting role determines your night action, but your FINAL role (after swaps) determines your team for winning\n"
    
    if Role.ROBBER in active_set:
        strategy_notes += "- The Robber knows their new role, but may have been swapped again by Troublemaker\n"
    if Role.DRUNK in active_set:
        strategy_notes += "- The Drunk doesn't know what they became\n"
    strategy_notes += "- Claims during day discussion may be true, false, or outdated (due to swaps)\n"
    strategy_notes += "- Pay attention to what information each role would and wouldn't have\n"
    
    return f"""=== ONE NIGHT ULTIMATE WEREWOLF RULES ===

OVERVIEW:
This is a single-night social deduction game. Each player is dealt a secret role card. During the night, players with special abilities wake up in a specific order and take actions. Some actions can SWAP cards - meaning your role might change without you knowing! After the night, there is one discussion period and one vote. The player(s) with the most votes die.

{win_conditions}

{roles_section}
{night_order_section}
{strategy_notes}"""

# Brief instruction for action formatting
REASONING_INSTRUCTION = """
Before acting, briefly state your thinking (1-2 sentences). Then use the appropriate tool."""


def get_role_system_prompt(role: Role, player_name: str, active_roles: list[Role]) -> str:
    """Generate the system prompt for a player based on their starting role."""
    
    role_specific = {
        Role.VILLAGER: """
=== YOUR STARTING ROLE: VILLAGER ===

You have no special night ability. During the night, you simply sleep.
Your goal is to help the village identify and vote out a werewolf.

Remember: You ARE on the village team, but someone might have swapped your card during the night!
""",
        Role.SEER: """
=== YOUR STARTING ROLE: SEER ===

During the night, you will wake up and choose ONE of these actions:
- Look at ONE other player's card to learn their role
- Look at TWO center cards to learn what roles are NOT in play

This information can be powerful, but be careful - the Robber or Troublemaker might have swapped cards AFTER you looked!
""",
        Role.ROBBER: """
=== YOUR STARTING ROLE: ROBBER ===

During the night, you will wake up and:
1. Choose another player to rob
2. Swap your card with theirs
3. Look at your NEW card (your new role)

Important: After robbing, you become whatever role you stole! Your NEW role determines which team you're on.
But the Troublemaker acts AFTER you, so your card might be swapped again without you knowing!
""",
        Role.TROUBLEMAKER: """
=== YOUR STARTING ROLE: TROUBLEMAKER ===

During the night, you will wake up and:
1. Choose TWO OTHER players (not yourself)
2. Swap their cards with each other
3. You do NOT see what roles they have

This creates confusion since those players won't know they've been swapped!
""",
        Role.DRUNK: """
=== YOUR STARTING ROLE: DRUNK ===

During the night, you will wake up and:
1. Choose one of the three center cards (position 0, 1, or 2)
2. Swap your card with that center card
3. You do NOT see what your new card is!

This means you don't know what team you're on anymore! You could be a Werewolf now.
During discussion, you should probably claim Drunk and see what others say about the center cards.
""",
        Role.INSOMNIAC: """
=== YOUR STARTING ROLE: INSOMNIAC ===

During the night, you wake up LAST (after all swaps have happened).
You look at your own card to see if anyone swapped it.

If your card is still Insomniac, no one swapped you.
If it's different, the Robber stole your card or the Troublemaker swapped you with someone.
""",
        Role.HUNTER: """
=== YOUR STARTING ROLE: HUNTER ===

You have no night action - you sleep through the night.

Your special ability: If you are killed in the vote, the player YOU voted for also dies!
This is a powerful deterrent and can help the village if you're certain about a werewolf.
""",
        Role.WEREWOLF: """
=== YOUR STARTING ROLE: WEREWOLF ===

During the night, you will:
- See who the other werewolf is (if there is one)
- If you're the ONLY werewolf, you may look at ONE center card

Your goal: Survive the vote! Don't get killed.
- Blend in, cast suspicion on others, make false claims
- If there are two werewolves, coordinate your stories
- Watch out for the Seer, who may have seen your card!

Remember: Even if you start as Werewolf, the Robber might steal your card and YOU become their old role!
""",
        Role.MINION: """
=== YOUR STARTING ROLE: MINION ===

During the night, you will see who the werewolf player(s) are.
The werewolves do NOT know you exist or that you're on their team.

Your goal: Help the werewolves survive! 
- You can claim to be a werewolf to draw votes away from them
- You can cast suspicion on villagers
- You WIN if no werewolf dies (even if you die!)
- If there are no werewolves in the game, you must ensure someone dies
""",
        Role.TANNER: """
=== YOUR STARTING ROLE: TANNER ===

You have no night action - you sleep through the night.

Your goal is UNIQUE: You WIN if and only if YOU get killed in the vote!
- Act suspicious to draw votes
- Make claims that make you seem like a werewolf
- But don't be TOO obvious or people will catch on

You are not on any team. You only win if you die.
"""
    }
    
    base = get_game_rules(active_roles) + role_specific.get(role, "")
    return base + f"\nYour name is {player_name}.\n" + REASONING_INSTRUCTION


def get_night_phase_prompt(player: Player, context: dict) -> str:
    """Generate the night phase prompt based on the player's original role."""
    role = player.original_role
    
    if role == Role.VILLAGER:
        return """<phase>NIGHT</phase>
<your_role>VILLAGER</your_role>
You have no night action. The night passes quietly for you."""

    elif role == Role.HUNTER:
        return """<phase>NIGHT</phase>
<your_role>HUNTER</your_role>
You have no night action. The night passes quietly for you.
Remember: If you die in the vote, whoever you voted for also dies!"""

    elif role == Role.TANNER:
        return """<phase>NIGHT</phase>
<your_role>TANNER</your_role>
You have no night action. The night passes quietly for you.
Remember: Your goal is to get yourself killed in the vote!"""

    elif role == Role.WEREWOLF:
        other_werewolves = context.get("other_werewolves", [])
        if other_werewolves:
            return f"""<phase>NIGHT - WEREWOLF</phase>
<your_role>WEREWOLF</your_role>
<teammate>{', '.join(other_werewolves)}</teammate>

You see your fellow werewolf: {', '.join(other_werewolves)}.
Coordinate your stories during the day!

Use acknowledge to confirm you've seen this information."""
        else:
            return """<phase>NIGHT - WEREWOLF (ALONE)</phase>
<your_role>WEREWOLF</your_role>
<status>You are the ONLY werewolf!</status>

Since you're alone, you may look at ONE center card to gain information.
This helps you know what roles are NOT with other players.

Use werewolf_look_center to peek at a center card (0, 1, or 2), or acknowledge to skip."""

    elif role == Role.MINION:
        werewolves = context.get("werewolves", [])
        if werewolves:
            return f"""<phase>NIGHT - MINION</phase>
<your_role>MINION</your_role>
<werewolves>{', '.join(werewolves)}</werewolves>

The werewolf player(s) are: {', '.join(werewolves)}.
They don't know you're helping them!

Your job during the day: Protect them. Mislead the village. Take the fall if needed.

Use acknowledge to confirm you've seen this information."""
        else:
            return """<phase>NIGHT - MINION</phase>
<your_role>MINION</your_role>
<werewolves>None!</werewolves>

There are NO werewolf players! Both werewolves must be in the center.
This is bad for you - the village wins if no one dies.

Your job: Make sure SOMEONE gets voted out (but not yourself)!

Use acknowledge to confirm you've seen this information."""

    elif role == Role.SEER:
        other_players = context.get("other_players", [])
        return f"""<phase>NIGHT - SEER</phase>
<your_role>SEER</your_role>
<other_players>{', '.join(other_players)}</other_players>

Choose ONE of these actions:
1. look_at_player: See one player's current card
2. look_at_center: See two center cards (positions 0, 1, 2)

Looking at a player tells you their role directly.
Looking at center tells you what's NOT in play."""

    elif role == Role.ROBBER:
        other_players = context.get("other_players", [])
        return f"""<phase>NIGHT - ROBBER</phase>
<your_role>ROBBER</your_role>
<targets>{', '.join(other_players)}</targets>

Choose a player to rob. You will:
1. Swap your Robber card with their card
2. See your NEW role

After this, you become whatever they were!

Use rob_player to choose your target."""

    elif role == Role.TROUBLEMAKER:
        other_players = context.get("other_players", [])
        return f"""<phase>NIGHT - TROUBLEMAKER</phase>
<your_role>TROUBLEMAKER</your_role>
<targets>{', '.join(other_players)}</targets>

Choose TWO other players to swap their cards.
You will NOT see what their cards are.

This creates chaos - they won't know they've been swapped!

Use swap_players to choose two targets."""

    elif role == Role.DRUNK:
        return """<phase>NIGHT - DRUNK</phase>
<your_role>DRUNK</your_role>
<center_positions>0, 1, 2</center_positions>

Choose a center card position (0, 1, or 2).
Your card will be swapped with that center card.
You will NOT see what your new card is!

Use drunk_swap to choose a position."""

    elif role == Role.INSOMNIAC:
        current_role = context.get("current_role", Role.INSOMNIAC)
        return f"""<phase>NIGHT - INSOMNIAC</phase>
<your_role>INSOMNIAC (checking card...)</your_role>
<your_current_card>{current_role.value}</your_current_card>

You look at your card and see: {current_role.value}

{"Your card was NOT swapped - you're still the Insomniac." if current_role == Role.INSOMNIAC else f"Your card was SWAPPED! You are now the {current_role.value}!"}

Use acknowledge to confirm you've seen this information."""

    return "<phase>NIGHT</phase>\nUnknown role - waiting for dawn."


def get_day_phase_prompt(
    player: Player, 
    player_names: list[str], 
    messages_so_far: list[PublicMessage],
    current_round: int,
    total_rounds: int,
) -> str:
    """Generate the prompt for a discussion round."""
    
    # Reminder about what they learned at night
    night_reminder = ""
    if player.original_role == Role.SEER:
        night_reminder = "\n<night_reminder>You are the Seer. Share or hide what you learned as you see fit.</night_reminder>"
    elif player.original_role == Role.ROBBER:
        night_reminder = f"\n<night_reminder>You were the Robber. Remember what role you stole!</night_reminder>"
    elif player.original_role == Role.INSOMNIAC:
        night_reminder = f"\n<night_reminder>You were the Insomniac. You checked your card at the end of the night.</night_reminder>"
    elif player.original_role == Role.DRUNK:
        night_reminder = "\n<night_reminder>You were the Drunk. You swapped with a center card but don't know what you are now!</night_reminder>"
    elif player.original_role == Role.WEREWOLF:
        night_reminder = "\n<night_reminder>You are a Werewolf. Lie, deflect, and survive!</night_reminder>"
    elif player.original_role == Role.MINION:
        night_reminder = "\n<night_reminder>You are the Minion. Protect the werewolves - they don't know you exist!</night_reminder>"
    elif player.original_role == Role.TANNER:
        night_reminder = "\n<night_reminder>You are the Tanner. Your goal is to get yourself killed!</night_reminder>"
    
    # Build message history grouped by round
    if messages_so_far:
        messages_xml = "\n<discussion_history>\n"
        # Group messages by round
        rounds_seen = set()
        for msg in messages_so_far:
            if msg.round_number > 0:  # Skip GM messages (round 0)
                rounds_seen.add(msg.round_number)
        
        for round_num in sorted(rounds_seen):
            round_msgs = [m for m in messages_so_far if m.round_number == round_num]
            messages_xml += f'  <round number="{round_num}">\n'
            for msg in round_msgs:
                messages_xml += f'    <message sender="{msg.sender_name}">{msg.content}</message>\n'
            messages_xml += f'  </round>\n'
        messages_xml += "</discussion_history>"
    else:
        messages_xml = "\n<discussion_history>\n  <!-- No messages yet -->\n</discussion_history>"
    
    rounds_remaining = total_rounds - current_round
    
    return f"""<phase>DAY - Discussion Round {current_round} of {total_rounds}</phase>{night_reminder}

<game_state>
  <players>{', '.join(player_names)}</players>
  <you>{player.name}</you>
  <your_starting_role>{player.original_role.value}</your_starting_role>
  <rounds_remaining>{rounds_remaining}</rounds_remaining>
</game_state>
{messages_xml}

IMPORTANT: All players submit their messages simultaneously each round. Messages are revealed together after everyone responds.
{"This is round 1 - you won't see others' messages until round 2." if current_round == 1 else ""}

You must either:
- send_message: Send a message to the group (it will be revealed with everyone else's messages)
- pass_turn: Stay silent this round

Discuss with other players. Try to figure out who the werewolves are (or hide if you are one!).
Remember: Cards may have been swapped during the night!"""


def get_voting_phase_prompt(player: Player, player_names: list[str], messages: Optional[list[PublicMessage]] = None) -> str:
    """Generate the prompt for the voting phase."""
    other_players = [p for p in player_names if p != player.name]
    
    # Summarize key claims from discussion
    claims_summary = ""
    if messages:
        claims_summary = "\n<discussion_summary>\n"
        for msg in messages[-15:]:  # Last 15 messages
            claims_summary += f"  {msg.sender_name}: {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}\n"
        claims_summary += "</discussion_summary>\n"
    
    vote_options = other_players + ["no_one"]
    
    # Role-specific voting reminder
    voting_reminder = ""
    if player.original_role == Role.TANNER:
        voting_reminder = "\n<reminder>You are the Tanner - you WIN if you get killed! Consider voting for someone unlikely to vote for you.</reminder>"
    elif player.original_role == Role.MINION:
        voting_reminder = "\n<reminder>You are the Minion - protect the werewolves! Vote for a villager or take suspicion off them.</reminder>"
    elif player.original_role == Role.WEREWOLF:
        voting_reminder = "\n<reminder>You are a Werewolf - don't vote for your teammate! Deflect to a villager.</reminder>"
    elif player.original_role == Role.HUNTER:
        voting_reminder = "\n<reminder>You are the Hunter - if you die, whoever you vote for also dies! Vote carefully.</reminder>"
    
    return f"""<phase>VOTING</phase>

<players>{', '.join(player_names)}</players>
<vote_options>{', '.join(vote_options)}</vote_options>
{claims_summary}{voting_reminder}
This is the final vote. The player(s) with the most votes will die.
- If a Werewolf dies: Village team wins
- If no Werewolf dies: Werewolf team wins  
- If Tanner dies: Tanner wins

Vote for a player name or "no_one" to vote for no lynch.

Use cast_vote to submit your vote."""


class PlayerAgent:
    """An AI-controlled player in One Night Ultimate Werewolf."""
    
    def __init__(
        self,
        player: Player,
        llm_client: CachedLLMClient,
        message_queue: asyncio.Queue,
        active_roles: list[Role],
        log_thought_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.player = player
        self.llm_client = llm_client
        self.message_queue = message_queue
        self._log_thought_callback = log_thought_callback
        self._running = False
        
        # Current phase tracking
        self._current_phase: str = "NIGHT"
        
        # Track whether we've logged the system prompt in player thoughts
        self._has_logged_system_prompt = False
        
        # Initialize chat history with system prompt
        self._system_prompt = get_role_system_prompt(player.original_role, player.name, active_roles)
        self.player.chat_history = [
            ChatMessage(role="system", content=self._system_prompt)
        ]
    
    def _cleanup_dangling_tool_calls(self) -> None:
        """Remove any assistant messages with tool_calls that don't have corresponding tool responses."""
        if not self.player.chat_history:
            return
        
        responded_ids = set()
        for msg in self.player.chat_history:
            if msg.role == "tool" and msg.tool_call_id:
                responded_ids.add(msg.tool_call_id)
        
        cleaned_history = []
        for msg in self.player.chat_history:
            if msg.role == "assistant" and msg.tool_calls:
                all_responded = all(
                    tc.get("id") in responded_ids 
                    for tc in msg.tool_calls
                )
                if not all_responded:
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
    
    async def _call_llm(self, prompt: str, tools: list[dict], tool_choice: Optional[str] = None) -> LLMResponse:
        """Make an LLM call with the current chat history."""
        self._cleanup_dangling_tool_calls()
        
        self.player.chat_history.append(ChatMessage(role="user", content=prompt))
        
        print(f"  [{self.player.name}] Calling LLM...")
        response = await self.llm_client.chat_completion(
            model=self.player.model,
            messages=self.player.chat_history,
            tools=tools if tools else None,
            tool_choice=tool_choice,
            temperature=1,
        )
        
        assistant_msg = ChatMessage(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls
        )
        self.player.chat_history.append(assistant_msg)
        
        # Broadcast thought event
        thought_data = {
            "player_name": self.player.name,
            "prompt": prompt,
            "response": response.content,
            "tool_calls": response.tool_calls,
            "phase": self._current_phase,
        }
        
        # Include system prompt in the first thought event so that 
        # appending all private thoughts recreates the full chat thread
        if not self._has_logged_system_prompt:
            thought_data["system_prompt"] = self._system_prompt
            self._has_logged_system_prompt = True
        
        broadcaster = get_broadcaster()
        await broadcaster.broadcast(
            GameEvent(event_type=EventType.PLAYER_THOUGHT, data=thought_data),
            channels=[f"player_{self.player.name}"]
        )
        
        if self._log_thought_callback:
            self._log_thought_callback(thought_data)
        
        return response
    
    def _handle_tool_call(self, tool_call: dict) -> str:
        """Execute a tool call and return the result."""
        func_name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])
        
        if func_name == "send_message":
            content = args.get("content", "")
            return f"SEND_MESSAGE:{content}"
        
        elif func_name == "pass_turn":
            return "PASS_TURN"
        
        elif func_name == "cast_vote":
            target = args["target"]
            return f"VOTE:{target}"
        
        # ONUW night action tools
        elif func_name == "look_at_player":
            target = args["target"]
            return f"LOOK_PLAYER:{target}"
        
        elif func_name == "look_at_center":
            pos1 = args["position1"]
            pos2 = args["position2"]
            return f"LOOK_CENTER:{pos1},{pos2}"
        
        elif func_name == "rob_player":
            target = args["target"]
            return f"ROB:{target}"
        
        elif func_name == "swap_players":
            player1 = args["player1"]
            player2 = args["player2"]
            return f"SWAP:{player1},{player2}"
        
        elif func_name == "drunk_swap":
            position = args["position"]
            return f"DRUNK_SWAP:{position}"
        
        elif func_name == "werewolf_look_center":
            position = args["position"]
            return f"WEREWOLF_LOOK:{position}"
        
        elif func_name == "acknowledge":
            return "ACKNOWLEDGE"
        
        else:
            return f"Unknown tool: {func_name}"
    
    async def run_night_phase(self, context: dict) -> Optional[NightAction]:
        """Run the night phase for this player and return their action."""
        self._current_phase = "NIGHT"
        role = self.player.original_role
        
        print(f"[{self.player.name}] Night phase - role: {role.value}")
        
        # Passive roles have no night action
        if role in PASSIVE_ROLES:
            prompt = get_night_phase_prompt(self.player, context)
            self.player.chat_history.append(ChatMessage(role="user", content=prompt))
            self.player.chat_history.append(ChatMessage(
                role="assistant", 
                content="I have no night action. I'll wait for day."
            ))
            return None
        
        # Get the appropriate tools for this role
        if role == Role.WEREWOLF:
            other_werewolves = context.get("other_werewolves", [])
            tools = WEREWOLF_TEAM_TOOLS if other_werewolves else WEREWOLF_LONE_TOOLS
        elif role == Role.MINION:
            tools = MINION_TOOLS
        elif role == Role.SEER:
            tools = SEER_TOOLS
        elif role == Role.ROBBER:
            tools = ROBBER_TOOLS
        elif role == Role.TROUBLEMAKER:
            tools = TROUBLEMAKER_TOOLS
        elif role == Role.DRUNK:
            tools = DRUNK_TOOLS
        elif role == Role.INSOMNIAC:
            tools = INSOMNIAC_TOOLS
        else:
            tools = [ACKNOWLEDGE_TOOL]
        
        prompt = get_night_phase_prompt(self.player, context)
        response = await self._call_llm(prompt, tools, tool_choice="required")
        
        action = None
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = self._handle_tool_call(tool_call)
                
                self.player.chat_history.append(ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
                
                # Parse the result into a NightAction
                if result.startswith("LOOK_PLAYER:"):
                    target = result[12:]
                    action = NightAction(
                        player_name=self.player.name,
                        original_role=role,
                        action_type="look_player",
                        targets=[target]
                    )
                elif result.startswith("LOOK_CENTER:"):
                    positions = result[12:].split(",")
                    action = NightAction(
                        player_name=self.player.name,
                        original_role=role,
                        action_type="look_center",
                        targets=[f"center_{p}" for p in positions]
                    )
                elif result.startswith("ROB:"):
                    target = result[4:]
                    action = NightAction(
                        player_name=self.player.name,
                        original_role=role,
                        action_type="rob",
                        targets=[target]
                    )
                elif result.startswith("SWAP:"):
                    players = result[5:].split(",")
                    action = NightAction(
                        player_name=self.player.name,
                        original_role=role,
                        action_type="swap",
                        targets=players
                    )
                elif result.startswith("DRUNK_SWAP:"):
                    position = result[11:]
                    action = NightAction(
                        player_name=self.player.name,
                        original_role=role,
                        action_type="drunk_swap",
                        targets=[f"center_{position}"]
                    )
                elif result.startswith("WEREWOLF_LOOK:"):
                    position = result[14:]
                    action = NightAction(
                        player_name=self.player.name,
                        original_role=role,
                        action_type="werewolf_look",
                        targets=[f"center_{position}"]
                    )
                elif result == "ACKNOWLEDGE":
                    # No action, just acknowledgment
                    action = NightAction(
                        player_name=self.player.name,
                        original_role=role,
                        action_type="acknowledge",
                        targets=[]
                    )
        
        print(f"[{self.player.name}] Night action: {action}")
        return action
    
    async def run_day_round(
        self,
        player_names: list[str],
        messages_so_far: list[PublicMessage],
        current_round: int,
        total_rounds: int,
    ) -> Optional[str]:
        """
        Run a single discussion round for this player.
        
        Returns the message content to send, or None if player passes.
        """
        self._current_phase = "DAY"
        print(f"[{self.player.name}] Round {current_round}/{total_rounds}")
        
        prompt = get_day_phase_prompt(
            self.player, 
            player_names, 
            messages_so_far,
            current_round=current_round,
            total_rounds=total_rounds,
        )
        
        # Require a tool call (either send_message or pass_turn)
        # No timeout - wait for the LLM to respond (errors will propagate up)
        response = await self._call_llm(prompt, DAY_TOOLS, tool_choice="required")
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = self._handle_tool_call(tool_call)
                
                self.player.chat_history.append(ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
                
                if result.startswith("SEND_MESSAGE:"):
                    content = result[13:]  # Remove "SEND_MESSAGE:" prefix
                    print(f"  [{self.player.name}] Sending message: {content[:50]}...")
                    return content
                elif result == "PASS_TURN":
                    print(f"  [{self.player.name}] Passed this round")
                    return None
        
        print(f"  [{self.player.name}] No valid tool call, treating as pass")
        return None
    
    async def run_voting_phase(self, player_names: list[str], messages: Optional[list[PublicMessage]] = None) -> tuple[Optional[str], Optional[str]]:
        """Run the voting phase and return (vote_target, reasoning) tuple."""
        self._current_phase = "VOTING"
        print(f"[{self.player.name}] Starting voting phase")
        
        prompt = get_voting_phase_prompt(self.player, player_names, messages)
        response = await self._call_llm(prompt, VOTING_TOOLS, tool_choice="required")
        
        reasoning = response.content
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = self._handle_tool_call(tool_call)
                
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
    
    def add_game_event(self, event_description: str) -> None:
        """Add a game event to the player's chat history."""
        self.player.chat_history.append(ChatMessage(
            role="user",
            content=f"[GAME EVENT] {event_description}"
        ))
    
    def set_phase(self, phase: str) -> None:
        """Set the current game phase."""
        self._current_phase = phase
    
    def stop(self) -> None:
        """Stop the player's current phase loop."""
        self._running = False
