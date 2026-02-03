"""Player agent with LLM integration for One Night Ultimate Werewolf."""

import asyncio
import json
from typing import Optional, Callable

from .onuw import (
    Player, Role, ChatMessage, PublicMessage,
    GameEvent, EventType, PASSIVE_ROLES
)
from .llm_client import (
    CachedLLMClient, LLMResponse,
    DAY_TOOLS, VOTING_TOOLS,
)
from .events import get_broadcaster


# Role descriptions - can be filtered based on active roles
ROLE_DESCRIPTIONS = {
    # Village Team
    Role.VILLAGER: "VILLAGER: No special ability.",
    Role.SEER: "SEER: Wake up and look at ONE player's card OR TWO center cards.",
    Role.ROBBER: "ROBBER: Wake up and swap your card with another player's card. You see your NEW card.",
    Role.TROUBLEMAKER: "TROUBLEMAKER: Wake up and swap TWO OTHER players' cards. You don't see the cards.",
    Role.DRUNK: "DRUNK: Wake up and swap your card with a center card. You DON'T see your new card.",
    Role.INSOMNIAC: "INSOMNIAC: Wake up LAST and look at your own card (to see if it was swapped).",
    Role.HUNTER: "HUNTER: No night action. If you die, the player you voted for also dies.",
    # Werewolf Team
    Role.WEREWOLF: "WEREWOLF: Wake up and see other werewolves. If you're the ONLY werewolf, you may look at one center card.",
    Role.MINION: "MINION: Wake up and see who the werewolves are. They don't know you exist. You win with werewolves, unless there are no werewolves, in which case you win if you are killed in the vote.",
    # Neutral
    Role.TANNER: "TANNER: No night action. You WIN if you get killed in the vote. You LOSE if you survive.",
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
    win_conditions = f"""WIN CONDITIONS:
- VILLAGE TEAM wins if at least one Werewolf player is killed
- WEREWOLF TEAM wins if no Werewolf player is killed
{"- If there are NO werewolves among players (all in center), Minion wins if they are killed, otherwise Village wins" if Role.MINION in active_set else ""}
- In case of a tied vote, ALL tied players die"""
    
    if Role.TANNER in active_set:
        win_conditions += "\n- TANNER wins if the Tanner is killed (this overrides all other win conditions)"
    
    # Build strategy notes (adjust based on active roles)
    strategy_notes = "\nIMPORTANT STRATEGY NOTES:\n"
    strategy_notes += "- Your starting role determines your night action, but your FINAL role (after swaps) determines your team for winning\n"
    strategy_notes += "- You should keep in mind the possibility that your role was swapped during the night. This is a key part of the game.\n"
    strategy_notes += "- From a town perspective, sharing information can help the town figure out what happened during the night. On the other hand, withholding information can be useful for catching werewolves in lies. Lying can similarly help catch werewolves in lies, but risks looking like a werewolf yourself and misleading the town.\n"
    strategy_notes += "- From a werewolf perspective, you need to claim some role to avoid being suspicious. Committing to a story later in the discussion reduces the risk of contradicting other players information, but claims made earlier are more trustworthy precisely because they are risky for werewolves to make.\n"
    if Role.TANNER in active_set:
        strategy_notes += "- From a tanner perspective, the game is a balancing act of being suspicious enough to seem like a werewolf, but not so suspicious that you seem like a tanner.\n"
    
    return f"""=== ONE NIGHT ULTIMATE WEREWOLF RULES ===

OVERVIEW:
This is a single-night social deduction game. Each player is dealt a secret role card. During the night, players with special abilities wake up in a specific order and take actions. Some actions can SWAP cards - meaning your role (and win conditions) might change without you knowing! After the night, there is one discussion period and one vote. The player(s) with the most votes die.

{win_conditions}

{roles_section}
{night_order_section}
{strategy_notes}"""

def get_role_system_prompt(role: Role, player_name: str, active_roles: list[Role]) -> str:
    """Generate the system prompt for a player based on their starting role."""
    
    role_specific = {
        Role.VILLAGER: """
=== YOUR STARTING ROLE: VILLAGER ===

You have no special night ability. During the night, you simply sleep.

The villager wins if at least one werewolf is killed.
Remember: someone might have swapped your role during the night!
""",
        Role.SEER: """
=== YOUR STARTING ROLE: SEER ===

During the night, you will wake up and choose ONE of these actions:
- Look at ONE other player's card to learn their role
- Look at TWO center cards to learn what roles are NOT in play

The seer wins if at least one werewolf is killed.
Remember: someone might have swapped your role during the night!
""",
        Role.ROBBER: """
=== YOUR STARTING ROLE: ROBBER ===

During the night, you will wake up and:
1. Choose another player to rob
2. Swap your card with theirs
3. Look at your NEW card (your new role)

After robbing, you become whatever role you stole! Your NEW role determines which team you're on and your win condition.
Remember: someone might have swapped your role during the night!
""",
        Role.TROUBLEMAKER: """
=== YOUR STARTING ROLE: TROUBLEMAKER ===

During the night, you will wake up and:
1. Choose TWO OTHER players (not yourself)
2. Swap their cards with each other
3. You do NOT see what roles they have

The troublemaker wins if at least one werewolf is killed.
Remember: someone might have swapped your role during the night!
""",
        Role.DRUNK: """
=== YOUR STARTING ROLE: DRUNK ===

During the night, you will wake up and:
1. Choose one of the three center cards (position 0, 1, or 2)
2. Swap your card with that center card
3. You do NOT see what your new card is!

You don't know what team you're on! You could win with the werewolves or with the town.
""",
        Role.INSOMNIAC: """
=== YOUR STARTING ROLE: INSOMNIAC ===

During the night, you wake up LAST (after all swaps have happened).
You look at your own card to see if anyone swapped it.

Your NEW role determines which team you're on and your win condition.
""",
        Role.HUNTER: """
=== YOUR STARTING ROLE: HUNTER ===

You have no night action - you sleep through the night.
Your special ability: If you are killed in the vote, the player YOU voted for also dies!

The hunter wins if at least one werewolf is killed.
Remember: someone might have swapped your role during the night! If you are no longer the hunter, your ability does not apply.
""",
        Role.WEREWOLF: """
=== YOUR STARTING ROLE: WEREWOLF ===

During the night, you will:
- See who the other werewolf is (if there is one)
- If you're the ONLY werewolf, you may look at ONE center card

The werewolf wins if no werewolf is killed.
Remember: someone might have swapped your role during the night!
""",
        Role.MINION: """
=== YOUR STARTING ROLE: MINION ===

During the night, you will see who the werewolf player(s) are.
The werewolves do NOT know you exist or that you're on their team.

The minion wins if no werewolf is killed.
Remember: someone might have swapped your role during the night!
""",
        Role.TANNER: """
=== YOUR STARTING ROLE: TANNER ===

You have no night action - you sleep through the night.

The tanner wins if they are killed in the vote.
Remember: someone might have swapped your role during the night!
"""
    }
    
    base = get_game_rules(active_roles) + role_specific.get(role, "")
    return base + f"\nYour name is {player_name}.\n"


def get_night_phase_prompt(player: Player, context: dict) -> str:
    """
    Generate an informational night phase prompt.
    
    Tells players what happened during their night action (pre-determined by seed).
    """
    role = player.original_role
    action = context.get("predetermined_action")
    result = context.get("action_result")
    
    if role == Role.VILLAGER:
        return """<phase>NIGHT</phase>
<your_role>VILLAGER</your_role>

You have no night action. The night passes quietly for you."""

    elif role == Role.HUNTER:
        return """<phase>NIGHT</phase>
<your_role>HUNTER</your_role>

You have no night action. The night passes quietly for you."""

    elif role == Role.TANNER:
        return """<phase>NIGHT</phase>
<your_role>TANNER</your_role>

You have no night action. The night passes quietly for you."""

    elif role == Role.WEREWOLF:
        other_werewolves = context.get("other_werewolves", [])
        if other_werewolves:
            return f"""<phase>NIGHT - WEREWOLF</phase>
<your_role>WEREWOLF</your_role>
<teammate>{', '.join(other_werewolves)}</teammate>

You wake up and see your fellow werewolf: {', '.join(other_werewolves)}.
You exchange a knowing glance."""
        else:
            # Lone werewolf looked at center
            position = action.targets[0] if action and action.targets else "center_0"
            pos_num = position.split("_")[1]
            return f"""<phase>NIGHT - WEREWOLF (ALONE)</phase>
<your_role>WEREWOLF</your_role>
<status>You are the ONLY werewolf!</status>
<center_peek>Position {pos_num}: {result}</center_peek>

Since you're alone, you peek at center card position {pos_num} and see: {result}.
This tells you that {result} is NOT held by any player - useful information for the day discussion!"""

    elif role == Role.MINION:
        werewolves = context.get("werewolves", [])
        if werewolves:
            return f"""<phase>NIGHT - MINION</phase>
<your_role>MINION</your_role>
<werewolves>{', '.join(werewolves)}</werewolves>

You wake up and see that the werewolf player(s) are: {', '.join(werewolves)}.
They don't know you exist!"""
        else:
            return """<phase>NIGHT - MINION</phase>
<your_role>MINION</your_role>
<werewolves>None!</werewolves>

You wake up and see... NO werewolves! Both werewolf cards must be in the center.
This means you win if you are killed in the vote."""

    elif role == Role.SEER:
        if action and action.action_type == "look_player":
            target = action.targets[0]
            return f"""<phase>NIGHT - SEER</phase>
<your_role>SEER</your_role>
<action>Looked at {target}'s card</action>
<result>{result}</result>

You wake up and look at {target}'s card.
You see that {target} is a {result}!"""
        else:
            # Looked at center
            positions = [t.split("_")[1] for t in action.targets] if action else ["0", "1"]
            roles_str = ", ".join(result) if isinstance(result, list) else str(result)
            return f"""<phase>NIGHT - SEER</phase>
<your_role>SEER</your_role>
<action>Looked at center cards</action>
<positions>{', '.join(positions)}</positions>
<result>{roles_str}</result>

You wake up and look at center card positions {' and '.join(positions)}.
You see: {roles_str}

These roles are NOT held by any player."""

    elif role == Role.ROBBER:
        target = action.targets[0] if action else "someone"
        return f"""<phase>NIGHT - ROBBER</phase>
<your_role>ROBBER (was)</your_role>
<action>Robbed {target}</action>
<new_role>{result}</new_role>

You wake up and swap your card with {target}'s card.
You look at your new card and see: {result}

You are now the {result}! This is your new role and team allegiance.
{target} now has your old Robber card (but doesn't know it)."""

    elif role == Role.TROUBLEMAKER:
        if action and len(action.targets) >= 2:
            player1, player2 = action.targets[0], action.targets[1]
        else:
            player1, player2 = "Player1", "Player2"
        return f"""<phase>NIGHT - TROUBLEMAKER</phase>
<your_role>TROUBLEMAKER</your_role>
<action>Swapped {player1} and {player2}</action>

You wake up and swap {player1}'s card with {player2}'s card.
You do NOT see what their cards are - you just know they've been swapped!

Neither {player1} nor {player2} knows they've been swapped."""

    elif role == Role.DRUNK:
        position = action.targets[0].split("_")[1] if action and action.targets else "0"
        return f"""<phase>NIGHT - DRUNK</phase>
<your_role>DRUNK (was)</your_role>
<action>Swapped with center position {position}</action>

You wake up in a daze and swap your card with center card position {position}.
You do NOT see what your new card is!"""

    elif role == Role.INSOMNIAC:
        current_role = context.get("current_role", Role.INSOMNIAC)
        was_swapped = current_role != Role.INSOMNIAC
        if was_swapped:
            return f"""<phase>NIGHT - INSOMNIAC</phase>
<your_role>INSOMNIAC (was)</your_role>
<current_card>{current_role.value}</current_card>
<swapped>YES</swapped>

You wake up LAST (after all other night actions) and look at your card.
Your card has been SWAPPED! You are now the {current_role.value}!
This is your new role and team allegiance."""
        else:
            return """<phase>NIGHT - INSOMNIAC</phase>
<your_role>INSOMNIAC</your_role>
<current_card>INSOMNIAC</current_card>
<swapped>NO</swapped>

You wake up LAST (after all other night actions) and look at your card.
Your card is still the Insomniac!"""

    return "<phase>NIGHT</phase>\nUnknown role - waiting for dawn."


def _build_day_context(
    player: Player, 
    player_names: list[str], 
    messages_so_far: list[PublicMessage],
    current_round: int,
    total_rounds: int,
) -> str:
    """Build the context portion of the day phase prompt."""
    
    # Reminder about what they learned at night
    night_reminder = ""
    if player.original_role == Role.SEER:
        night_reminder = "\n<night_reminder>You started as the Seer, and you saw what you saw. But remember, your role may have been swapped during the night!</night_reminder>"
    elif player.original_role == Role.ROBBER:
        night_reminder = "\n<night_reminder>You started as the Robber. Remember, the troublemaker may have swapped your card again after you did!</night_reminder>"
    elif player.original_role == Role.INSOMNIAC:
        night_reminder = "\n<night_reminder>You started as the Insomniac. You checked your card at the end of the night, so you know your new role and whether you were swapped or not.</night_reminder>"
    elif player.original_role == Role.DRUNK:
        night_reminder = "\n<night_reminder>You started as the Drunk. You swapped with a center card but don't know what you are now!</night_reminder>"
    elif player.original_role == Role.WEREWOLF:
        night_reminder = "\n<night_reminder>You started as a Werewolf. Remember, your role may have been swapped during the night!</night_reminder>"
    elif player.original_role == Role.MINION:
        night_reminder = "\n<night_reminder>You started as the Minion. Remember, your role may have been swapped during the night!</night_reminder>"
    elif player.original_role == Role.TANNER:
        night_reminder = "\n<night_reminder>You started as the Tanner. Remember, your role may have been swapped during the night!</night_reminder>"
    
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

All players submit their messages simultaneously each round. Messages are revealed together after everyone responds.
{"This is round 1 - you won't see others' messages until round 2." if current_round == 1 else ""}"""


def get_day_thinking_prompt(
    player: Player, 
    player_names: list[str], 
    messages_so_far: list[PublicMessage],
    current_round: int,
    total_rounds: int,
) -> str:
    """Generate the thinking prompt for a discussion round (no tools)."""
    context = _build_day_context(player, player_names, messages_so_far, current_round, total_rounds)
    
    return f"""{context}

THINKING PHASE: Before deciding what to do, analyze the current situation.

Consider:
1. What do you know from your night action?
2. What have other players claimed? Are there contradictions?
3. Who might be lying? Who seems trustworthy?
4. What is your current win condition (considering possible role swaps)?
5. Should you speak this round or stay silent? What would you say?

Think through this carefully. After you respond, you'll be asked to take an action."""


def get_day_action_prompt() -> str:
    """Generate the action prompt for a discussion round (with tools)."""
    return """Now take your action for this round.

You must either:
- send_message: Send a PUBLIC message to the group (all players will see this!)
- pass_turn: Stay silent this round

If you send a message, include ONLY what you want to say out loud. No labels, no reasoning - just your actual words."""


def get_voting_phase_prompt(player: Player, player_names: list[str], messages: Optional[list[PublicMessage]] = None) -> str:
    """Generate the prompt for the voting phase."""
    other_players = [p for p in player_names if p != player.name]
    
    # Summarize key claims from discussion
    claims_summary = ""
    if messages:
        claims_summary = "\n<discussion_summary>\n"
        for msg in messages[-15:]:  # Last 15 messages
            claims_summary += f"  {msg.sender_name}: {msg.content}\n"
        claims_summary += "</discussion_summary>\n"
    
    return f"""<phase>VOTING</phase>

<players>{', '.join(player_names)}</players>
<vote_options>{', '.join(other_players)}</vote_options>
{claims_summary}
This is the final vote. The player(s) with the most votes will die.
In case of a tie, ALL tied players die.
- If a Werewolf dies: Village team wins
- If no Werewolf dies: Werewolf team wins  
- If Tanner dies: Tanner wins

You must vote for another player. Use cast_vote to submit your vote."""


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
        
        # Use reasoning summary for gpt-5-mini to get detailed reasoning output
        reasoning_settings = None
        if "gpt-5-mini" in self.player.model:
            reasoning_settings = {"effort": "medium", "summary": "detailed"}
        
        response = await self.llm_client.chat_completion(
            model=self.player.model,
            messages=self.player.chat_history,
            tools=tools if tools else None,
            tool_choice=tool_choice,
            temperature=1,
            reasoning=reasoning_settings,
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
        
        else:
            return f"Unknown tool: {func_name}"
    
    async def run_night_phase(self, context: dict) -> None:
        """
        Run the night phase with pre-determined actions (informational only).
        
        The player receives information about what happened during their night
        action and can respond/acknowledge. No tool calls are required since
        the action was already determined by the seed.
        """
        self._current_phase = "NIGHT"
        role = self.player.original_role
        
        print(f"[{self.player.name}] Night phase - role: {role.value}")
        
        # Generate informational prompt
        prompt = get_night_phase_prompt(self.player, context)
        
        # For passive roles, just add the prompt to history (no LLM call needed)
        if role in PASSIVE_ROLES:
            self.player.chat_history.append(ChatMessage(role="user", content=prompt))
            self.player.chat_history.append(ChatMessage(
                role="assistant", 
                content="I understand. I have no night action and will wait for the day phase."
            ))
            return
        
        # For active roles, call LLM to let them acknowledge/process the information
        # No tools required - they just receive and acknowledge the information
        await self._call_llm(prompt, tools=[], tool_choice=None)
        
        print(f"  [{self.player.name}] Acknowledged night action")
    
    async def run_day_round(
        self,
        player_names: list[str],
        messages_so_far: list[PublicMessage],
        current_round: int,
        total_rounds: int,
    ) -> Optional[str]:
        """
        Run a single discussion round for this player.
        
        Uses a two-step process:
        1. First, prompt the model to think (no tools) - generates reasoning
        2. Then, prompt the model to take action (with tools)
        
        Returns the message content to send, or None if player passes.
        """
        self._current_phase = "DAY"
        print(f"[{self.player.name}] Round {current_round}/{total_rounds}")
        
        # Step 1: Thinking phase (no tools)
        thinking_prompt = get_day_thinking_prompt(
            self.player, 
            player_names, 
            messages_so_far,
            current_round=current_round,
            total_rounds=total_rounds,
        )
        
        print(f"  [{self.player.name}] Thinking...")
        await self._call_llm(thinking_prompt, tools=[], tool_choice=None)
        
        # Step 2: Action phase (with tools)
        action_prompt = get_day_action_prompt()
        
        print(f"  [{self.player.name}] Taking action...")
        response = await self._call_llm(action_prompt, DAY_TOOLS, tool_choice="required")
        
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
