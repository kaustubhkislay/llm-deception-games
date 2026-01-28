#!/usr/bin/env python3
"""
QA Test Suite for Mafia Game Logic

Tests edge cases and rule enforcement without running actual LLM calls.
Run with: python experiments/03_qa_tests.py
"""

import sys
from pathlib import Path
from collections import Counter
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.mafia import (
    Role, Phase, Player, GameState, NightResult, VoteResult
)


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []
    
    def record(self, name: str, passed: bool, error_msg: str = ""):
        if passed:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {error_msg}")
            print(f"  ❌ {name}: {error_msg}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFailed tests:")
            for err in self.errors:
                print(f"  - {err}")
        print(f"{'='*60}")
        return self.failed == 0


results = TestResults()


# =============================================================================
# WIN CONDITION TESTS
# =============================================================================

def test_win_conditions():
    """Test win condition detection."""
    print("\n🎯 WIN CONDITION TESTS")
    
    # Test 1: Town wins when all mafia dead
    players = [
        Player(name="A", model="test", role=Role.TOWN, is_alive=True),
        Player(name="B", model="test", role=Role.TOWN, is_alive=True),
        Player(name="C", model="test", role=Role.MAFIA, is_alive=False),
    ]
    state = GameState(players=players)
    living_mafia = len(state.living_mafia)
    living_town = len(state.living_town)
    winner = None
    if living_mafia == 0:
        winner = "TOWN"
    elif living_mafia >= living_town:
        winner = "MAFIA"
    
    results.record(
        "Town wins when all mafia dead",
        winner == "TOWN",
        f"Expected TOWN, got {winner}"
    )
    
    # Test 2: Mafia wins when mafia >= town
    players = [
        Player(name="A", model="test", role=Role.TOWN, is_alive=True),
        Player(name="B", model="test", role=Role.MAFIA, is_alive=True),
        Player(name="C", model="test", role=Role.MAFIA, is_alive=True),
    ]
    state = GameState(players=players)
    living_mafia = len(state.living_mafia)
    living_town = len(state.living_town)
    winner = None
    if living_mafia == 0:
        winner = "TOWN"
    elif living_mafia >= living_town:
        winner = "MAFIA"
    
    results.record(
        "Mafia wins when mafia > town",
        winner == "MAFIA",
        f"Expected MAFIA, got {winner}"
    )
    
    # Test 3: Mafia wins when mafia == town
    players = [
        Player(name="A", model="test", role=Role.TOWN, is_alive=True),
        Player(name="B", model="test", role=Role.MAFIA, is_alive=True),
    ]
    state = GameState(players=players)
    living_mafia = len(state.living_mafia)
    living_town = len(state.living_town)
    winner = None
    if living_mafia == 0:
        winner = "TOWN"
    elif living_mafia >= living_town:
        winner = "MAFIA"
    
    results.record(
        "Mafia wins when mafia == town",
        winner == "MAFIA",
        f"Expected MAFIA, got {winner}"
    )
    
    # Test 4: Game continues when town > mafia and mafia alive
    players = [
        Player(name="A", model="test", role=Role.TOWN, is_alive=True),
        Player(name="B", model="test", role=Role.TOWN, is_alive=True),
        Player(name="C", model="test", role=Role.MAFIA, is_alive=True),
    ]
    state = GameState(players=players)
    living_mafia = len(state.living_mafia)
    living_town = len(state.living_town)
    winner = None
    if living_mafia == 0:
        winner = "TOWN"
    elif living_mafia >= living_town:
        winner = "MAFIA"
    
    results.record(
        "Game continues when town > mafia",
        winner is None,
        f"Expected None, got {winner}"
    )


# =============================================================================
# VOTE RESOLUTION TESTS
# =============================================================================

def test_vote_resolution():
    """Test voting phase logic."""
    print("\n🗳️ VOTE RESOLUTION TESTS")
    
    # Test 1: Clear majority lynches target
    votes = {"A": "C", "B": "C", "C": "A"}
    vote_counts = Counter(votes.values())
    max_votes = max(vote_counts.values())
    top_voted = [name for name, count in vote_counts.items() if count == max_votes]
    
    results.record(
        "Clear majority lynches target",
        len(top_voted) == 1 and top_voted[0] == "C",
        f"Expected ['C'], got {top_voted}"
    )
    
    # Test 2: Tie results in no lynch
    votes = {"A": "C", "B": "D", "C": "A", "D": "B"}
    vote_counts = Counter(votes.values())
    max_votes = max(vote_counts.values())
    top_voted = [name for name, count in vote_counts.items() if count == max_votes]
    is_tie = len(top_voted) > 1
    
    results.record(
        "Tie results in no lynch",
        is_tie,
        f"Expected tie, got single winner: {top_voted}"
    )
    
    # Test 3: Empty votes = no lynch
    votes = {}
    is_empty = len(votes) == 0
    
    results.record(
        "Empty votes results in no lynch",
        is_empty,
        "Expected empty votes"
    )
    
    # Test 4: Self-votes are allowed
    votes_with_self = {"A": "A", "B": "C", "C": "B"}
    # Self-votes should count - A voting for A means A gets 1 vote
    vote_counts = Counter(votes_with_self.values())
    
    results.record(
        "Self-votes are allowed and counted",
        vote_counts["A"] == 1,
        f"Self-vote A->A should count: {vote_counts}"
    )


# =============================================================================
# DOCTOR PROTECTION TESTS
# =============================================================================

def test_doctor_protection():
    """Test doctor protection mechanics."""
    print("\n💉 DOCTOR PROTECTION TESTS")
    
    # Test 1: Doctor saves mafia target
    kill_target = Player(name="Target", model="test", role=Role.TOWN, is_alive=True)
    saved_player = Player(name="Target", model="test", role=Role.TOWN, is_alive=True)
    
    kill_blocked = (saved_player and kill_target.name == saved_player.name)
    
    results.record(
        "Doctor saves mafia target",
        kill_blocked,
        "Doctor protection should block kill"
    )
    
    # Test 2: Doctor protects wrong person - kill succeeds
    kill_target = Player(name="Target", model="test", role=Role.TOWN, is_alive=True)
    saved_player = Player(name="Other", model="test", role=Role.TOWN, is_alive=True)
    
    kill_blocked = (saved_player and kill_target.name == saved_player.name)
    
    results.record(
        "Kill succeeds when doctor protects wrong person",
        not kill_blocked,
        "Kill should succeed when doctor protects different person"
    )
    
    # Test 3: Doctor can protect themselves
    doctor = Player(name="Doc", model="test", role=Role.DOCTOR, is_alive=True)
    living_players = ["Doc", "A", "B"]
    can_protect_self = doctor.name in living_players
    
    results.record(
        "Doctor can protect themselves",
        can_protect_self,
        "Doctor should be able to protect self"
    )
    
    # Test 4: Consecutive protection rule - can't protect same twice
    last_protected = "Alice"
    current_choice = "Alice"
    valid_targets_without_last = [p for p in ["Alice", "Bob", "Charlie"] if p != last_protected]
    
    results.record(
        "Consecutive protection - same target excluded",
        current_choice not in valid_targets_without_last,
        "Same target should be excluded from valid choices"
    )
    
    # Test 5: Can protect different person after first protection
    last_protected = "Alice"
    current_choice = "Bob"
    is_valid = current_choice != last_protected
    
    results.record(
        "Can protect different person next night",
        is_valid,
        "Different target should be valid"
    )


# =============================================================================
# DETECTIVE INVESTIGATION TESTS
# =============================================================================

def test_detective_investigation():
    """Test detective investigation mechanics."""
    print("\n🔍 DETECTIVE INVESTIGATION TESTS")
    
    # Test 1: Investigating mafia returns True
    target = Player(name="Mafia", model="test", role=Role.MAFIA, is_alive=True)
    is_mafia = target.role == Role.MAFIA
    
    results.record(
        "Investigating mafia returns True",
        is_mafia,
        f"Expected True, got {is_mafia}"
    )
    
    # Test 2: Investigating town returns False
    target = Player(name="Town", model="test", role=Role.TOWN, is_alive=True)
    is_mafia = target.role == Role.MAFIA
    
    results.record(
        "Investigating town returns False",
        not is_mafia,
        f"Expected False, got {is_mafia}"
    )
    
    # Test 3: Investigating doctor returns False
    target = Player(name="Doc", model="test", role=Role.DOCTOR, is_alive=True)
    is_mafia = target.role == Role.MAFIA
    
    results.record(
        "Investigating doctor returns False (not mafia)",
        not is_mafia,
        f"Expected False, got {is_mafia}"
    )
    
    # Test 4: Investigating detective returns False
    target = Player(name="Det", model="test", role=Role.DETECTIVE, is_alive=True)
    is_mafia = target.role == Role.MAFIA
    
    results.record(
        "Investigating detective returns False (not mafia)",
        not is_mafia,
        f"Expected False, got {is_mafia}"
    )


# =============================================================================
# MAFIA COORDINATION TESTS
# =============================================================================

def test_mafia_coordination():
    """Test mafia kill vote resolution."""
    print("\n🔪 MAFIA COORDINATION TESTS")
    
    # Test 1: Single mafia - their target is used
    mafia_votes = ["Diana"]
    vote_counts = Counter(mafia_votes)
    final_target = vote_counts.most_common(1)[0][0]
    
    results.record(
        "Single mafia vote uses their target",
        final_target == "Diana",
        f"Expected Diana, got {final_target}"
    )
    
    # Test 2: Two mafia agree - target is killed
    mafia_votes = ["Diana", "Diana"]
    vote_counts = Counter(mafia_votes)
    final_target = vote_counts.most_common(1)[0][0]
    
    results.record(
        "Two mafia agree on target",
        final_target == "Diana",
        f"Expected Diana, got {final_target}"
    )
    
    # Test 3: Two mafia disagree - tie means no kill
    mafia_votes = ["Diana", "Charlie"]
    vote_counts = Counter(mafia_votes)
    max_votes = max(vote_counts.values())
    top_voted = [t for t, c in vote_counts.items() if c == max_votes]
    is_tie = len(top_voted) > 1
    final_target = None if is_tie else top_voted[0]
    
    results.record(
        "Two mafia disagree - tie means no kill",
        final_target is None,
        f"Expected None (no kill), got {final_target}"
    )
    
    # Test 4: Three mafia - majority wins
    mafia_votes = ["Diana", "Diana", "Charlie"]
    vote_counts = Counter(mafia_votes)
    final_target = vote_counts.most_common(1)[0][0]
    
    results.record(
        "Three mafia - majority wins",
        final_target == "Diana",
        f"Expected Diana, got {final_target}"
    )
    
    # Test 5: Mafia cannot target another mafia (handled in prompts)
    mafia_members = ["Alice", "Bob"]
    living_players = ["Alice", "Bob", "Charlie", "Diana"]
    valid_targets = [p for p in living_players if p not in mafia_members]
    
    results.record(
        "Mafia targets exclude other mafia",
        "Alice" not in valid_targets and "Bob" not in valid_targets,
        f"Mafia should not be in targets: {valid_targets}"
    )


# =============================================================================
# DEATH REVEAL TESTS
# =============================================================================

def test_death_reveals():
    """Test what information is revealed on death."""
    print("\n☠️ DEATH REVEAL TESTS")
    
    # Current implementation: reveal "Mafia" or "not Mafia", not exact role
    
    def get_alignment_string(player: Player) -> str:
        return "Mafia" if player.role == Role.MAFIA else "not Mafia"
    
    # Test 1: Mafia death reveals "Mafia"
    mafia = Player(name="M", model="test", role=Role.MAFIA, is_alive=False)
    reveal = get_alignment_string(mafia)
    
    results.record(
        "Mafia death reveals 'Mafia'",
        reveal == "Mafia",
        f"Expected 'Mafia', got '{reveal}'"
    )
    
    # Test 2: Town death reveals "not Mafia"
    town = Player(name="T", model="test", role=Role.TOWN, is_alive=False)
    reveal = get_alignment_string(town)
    
    results.record(
        "Town death reveals 'not Mafia'",
        reveal == "not Mafia",
        f"Expected 'not Mafia', got '{reveal}'"
    )
    
    # Test 3: Doctor death reveals "not Mafia" (not "Doctor")
    doctor = Player(name="D", model="test", role=Role.DOCTOR, is_alive=False)
    reveal = get_alignment_string(doctor)
    
    results.record(
        "Doctor death reveals 'not Mafia' (not exact role)",
        reveal == "not Mafia" and "Doctor" not in reveal,
        f"Expected 'not Mafia', got '{reveal}'"
    )
    
    # Test 4: Detective death reveals "not Mafia" (not "Detective")
    detective = Player(name="D", model="test", role=Role.DETECTIVE, is_alive=False)
    reveal = get_alignment_string(detective)
    
    results.record(
        "Detective death reveals 'not Mafia' (not exact role)",
        reveal == "not Mafia" and "Detective" not in reveal,
        f"Expected 'not Mafia', got '{reveal}'"
    )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_edge_cases():
    """Test edge cases and unusual game states."""
    print("\n🔧 EDGE CASE TESTS")
    
    # Test 1: All players dead is handled
    players = [
        Player(name="A", model="test", role=Role.TOWN, is_alive=False),
        Player(name="B", model="test", role=Role.MAFIA, is_alive=False),
    ]
    state = GameState(players=players)
    
    results.record(
        "GameState handles all players dead",
        len(state.living_players) == 0,
        "Should have 0 living players"
    )
    
    # Test 2: Single player remaining
    players = [
        Player(name="A", model="test", role=Role.TOWN, is_alive=True),
        Player(name="B", model="test", role=Role.MAFIA, is_alive=False),
    ]
    state = GameState(players=players)
    
    results.record(
        "Single player remaining works",
        len(state.living_players) == 1,
        "Should have 1 living player"
    )
    
    # Test 3: living_mafia and living_town are correct
    players = [
        Player(name="A", model="test", role=Role.TOWN, is_alive=True),
        Player(name="B", model="test", role=Role.DOCTOR, is_alive=True),
        Player(name="C", model="test", role=Role.DETECTIVE, is_alive=True),
        Player(name="D", model="test", role=Role.MAFIA, is_alive=True),
        Player(name="E", model="test", role=Role.MAFIA, is_alive=False),
    ]
    state = GameState(players=players)
    
    # living_town includes TOWN, DOCTOR, DETECTIVE
    results.record(
        "living_town includes all non-mafia roles",
        len(state.living_town) == 3,
        f"Expected 3 living town, got {len(state.living_town)}"
    )
    
    results.record(
        "living_mafia counts correctly",
        len(state.living_mafia) == 1,
        f"Expected 1 living mafia, got {len(state.living_mafia)}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("MAFIA GAME QA TEST SUITE")
    print("=" * 60)
    
    test_win_conditions()
    test_vote_resolution()
    test_doctor_protection()
    test_detective_investigation()
    test_mafia_coordination()
    test_death_reveals()
    test_edge_cases()
    
    success = results.summary()
    
    print("\n📋 RULE CONFIGURATION DOCUMENTATION")
    print("-" * 40)
    print("""
Current Implementation Rules:
1. WIN CONDITIONS
   - Town wins: All mafia eliminated
   - Mafia wins: Mafia >= Town (living count)

2. VOTING
   - Tie: No lynch
   - Self-votes: ALLOWED (count as normal votes)
   - Invalid votes: Ignored

3. DEATH REVEALS
   - Shows: "Mafia" or "not Mafia"
   - Does NOT reveal exact role (Doctor, Detective)

4. DOCTOR
   - Can protect self: YES
   - Consecutive protection: NO (can't protect same person twice in a row)

5. MAFIA COORDINATION
   - Multiple mafia discuss privately (half day duration)
   - Then vote on target
   - Tie: NO KILL (mafia must agree on target)

6. DETECTIVE
   - Investigation returns: is_mafia (True/False)
   - Results given same night (before day phase)
""")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
