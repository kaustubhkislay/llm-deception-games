#!/usr/bin/env python3
"""
QA Tests for One Night Ultimate Werewolf game logic.

Run with: python experiments/03_qa_tests.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.onuw import (
    Role, Player, GameState, determine_winner, select_roles_for_game,
    get_team, VILLAGE_TEAM, WEREWOLF_TEAM, NEUTRAL_TEAM
)


def test_role_selection():
    """Test that role selection works correctly."""
    print("\n=== Test: Role Selection ===")
    
    # Test 5 players -> should get 5 player roles + 3 center
    pool = [
        Role.WEREWOLF, Role.WEREWOLF,
        Role.SEER, Role.ROBBER, Role.TROUBLEMAKER,
        Role.VILLAGER, Role.VILLAGER, Role.DRUNK
    ]
    
    player_roles, center_roles = select_roles_for_game(5, pool)
    
    assert len(player_roles) == 5, f"Expected 5 player roles, got {len(player_roles)}"
    assert len(center_roles) == 3, f"Expected 3 center roles, got {len(center_roles)}"
    assert len(player_roles) + len(center_roles) == 8, "Total should be 8 roles"
    
    # All roles should be from the pool
    all_roles = player_roles + center_roles
    for role in all_roles:
        assert role in pool, f"Role {role} not in pool"
    
    print("  ✓ Role selection creates correct number of player and center roles")
    print("  ✓ All selected roles are from the pool")


def test_team_assignments():
    """Test team assignment for each role."""
    print("\n=== Test: Team Assignments ===")
    
    assert get_team(Role.WEREWOLF) == "WEREWOLF"
    assert get_team(Role.MINION) == "WEREWOLF"
    assert get_team(Role.VILLAGER) == "VILLAGE"
    assert get_team(Role.SEER) == "VILLAGE"
    assert get_team(Role.ROBBER) == "VILLAGE"
    assert get_team(Role.TROUBLEMAKER) == "VILLAGE"
    assert get_team(Role.DRUNK) == "VILLAGE"
    assert get_team(Role.INSOMNIAC) == "VILLAGE"
    assert get_team(Role.HUNTER) == "VILLAGE"
    assert get_team(Role.TANNER) == "NEUTRAL"
    
    print("  ✓ Werewolf team roles correctly assigned")
    print("  ✓ Village team roles correctly assigned")
    print("  ✓ Neutral roles correctly assigned")


def test_card_swapping():
    """Test card swapping mechanics."""
    print("\n=== Test: Card Swapping ===")
    
    # Create a game state
    players = [
        Player(name="Alice", model="test", original_role=Role.SEER, current_role=Role.SEER),
        Player(name="Bob", model="test", original_role=Role.ROBBER, current_role=Role.ROBBER),
        Player(name="Charlie", model="test", original_role=Role.WEREWOLF, current_role=Role.WEREWOLF),
    ]
    center = [Role.VILLAGER, Role.DRUNK, Role.TROUBLEMAKER]
    
    state = GameState(
        players=players,
        center_cards=center,
        original_assignments={p.name: p.original_role for p in players}
    )
    
    # Test player-to-player swap (Robber robs Alice)
    state.swap_player_cards("Bob", "Alice")
    
    assert state.get_player_by_name("Bob").current_role == Role.SEER, \
        "Bob should now have Seer"
    assert state.get_player_by_name("Alice").current_role == Role.ROBBER, \
        "Alice should now have Robber"
    
    print("  ✓ Player-to-player swap works correctly")
    
    # Test player-to-center swap (Drunk swaps with center)
    state.swap_player_with_center("Charlie", 0)
    
    assert state.get_player_by_name("Charlie").current_role == Role.VILLAGER, \
        "Charlie should now have Villager"
    assert state.center_cards[0] == Role.WEREWOLF, \
        "Center position 0 should now have Werewolf"
    
    print("  ✓ Player-to-center swap works correctly")
    
    # Verify original roles unchanged
    assert state.get_player_by_name("Bob").original_role == Role.ROBBER
    assert state.get_player_by_name("Alice").original_role == Role.SEER
    assert state.get_player_by_name("Charlie").original_role == Role.WEREWOLF
    
    print("  ✓ Original roles preserved after swaps")


def test_win_condition_werewolf_killed():
    """Test: Village wins when a werewolf is killed."""
    print("\n=== Test: Village Wins (Werewolf Killed) ===")
    
    players = [
        Player(name="Alice", model="test", original_role=Role.WEREWOLF, current_role=Role.WEREWOLF),
        Player(name="Bob", model="test", original_role=Role.SEER, current_role=Role.SEER),
        Player(name="Charlie", model="test", original_role=Role.VILLAGER, current_role=Role.VILLAGER),
    ]
    
    state = GameState(
        players=players,
        center_cards=[Role.ROBBER, Role.DRUNK, Role.TROUBLEMAKER],
        original_assignments={p.name: p.original_role for p in players}
    )
    
    # Everyone votes for Alice (the werewolf)
    state.current_votes = {
        "Alice": "Bob",      # Werewolf votes for Seer
        "Bob": "Alice",      # Seer votes for Werewolf
        "Charlie": "Alice"   # Villager votes for Werewolf
    }
    
    winner, killed = determine_winner(state)
    
    assert winner == "VILLAGE", f"Expected VILLAGE to win, got {winner}"
    assert "Alice" in killed, "Alice (werewolf) should be killed"
    
    print("  ✓ Village wins when werewolf receives most votes")


def test_win_condition_no_werewolf_killed():
    """Test: Werewolf wins when no werewolf is killed."""
    print("\n=== Test: Werewolf Wins (No Werewolf Killed) ===")
    
    players = [
        Player(name="Alice", model="test", original_role=Role.WEREWOLF, current_role=Role.WEREWOLF),
        Player(name="Bob", model="test", original_role=Role.SEER, current_role=Role.SEER),
        Player(name="Charlie", model="test", original_role=Role.VILLAGER, current_role=Role.VILLAGER),
    ]
    
    state = GameState(
        players=players,
        center_cards=[Role.ROBBER, Role.DRUNK, Role.TROUBLEMAKER],
        original_assignments={p.name: p.original_role for p in players}
    )
    
    # Everyone votes for Bob (not a werewolf)
    state.current_votes = {
        "Alice": "Bob",
        "Bob": "Charlie",
        "Charlie": "Bob"
    }
    
    winner, killed = determine_winner(state)
    
    assert winner == "WEREWOLF", f"Expected WEREWOLF to win, got {winner}"
    assert "Bob" in killed, "Bob should be killed"
    
    print("  ✓ Werewolf wins when non-werewolf receives most votes")


def test_win_condition_no_werewolves_in_game():
    """Test: Village wins if no werewolves and no one dies."""
    print("\n=== Test: No Werewolves in Game ===")
    
    # All werewolves in center
    players = [
        Player(name="Alice", model="test", original_role=Role.SEER, current_role=Role.SEER),
        Player(name="Bob", model="test", original_role=Role.ROBBER, current_role=Role.ROBBER),
        Player(name="Charlie", model="test", original_role=Role.VILLAGER, current_role=Role.VILLAGER),
    ]
    
    state = GameState(
        players=players,
        center_cards=[Role.WEREWOLF, Role.WEREWOLF, Role.DRUNK],
        original_assignments={p.name: p.original_role for p in players}
    )
    
    # Everyone votes no_one
    state.current_votes = {
        "Alice": "no_one",
        "Bob": "no_one",
        "Charlie": "no_one"
    }
    
    winner, killed = determine_winner(state)
    
    assert winner == "VILLAGE", f"Expected VILLAGE to win (no werewolves, no kill), got {winner}"
    assert len(killed) == 0, "No one should be killed"
    
    print("  ✓ Village wins when no werewolves and no one is killed")
    
    # Test: If someone IS killed and no werewolves, werewolf team wins
    state.current_votes = {
        "Alice": "Bob",
        "Bob": "Alice",
        "Charlie": "Alice"
    }
    
    winner, killed = determine_winner(state)
    
    assert winner == "WEREWOLF", f"Expected WEREWOLF to win (no werewolves but kill), got {winner}"
    
    print("  ✓ Werewolf team wins when no werewolves but village kills someone")


def test_win_condition_tanner():
    """Test: Tanner wins if Tanner is killed."""
    print("\n=== Test: Tanner Win Condition ===")
    
    players = [
        Player(name="Alice", model="test", original_role=Role.TANNER, current_role=Role.TANNER),
        Player(name="Bob", model="test", original_role=Role.WEREWOLF, current_role=Role.WEREWOLF),
        Player(name="Charlie", model="test", original_role=Role.VILLAGER, current_role=Role.VILLAGER),
    ]
    
    state = GameState(
        players=players,
        center_cards=[Role.SEER, Role.ROBBER, Role.DRUNK],
        original_assignments={p.name: p.original_role for p in players}
    )
    
    # Everyone votes for Alice (the Tanner)
    state.current_votes = {
        "Alice": "Bob",
        "Bob": "Alice",
        "Charlie": "Alice"
    }
    
    winner, killed = determine_winner(state)
    
    assert winner == "TANNER", f"Expected TANNER to win, got {winner}"
    assert "Alice" in killed, "Alice (Tanner) should be killed"
    
    print("  ✓ Tanner wins when they are killed")


def test_vote_tie():
    """Test: Tied votes result in multiple deaths."""
    print("\n=== Test: Vote Tie ===")
    
    players = [
        Player(name="Alice", model="test", original_role=Role.WEREWOLF, current_role=Role.WEREWOLF),
        Player(name="Bob", model="test", original_role=Role.SEER, current_role=Role.SEER),
        Player(name="Charlie", model="test", original_role=Role.VILLAGER, current_role=Role.VILLAGER),
        Player(name="Diana", model="test", original_role=Role.ROBBER, current_role=Role.ROBBER),
    ]
    
    state = GameState(
        players=players,
        center_cards=[Role.DRUNK, Role.TROUBLEMAKER, Role.VILLAGER],
        original_assignments={p.name: p.original_role for p in players}
    )
    
    # Tie between Alice and Bob
    state.current_votes = {
        "Alice": "Bob",
        "Bob": "Alice",
        "Charlie": "Alice",
        "Diana": "Bob"
    }
    
    winner, killed = determine_winner(state)
    
    assert "Alice" in killed and "Bob" in killed, \
        f"Both Alice and Bob should die in tie, got {killed}"
    assert winner == "VILLAGE", f"Village should win (werewolf Alice died), got {winner}"
    
    print("  ✓ Tied votes result in multiple deaths")
    print("  ✓ Village wins if werewolf is among tied players")


def test_hunter_ability():
    """Test: Hunter kills their vote target when they die."""
    print("\n=== Test: Hunter Ability ===")
    
    players = [
        Player(name="Alice", model="test", original_role=Role.HUNTER, current_role=Role.HUNTER),
        Player(name="Bob", model="test", original_role=Role.WEREWOLF, current_role=Role.WEREWOLF),
        Player(name="Charlie", model="test", original_role=Role.VILLAGER, current_role=Role.VILLAGER),
    ]
    
    state = GameState(
        players=players,
        center_cards=[Role.SEER, Role.ROBBER, Role.DRUNK],
        original_assignments={p.name: p.original_role for p in players}
    )
    
    # Alice (Hunter) gets killed, but voted for Bob (Werewolf)
    state.current_votes = {
        "Alice": "Bob",      # Hunter votes for Werewolf
        "Bob": "Alice",      # Werewolf votes for Hunter
        "Charlie": "Alice"   # Villager votes for Hunter
    }
    
    winner, killed = determine_winner(state)
    
    # Hunter should take the werewolf with them
    assert "Alice" in killed, "Alice (Hunter) should die"
    assert "Bob" in killed, "Bob (Werewolf) should die from Hunter ability"
    assert winner == "VILLAGE", f"Village should win (werewolf died), got {winner}"
    
    print("  ✓ Hunter kills their vote target when they die")
    print("  ✓ Village can win via Hunter ability")


def test_robber_swap_changes_winner():
    """Test: Robber swap can change who wins."""
    print("\n=== Test: Robber Swap Changes Teams ===")
    
    # Setup: Bob (Robber) steals Alice's (Werewolf) card
    players = [
        Player(name="Alice", model="test", original_role=Role.WEREWOLF, current_role=Role.ROBBER),  # Was robbed
        Player(name="Bob", model="test", original_role=Role.ROBBER, current_role=Role.WEREWOLF),   # Now werewolf!
        Player(name="Charlie", model="test", original_role=Role.VILLAGER, current_role=Role.VILLAGER),
        Player(name="Diana", model="test", original_role=Role.SEER, current_role=Role.SEER),
    ]
    
    state = GameState(
        players=players,
        center_cards=[Role.DRUNK, Role.TROUBLEMAKER, Role.VILLAGER],
        original_assignments={p.name: p.original_role for p in players}
    )
    
    # Village votes for Alice (who they think is werewolf)
    # But Alice is now Robber, Bob is the actual werewolf!
    state.current_votes = {
        "Alice": "Bob",
        "Bob": "Alice",
        "Charlie": "Alice",
        "Diana": "Alice"
    }
    
    winner, killed = determine_winner(state)
    
    assert "Alice" in killed, "Alice should be killed"
    assert "Bob" not in killed, "Bob should NOT be killed"
    assert winner == "WEREWOLF", f"Werewolf (Bob) should win since Alice is now Robber, got {winner}"
    
    print("  ✓ Robber swap correctly changes team affiliations")
    print("  ✓ Win condition uses current roles, not original roles")


def test_drunk_swap():
    """Test: Drunk swap with center."""
    print("\n=== Test: Drunk Swap with Center ===")
    
    players = [
        Player(name="Alice", model="test", original_role=Role.DRUNK, current_role=Role.DRUNK),
        Player(name="Bob", model="test", original_role=Role.SEER, current_role=Role.SEER),
        Player(name="Charlie", model="test", original_role=Role.VILLAGER, current_role=Role.VILLAGER),
    ]
    
    state = GameState(
        players=players,
        center_cards=[Role.WEREWOLF, Role.ROBBER, Role.TROUBLEMAKER],
        original_assignments={p.name: p.original_role for p in players}
    )
    
    # Drunk swaps with center position 0 (Werewolf)
    state.swap_player_with_center("Alice", 0)
    
    assert state.get_player_by_name("Alice").current_role == Role.WEREWOLF, \
        "Alice should now be werewolf"
    assert state.center_cards[0] == Role.DRUNK, \
        "Center 0 should now have Drunk"
    
    # Alice is now a werewolf but doesn't know it!
    # If she gets killed, village wins
    state.current_votes = {
        "Alice": "Bob",
        "Bob": "Alice",
        "Charlie": "Alice"
    }
    
    winner, killed = determine_winner(state)
    
    assert winner == "VILLAGE", f"Village should win (Alice is now werewolf), got {winner}"
    
    print("  ✓ Drunk swap changes player role")
    print("  ✓ Drunk can unknowingly become werewolf")


def run_all_tests():
    """Run all QA tests."""
    print("=" * 60)
    print("ONE NIGHT ULTIMATE WEREWOLF - QA TESTS")
    print("=" * 60)
    
    tests = [
        test_role_selection,
        test_team_assignments,
        test_card_swapping,
        test_win_condition_werewolf_killed,
        test_win_condition_no_werewolf_killed,
        test_win_condition_no_werewolves_in_game,
        test_win_condition_tanner,
        test_vote_tie,
        test_hunter_ability,
        test_robber_swap_changes_winner,
        test_drunk_swap,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n  ✗ FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"\n  ✗ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
