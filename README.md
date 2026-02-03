# LLM One Night Ultimate Werewolf

An AI-powered game of One Night Ultimate Werewolf where LLM agents play against each other with a live web viewer.

## Features

- **5+ AI Players**: Configurable number of players
- **10 ONUW Roles**: Werewolf, Seer, Robber, Troublemaker, Drunk, Insomniac, Hunter, Minion, Tanner, Villager
- **Card Swapping Mechanics**: Robber, Troublemaker, and Drunk can swap cards during the night
- **Real-time Web Viewer**: Watch the game unfold live with SSE updates
- **God View**: See true roles, center cards, and all swaps as they happen
- **Player Thoughts Viewer**: See inside any player's mind - their full LLM conversation

## Quick Start

```bash
# Set up environment
uv venv
source .venv/bin/activate
uv pip install -e .

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run a game
python experiments/01_run_game.py
```

Then open http://localhost:9000 in your browser to watch the game.

## Game Flow

One Night Ultimate Werewolf is a single-night social deduction game:

1. **Setup**: Deal N+3 role cards - N to players, 3 to the center
2. **Night Phase**: Roles wake up in order and perform their actions:
   - Werewolves see each other (or peek at center if alone)
   - Minion sees who the werewolves are
   - Seer looks at one player's card OR two center cards
   - Robber steals another player's card and sees their new role
   - Troublemaker swaps two other players' cards
   - Drunk swaps their card with a center card (blind)
   - Insomniac looks at their own card
3. **Day Phase**: Discussion and accusations
4. **Voting**: Everyone points at who they think is a werewolf
5. **Game Over**: 
   - Village wins if at least one Werewolf is killed
   - Werewolf team wins if no Werewolf is killed
   - Tanner wins if they get killed

## Win Conditions

- **Village Team** (Villager, Seer, Robber, Troublemaker, Drunk, Insomniac, Hunter): Win if at least one Werewolf player is killed
- **Werewolf Team** (Werewolf, Minion): Win if no Werewolf player is killed
- **Tanner**: Wins only if they are killed
- **No Werewolves in Game**: Village wins only if no one is killed

## Command Line Options

```bash
python experiments/01_run_game.py --help

Options:
  --day-duration SECONDS  Duration of day phase (default: 300 = 5 minutes)
  --port PORT            Web viewer port (default: 9000)
  --model MODEL          OpenAI model to use (default: gpt-4o-mini)
  --players N            Number of players (default: 5)
```

## Architecture

```
llm-mafia/
├── infra/
│   ├── onuw.py           # Core game state and data classes
│   ├── player.py         # Player agent with LLM integration
│   ├── game_engine.py    # Game loop and phase management
│   ├── llm_client.py     # Cached OpenAI client with ONUW tools
│   └── events.py         # Event broadcasting for web viewer
├── web/
│   ├── app.py            # Flask application with SSE
│   ├── templates/        # HTML templates
│   └── static/           # CSS styling
├── experiments/
│   ├── 01_run_game.py    # Run single game with web viewer
│   ├── 02_replay_game.py # Replay from log file
│   ├── 03_qa_tests.py    # QA tests for game logic
│   └── 04_batch_games.py # Run multiple games for statistics
└── llm_cache/            # Cached LLM responses (gitignored)
```

## Role Abilities

| Role | Team | Night Action |
|------|------|--------------|
| Werewolf | Werewolf | See other werewolves; if alone, may look at one center card |
| Minion | Werewolf | See who the werewolves are (they don't know you) |
| Seer | Village | Look at one player's card OR two center cards |
| Robber | Village | Swap your card with another player's, see your new role |
| Troublemaker | Village | Swap two other players' cards (don't see them) |
| Drunk | Village | Swap your card with a center card (don't see it) |
| Insomniac | Village | Look at your own card at the end of night |
| Hunter | Village | No night action. If you die, your vote target also dies |
| Tanner | Neutral | No night action. You win only if you die |
| Villager | Village | No night action |

## The Challenge of ONUW for LLMs

This game is particularly challenging for LLMs because:

1. **Hidden Information**: Players don't know their final role if swapped
2. **Deductive Reasoning**: Must piece together information from claims
3. **Bluffing Detection**: Must identify when others are lying
4. **Strategic Deception**: Werewolves must convincingly lie
5. **Information Asymmetry**: Each role knows different things

The Robber knows their new role but not if Troublemaker swapped them again. The Drunk doesn't know what they became. The Seer's information may be outdated if swaps happened after they looked.

## QA Tests

Run the QA tests to verify game logic:

```bash
python experiments/03_qa_tests.py
```

Tests cover: role selection, card swapping, win conditions (village, werewolf, tanner), vote ties, Hunter ability, and swap interactions.
