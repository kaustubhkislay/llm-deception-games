# LLM Mafia

An AI-powered game of Mafia where LLM agents play against each other with a live web viewer.

## Features

- **7 AI Players**: Alice, Bob, Charlie, Diana, Edward, Fiona, and George
- **4 Roles**: Mafia (2), Doctor (1), Detective (1), Town (3)
- **Real-time Web Viewer**: Watch the game unfold live with SSE updates
- **Player Thoughts Viewer**: See inside any player's mind - their full LLM conversation
- **Blind Writing Mechanic**: Players are "blind" during inference, creating realistic crosstalk

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

Then open http://localhost:5000 in your browser to watch the game.

## Game Flow

1. **Night Phase**: Mafia chooses a kill target, Doctor chooses someone to save, Detective investigates
2. **Day Phase**: 5 minutes of discussion in the group chat
3. **Voting Phase**: Players vote to lynch someone
4. Repeat until Mafia or Town wins

## Command Line Options

```bash
python experiments/01_run_game.py --help

Options:
  --day-duration SECONDS  Duration of day phase (default: 300 = 5 minutes)
  --port PORT            Web viewer port (default: 5000)
  --model MODEL          OpenAI model to use (default: gpt-5-mini)
```

## Architecture

```
llm-mafia/
├── infra/
│   ├── mafia.py          # Core game state and data classes
│   ├── player.py         # Player agent with LLM integration
│   ├── game_engine.py    # Game loop and phase management
│   ├── llm_client.py     # Cached OpenAI client
│   └── events.py         # Event broadcasting for web viewer
├── web/
│   ├── app.py            # Flask application with SSE
│   ├── templates/        # HTMX templates
│   └── static/           # CSS styling
├── experiments/
│   └── 01_run_game.py    # Main run script
└── llm_cache/            # Cached LLM responses (gitignored)
```

## The "Blind Writing" Mechanic

When a player decides to send a message, they only see the chat state at the moment they start their LLM inference. Any messages posted during inference are invisible until after they send their message. This creates realistic "crosstalk" where players may:

- Respond to outdated information
- Talk over each other
- Have their points addressed before they finish "typing"

Messages are ordered by their completion timestamp, so faster responses appear first.
