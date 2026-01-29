#!/usr/bin/env python3
"""Stop a running LLM Mafia game server."""

import argparse
import requests
import sys


def main():
    parser = argparse.ArgumentParser(description="Stop a running LLM Mafia game")
    parser.add_argument("--port", type=int, default=9000, help="Port the game is running on (default: 9000)")
    args = parser.parse_args()
    
    url = f"http://localhost:{args.port}/api/stop"
    
    try:
        print(f"Stopping game on port {args.port}...")
        response = requests.post(url, timeout=5)
        if response.ok:
            print("✓ Game server stopping")
        else:
            print(f"✗ Failed to stop: {response.status_code}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"✗ No game running on port {args.port}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
