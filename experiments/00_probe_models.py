#!/usr/bin/env python3
"""Probe every model slug used in the suite through OpenRouter."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.llm_client import CachedLLMClient
from infra.onuw import ChatMessage

MODELS = [
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.6-sol",
    "anthropic/claude-fable-5",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.2",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.8-max",
]


async def main():
    client = CachedLLMClient(use_cache=False)
    failures = []
    for model in MODELS:
        try:
            resp = await client.chat_completion(
                model=model,
                messages=[ChatMessage(role="user", content="Reply with the word OK.")],
                temperature=1,
                reasoning={"effort": "medium"},
            )
            print(f"PASS  {model}: {(resp.content or '')[:40]!r}")
        except Exception as e:
            failures.append(model)
            print(f"FAIL  {model}: {e}")
    if failures:
        print(f"\n{len(failures)} slug(s) failed: {failures}")
        sys.exit(1)
    print("\nAll slugs OK.")


asyncio.run(main())
