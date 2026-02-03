"""Cached LLM client with function calling support."""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

from openai import AsyncOpenAI

from .onuw import ChatMessage


# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "llm_cache"


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: Optional[str]
    tool_calls: Optional[list[dict]]
    raw_response: dict


def _compute_cache_key(
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]],
    tool_choice: Optional[str | dict]
) -> str:
    """Compute a cache key for an LLM request."""
    cache_data = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice
    }
    cache_str = json.dumps(cache_data, sort_keys=True)
    return hashlib.sha256(cache_str.encode()).hexdigest()


def _get_cache_path(cache_key: str) -> Path:
    """Get the file path for a cache entry."""
    return CACHE_DIR / f"{cache_key}.json"


def _load_from_cache(cache_key: str) -> Optional[dict]:
    """Load a cached response if it exists."""
    cache_path = _get_cache_path(cache_key)
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _save_to_cache(cache_key: str, response: dict) -> None:
    """Save a response to the cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_cache_path(cache_key)
    with open(cache_path, "w") as f:
        json.dump(response, f, indent=2)


class CachedLLMClient:
    """Async LLM client with caching support."""
    
    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        self.client = AsyncOpenAI(api_key=api_key)
        self.use_cache = use_cache
        self._cache_hits = 0
        self._cache_misses = 0
        # Token usage tracking
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._api_calls = 0
    
    async def chat_completion(
        self,
        model: str,
        messages: list[ChatMessage] | list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str | dict] = None,
        temperature: float = 1,
    ) -> LLMResponse:
        """
        Make a chat completion request with caching.
        
        Args:
            model: The model to use (e.g., "gpt-5-nano")
            messages: List of ChatMessage objects or dicts
            tools: Optional list of tool definitions
            tool_choice: Optional tool choice specification
            temperature: Sampling temperature
            
        Returns:
            LLMResponse with content and/or tool calls
        """
        # Convert ChatMessage objects to dicts
        message_dicts = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                msg_dict = {"role": msg.role, "content": msg.content}
                if msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                if msg.tool_call_id:
                    msg_dict["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    msg_dict["name"] = msg.name
                message_dicts.append(msg_dict)
            else:
                message_dicts.append(msg)
        
        # Check cache
        if self.use_cache:
            cache_key = _compute_cache_key(model, message_dicts, tools, tool_choice)
            cached = _load_from_cache(cache_key)
            if cached is not None:
                self._cache_hits += 1
                print(f"  [Cache HIT] {cache_key[:12]}...")
                return self._parse_response(cached)
            self._cache_misses += 1
            print(f"  [Cache MISS] {cache_key[:12]}...")
        
        # Make API request
        if tools:
            response = await self.client.chat.completions.create(
                model=model,
                messages=message_dicts,  # type: ignore
                temperature=temperature,
                tools=tools,  # type: ignore
                tool_choice=tool_choice if tool_choice else "auto",  # type: ignore
            )
        else:
            response = await self.client.chat.completions.create(
                model=model,
                messages=message_dicts,  # type: ignore
                temperature=temperature,
            )
        
        # Track token usage (only for actual API calls, not cache hits)
        self._api_calls += 1
        if response.usage:
            self._total_prompt_tokens += response.usage.prompt_tokens
            self._total_completion_tokens += response.usage.completion_tokens
        
        # Convert to dict for caching
        response_dict = {
            "content": response.choices[0].message.content,
            "tool_calls": None,
            "finish_reason": response.choices[0].finish_reason,
        }
        
        if response.choices[0].message.tool_calls:
            response_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in response.choices[0].message.tool_calls
            ]
        
        # Save to cache
        if self.use_cache:
            _save_to_cache(cache_key, response_dict)
        
        return self._parse_response(response_dict)
    
    def _parse_response(self, response_dict: dict) -> LLMResponse:
        """Parse a response dict into an LLMResponse."""
        return LLMResponse(
            content=response_dict.get("content"),
            tool_calls=response_dict.get("tool_calls"),
            raw_response=response_dict
        )
    
    @property
    def cache_stats(self) -> dict[str, float]:
        """Get cache statistics."""
        return {
            "hits": float(self._cache_hits),
            "misses": float(self._cache_misses),
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        }
    
    @property
    def usage_stats(self) -> dict[str, Any]:
        """Get token usage statistics."""
        return {
            "api_calls": self._api_calls,
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
        }
    
    def get_cost_estimate(self, input_price_per_1m: float = 0.0, output_price_per_1m: float = 0.0) -> dict[str, float]:
        """
        Estimate API costs based on token usage.
        
        Args:
            input_price_per_1m: Price per 1M input tokens (default: 0 for unknown models)
            output_price_per_1m: Price per 1M output tokens (default: 0 for unknown models)
        
        Returns:
            Dict with input_cost, output_cost, total_cost
        """
        input_cost = (self._total_prompt_tokens / 1_000_000) * input_price_per_1m
        output_cost = (self._total_completion_tokens / 1_000_000) * output_price_per_1m
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }
    
    def reset_stats(self) -> None:
        """Reset all usage statistics."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._api_calls = 0


# Tool definitions for players - Round-based messaging
SEND_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": "Send a message to the group chat for this round. All players' messages are revealed simultaneously at the end of the round.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message text to send"
                }
            },
            "required": ["content"]
        }
    }
}

PASS_TURN_TOOL = {
    "type": "function", 
    "function": {
        "name": "pass_turn",
        "description": "Pass this round without sending a message. Use when you want to stay quiet and observe.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

VOTE_TOOL = {
    "type": "function",
    "function": {
        "name": "cast_vote",
        "description": "Cast your vote for who should be lynched. Vote for a living player's name, or 'no_lynch' to vote against lynching anyone.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "The name of the player you want to vote to lynch, or 'no_lynch' to skip"
                }
            },
            "required": ["target"]
        }
    }
}

# Day phase tools (round-based)
DAY_TOOLS = [SEND_MESSAGE_TOOL, PASS_TURN_TOOL]

# Voting phase tools
VOTING_TOOLS = [VOTE_TOOL]


def get_llm_client(use_cache: bool = True) -> CachedLLMClient:
    """Get a cached LLM client instance."""
    return CachedLLMClient(use_cache=use_cache)
