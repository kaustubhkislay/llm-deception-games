"""Cached LLM client using the OpenAI Responses API."""

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
    reasoning_summary: Optional[str] = None


def _compute_cache_key(
    model: str,
    input_items: list[dict],
    instructions: Optional[str],
    tools: Optional[list[dict]],
    tool_choice: Optional[str | dict],
    reasoning: Optional[dict] = None,
) -> str:
    """Compute a cache key for an LLM request."""
    cache_data = {
        "model": model,
        "input": input_items,
        "instructions": instructions,
        "tools": tools,
        "tool_choice": tool_choice,
        "reasoning": reasoning,
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
    """Async LLM client with caching support using the Responses API."""
    
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
        reasoning: Optional[dict] = None,
    ) -> LLMResponse:
        """
        Make a request using the Responses API with caching.
        
        Args:
            model: The model to use (e.g., "gpt-5-mini")
            messages: List of ChatMessage objects or dicts (converted to Responses API format)
            tools: Optional list of tool definitions
            tool_choice: Optional tool choice specification
            temperature: Sampling temperature
            reasoning: Optional reasoning settings (e.g., {"effort": "medium", "summary": "detailed"})
            
        Returns:
            LLMResponse with content and/or tool calls
        """
        # Convert ChatMessage objects to Responses API input format
        # Extract system message as instructions, convert rest to input items
        instructions: Optional[str] = None
        input_items: list[dict] = []
        
        for msg in messages:
            if isinstance(msg, ChatMessage):
                role = msg.role
                content = msg.content
                tool_calls = msg.tool_calls
                tool_call_id = msg.tool_call_id
            else:
                role = msg.get("role")
                content = msg.get("content")
                tool_calls = msg.get("tool_calls")
                tool_call_id = msg.get("tool_call_id")
            
            if role == "system":
                # System messages become instructions
                instructions = content
            elif role == "user":
                input_items.append({
                    "type": "message",
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                if tool_calls:
                    # Assistant message with tool calls
                    for tc in tool_calls:
                        # Convert call_ prefix to fc_ for Responses API
                        original_id = tc.get("id", "")
                        fc_id = original_id.replace("call_", "fc_") if original_id.startswith("call_") else original_id
                        input_items.append({
                            "type": "function_call",
                            "id": fc_id,
                            "call_id": fc_id,
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("function", {}).get("arguments", "{}")
                        })
                else:
                    # Regular assistant message
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": content or ""
                    })
            elif role == "tool":
                # Tool response - convert call_ prefix to fc_ for Responses API
                fc_call_id = tool_call_id.replace("call_", "fc_") if tool_call_id and tool_call_id.startswith("call_") else (tool_call_id or "")
                input_items.append({
                    "type": "function_call_output",
                    "call_id": fc_call_id,
                    "output": content or ""
                })
        
        # Check cache (include reasoning settings in cache key)
        if self.use_cache:
            cache_key = _compute_cache_key(model, input_items, instructions, tools, tool_choice, reasoning)
            cached = _load_from_cache(cache_key)
            if cached is not None:
                self._cache_hits += 1
                print(f"  [Cache HIT] {cache_key[:12]}...")
                return self._parse_response(cached)
            self._cache_misses += 1
            print(f"  [Cache MISS] {cache_key[:12]}...")
        
        # Build API request kwargs
        api_kwargs: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "temperature": temperature,
            "store": False,  # Don't store responses on OpenAI's side
        }
        
        if instructions:
            api_kwargs["instructions"] = instructions
        
        if tools:
            # Convert tools from Chat Completions format to Responses API format
            # Chat Completions: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
            # Responses API: {"type": "function", "name": ..., "description": ..., "parameters": ...}
            converted_tools = []
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    func = tool["function"]
                    converted_tools.append({
                        "type": "function",
                        "name": func.get("name"),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    })
                else:
                    # Already in correct format or different type
                    converted_tools.append(tool)
            api_kwargs["tools"] = converted_tools
            if tool_choice:
                api_kwargs["tool_choice"] = tool_choice
        
        if reasoning:
            api_kwargs["reasoning"] = reasoning
        
        # Make API request using Responses API
        response = await self.client.responses.create(**api_kwargs)  # type: ignore
        
        # Track token usage (only for actual API calls, not cache hits)
        self._api_calls += 1
        if response.usage:
            self._total_prompt_tokens += response.usage.input_tokens
            self._total_completion_tokens += response.usage.output_tokens
        
        # Parse response output to extract content, reasoning summary, and tool calls
        content = None
        reasoning_summary = None
        tool_calls_list = []
        
        for item in response.output:
            if item.type == "message":
                # Extract text content from message
                for content_item in item.content:
                    if content_item.type == "output_text":
                        content = content_item.text
            elif item.type == "reasoning":
                # Extract reasoning summary if available
                if hasattr(item, 'summary') and item.summary:
                    summary_texts = []
                    for summary_item in item.summary:
                        if hasattr(summary_item, 'text'):
                            summary_texts.append(summary_item.text)
                    if summary_texts:
                        reasoning_summary = "\n".join(summary_texts)
            elif item.type == "function_call":
                # Extract function call - convert fc_ back to call_ for compatibility
                call_id = item.call_id
                if call_id and call_id.startswith("fc_"):
                    call_id = call_id.replace("fc_", "call_", 1)
                tool_calls_list.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments
                    }
                })
        
        # Use reasoning summary as content if no message content but reasoning exists
        if content is None and reasoning_summary:
            content = reasoning_summary
        elif content and reasoning_summary:
            # Prepend reasoning to content
            content = reasoning_summary + "\n\n" + content
        
        # Convert to dict for caching
        response_dict = {
            "content": content,
            "reasoning_summary": reasoning_summary,
            "tool_calls": tool_calls_list if tool_calls_list else None,
            "status": response.status,
        }
        
        # Save to cache
        if self.use_cache:
            _save_to_cache(cache_key, response_dict)
        
        return self._parse_response(response_dict)
    
    def _parse_response(self, response_dict: dict) -> LLMResponse:
        """Parse a response dict into an LLMResponse."""
        return LLMResponse(
            content=response_dict.get("content"),
            tool_calls=response_dict.get("tool_calls"),
            raw_response=response_dict,
            reasoning_summary=response_dict.get("reasoning_summary"),
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
        "description": "Send a PUBLIC message to the group chat. This message will be seen by ALL players. ONLY include the actual words you want to say out loud to the group. Do NOT include: private reasoning, strategy notes, meta-commentary like 'I can't share my reasoning', labels like 'Public message:', or anything you wouldn't literally say in conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The exact words you want to say out loud to the group. No labels, no reasoning, just your actual message."
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
