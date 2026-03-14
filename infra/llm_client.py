"""Cached LLM client supporting OpenAI, Anthropic, and OpenRouter."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from openai import AsyncOpenAI
import anthropic

# Load environment variables from .env file (override=True to take precedence over shell env)
load_dotenv(override=True)

from .onuw import ChatMessage


# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "llm_cache"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


def detect_provider(model: str) -> LLMProvider:
    """
    Detect the LLM provider based on the model name.
    
    - Models starting with 'claude-' → Anthropic
    - Models containing '/' → OpenRouter (uses path-based model names like 'anthropic/claude-3-opus')
    - Everything else → OpenAI (default)
    """
    if "/" in model:
        return LLMProvider.OPENROUTER
    if model.startswith("claude-"):
        return LLMProvider.ANTHROPIC
    return LLMProvider.OPENAI


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: Optional[str]
    tool_calls: Optional[list[dict]]
    raw_response: dict
    usage: Optional[dict] = None
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
    """Async LLM client with caching support for OpenAI, Anthropic, and OpenRouter."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        use_cache: bool = True,
        anthropic_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ):
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=api_key)
        
        # Initialize Anthropic client
        anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_key) if anthropic_key else None
        
        # Initialize OpenRouter client (uses OpenAI-compatible API)
        openrouter_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self.openrouter_client = AsyncOpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        ) if openrouter_key else None
        
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
        Make a request to the appropriate LLM provider with caching.
        
        Args:
            model: The model to use. Provider is auto-detected:
                   - "claude-*" → Anthropic
                   - "provider/model" → OpenRouter  
                   - Otherwise → OpenAI
            messages: List of ChatMessage objects or dicts
            tools: Optional list of tool definitions (Chat Completions format)
            tool_choice: Optional tool choice specification
            temperature: Sampling temperature
            reasoning: Optional reasoning settings (OpenAI only)
            
        Returns:
            LLMResponse with content and/or tool calls
        """
        provider = detect_provider(model)
        
        if provider == LLMProvider.ANTHROPIC:
            return await self._anthropic_completion(model, messages, tools, tool_choice, temperature)
        elif provider == LLMProvider.OPENROUTER:
            return await self._openrouter_completion(model, messages, tools, tool_choice, temperature)
        else:
            return await self._openai_completion(model, messages, tools, tool_choice, temperature, reasoning)
    
    async def _openai_completion(
        self,
        model: str,
        messages: list[ChatMessage] | list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str | dict] = None,
        temperature: float = 1,
        reasoning: Optional[dict] = None,
    ) -> LLMResponse:
        """OpenAI implementation using the Responses API."""
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
        
        # Inject summary=auto into reasoning before cache key + API call
        effective_reasoning = reasoning
        if reasoning:
            effective_reasoning = dict(reasoning)
            effective_reasoning.setdefault("summary", "auto")

        # Check cache (include reasoning settings in cache key)
        if self.use_cache:
            cache_key = _compute_cache_key(model, input_items, instructions, tools, tool_choice, effective_reasoning)
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
            "store": False,
        }
        
        if instructions:
            api_kwargs["instructions"] = instructions
        
        if tools:
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
                    converted_tools.append(tool)
            api_kwargs["tools"] = converted_tools
            if tool_choice:
                api_kwargs["tool_choice"] = tool_choice
        
        if effective_reasoning:
            api_kwargs["reasoning"] = effective_reasoning
        
        # Make API request using Responses API
        response = await self.openai_client.responses.create(**api_kwargs)  # type: ignore
        
        # Track token usage (only for actual API calls, not cache hits)
        self._api_calls += 1
        usage_dict = None
        if response.usage:
            self._total_prompt_tokens += response.usage.input_tokens
            self._total_completion_tokens += response.usage.output_tokens
            usage_dict = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            details = getattr(response.usage, "output_tokens_details", None)
            if details:
                reasoning_toks = getattr(details, "reasoning_tokens", 0)
                usage_dict["reasoning_tokens"] = reasoning_toks
        
        # Parse response output
        content = None
        tool_calls_list = []
        reasoning_summary = None
        
        for item in response.output:
            if item.type == "reasoning":
                summaries = getattr(item, "summary", None)
                if summaries:
                    reasoning_summary = "\n\n".join(
                        s.text for s in summaries if getattr(s, "text", None)
                    ) or None
            elif item.type == "message":
                for content_item in item.content:
                    if content_item.type == "output_text":
                        content = content_item.text
            elif item.type == "function_call":
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
        
        # Convert to dict for caching
        response_dict = {
            "content": content,
            "tool_calls": tool_calls_list if tool_calls_list else None,
            "status": response.status,
            "usage": usage_dict,
            "reasoning_summary": reasoning_summary,
        }
        
        # Save to cache
        if self.use_cache:
            _save_to_cache(cache_key, response_dict)
        
        return self._parse_response(response_dict)
    
    async def _anthropic_completion(
        self,
        model: str,
        messages: list[ChatMessage] | list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str | dict] = None,
        temperature: float = 1,
    ) -> LLMResponse:
        """Anthropic implementation using the Messages API."""
        if self.anthropic_client is None:
            raise ValueError("Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable.")
        
        # Convert ChatMessage objects to Anthropic Messages API format
        system_prompt: Optional[str] = None
        api_messages: list[dict] = []
        
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
                system_prompt = content
            elif role == "user":
                api_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                if tool_calls:
                    # Assistant message with tool use
                    tool_use_blocks = []
                    for tc in tool_calls:
                        try:
                            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}
                        tool_use_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "input": args
                        })
                    api_messages.append({
                        "role": "assistant",
                        "content": tool_use_blocks
                    })
                else:
                    api_messages.append({
                        "role": "assistant",
                        "content": content or ""
                    })
            elif role == "tool":
                # Tool result - Anthropic expects this as a user message with tool_result block
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id or "",
                        "content": content or ""
                    }]
                })
        
        # Check cache
        if self.use_cache:
            cache_key = _compute_cache_key(model, api_messages, system_prompt, tools, tool_choice)
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
            "messages": api_messages,
            "max_tokens": 4096,
            "temperature": temperature,
        }
        
        if system_prompt:
            api_kwargs["system"] = system_prompt
        
        if tools:
            # Convert tools from Chat Completions format to Anthropic format
            anthropic_tools = []
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    func = tool["function"]
                    anthropic_tools.append({
                        "name": func.get("name"),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                    })
            api_kwargs["tools"] = anthropic_tools
            
            # Handle tool_choice
            if tool_choice:
                if tool_choice == "required":
                    api_kwargs["tool_choice"] = {"type": "any"}
                elif tool_choice == "auto":
                    api_kwargs["tool_choice"] = {"type": "auto"}
                elif tool_choice == "none":
                    # Don't send tools if none is specified
                    del api_kwargs["tools"]
                elif isinstance(tool_choice, dict) and "function" in tool_choice:
                    api_kwargs["tool_choice"] = {
                        "type": "tool",
                        "name": tool_choice["function"]["name"]
                    }
        
        # Make API request
        response = await self.anthropic_client.messages.create(**api_kwargs)
        
        # Track token usage
        self._api_calls += 1
        if response.usage:
            self._total_prompt_tokens += response.usage.input_tokens
            self._total_completion_tokens += response.usage.output_tokens
        
        # Parse response
        content = None
        tool_calls_list = []
        
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls_list.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
        
        # Convert to dict for caching
        response_dict = {
            "content": content,
            "tool_calls": tool_calls_list if tool_calls_list else None,
            "stop_reason": response.stop_reason,
        }
        
        # Save to cache
        if self.use_cache:
            _save_to_cache(cache_key, response_dict)
        
        return self._parse_response(response_dict)
    
    async def _openrouter_completion(
        self,
        model: str,
        messages: list[ChatMessage] | list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str | dict] = None,
        temperature: float = 1,
    ) -> LLMResponse:
        """OpenRouter implementation using their OpenAI-compatible Chat Completions API."""
        if self.openrouter_client is None:
            raise ValueError("OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable.")
        
        # Convert ChatMessage objects to Chat Completions format
        api_messages: list[dict] = []
        
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
                api_messages.append({"role": "system", "content": content})
            elif role == "user":
                api_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                msg_dict: dict[str, Any] = {"role": "assistant"}
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                    msg_dict["content"] = content or ""
                else:
                    msg_dict["content"] = content or ""
                api_messages.append(msg_dict)
            elif role == "tool":
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id or "",
                    "content": content or ""
                })
        
        # Check cache
        if self.use_cache:
            cache_key = _compute_cache_key(model, api_messages, None, tools, tool_choice)
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
            "messages": api_messages,
            "temperature": temperature,
        }
        
        if tools:
            api_kwargs["tools"] = tools
            if tool_choice:
                api_kwargs["tool_choice"] = tool_choice
        
        # Make API request using Chat Completions API
        response = await self.openrouter_client.chat.completions.create(**api_kwargs)
        
        # Track token usage
        self._api_calls += 1
        if response.usage:
            self._total_prompt_tokens += response.usage.prompt_tokens
            self._total_completion_tokens += response.usage.completion_tokens
        
        # Parse response
        choice = response.choices[0]
        message = choice.message
        
        content = message.content
        tool_calls_list = None
        
        if message.tool_calls:
            tool_calls_list = []
            for tc in message.tool_calls:
                tool_calls_list.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
        
        # Convert to dict for caching
        response_dict = {
            "content": content,
            "tool_calls": tool_calls_list,
            "finish_reason": choice.finish_reason,
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
            usage=response_dict.get("usage"),
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
        "description": "Cast your vote for who should be eliminated. You MUST vote for another player - there is no option to abstain.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "The name of the player you want to vote to eliminate. Must be one of the other players."
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
