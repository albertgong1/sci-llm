"""LLM utilities module for sci-llm project."""

from pathlib import Path
import yaml

from llm_utils.common import (
    Conversation,
    InferenceGenerationConfig,
    LLMChat,
    LLMChatResponse,
    Message,
    aggregate_usage,
)

# Load model pricing configuration from YAML
_PRICING_CONFIG_PATH = Path(__file__).parent / "model_pricing.yaml"
with open(_PRICING_CONFIG_PATH, "r") as f:
    MODEL_PRICING: dict = yaml.safe_load(f)

__all__ = [
    "Conversation",
    "InferenceGenerationConfig",
    "LLMChat",
    "LLMChatResponse",
    "Message",
    "aggregate_usage",
    "get_llm",
    "MODEL_PRICING",
    "calculate_cost",
]

# Supported LLM servers
SUPPORTED_SERVERS = ["gemini", "openai", "qwen"]


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    thinking_tokens: int = 0,
) -> float | None:
    """Calculate dollar cost based on token usage.

    Args:
        model: Model name (must be in MODEL_PRICING).
        prompt_tokens: Total prompt/input tokens.
        completion_tokens: Completion/output tokens.
        cached_tokens: Cached input tokens (subtracted from prompt_tokens for pricing).
        thinking_tokens: Thinking/reasoning tokens (priced as output).

    Returns:
        Total cost in USD, or None if model pricing is unavailable.

    """
    # Strip provider prefix (e.g., "openai/gpt-5" -> "gpt-5")
    model_key = model.split("/")[-1] if "/" in model else model

    if model_key not in MODEL_PRICING:
        return None

    prices = MODEL_PRICING[model_key]
    if prices is None or not isinstance(prices, dict):
        return None

    input_price = prices.get("input")  # USD per 1M tokens
    cache_price = prices.get("context_cache_read") or prices.get(
        "cached_input"
    )  # USD per 1M tokens
    output_price = prices.get("output")  # USD per 1M tokens

    if input_price is None or output_price is None:
        return None

    # Calculate cost (divide by 1M since prices are per 1M tokens)
    prompt_cost = (prompt_tokens - cached_tokens) * input_price / 1_000_000
    cache_cost = cached_tokens * (cache_price or 0) / 1_000_000
    thinking_cost = thinking_tokens * output_price / 1_000_000
    completion_cost = completion_tokens * output_price / 1_000_000

    return prompt_cost + cache_cost + thinking_cost + completion_cost


def get_llm(server: str, model_name: str) -> LLMChat:
    """Factory function to get an LLM chat instance.

    Args:
        server: The LLM server.
        model_name: The name of the model to use.

    Returns:
        An LLMChat instance for the specified server.

    Raises:
        ValueError: If the server is not supported.

    """
    match server.lower():
        case "gemini":
            from llm_utils.gemini import GeminiChat

            return GeminiChat(model_name)
        case "openai":
            from llm_utils.openai import OpenAIChat

            return OpenAIChat(model_name)
        case "qwen":
            from llm_utils.qwen import QwenChat

            return QwenChat(model_name)
        case _:
            raise ValueError(
                f"Unsupported server: {server}. "
                f"Supported servers: {', '.join(SUPPORTED_SERVERS)}"
            )
