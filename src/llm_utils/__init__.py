"""LLM utilities module for sci-llm project."""

from llm_utils.common import (
    Conversation,
    InferenceGenerationConfig,
    LLMChat,
    LLMChatResponse,
    Message,
)

__all__ = [
    "Conversation",
    "InferenceGenerationConfig",
    "LLMChat",
    "LLMChatResponse",
    "Message",
    "get_llm",
]

# Supported LLM servers
SUPPORTED_SERVERS = ["gemini", "openai"]


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
        case _:
            raise ValueError(
                f"Unsupported server: {server}. "
                f"Supported servers: {', '.join(SUPPORTED_SERVERS)}"
            )
