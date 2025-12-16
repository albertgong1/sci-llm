"""Common utilities and base classes for LLM interactions."""

import abc
from typing import Any, Literal

from pydantic import BaseModel, Field


class InferenceGenerationConfig(BaseModel):
    """Configuration for LLM generation parameters."""

    # Non-optional parameters
    max_output_tokens: int = Field(default=1024, ge=1)

    # Optional parameters
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop_sequences: list[str] | None = Field(default=None)

    # Supported by OpenAI Responses API
    reasoning_effort: str | None = Field(default=None)

    # Supported by Gemini API
    thinking_budget: int = Field(default=-1)


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class Conversation(BaseModel):
    """A conversation consisting of multiple messages."""

    messages: list[Message]


class LLMChatResponse(BaseModel):
    """Response from an LLM chat completion."""

    pred: str
    usage: dict[str, Any] = Field(default={})
    error: str | None = None


class LLMChat(abc.ABC):
    """Abstract base class for LLM chat interfaces."""

    def __init__(self, model_name: str) -> None:
        """Initialize the LLM chat interface.

        Args:
            model_name: The name of the model to use.

        """
        self.model_name = model_name

    def generate_response(
        self,
        conv: Conversation,
        inf_gen_config: InferenceGenerationConfig,
    ) -> LLMChatResponse:
        """Generate a response from the LLM.

        Args:
            conv: The conversation history.
            inf_gen_config: Inference generation configuration.

        Returns:
            The LLM's response.

        """
        messages = self._convert_conv_to_api_format(conv)
        response = self._call_api(messages, inf_gen_config)
        return self._parse_api_output(response)

    @abc.abstractmethod
    def _call_api(
        self,
        messages: list[dict[str, Any]],
        inf_gen_config: InferenceGenerationConfig,
    ) -> Any:
        """Call the server's API.

        Args:
            messages: Messages in server-specific format.
            inf_gen_config: Inference generation configuration.

        Returns:
            Raw API response.

        """
        pass

    @abc.abstractmethod
    def _parse_api_output(self, response: Any) -> LLMChatResponse:
        """Parse the API response into a standardized format.

        Args:
            response: Raw API response.

        Returns:
            Parsed LLM chat response.

        """
        pass

    def _convert_conv_to_api_format(self, conv: Conversation) -> list[dict[str, Any]]:
        """Convert conversation to server-specific format.
        The default implementation converts a conversation to a list of messages
        of the form {"role": "user" | "assistant" | "system", "content": str}.

        Args:
            conv: The conversation to convert.

        Returns:
            Messages in server-specific format.

        """
        return [{"role": msg.role, "content": msg.content} for msg in conv.messages]
