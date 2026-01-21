"""Common utilities and base classes for LLM interactions."""

import abc
from pathlib import Path
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
    # Docs: https://ai.google.dev/api/generate-content#ThinkingConfig
    thinking_budget: int = Field(default=-1)

    # Output format
    # Docs: https://ai.google.dev/api/generate-content#generationconfig
    output_format: Literal["text", "json"] = Field(default="text")

    # Web search / grounding
    # Gemini: https://ai.google.dev/gemini-api/docs/google-search
    # OpenAI: https://platform.openai.com/docs/guides/tools-web-search
    use_web_search: bool = Field(default=False)


class File(BaseModel):
    """A file wrapper to interface with the LLM server."""

    path: Path
    uploaded_handle: Any | None = None

    def is_uploaded(self) -> bool:
        """Check if a file is uploaded to the LLM server."""
        return self.uploaded_handle is not None


class Message(BaseModel):
    """A single message in a conversation. The content is list of either text string or a file."""

    role: Literal["user", "assistant", "system"]
    content: list[str | File]


class Conversation(BaseModel):
    """A conversation consisting of multiple messages."""

    messages: list[Message]


class WebSearchMetadata(BaseModel):
    """Metadata about the web search uris.
    
    Args:
        queries: The queries used to search the web.
        uris: The uris of the web search results.
    """

    queries: list[str]
    uris: list[str]


class LLMChatResponse(BaseModel):
    """Response from an LLM chat completion."""

    pred: str | dict[str, Any]
    usage: dict[str, Any]
    error: str | None
    thought: str | None = None
    web_search_metadata: WebSearchMetadata | None = None


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
        # Upload all files first
        for msg in conv.messages:
            for content in msg.content:
                if isinstance(content, File):
                    self.upload_file(content)

        messages = self._convert_conv_to_api_format(conv)
        response = self._call_api(messages, inf_gen_config)
        return self._parse_api_output(response, inf_gen_config)

    async def generate_response_async(
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
        # Upload all files first
        for msg in conv.messages:
            for content in msg.content:
                if isinstance(content, File):
                    self.upload_file(content)

        messages = self._convert_conv_to_api_format(conv)
        response = await self._call_api(messages, inf_gen_config, use_async=True)
        return self._parse_api_output(response, inf_gen_config)

    @abc.abstractmethod
    def _convert_conv_to_api_format(self, conv: Conversation) -> list[dict[str, Any]]:
        """Convert conversation to server-specific format.

        Args:
            conv: The conversation to convert.

        Returns:
            Messages in server-specific format.

        """
        pass

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
    def _parse_api_output(
        self, response: Any, inf_gen_config: InferenceGenerationConfig
    ) -> LLMChatResponse:
        """Parse the API response into a standardized format.

        Args:
            response: Raw API response.
            inf_gen_config: Inference generation configuration.

        Returns:
            Parsed LLM chat response.

        """
        pass

    @abc.abstractmethod
    def upload_file(self, file: File) -> None:
        """Upload a file to the LLM server, if it does not already exist.

        Args:
            file: File to upload.

        """
        pass

    @abc.abstractmethod
    def delete_file(self, file: File) -> None:
        """Delete an uploaded file from the LLM server, if it exists.

        Args:
            file: The file to delete.

        """
        pass
