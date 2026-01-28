"""Common utilities and base classes for LLM interactions."""

import abc
import json
import re
from copy import deepcopy
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

    # Tool choice
    # OpenAI: "auto", "required", or {"type": "function", "function": {"name": "my_function"}}
    tool_choice: str | dict | None = Field(default=None)


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
    titles: list[str] = Field(default_factory=list)
    grounding_supports: list[dict[str, Any]] = Field(default_factory=list)
    num_tool_calls: int = 0


class LLMChatResponse(BaseModel):
    """Response from an LLM chat completion."""

    pred: str | dict[str, Any]
    usage: dict[str, Any]
    error: str | None
    finish_reason: str | None = None
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


def parse_json_response(response_text: str | dict[str, Any]) -> dict[str, Any]:
    """Parse a JSON response from the LLM, or search the string for a JSON code block and parse it.

    Args:
        response_text: The response text from the LLM.
        If the response is a dictionary, it will be returned as is.

    Returns:
        The parsed JSON response.

    Raises:
        ValueError: If the response text is not a valid JSON or if the string
            does not contain a JSON code block.

    """
    if not response_text:
        raise ValueError("Empty response from LLM")

    if isinstance(response_text, dict):
        return response_text

    # Try to parse the entire text as JSON
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON in Markdown code block then parse it
    try:
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1).strip())
    except json.JSONDecodeError:
        pass

    # Try to find something that looks like a JSON object (starts with { and ends with })
    # This is a fallback for when the model includes chatter but no backticks
    try:
        # Find the first { and the last }
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            potential_json = response_text[start : end + 1]
            return json.loads(potential_json.strip())
    except json.JSONDecodeError:
        pass

    # If all above fails, raise error with a snippet of the response
    snippet = response_text[:200] + "..." if len(response_text) > 200 else response_text
    raise ValueError(f"Failed to parse JSON response: {snippet}")


def aggregate_usage(usage_list: list[dict]) -> dict:
    """Recursively sum the values of the usage dict.

    Uses the first non-empty dict as reference schema for keys.

    NOTE: assumes that each usage dict in the list shares a common schema.
    Otherwise, value errors may occur.

    NOTE: assumes that the usage values are summable.

    Args:
        usage_list: List of usage dict objects.

    Returns:
        The aggregated usage dict.

    """
    if len(usage_list) == 0:
        return {}

    # Find the first non-empty dict in usage_list and assign it to result
    # Use that dict as reference schema for the aggregated usage dict
    result = {}
    for usage in usage_list:
        if usage:
            result = deepcopy(usage)
            break

    for key in result.keys():
        if isinstance(result[key], dict):
            # result[key] is a dict, nested within the usage dict
            # Since usage lists share a common schema, we can assume that
            # the nested dicts are of the same schema
            # Recursively sum the values of the nested dicts
            # Use .get method to default to empty dict if key not present
            result[key] = aggregate_usage([usage.get(key, {}) for usage in usage_list])
        else:
            # Assume that the values are summable (default 0 if key not present)
            # key may exist in usage dict but have a None value, so convert that to 0
            # Otherwise sum() will complain that it cannot sum None with integer
            result[key] = sum([usage.get(key, 0) or 0 for usage in usage_list])
    return result
