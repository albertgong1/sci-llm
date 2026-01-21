"""OpenAI chat implementation."""

import os
from typing import Any

from openai import OpenAI

from llm_utils.common import (
    Conversation,
    File,
    InferenceGenerationConfig,
    LLMChat,
    LLMChatResponse,
)


class OpenAIChat(LLMChat):
    """Chat interface for OpenAI models."""

    def __init__(self, model_name: str) -> None:
        """Initialize the OpenAI chat interface.

        Args:
            model_name: The name of the OpenAI model to use.

        """
        super().__init__(model_name)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

    def _convert_conv_to_api_format(self, conv: Conversation) -> list[dict[str, Any]]:
        """Convert conversation to OpenAI's Responses API format.
        # https://platform.openai.com/docs/guides/pdf-files

        Requires:
        - All files in the conversation must be uploaded before calling this method.

        Examples of text and file messages:
        ```
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Hello, how are you?"},
                {"type": "input_file", "file_id": file.uploaded_handle.id}
            ]
        }
        ```

        Args:
            conv: The conversation to convert.

        Returns:
            Messages in OpenAI's format.

        """
        messages = []
        for msg in conv.messages:
            content: list[dict[str, Any]] = []
            for c in msg.content:
                if isinstance(c, str):
                    content.append({"type": "input_text", "text": c})
                elif isinstance(c, File):
                    content.append(
                        {
                            "type": "input_file",
                            "file_id": c.uploaded_handle.id,
                        }
                    )
                else:
                    raise ValueError(f"Invalid message content type: {type(c)}")
            messages.append({"role": msg.role, "content": content})
        return messages

    def _call_api(
        self,
        messages: list[dict[str, Any]],
        inf_gen_config: InferenceGenerationConfig,
    ) -> Any:
        """Call the OpenAI Responses API.
        https://platform.openai.com/docs/guides/migrate-to-responses

        Args:
            messages: Messages in OpenAI's format.
            inf_gen_config: Inference generation configuration.

        Returns:
            Raw API response.

        """
        # Build generation kwargs for OpenAI API
        gen_kwargs: dict[str, Any] = {
            "max_output_tokens": inf_gen_config.max_output_tokens,
        }

        # Add optional parameters
        if inf_gen_config.reasoning_effort:
            gen_kwargs["reasoning"] = {"effort": inf_gen_config.reasoning_effort}
        if inf_gen_config.temperature:
            gen_kwargs["temperature"] = inf_gen_config.temperature
        if inf_gen_config.top_p:
            gen_kwargs["top_p"] = inf_gen_config.top_p

        # NOTE: InferenceGenerationConfig params not supported by OpenAI Responses API:
        # - stop sequences

        # If messages has a system message, add it to kwargs["instructions"]
        # and remove it from messages
        if messages[0]["role"] == "system":
            gen_kwargs["instructions"] = messages[0]["content"][0]["text"]
            messages = messages[1:]

        response = self.client.responses.create(
            model=self.model_name,
            input=messages,
            **gen_kwargs,
        )
        return response

    def _parse_api_output(
        self, response: Any, inf_gen_config: InferenceGenerationConfig
    ) -> LLMChatResponse:
        """Parse OpenAI's response.

        Args:
            response: Raw OpenAI API response.
            inf_gen_config: Inference generation configuration.

        Returns:
            Parsed LLM chat response.

        """
        try:
            pred = response.output_text
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            # TODO: Add web_search_metadata extraction when implemented
            return LLMChatResponse(pred=pred, usage=usage, error=None, web_search_metadata=None)
        except Exception as e:
            return LLMChatResponse(pred="", usage={}, error=str(e), web_search_metadata=None)

    def upload_file(self, file: File) -> None:
        """Upload a file to OpenAI server.

        Args:
            file: File to upload.

        """
        if not file.is_uploaded():
            with open(file.path, "rb") as f:
                file_handle = self.client.files.create(file=f, purpose="user_data")
                file.uploaded_handle = file_handle

    def delete_file(self, file: File) -> None:
        """Delete an uploaded file from OpenAI server.

        Args:
            file: The file to delete.

        """
        if file.is_uploaded():
            self.client.files.delete(file.uploaded_handle.id)
            file.uploaded_handle = None
