"""Google Gemini chat implementation."""

import os
from typing import Any

from google import genai
from google.genai import types as genai_types

from llm_utils.common import (
    Conversation,
    File,
    InferenceGenerationConfig,
    LLMChat,
    LLMChatResponse,
)


class GeminiChat(LLMChat):
    """Chat interface for Google Gemini models."""

    def __init__(self, model_name: str) -> None:
        """Initialize the Gemini chat interface.

        Args:
            model_name: The name of the Gemini model to use.

        """
        super().__init__(model_name)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)

    def _convert_conv_to_api_format(self, conv: Conversation) -> list[dict[str, Any]]:
        """Convert conversation to Gemini's format.
        https://ai.google.dev/gemini-api/docs/document-processing#large-pdfs

        Requires:
        - All files in the conversation must be uploaded before calling this method.

        Examples of text and file messages:
        ```
        {
            "role": "user",
            "parts": ["Hello, how are you?"]
        }
        {
            "role": "user",
            "parts": [file.uploaded_handle]
        }
        ```

        Args:
            conv: The conversation to convert.

        Returns:
            Messages in Gemini's format.

        """
        messages = []
        for msg in conv.messages:
            # Gemini uses "model" instead of "assistant"
            role = "model" if msg.role == "assistant" else msg.role
            if isinstance(msg.content, str):
                messages.append({"role": role, "parts": [msg.content]})
            elif isinstance(msg.content, File):
                messages.append({"role": role, "parts": [msg.content.uploaded_handle]})
            else:
                raise ValueError(f"Invalid message content type: {type(msg.content)}")
        return messages

    def _call_api(
        self,
        messages: list[dict[str, Any]],
        inf_gen_config: InferenceGenerationConfig,
    ) -> Any:
        """Call the Gemini API.

        Args:
            messages: Messages in Gemini's format.
            inf_gen_config: Inference generation configuration.

        Returns:
            Raw API response.

        """
        gen_kwargs: dict[str, Any] = {
            "max_output_tokens": inf_gen_config.max_output_tokens,
            "thinking_config": genai_types.ThinkingConfig(
                thinking_budget=inf_gen_config.thinking_budget
            ),
        }
        if inf_gen_config.temperature:
            gen_kwargs["temperature"] = inf_gen_config.temperature
        if inf_gen_config.top_p:
            gen_kwargs["top_p"] = inf_gen_config.top_p
        if inf_gen_config.stop_sequences:
            gen_kwargs["stop_sequences"] = inf_gen_config.stop_sequences

        # If messages has a system message, add it to gen_kwargs["system_instruction"]
        # and remove it from messages
        if messages and messages[0]["role"] == "system":
            gen_kwargs["system_instruction"] = messages[0]["parts"][0]["text"]
            messages = messages[1:]

        # Gemini's chat expects history (all but last message) and current message
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=messages,
            config=genai_types.GenerateContentConfig(**gen_kwargs),
        )

        return response

    def _parse_api_output(self, response: Any) -> LLMChatResponse:
        """Parse Gemini's response.

        Args:
            response: Raw Gemini API response.

        Returns:
            Parsed LLM chat response.

        """
        try:
            pred = response.text
            usage = None
            if hasattr(response, "usage_metadata"):
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }
            return LLMChatResponse(pred=pred, usage=usage, error=None)
        except Exception as e:
            # Handle cases where Gemini refuses to generate
            return LLMChatResponse(
                pred="",
                usage={},
                error=str(e),
            )

    def upload_file(self, file: File) -> None:
        """Upload a file to Gemini.

        Args:
            file: File to upload.

        """
        if not file.is_uploaded():
            file_handle = self.client.files.upload(file=file.path)
            file.uploaded_handle = file_handle

    def delete_file(self, file: File) -> None:
        """Delete an uploaded file from Gemini.

        Args:
            file: The file to delete.

        """
        if file.is_uploaded():
            self.client.files.delete(name=file.uploaded_handle.name)
            file.uploaded_handle = None
