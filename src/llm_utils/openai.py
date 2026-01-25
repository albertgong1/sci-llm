"""OpenAI chat implementation."""

import json
import os
from typing import Any

from openai import AsyncOpenAI, OpenAI, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from llm_utils.common import (
    Conversation,
    File,
    InferenceGenerationConfig,
    LLMChat,
    LLMChatResponse,
)


logger = logging.getLogger(__name__)

# Retry decorator for Gemini API calls
_retry_decorator = retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(initial=1, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
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
        self.aio_client = AsyncOpenAI(api_key=api_key)

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

    def _build_gen_kwargs(
        self, messages: list[dict[str, Any]], inf_gen_config: InferenceGenerationConfig
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build generation kwargs for the OpenAI API.

        Args:
            messages: Messages in OpenAI's format.
            inf_gen_config: Inference generation configuration.

        Returns:
            Tuple of (possibly modified messages, generation kwargs).

        """
        # Build generation kwargs for OpenAI API
        gen_kwargs: dict[str, Any] = {
            "max_output_tokens": inf_gen_config.max_output_tokens,
        }

        # Add optional parameters
        if inf_gen_config.reasoning_effort:
            # TODO: add reasoning summary after verifying organization
            # 2026-01-24 22:17:13 - pbench_eval.zeroshot - ERROR - Error processing 60c74e7e337d6c10b9e27fad: Error code: 400 - {'error': {'message': 'Your organization must be verified to generate reasoning summaries. Please go to: https://platform.openai.com/settings/organization/general and click on Verify Organization. If you just verified, it can take up to 15 minutes for access to propagate.', 'type': 'invalid_request_error', 'param': 'reasoning.summary', 'code': 'unsupported_value'}}
            gen_kwargs["reasoning"] = {"effort": inf_gen_config.reasoning_effort}
        if inf_gen_config.temperature:
            gen_kwargs["temperature"] = inf_gen_config.temperature
        if inf_gen_config.top_p:
            gen_kwargs["top_p"] = inf_gen_config.top_p
        if inf_gen_config.use_web_search:
            gen_kwargs["tools"] = [{"type": "web_search"}]

        # https://platform.openai.com/docs/api-reference/responses/create#responses_create-text
        if inf_gen_config.output_format == "json":
            gen_kwargs["text"] = {"format": {"type": "json_object"}}
        elif inf_gen_config.output_format == "text":
            gen_kwargs["text"] = {"format": {"type": "text"}}
        else:
            raise ValueError(f"Invalid output format: {inf_gen_config.output_format}")

        # NOTE: InferenceGenerationConfig params not supported by OpenAI Responses API:
        # - stop sequences

        # If messages has a system message, add it to kwargs["instructions"]
        # and remove it from messages
        if messages[0]["role"] == "system":
            gen_kwargs["instructions"] = messages[0]["content"][0]["text"]
            messages = messages[1:]

        return messages, gen_kwargs

    def _call_api(
        self,
        messages: list[dict[str, Any]],
        inf_gen_config: InferenceGenerationConfig,
        use_async: bool = False,
    ) -> Any:
        """Call the OpenAI Responses API.
        https://platform.openai.com/docs/guides/migrate-to-responses

        Args:
            messages: Messages in OpenAI's format.
            inf_gen_config: Inference generation configuration.
            use_async: Whether to return an async coroutine.

        Returns:
            Raw API response.

        """
        messages, gen_kwargs = self._build_gen_kwargs(messages, inf_gen_config)

        if use_async:
            return self._call_api_async_with_retry(messages, gen_kwargs)
        else:
            return self._call_api_sync_with_retry(messages, gen_kwargs)

    @_retry_decorator
    def _call_api_sync_with_retry(
        self,
        messages: list[dict[str, Any]],
        gen_kwargs: dict[str, Any],
    ) -> Any:
        """Call the OpenAI Responses API synchronously with retry/backoff."""
        return self.client.responses.create(
            model=self.model_name,
            input=messages,
            **gen_kwargs,
        )

    @_retry_decorator
    async def _call_api_async_with_retry(
        self,
        messages: list[dict[str, Any]],
        gen_kwargs: dict[str, Any],
    ) -> Any:
        """Call the OpenAI Responses API asynchronously with retry/backoff."""
        return await self.aio_client.responses.create(
            model=self.model_name,
            input=messages,
            **gen_kwargs,
        )

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
        # import pdb; pdb.set_trace()
        # thought = None
        # for part in response.output:
        #     if part["type"] != "reasoning":
        #         continue
        #     thought = part["summary"][0]["text"]
        # Reference: https://platform.openai.com/docs/api-reference/responses/object#responses-object-usage
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "cached_tokens": response.usage.input_tokens_details.cached_tokens,
            "output_tokens": response.usage.output_tokens,
            "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        try:
            if inf_gen_config.output_format == "json":
                pred = json.loads(response.output_text)
            elif inf_gen_config.output_format == "text":
                pred = response.output_text
            else:
                raise ValueError(
                    f"Invalid output format: {inf_gen_config.output_format}"
                )

            # TODO: Difficult to get web_search_metadata extraction from the Responses API response, do it later.
            return LLMChatResponse(
                pred=pred, usage=usage, error=None, web_search_metadata=None
            )
        except Exception as e:
            return LLMChatResponse(
                pred="", usage=usage, error=str(e), web_search_metadata=None
            )

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
