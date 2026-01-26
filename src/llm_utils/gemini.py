"""Google Gemini chat implementation."""

import os
from typing import Any
import json

from google import genai
from google.genai import types as genai_types
from google.genai.errors import ServerError
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
    WebSearchMetadata,
)


logger = logging.getLogger(__name__)

# Retry decorator for Gemini API calls
_retry_decorator = retry(
    retry=retry_if_exception_type(ServerError),
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(initial=1, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
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
            "parts": [
                {"text": "Hello, how are you?"},
                {"file_data": {
                    "mime_type": file.uploaded_handle.mime_type,
                    "file_uri": file.uploaded_handle.uri,
                }}
            ]
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
            parts: list[dict[str, Any]] = []
            for c in msg.content:
                if isinstance(c, str):
                    parts.append({"text": c})
                elif isinstance(c, File):
                    parts.append(
                        {
                            "file_data": {
                                "mime_type": c.uploaded_handle.mime_type,
                                "file_uri": c.uploaded_handle.uri,
                            }
                        }
                    )
                else:
                    raise ValueError(f"Invalid message content type: {type(c)}")
            messages.append({"role": role, "parts": parts})

        return messages

    def _build_gen_kwargs(
        self, messages: list[dict[str, Any]], inf_gen_config: InferenceGenerationConfig
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build generation kwargs for the Gemini API.

        Args:
            messages: Messages in Gemini's format.
            inf_gen_config: Inference generation configuration.

        Returns:
            Tuple of (possibly modified messages, generation kwargs).

        """
        gen_kwargs: dict[str, Any] = {
            "max_output_tokens": inf_gen_config.max_output_tokens,
            "thinking_config": genai_types.ThinkingConfig(
                thinking_budget=inf_gen_config.thinking_budget,
                include_thoughts=True,
            ),
        }
        if inf_gen_config.temperature:
            gen_kwargs["temperature"] = inf_gen_config.temperature
        if inf_gen_config.top_p:
            gen_kwargs["top_p"] = inf_gen_config.top_p
        if inf_gen_config.stop_sequences:
            gen_kwargs["stop_sequences"] = inf_gen_config.stop_sequences

        # See https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#generationconfig for all available response mime types
        if inf_gen_config.output_format == "json":
            gen_kwargs["response_mime_type"] = "application/json"
        elif inf_gen_config.output_format == "text":
            gen_kwargs["response_mime_type"] = "text/plain"
        else:
            raise ValueError(f"Invalid output format: {inf_gen_config.output_format}")

        # Enable Google Search grounding if requested
        # https://ai.google.dev/gemini-api/docs/google-search
        if inf_gen_config.use_web_search:
            gen_kwargs["tools"] = [genai_types.Tool(google_search=genai_types.GoogleSearch())]

        # If messages has a system message, add it to gen_kwargs["system_instruction"]
        # and remove it from messages
        if messages and messages[0]["role"] == "system":
            gen_kwargs["system_instruction"] = messages[0]["parts"][0]
            messages = messages[1:]

        return messages, gen_kwargs

    def _call_api(
        self,
        messages: list[dict[str, Any]],
        inf_gen_config: InferenceGenerationConfig,
        use_async: bool = False,
    ) -> Any:
        """Call the Gemini API with retry/backoff for transient errors.

        Args:
            messages: Messages in Gemini's format.
            inf_gen_config: Inference generation configuration.
            use_async: Whether to return an async coroutine.

        Returns:
            Raw API response or async coroutine.

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
        """Call the Gemini API synchronously with retry/backoff."""
        return self.client.models.generate_content(
            model=self.model_name,
            contents=messages,
            config=genai_types.GenerateContentConfig(**gen_kwargs),
        )

    @_retry_decorator
    async def _call_api_async_with_retry(
        self,
        messages: list[dict[str, Any]],
        gen_kwargs: dict[str, Any],
    ) -> Any:
        """Call the Gemini API asynchronously with retry/backoff."""
        return await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=messages,
            config=genai_types.GenerateContentConfig(**gen_kwargs),
        )

    def _parse_api_output(
        self, response: Any, inf_gen_config: InferenceGenerationConfig
    ) -> LLMChatResponse:
        """Parse Gemini's response.

        Args:
            response: Raw Gemini API response.
            inf_gen_config: Inference generation configuration.

        Returns:
            Parsed LLM chat response.

        """
        # See https://ai.google.dev/gemini-api/docs/thinking#summaries for instructions
        # on how to parse thoughts from the response.
        thought = None
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thought = part.text
                break
        try:
            if inf_gen_config.output_format == "json":
                pred = json.loads(response.text)
            elif inf_gen_config.output_format == "text":
                pred = response.text
            else:
                raise ValueError(
                    f"Invalid output format: {inf_gen_config.output_format}"
                )
            # https://ai.google.dev/gemini-api/docs/tokens?lang=python
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "cached_tokens": response.usage_metadata.cached_content_token_count
                if response.usage_metadata.cached_content_token_count
                else 0,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "thinking_tokens": response.usage_metadata.thoughts_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

            # Extract grounding metadata if web search was used
            # https://ai.google.dev/gemini-api/docs/google-search
            if grounding_metadata := response.candidates[0].grounding_metadata:
                logger.debug(f"Grounding Metadata: {grounding_metadata}")
                # Convert GroundingChunk objects to serializable dicts
                queries = grounding_metadata.web_search_queries
                uris = []
                titles = []
                for chunk in grounding_metadata.grounding_chunks or []:
                    if hasattr(chunk, "web") and chunk.web:
                        uris.append(chunk.web.uri)
                        titles.append(chunk.web.title)
                
                # Convert GroundingSupport objects to dicts
                grounding_supports = []
                for support in grounding_metadata.grounding_supports or []:
                    support_dict = {
                        "grounding_chunk_indices": list(support.grounding_chunk_indices),
                        "segment": {
                            "text": support.segment.text,
                            "start_index": support.segment.start_index,
                            "end_index": support.segment.end_index,
                        } if support.segment else None,
                    }
                    grounding_supports.append(support_dict)

                web_search_metadata = WebSearchMetadata(
                    queries=queries,
                    uris=uris,
                    titles=titles,
                    grounding_supports=grounding_supports,
                    num_tool_calls=1,
                )
            else:
                web_search_metadata = None

            finish_reason = None
            if response.candidates and response.candidates[0].finish_reason:
                # Convert enum to string (e.g. FinishReason.STOP -> "STOP")
                finish_reason = str(response.candidates[0].finish_reason)

            return LLMChatResponse(
                pred=pred, 
                usage=usage, 
                error=None, 
                thought=thought, 
                web_search_metadata=web_search_metadata,
                finish_reason=finish_reason
            )
        except Exception as e:
            # Handle cases where Gemini refuses to generate
            return LLMChatResponse(
                pred="",
                usage={},
                error=str(e),
                thought=None,
                web_search_metadata=None,
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
