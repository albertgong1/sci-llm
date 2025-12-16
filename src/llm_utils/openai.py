"""OpenAI chat implementation."""

import os
from typing import Any

from openai import OpenAI

from llm_utils.common import (
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
            gen_kwargs["instructions"] = messages[0]["content"]
            messages = messages[1:]

        response = self.client.responses.create(
            model=self.model_name,
            input=messages,
            **gen_kwargs,
        )
        return response

    def _parse_api_output(self, response: Any) -> LLMChatResponse:
        """Parse OpenAI's response.

        Args:
            response: Raw OpenAI API response.

        Returns:
            Parsed LLM chat response.

        """
        try:
            pred = response.output_text
            usage = response.get("usage", {})
            return LLMChatResponse(pred=pred, usage=usage, error=None)
        except Exception as e:
            return LLMChatResponse(pred="", usage={}, error=str(e))
