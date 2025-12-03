"""
Gemini API utilities for querying materials.

Adapted from phantom-wiki's gemini.py with simplified rate limiting.
Uses the new Google Gen AI SDK: https://googleapis.github.io/python-genai/
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from google import genai
from google.genai import types


@dataclass
class InferenceGenerationConfig:
    """Configuration for inference generation."""

    max_tokens: int = 1024
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    seed: Optional[int] = None
    max_retries: int = 3
    wait_seconds: float = 1.0


@dataclass
class LLMChatResponse:
    """Response from LLM chat."""

    pred: str
    usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None


class GeminiChat:
    """
    Simplified Gemini chat client with rate limiting.

    Uses the new Google Gen AI SDK (google-genai package).
    Based on phantom-wiki's GeminiChat class.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        usage_tier: int = 1,
    ):
        self.model_name = model_name

        # Configure API key
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set"
            )

        # Create client using the new SDK
        self.client = genai.Client(api_key=api_key)

        # Simple rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    def _call_api(
        self,
        prompt: str,
        inf_gen_config: InferenceGenerationConfig,
    ) -> object:
        """Call Gemini API with rate limiting."""
        # Simple rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

        # Use the new SDK's client.models.generate_content
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=inf_gen_config.temperature,
                top_p=inf_gen_config.top_p,
                max_output_tokens=inf_gen_config.max_tokens,
                # NOTE: API does not support top_k>40
                top_k=min(inf_gen_config.top_k, 40),
            ),
        )
        return response

    async def _call_api_async(
        self,
        prompt: str,
        inf_gen_config: InferenceGenerationConfig,
    ) -> object:
        """Call Gemini API asynchronously with rate limiting."""
        # Simple rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

        # Use the new SDK's async client (client.aio)
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=inf_gen_config.temperature,
                top_p=inf_gen_config.top_p,
                max_output_tokens=inf_gen_config.max_tokens,
                # NOTE: API does not support top_k>40
                top_k=min(inf_gen_config.top_k, 40),
            ),
        )
        return response

    def _parse_api_output(
        self, response: object, inf_gen_config: Optional[InferenceGenerationConfig] = None
    ) -> LLMChatResponse:
        """Parse API output into LLMChatResponse."""
        # Try to get response text. If failed due to any reason, output empty prediction
        try:
            pred = response.text
            error = None
        except Exception as e:
            pred = ""
            error = str(e)

        return LLMChatResponse(
            pred=pred,
            usage={
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "response_token_count": response.usage_metadata.candidates_token_count,
                "total_token_count": response.usage_metadata.total_token_count,
                "cached_content_token_count": response.usage_metadata.cached_content_token_count,
            },
            error=error,
        )

    def chat(
        self, prompt: str, inf_gen_config: InferenceGenerationConfig
    ) -> LLMChatResponse:
        """Synchronous chat."""
        for attempt in range(inf_gen_config.max_retries):
            try:
                response = self._call_api(prompt, inf_gen_config)
                return self._parse_api_output(response, inf_gen_config)
            except Exception as e:
                if attempt == inf_gen_config.max_retries - 1:
                    return LLMChatResponse(pred="", error=str(e))
                wait_time = inf_gen_config.wait_seconds * (2**attempt)
                print(f"Error on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)

        return LLMChatResponse(pred="", error="Max retries exceeded")

    async def chat_async(
        self, prompt: str, inf_gen_config: InferenceGenerationConfig
    ) -> LLMChatResponse:
        """Asynchronous chat with exponential backoff retry."""
        for attempt in range(inf_gen_config.max_retries):
            try:
                response = await self._call_api_async(prompt, inf_gen_config)
                return self._parse_api_output(response, inf_gen_config)
            except Exception as e:
                if attempt == inf_gen_config.max_retries - 1:
                    return LLMChatResponse(pred="", error=str(e))
                wait_time = inf_gen_config.wait_seconds * (2**attempt)
                print(f"Error on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

        return LLMChatResponse(pred="", error="Max retries exceeded")

