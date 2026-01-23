from __future__ import annotations

import base64
import inspect
import json
import os
import re
from typing import Any

import requests

try:
    import httpx 
except Exception: 
    httpx = None

from llm_utils.common import (
    Conversation,
    File,
    InferenceGenerationConfig,
    LLMChat,
    LLMChatResponse,
    WebSearchMetadata,
)


class QwenClient:
    """
    Minimal OpenRouter client for OpenAI-compatible Chat Completions.

    Env vars supported:
      - OPENROUTER_API_KEY (preferred)
      - OPENAI_API_KEY (fallback)
      - OPENROUTER_BASE_URL (default: https://openrouter.ai/api/v1)
      - OPENROUTER_SITE_URL (optional, for OpenRouter attribution headers)
      - OPENROUTER_APP_NAME (optional, for OpenRouter attribution headers)
      - OPENROUTER_TIMEOUT_S (default: 120)
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY."
            )

        self.timeout_s = float(os.getenv("OPENROUTER_TIMEOUT_S", "120"))
        self.site_url = os.getenv("OPENROUTER_SITE_URL")
        self.app_name = os.getenv("OPENROUTER_APP_NAME")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    async def chat_completions_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        if httpx is None:
            raise RuntimeError("httpx is required for async calls. Please `pip install httpx`.")
        url = f"{self.base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()


class QwenChat(LLMChat):
    """Chat interface for Qwen models (via OpenRouter)."""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.client = QwenClient(model_name)

        # OpenRouter inline-file cache: key -> {"filename":..., "file_data":...}
        self._file_cache: dict[str, dict[str, Any]] = {}

    def _convert_conv_to_api_format(self, conv: Conversation) -> list[dict[str, Any]]:
        """
        Convert llm_utils.common.Conversation into OpenRouter/OpenAI-style messages.

        Input (common.py):
          Conversation(messages=[Message(role=..., content=[str|File, ...]), ...])

        Output (OpenRouter):
          [{"role": "...", "content": [{"type":"text","text":"..."}, {"type":"file","file":{...}}]}]

        We always use the list-of-parts format to support mixed text + PDFs.
        """
        out: list[dict[str, Any]] = []

        for msg in conv.messages:
            role = msg.role
            parts: list[dict[str, Any]] = []
            text_accum: list[str] = []

            for c in msg.content:
                if isinstance(c, str):
                    text_accum.append(c)
                elif isinstance(c, File):
                    if text_accum:
                        parts.append({"type": "text", "text": "\n".join(text_accum).strip()})
                        text_accum = []

                    key = self._file_cache_key(c)
                    cached = self._file_cache.get(key)
                    if cached is None:
                        self.upload_file(c)
                        cached = self._file_cache.get(key)

                    if cached is None:
                        raise RuntimeError("Failed to cache file for OpenRouter message formatting.")

                    parts.append({"type": "file", "file": cached})
                else:
                    text_accum.append(str(c))

            if text_accum:
                parts.append({"type": "text", "text": "\n".join(text_accum).strip()})

            out.append({"role": role, "content": parts})

        return out

    def _call_api(
        self,
        messages: list[dict[str, Any]],
        inf_gen_config: InferenceGenerationConfig,
        use_async: bool = False,
    ) -> Any:
        """
        Call OpenRouter Chat Completions endpoint.

        Uses common.InferenceGenerationConfig field names.
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        # -----------------------------
        # Generation params (common.py)
        # -----------------------------
        payload["max_tokens"] = inf_gen_config.max_output_tokens

        if inf_gen_config.temperature is not None:
            payload["temperature"] = inf_gen_config.temperature
        if inf_gen_config.top_p is not None:
            payload["top_p"] = inf_gen_config.top_p
        if inf_gen_config.stop_sequences is not None:
            payload["stop"] = inf_gen_config.stop_sequences


        # -----------------------------
        # Plugins: PDF parsing
        # -----------------------------
        if self._messages_contain_files(messages):
            payload.setdefault("plugins", [])
            payload["plugins"].append(
                {
                    "id": "file-parser",
                    "pdf": {"engine": "pdf-text"},
                }
            )

        # -----------------------------
        # Plugins: Web search
        # -----------------------------
        if inf_gen_config.use_web_search:
            payload.setdefault("plugins", [])
            payload["plugins"].append({"id": "web"})

        if use_async:
            return self.client.chat_completions_async(payload)
        return self.client.chat_completions(payload)

    def _parse_api_output(self, response: Any, inf_gen_config: InferenceGenerationConfig) -> LLMChatResponse:
        """
        Parse OpenRouter ChatCompletion response into llm_utils.common.LLMChatResponse.

        - pred: text or parsed json
        - web_search_metadata: WebSearchMetadata | None
        """
        if inspect.iscoroutine(response):
            raise RuntimeError("Async response not awaited; did you call generate_response_async correctly?")

        text = ""
        usage: dict[str, Any] = {}
        web_meta: WebSearchMetadata | None = None

        annotations: list[dict[str, Any]] = []

        if isinstance(response, dict):
            usage = response.get("usage") or {}
            choices = response.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                annotations = msg.get("annotations") or []

                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    chunks: list[str] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            chunks.append(part.get("text", ""))
                    text = "\n".join([c for c in chunks if c]).strip()
                else:
                    text = str(content) if content is not None else ""

        # Build WebSearchMetadata if we can find URLs (structured annotations or inline text)
        uris: list[str] = []

        for a in annotations:
            if isinstance(a, dict) and a.get("type") == "url_citation":
                url = a.get("url")
                if isinstance(url, str) and url:
                    uris.append(url)

        if not uris and text:
            uris = re.findall(r"https?://\S+", text)

        if uris:
            dedup = list(dict.fromkeys(uris))
            web_meta = WebSearchMetadata(queries=[], uris=dedup)

        try:
            if inf_gen_config.output_format == "json":
                pred = json.loads(text) if text else {}
            elif inf_gen_config.output_format == "text":
                pred = text
            else:
                raise ValueError(f"Invalid output format: {inf_gen_config.output_format}")

            return LLMChatResponse(
                pred=pred,
                usage=usage,
                error=None,
                thought=None,
                web_search_metadata=web_meta,
            )
        except Exception as e:
            return LLMChatResponse(
                pred="",
                usage=usage,
                error=str(e),
                thought=None,
                web_search_metadata=web_meta,
            )

    def upload_file(self, file: File) -> None:
        """
        common.File only has .path; OpenRouter accepts inline data URLs for PDFs.

        We cache base64 encoding and set file.uploaded_handle so File.is_uploaded() works.
        """
        key = self._file_cache_key(file)
        if key in self._file_cache:
            file.uploaded_handle = key
            return

        path = file.path
        filename = path.name
        mime_type = "application/pdf"

        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        self._file_cache[key] = {
            "filename": filename,
            "file_data": data_url,
        }
        file.uploaded_handle = key

    def delete_file(self, file: File) -> None:
        """
        OpenRouter inline files: no server deletion. Clear local cache and uploaded_handle.
        """
        key = self._file_cache_key(file)
        self._file_cache.pop(key, None)
        file.uploaded_handle = None

    # -------------------- helpers --------------------

    def _messages_contain_files(self, messages: list[dict[str, Any]]) -> bool:
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "file":
                        return True
        return False

    def _file_cache_key(self, file: File) -> str:
        return str(file.path)