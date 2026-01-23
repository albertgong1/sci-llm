from __future__ import annotations

import base64
import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

import requests

try:
    import httpx  # optional, used for async calls
except Exception:  # pragma: no cover
    httpx = None

from llm_utils.common import (
    Conversation,
    File,
    InferenceGenerationConfig,
    LLMChat,
    LLMChatResponse,
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
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            h["HTTP-Referer"] = self.site_url
        if self.app_name:
            h["X-Title"] = self.app_name
        return h

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
        self._file_cache: dict[str, dict[str, Any]] = {}

    def generate_response(
        self,
        conv: Conversation,
        inf_gen_config: InferenceGenerationConfig,
    ) -> LLMChatResponse:

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
        for msg in conv.messages:
            for content in msg.content:
                if isinstance(content, File):
                    self.upload_file(content)

        messages = self._convert_conv_to_api_format(conv)
        response = await self._call_api(messages, inf_gen_config, use_async=True)
        return self._parse_api_output(response, inf_gen_config)

    def _convert_conv_to_api_format(self, conv: Conversation) -> list[dict[str, Any]]:
        """
        OpenRouter uses OpenAI-style messages:
          [{"role": "...", "content": "..." | [{"type":"text","text":"..."}, {"type":"file","file":{...}}]}]

        For PDFs, OpenRouter supports:
          {"type":"file","file":{"filename":"x.pdf","file_data":"data:application/pdf;base64,..."}}
        or:
          {"type":"file","file":{"filename":"x.pdf","fileData":"https://..."}}
         [oai_citation:3‡OpenRouter](https://openrouter.ai/docs/guides/overview/multimodal/pdfs)
        """
        out: list[dict[str, Any]] = []

        for msg in conv.messages:
            role = getattr(msg, "role", None) or getattr(msg, "speaker", None)
            if role is None:
                raise ValueError("Conversation message missing role/speaker.")

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
                    if not cached:
                        # If user didn't call upload_file() first for some reason, do it now.
                        self.upload_file(c)
                        cached = self._file_cache.get(key)

                    if not cached:
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
        Calls OpenRouter Chat Completions endpoint.  [oai_citation:4‡OpenRouter](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        def _get(obj: Any, name: str, default: Any = None) -> Any:
            return getattr(obj, name, default)

        temperature = _get(inf_gen_config, "temperature", None)
        top_p = _get(inf_gen_config, "top_p", None)
        max_tokens = _get(inf_gen_config, "max_tokens", None) or _get(inf_gen_config, "max_output_tokens", None)
        stop = _get(inf_gen_config, "stop", None)
        seed = _get(inf_gen_config, "seed", None)

        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed

        # ---- PDF  configuration ---- [oai_citation:5‡OpenRouter](https://openrouter.ai/docs/guides/overview/multimodal/pdfs)
        pdf_engine = "pdf-text"

        if self._messages_contain_files(messages):
            payload["plugins"] = [
                {
                    "id": "file-parser",
                    "pdf": {"engine": pdf_engine},
                }
            ]
        
        # =====================================================
        # ---- OpenRouter web search (plugin) ----
        # =====================================================
        use_web_search = (
            _get(inf_gen_config, "use_web_search", False)
            or _get(inf_gen_config, "web_search", False)
        )
        web_max_results = _get(inf_gen_config, "web_max_results", None)
        web_engine = _get(inf_gen_config, "web_engine", None)      # "native" | "exa"
        web_search_prompt = _get(inf_gen_config, "web_search_prompt", None)

        if use_web_search:
            payload.setdefault("plugins", [])

            web_plugin: dict[str, Any] = {"id": "web"}

            if web_engine is not None:
                web_plugin["engine"] = web_engine
            if web_max_results is not None:
                web_plugin["max_results"] = web_max_results
            if web_search_prompt is not None:
                web_plugin["search_prompt"] = web_search_prompt

            payload["plugins"].append(web_plugin)
        # =====================================================


        if use_async:
            return self.client.chat_completions_async(payload)
        return self.client.chat_completions(payload)

    def _parse_api_output(
        self, response: Any, inf_gen_config: InferenceGenerationConfig
    ) -> LLMChatResponse:
        """
        OpenRouter returns OpenAI-style ChatCompletion JSON:
          { choices: [{ message: { role, content }, finish_reason }], usage: {...}, ... }
         [oai_citation:6‡OpenRouter](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)
        """
        # If async path returned a coroutine (because caller didn't await), fail loudly.
        if inspect.iscoroutine(response):
            raise RuntimeError("Async response not awaited; did you call generate_response_async correctly?")

        text = ""
        finish_reason = None
        usage = None
        model = None

        if isinstance(response, dict):
            model = response.get("model")
            usage = response.get("usage")
            choices = response.get("choices") or []
            if choices:
                finish_reason = choices[0].get("finish_reason")
                msg = choices[0].get("message") or {}
                content = msg.get("content")

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

        try:
            if inf_gen_config.output_format == "json":
                pred = json.loads(text) if text else {}
            elif inf_gen_config.output_format == "text":
                pred = text
            else:
                raise ValueError(f"Invalid output format: {inf_gen_config.output_format}")

            return LLMChatResponse(
                pred=pred,
                usage=usage or {},
                error=None,
                thought=None,
                web_search_metadata=None,
            )
        except Exception as e:
            return LLMChatResponse(
                pred="",
                usage=usage or {},
                error=str(e),
                thought=None,
                web_search_metadata=None,
            )

    def upload_file(self, file: File) -> None:
        """
        OpenRouter does not require a pre-upload step for PDFs; you include them inline
        as URL or data: URL in the message content.  [oai_citation:7‡OpenRouter](https://openrouter.ai/docs/guides/overview/multimodal/pdfs)

        We cache the converted representation to avoid repeated base64 encoding.
        """
        key = self._file_cache_key(file)
        if key in self._file_cache:
            return

        filename = getattr(file, "filename", None) or getattr(file, "name", None) or "document.pdf"
        mime_type = getattr(file, "mime_type", None) or getattr(file, "content_type", None) or "application/pdf"

        url = getattr(file, "url", None) or getattr(file, "file_url", None)
        if url:
            self._file_cache[key] = {
                "filename": filename,
                "fileData": url,
            }
            return

        data: Optional[bytes] = getattr(file, "data", None) or getattr(file, "bytes", None) or getattr(file, "content", None)
        path = getattr(file, "path", None)

        if data is None and path:
            with open(path, "rb") as f:
                data = f.read()

        if data is None:
            raise ValueError("File has no url and no bytes/path to read from.")

        b64 = base64.b64encode(data).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"

        self._file_cache[key] = {
            "filename": filename,
            # OpenRouter examples show `file_data` for base64 data URLs.  [oai_citation:9‡OpenRouter](https://openrouter.ai/docs/guides/overview/multimodal/pdfs)
            "file_data": data_url,
        }

    def delete_file(self, file: File) -> None:
        """
        No server-side deletion required for OpenRouter inline files.
        We just drop the local cache entry.
        """
        key = self._file_cache_key(file)
        self._file_cache.pop(key, None)

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
        # Try stable identifiers first
        fid = getattr(file, "id", None) or getattr(file, "file_id", None)
        if fid:
            return str(fid)

        # Fall back to a deterministic key based on name+path; last resort is object id.
        name = getattr(file, "filename", None) or getattr(file, "name", None) or ""
        path = getattr(file, "path", None) or ""
        if name or path:
            return f"{name}::{path}"

        return f"obj::{id(file)}"

    def _make_llm_chat_response(
        self,
        text: str,
        raw: Any,
        model: str,
        finish_reason: Any = None,
        usage: Any = None,
    ) -> LLMChatResponse:
        """
        Construct LLMChatResponse without assuming its exact constructor signature.
        """
        sig = inspect.signature(LLMChatResponse)  # works for dataclasses and normal classes
        kwargs: dict[str, Any] = {}

        # common names used across codebases
        candidates = {
            "text": text,
            "content": text,
            "message": text,
            "output_text": text,
            "raw": raw,
            "raw_response": raw,
            "response": raw,
            "model": model,
            "finish_reason": finish_reason,
            "stop_reason": finish_reason,
            "usage": usage,
        }

        for name, val in candidates.items():
            if name in sig.parameters and val is not None:
                kwargs[name] = val

        try:
            return LLMChatResponse(**kwargs) 
        except TypeError:
            try:
                return LLMChatResponse(text)  
            except TypeError:
                raise RuntimeError(
                    f"Could not construct LLMChatResponse; inferred kwargs={kwargs} "
                    f"signature={sig}"
                )