"""Minimal Gemini client wrapper.

This module provides a tiny LLM-like class `GeminiLLM` with an `invoke(prompt)`
method that returns an object with `.content` (a string). It uses the REST
API via `requests`. If you prefer a different integration (google-cloud client
library or LangChain provider), replace this adapter.

Usage:
    from src.gemini_client import GeminiLLM
    llm = GeminiLLM(api_key="...")
    resp = llm.invoke("Hello")
    print(resp.content)
"""
from __future__ import annotations
from types import SimpleNamespace
import os
import json
import requests
from typing import Optional


class GeminiLLM:
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        # Official Generative Language API endpoint style:
        # https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}
        self.endpoint = (
            endpoint
            or os.environ.get("GEMINI_ENDPOINT")
            or "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )
        # allow GEMINI_MODEL env override
        self.model = os.environ.get("GEMINI_MODEL", model)
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not provided to GeminiLLM")

    def _build_url(self) -> str:
        base = self.endpoint.format(model=self.model)
        # Use API key as query param per public docs
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}key={self.api_key}"

    def invoke(self, prompt: str):
        """Call Gemini REST endpoint synchronously and return an object with `.content`.

        Note: This is a minimal implementation. The exact request/response fields
        may need to be adjusted to match the specific Gemini model and API version
        you plan to call (chat vs text completion endpoints).
        """
        url = self._build_url()
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": str(prompt)}]}
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 4096,  # Much higher token limit
                "topP": 1,
                "topK": 1,
                "stopSequences": []  # Don't stop early
            }
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            # Try common fields; be conservative and fallback to stringified JSON
            if isinstance(data, dict):
                # PaLM/Generative API commonly returns `candidates` or `output` fields
                text = None
                if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
                    cand0 = data["candidates"][0]
                    content = cand0.get("content", {})
                    parts = content.get("parts") if isinstance(content, dict) else None
                    if parts and len(parts) > 0:
                        text = parts[0].get("text")
                if not text:
                    text = json.dumps(data, ensure_ascii=False)
            else:
                text = str(data)
            return SimpleNamespace(content=text)
        except Exception as e:
            # Don't raise here — keep parity with other LLM fallbacks in repo;
            # return an object with error text so upstream can handle it.
            return SimpleNamespace(content=f"__gemini_error__: {e}")
