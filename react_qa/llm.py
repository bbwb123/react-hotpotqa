from __future__ import annotations

import os
from typing import Protocol


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str: ...


class OpenAIResponsesModel:
    """Thin adapter around the OpenAI Responses API."""

    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
        )
        return response.output_text.strip()
