from __future__ import annotations

import re
from typing import Protocol

class KnowledgeEnvironment(Protocol):
    def search(self, query: str) -> str: ...
    def lookup(self, keyword: str) -> str: ...


class WikipediaEnvironment:
    API_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, timeout: float = 10.0) -> None:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
        self.timeout = timeout
        self.active_title: str | None = None
        self.active_text = ""
        self.lookup_cursor: dict[str, int] = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ReAct-HotpotQA/1.0 (academic reproduction)"})

    def _request(self, params: dict) -> dict:
        response = self.session.get(self.API_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def search(self, query: str) -> str:
        data = self._request({
            "action": "query", "generator": "search", "gsrsearch": query,
            "gsrlimit": 3, "prop": "extracts", "explaintext": 1,
            "exintro": 1, "format": "json", "formatversion": 2,
        })
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return f"No Wikipedia page found for: {query}"
        page = pages[0]
        self.active_title = page.get("title", query)
        full = self._request({
            "action": "query", "prop": "extracts", "explaintext": 1,
            "titles": self.active_title, "format": "json", "formatversion": 2,
        })
        full_pages = full.get("query", {}).get("pages", [])
        self.active_text = full_pages[0].get("extract", "") if full_pages else ""
        self.lookup_cursor.clear()
        summary = page.get("extract", "").strip()
        alternatives = ", ".join(p.get("title", "") for p in pages[1:])
        suffix = f" Related results: {alternatives}." if alternatives else ""
        return f"{self.active_title}: {summary[:1800]}{suffix}".strip()

    def lookup(self, keyword: str) -> str:
        if not self.active_text:
            return "Lookup requires an active page. Call Search first."
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", self.active_text) if s.strip()]
        matches = [s for s in sentences if keyword.casefold() in s.casefold()]
        index = self.lookup_cursor.get(keyword.casefold(), 0)
        if index >= len(matches):
            return f"No more sentences containing '{keyword}' in {self.active_title}."
        self.lookup_cursor[keyword.casefold()] = index + 1
        return f"({index + 1}/{len(matches)}) {matches[index][:1800]}"
