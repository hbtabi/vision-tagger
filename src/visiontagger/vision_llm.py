"""Optional vision-LLM captioning through any OpenAI-compatible chat endpoint that accepts images.

Environment: ``OPENAI_API_KEY``, ``OPENAI_BASE_URL`` (default https://api.openai.com/v1) and
``VISIONTAGGER_MODEL`` (default ``gpt-4o-mini``; e.g. ``llava`` on Ollama).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from .features import load_image
from .tagger import Tag, TagResult

PROMPT = ('Describe this image for a photo library. Reply with JSON only: {"caption": one sentence, '
          '"tags": up to 10 lowercase single- or two-word tags, most important first}.')


class VisionLLMError(RuntimeError):
    pass


class VisionLLM:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None,
                 max_side: int = 768, timeout: float = 90.0) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("VISIONTAGGER_MODEL", "gpt-4o-mini")
        self.max_side = max_side
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _data_uri(self, path: str | Path) -> str:
        img: Image.Image = load_image(path)
        img.thumbnail((self.max_side, self.max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _post(self, payload: dict) -> dict:  # pragma: no cover - network
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError) as exc:
            raise VisionLLMError(str(exc)) from exc

    def describe(self, path: str | Path) -> tuple[str, list[str]]:
        if not self.available:
            raise VisionLLMError("OPENAI_API_KEY is not set")
        payload = {"model": self.model, "temperature": 0.2, "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": self._data_uri(path)}},
        ]}]}
        data = self._post(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise VisionLLMError("unexpected response") from exc
        return parse_reply(content)


def parse_reply(content: str) -> tuple[str, list[str]]:
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise VisionLLMError("no JSON in reply")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise VisionLLMError(f"bad JSON: {exc}") from exc
    caption = str(obj.get("caption", "")).strip()
    tags = []
    for t in obj.get("tags", []):
        t = str(t).strip().lower()
        if t and t not in tags:
            tags.append(t)
    if not caption:
        raise VisionLLMError("empty caption")
    return caption, tags[:10]


def enrich(result: TagResult, llm: VisionLLM) -> TagResult:
    """Replace the heuristic caption with the LLM's and put its tags first; keep colour/layout tags."""
    if result.error or not llm.available:
        return result
    try:
        caption, tags = llm.describe(result.file)
    except VisionLLMError as exc:
        result.error = None
        result.source = f"heuristic (llm failed: {exc})"
        return result
    llm_tags = [Tag(t, round(1.0 - i * 0.05, 3), "llm") for i, t in enumerate(tags)]
    keep = [t for t in result.tags if t.kind in ("colour", "layout") and t.name not in tags]
    result.caption, result.tags, result.source = caption, llm_tags + keep, f"llm:{llm.model}"
    return result
