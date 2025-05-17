"""
title: OpenAI Reasoning Pipe with Summary via Responses API
author: GeorgeTse
author_url: https://github.com/GeorgeXie2333
version: 1.0
"""

from __future__ import annotations

import json
import re
from typing import (
    Dict,
    AsyncGenerator,
    List,
    Union,
)

import httpx
from pydantic import BaseModel, Field


class Pipe:

    # ------------------------------------------------------------------
    #                       User settings (Valves)
    # ------------------------------------------------------------------

    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="Base URL of the OpenAI API (no trailing slash).",
        )
        OPENAI_API_KEY: str = Field(
            default="",
            description="Your OpenAI API key (sk‑…). Required for outbound calls.",
        )
        SUMMARY_MODE: str = Field(
            default="auto",
            description="Reasoning summary style: auto | concise | detailed",
        )
        TARGET_MODELS: str = Field(
            default="o3,o4-mini",
            description="Comma‑separated base model IDs that should get reasoning injection.",
        )
        NAME_PREFIX: str = Field(
            default="REASONING/",
            description="Prefix shown before each exposed model name.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------
    #                  Expose selectable models to WebUI
    # ------------------------------------------------------------------

    def pipes(self):
        suffixes = ["", "-high"]
        bases = [m.strip() for m in self.valves.TARGET_MODELS.split(",") if m.strip()]
        return [
            {
                "id": f"responses-{b}{s}".lower().replace("/", "-"),
                "name": f"{self.valves.NAME_PREFIX}{b}{s}",
            }
            for b in bases
            for s in suffixes
        ]

    # ------------------------------------------------------------------
    #                           Helper utils
    # ------------------------------------------------------------------

    @classmethod
    def _pick_text(cls, payload: Union[str, Dict]) -> str:
        if isinstance(payload, str):
            return payload
        if not isinstance(payload, dict):
            return ""
        for key in ("text", "content"):
            val = payload.get(key)
            if isinstance(val, str) and val:
                return val
        for key in ("delta", "data", "value", "choices"):
            child = payload.get(key)
            if child:
                txt = cls._pick_text(child)
                if txt:
                    return txt
        for val in payload.values():
            txt = cls._pick_text(val)
            if txt:
                return txt
        return ""

    @staticmethod
    def _envelope(text: str) -> bytes:
        return json.dumps({"choices": [{"delta": {"content": text}}]}).encode()

    # ------------------------------------------------------------------
    #                           Main logic
    # ------------------------------------------------------------------

    async def pipe(
        self, body: dict
    ) -> AsyncGenerator[bytes, None]:  # Changed to async def and AsyncGenerator
        # ---- convert Chat → Responses schema -----------------------
        if "input" not in body and "messages" in body:
            input_blocks: List[Dict] = []
            for msg in body.pop("messages"):
                role = msg.get("role", "user")
                if role not in {"user", "system"}:  # assistant history ignored
                    continue

                raw_content = msg.get("content", [])
                if isinstance(raw_content, str):
                    raw_content = [raw_content]

                parts: List[Dict] = []
                for part in raw_content:
                    # image_url => input_image
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url")
                        if url:
                            parts.append({"type": "input_image", "image_url": url})
                        continue
                    # fallback to text
                    txt = part if isinstance(part, str) else str(part)
                    if txt:
                        parts.append({"type": "input_text", "text": txt})

                if parts:
                    input_blocks.append(
                        {"type": "message", "role": role, "content": parts}
                    )

            if not input_blocks:
                input_blocks.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": ""}],
                    }
                )
            body["input"] = input_blocks

        # ---- model sanitise ---------------------------------------
        raw = str(body.get("model", ""))
        name = raw.rsplit(".", 1)[-1]
        for pre in ("openai_responses_pipe.", "responses-"):
            if name.startswith(pre):
                name = name[len(pre) :]
        m = re.search(r"-(low|medium|high)$", name, re.I)
        effort = m.group(1).lower() if m else None
        base = name[: -len(m.group(0))] if m else name

        if base.lower() in {
            m.strip().lower() for m in self.valves.TARGET_MODELS.split(",") if m.strip()
        }:
            reasoning = body.setdefault("reasoning", {})
            reasoning.setdefault("summary", self.valves.SUMMARY_MODE)
            if effort and "effort" not in reasoning:
                reasoning["effort"] = effort

        body.update({"model": base, "stream": True})

        # ---- upstream call (async) --------------------------------
        async with httpx.AsyncClient(timeout=600.0) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.valves.OPENAI_API_BASE_URL}/responses",
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                ) as resp:
                    resp.raise_for_status()

                    # ---- SSE transform (async) ---------------------------------------
                    # MODIFICATION START: Send <think> tag immediately and set think_open to True
                    yield b"data: " + self._envelope(
                        "<think>\nReasoning Started...\n"
                    ) + b"\n\n"
                    think_open = True
                    # MODIFICATION END

                    current_event: str | None = None
                    summary_started = (
                        False  # Used to add newlines between summary parts
                    )

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("event:"):
                            current_event = line[6:].strip()
                            continue
                        if not line.startswith("data:"):
                            continue
                        try:
                            payload = json.loads(line[5:].strip())
                        except Exception:
                            continue

                        if current_event and current_event.startswith(
                            "response.reasoning_summary"
                        ):
                            if current_event.endswith("part.added"):
                                # MODIFICATION START: <think> is already open.
                                # This logic now only handles newlines between summary parts.
                                if (
                                    summary_started
                                ):  # If a previous summary part was already added
                                    yield b"data: " + self._envelope("\n\n") + b"\n\n"
                                summary_started = True  # Mark that at least one summary part has been initiated
                                # MODIFICATION END
                                continue
                            if current_event.endswith(
                                "text.delta"
                            ):  # and think_open is implicitly True
                                txt = self._pick_text(payload)
                                if txt:
                                    yield b"data: " + self._envelope(txt) + b"\n\n"
                                continue
                            continue

                        if current_event == "response.output_text.delta":
                            if think_open:
                                yield b"data: " + self._envelope("</think>\n") + b"\n\n"
                                think_open = False
                            token = self._pick_text(payload)
                            if token:
                                yield b"data: " + self._envelope(token) + b"\n\n"

                    if think_open:
                        yield b"data: " + self._envelope("</think>\n") + b"\n\n"

            except httpx.HTTPStatusError as e:
                error_content = ""
                try:
                    error_content_bytes = await e.response.aread()
                    error_content = error_content_bytes.decode(errors="replace")
                except Exception:
                    error_content = str(e)
                raise Exception(
                    f"Upstream API Error: {e.response.status_code} - {error_content}"
                ) from e
            except httpx.RequestError as e:
                raise Exception(f"Request failed: {str(e)}") from e
