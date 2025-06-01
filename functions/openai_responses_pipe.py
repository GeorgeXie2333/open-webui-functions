"""
title: OpenAI Responses Pipe with Reasoning Summary
author_url:https://github.com/GeorgeXie2333
author: George
version: 0.1.1
license: MIT
"""

from pydantic import BaseModel, Field
import httpx
import json
import random
import re
from typing import Dict, AsyncGenerator, List, Union, Literal


class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="OpenAI: ",
            description="Prefix to be added before model names.",
        )
        BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="Base URL for OpenAI API.",
        )
        API_KEYS: str = Field(
            default="",
            description="API keys for OpenAI, use , to split",
        )
        NORMAL_MODELS: str = Field(
            default="gpt-4.5-preview,chatgpt-4o-latest",
            description="Comma-separated normal model IDs (no reasoning).",
        )
        REASONING_MODELS: str = Field(
            default="o4-mini,o3,o3-mini,o1",
            description="Comma-separated reasoning model IDs (with thinking process).",
        )
        SUMMARY_MODE: Literal["auto", "concise", "detailed", "None"] = Field(
            default="auto",
            description="Reasoning summary mode for reasoning models: auto | concise | detailed | None",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        res = []

        # Add normal models
        normal_models = [
            m.strip() for m in self.valves.NORMAL_MODELS.split(",") if m.strip()
        ]
        for model in normal_models:
            res.append({"name": f"{self.valves.NAME_PREFIX}{model}", "id": model})

        # Add reasoning models (both normal and -high variants)
        reasoning_models = [
            m.strip() for m in self.valves.REASONING_MODELS.split(",") if m.strip()
        ]
        for model in reasoning_models:
            res.append({"name": f"{self.valves.NAME_PREFIX}{model}", "id": model})
            res.append(
                {
                    "name": f"{self.valves.NAME_PREFIX}{model}-high",
                    "id": f"{model}-high",
                }
            )

        return res

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

    def _is_reasoning_model(self, model_id: str) -> bool:
        """Check if the model is a reasoning model"""
        base_model = model_id.replace("-high", "")
        reasoning_models = [
            m.strip().lower()
            for m in self.valves.REASONING_MODELS.split(",")
            if m.strip()
        ]
        return base_model.lower() in reasoning_models

    def _is_normal_model(self, model_id: str) -> bool:
        """Check if the model is a normal model"""
        normal_models = [
            m.strip().lower() for m in self.valves.NORMAL_MODELS.split(",") if m.strip()
        ]
        return model_id.lower() in normal_models

    async def _handle_reasoning_model(
        self, body: dict, model_id: str, headers: dict
    ) -> AsyncGenerator[str, None]:
        """Handle reasoning models with summary output"""
        # Convert messages to input format for Responses API
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
                    # Handle image_url => input_image
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url")
                        if url:
                            parts.append({"type": "input_image", "image_url": url})
                        continue
                    # Handle text content
                    if isinstance(part, dict) and part.get("type") == "text":
                        txt = part.get("text", "")
                    else:
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

        # Handle model name and effort
        base_model = model_id.replace("-high", "")
        effort = "high" if model_id.endswith("-high") else "medium"

        # Set reasoning parameters
        reasoning = body.setdefault("reasoning", {})
        reasoning.setdefault("summary", self.valves.SUMMARY_MODE)
        reasoning.setdefault("effort", effort)

        body.update({"model": base_model, "stream": True})

        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream(
                "POST",
                f"{self.valves.BASE_URL}/responses",
                json=body,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"Error: {response.status_code} {error_text.decode('utf-8')}"
                    return

                # Send <think> tag immediately
                yield "<think>\nReasoning Started...\n"
                think_open = True
                current_event: str | None = None
                summary_started = False

                async for line in response.aiter_lines():
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

                    # Handle reasoning summary
                    if current_event and current_event.startswith(
                        "response.reasoning_summary"
                    ):
                        if current_event.endswith("part.added"):
                            if summary_started:
                                yield "\n\n"
                            summary_started = True
                            continue
                        if current_event.endswith("text.delta"):
                            txt = self._pick_text(payload)
                            if txt:
                                yield txt
                            continue
                        continue

                    # Handle main output
                    if current_event == "response.output_text.delta":
                        if think_open:
                            yield "</think>\n"
                            think_open = False
                        token = self._pick_text(payload)
                        if token:
                            yield token

                if think_open:
                    yield "</think>\n"

    async def _handle_normal_model(
        self, body: dict, model_id: str, headers: dict
    ) -> AsyncGenerator[str, None]:
        """Handle normal models without reasoning"""
        # Convert messages format for normal models
        new_messages = []
        messages = body["messages"]
        for message in messages:
            try:
                if message["role"] == "user":
                    if isinstance(message["content"], list):
                        content = []
                        for i in message["content"]:
                            if i["type"] == "text":
                                content.append(
                                    {"type": "input_text", "text": i["text"]}
                                )
                            elif i["type"] == "image_url":
                                content.append(
                                    {
                                        "type": "input_image",
                                        "image_url": i["image_url"]["url"],
                                    }
                                )
                        new_messages.append({"role": "user", "content": content})
                    else:
                        new_messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": message["content"]}
                                ],
                            }
                        )
                elif message["role"] == "assistant":
                    new_messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": message["content"]}
                            ],
                        }
                    )
                elif message["role"] == "system":
                    new_messages.append(
                        {
                            "role": "system",
                            "content": [
                                {"type": "input_text", "text": message["content"]}
                            ],
                        }
                    )
            except Exception as e:
                yield f"Error: {message} - {str(e)}"
                return

        payload = {**body, "model": model_id}
        payload.pop("messages")
        payload["input"] = new_messages
        payload["stream"] = True

        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream(
                "POST",
                f"{self.valves.BASE_URL}/responses",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"Error: {response.status_code} {error_text.decode('utf-8')}"
                    return

                async for line in response.aiter_lines():
                    if line:
                        try:
                            if line.startswith("data:"):
                                line_data = json.loads(line[5:])
                                yield line_data["delta"]
                        except Exception:
                            pass

    async def pipe(self, body: dict, __user__: dict):
        self.key = random.choice(self.valves.API_KEYS.split(",")).strip()
        print(f"pipe:{__name__}")

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

        model_id = body["model"][body["model"].find(".") + 1 :]

        try:
            if self._is_reasoning_model(model_id):
                async for chunk in self._handle_reasoning_model(
                    body, model_id, headers
                ):
                    yield chunk
            elif self._is_normal_model(model_id):
                async for chunk in self._handle_normal_model(body, model_id, headers):
                    yield chunk
            else:
                yield f"Error: Model {model_id} not found in NORMAL_MODELS or REASONING_MODELS"
                return
        except Exception as e:
            yield f"Error: {e}"
            return
