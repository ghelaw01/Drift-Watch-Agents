"""One interface for OpenAI and Claude tool calling.

Agents keep a neutral message history:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str | None, "tool_calls": [ToolCall]}
  {"role": "tool", "id": str, "name": str, "content": str}
Each provider converts it to its own format on every call.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict


class OpenAIModel:
    def __init__(self, model: str | None = None, temperature: float = 0.0):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("DW_OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature

    def chat(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> Reply:
        converted = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content")}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                        "name": c.name, "arguments": json.dumps(c.args)}} for c in m["tool_calls"]]
                converted.append(msg)
            elif m["role"] == "tool":
                converted.append({"role": "tool", "tool_call_id": m["id"], "content": m["content"]})
            else:
                converted.append({"role": m["role"], "content": m["content"]})
        kwargs = {}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.schema}} for t in tools]
        if self.temperature is not None and not self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["temperature"] = self.temperature   # reasoning models only accept the default
        resp = self.client.chat.completions.create(model=self.model, messages=converted, **kwargs)
        msg = resp.choices[0].message
        calls = [ToolCall(c.id, c.function.name, json.loads(c.function.arguments or "{}"))
                 for c in (msg.tool_calls or [])]
        return Reply(msg.content, calls, resp.usage.prompt_tokens, resp.usage.completion_tokens)


class ClaudeModel:
    def __init__(self, model: str | None = None, temperature: float = 0.0, max_tokens: int = 1024):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("DW_CLAUDE_MODEL", "claude-sonnet-4-5")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> Reply:
        converted: list[dict] = []
        for m in messages:
            if m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                           for c in m.get("tool_calls", [])]
                converted.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["id"], "content": m["content"]}
                if converted and converted[-1]["role"] == "user" and isinstance(converted[-1]["content"], list):
                    converted[-1]["content"].append(block)   # Claude groups results in one user turn
                else:
                    converted.append({"role": "user", "content": [block]})
            else:
                converted.append({"role": "user", "content": m["content"]})
        kwargs = {}
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.schema}
                               for t in tools]
        resp = self.client.messages.create(model=self.model, system=system, messages=converted,
                                           max_tokens=self.max_tokens, temperature=self.temperature, **kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text") or None
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return Reply(text, calls, resp.usage.input_tokens, resp.usage.output_tokens)


def make_model(provider: str = "openai", model: str | None = None):
    provider = provider.lower()
    if provider == "openai":
        return OpenAIModel(model)
    if provider in ("claude", "anthropic"):
        return ClaudeModel(model)
    raise ValueError(f"Unknown provider: {provider}")
