"""Harnesses where the client owns the message list: an edit is simply sending something different.

The Claude API, OpenAI's Responses API, the Gemini API and a local Ollama model share this shape. The
paid ones go through a `SpendGate` and read their key only from a variable named for this kit, so no key
a machine happens to hold for something else can be picked up by accident. All of them take an injected
transport, so they are tested offline against recorded shapes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time

from eunoe.drivers.base import OfflineError, SpendGate, http_post, is_offline
from eunoe.editors.base import Harness, Turn, replace_in_json
from eunoe.editors.claude_cli import claude_usage
from eunoe.interfaces import CallUsage, RenderedContext, SessionRef
from eunoe.wire import to_anthropic, to_chat_messages, to_gemini, to_openai_input

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1/responses"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
BETA_CONTEXT_EDITING = "context-management-2025-06-27"
BETA_COMPACTION = "compact-2026-01-12"
BETA_THINKING_CONTROLS = "thinking-binding-controls-2026-08-01"


class _Stateless(Harness):
    key_env = ""
    paid = True

    def __init__(self, model=None, gate=None, transport=http_post, environ=None) -> None:
        super().__init__(model, gate or SpendGate())
        self.transport = transport
        self.environ = environ if environ is not None else os.environ
        self.messages: list = []
        self.thinking = False
        self.thinking_seen = 0

    @property
    def spends(self) -> bool:
        return self.paid

    def available(self) -> tuple:
        if is_offline(self.environ):
            return False, "offline: EUNOE_OFFLINE is set"
        if self.key_env and not self.environ.get(self.key_env):
            return False, f"needs {self.key_env}, and a key funded for this experiment"
        if self.paid and not self.gate.allow:
            return False, "spends API credit: pass --allow-spend once its funding is confirmed"
        return True, ""

    def version(self) -> str:
        return self.model

    def start(self, prompt: str, thinking: bool = False) -> Turn:
        self.messages, self.thinking, self.thinking_seen = [], thinking, 0
        return self._turn(prompt)

    def resume(self, prompt: str) -> Turn:
        return self._turn(prompt)

    def _turn(self, prompt: str) -> Turn:
        self.messages.append(self._user(prompt))
        try:
            if self.paid:
                self.gate.check(self.name)
            turn = self._send(self.messages)
        except OfflineError as exc:
            self.messages.pop()
            return Turn(error=str(exc))
        if turn.error:
            self.messages.pop()
            return turn
        if self.paid and self.gate is not None:
            self.gate.record(self.name, self.model, turn.usage, self._price(turn.usage))
        return turn

    def _price(self, usage: CallUsage) -> float:
        return 0.0

    def replace(self, old: str, new: str) -> int:
        self.messages, count = replace_in_json(self.messages, old, new)
        return count

    def fork(self, prompt: str) -> Turn:
        keep = copy.deepcopy(self.messages)
        turn = self._turn(prompt)
        self.messages = keep
        return turn

    def snapshot(self) -> str:
        return hashlib.sha256(json.dumps(self.messages, sort_keys=True).encode("utf-8")).hexdigest()

    def thinking_block_count(self) -> int:
        return self.thinking_seen

    def apply(self, session, rendered: RenderedContext) -> SessionRef:
        self.messages = self._wire(rendered)
        self.thinking = False
        return SessionRef(self.name, "client-side")

    # ---- per provider

    def _user(self, prompt: str):
        raise NotImplementedError

    def _send(self, messages: list) -> Turn:
        raise NotImplementedError

    def _wire(self, rendered: RenderedContext) -> list:
        raise NotImplementedError


class ClaudeApiHarness(_Stateless):
    """The Messages API, client-side: you own `messages`, and thinking blocks go back unchanged."""

    name = "claude-api"
    default_model = "claude-haiku-5-5"
    key_env = "EUNOE_ANTHROPIC_API_KEY"
    apply_supported = True

    def __init__(self, model=None, gate=None, transport=http_post, environ=None, ttl: str = "5m", cost=None) -> None:
        super().__init__(model, gate, transport, environ)
        self.ttl = ttl
        self.cost = cost

    def _price(self, usage: CallUsage) -> float:
        if self.cost is None:
            from eunoe.cost import ListPrice

            self.cost = ListPrice(self.model)
        return self.cost.price(usage, self.ttl)

    def _user(self, prompt: str):
        return {"role": "user", "content": [{"type": "text", "text": prompt}]}

    def _wire(self, rendered: RenderedContext) -> list:
        return to_anthropic(rendered, thinking_replay=False)["messages"]

    def request_body(self, messages: list, **extra) -> dict:
        control = {"type": "ephemeral"}
        if self.ttl == "1h":
            control["ttl"] = "1h"
        body = {"model": self.model, "max_tokens": 1024, "messages": messages, "cache_control": control, **extra}
        if self.thinking:
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": "high"}
        return body

    def headers(self, *betas) -> dict:
        headers = {"x-api-key": self.environ.get(self.key_env, ""), "anthropic-version": ANTHROPIC_VERSION}
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        return headers

    def _send(self, messages: list) -> Turn:
        status, data = self.transport(ANTHROPIC_URL, self.headers(), self.request_body(messages))
        if status != 200:
            return Turn(error=f"{status} {_error_text(data)}")
        content = data.get("content") or []
        self.thinking_seen += sum(1 for b in content if b.get("type") == "thinking")
        messages.append({"role": "assistant", "content": content})
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        return Turn(text, claude_usage(data.get("usage") or {}), data.get("id", ""))


class ClaudeApiServerHarness(ClaudeApiHarness):
    """The server-side edits, which is what controls C1 and C4 use when they run for real.

    The steps differ from a client edit: the server clears or compacts, and the client's history stays
    full. So the codeword steps prove the same thing through `context_management`.
    """

    name = "claude-api-server"
    apply_supported = False

    def context_management_body(self, messages: list, edits: list) -> dict:
        return self.request_body(messages, context_management={"edits": edits})

    def clear_tool_uses_edit(self, keep: int = 1, trigger: int = 1, clear_at_least: int = 0) -> dict:
        edit = {"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": trigger},
                "keep": {"type": "tool_uses", "value": keep}}
        if clear_at_least:
            edit["clear_at_least"] = {"type": "input_tokens", "value": clear_at_least}
        return edit

    def send_with_edits(self, messages: list, edits: list, *betas) -> tuple:
        self.gate.check(self.name)
        return self.transport(ANTHROPIC_URL, self.headers(*(betas or (BETA_CONTEXT_EDITING,))),
                              self.context_management_body(messages, edits))


class OpenAiResponsesHarness(_Stateless):
    """The Responses API sent whole. `previous_response_id` would hide the list, so it is never used."""

    name = "openai-api"
    default_model = "gpt-6-luna"
    key_env = "EUNOE_OPENAI_API_KEY"
    apply_supported = True

    def _user(self, prompt: str):
        return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}

    def _wire(self, rendered: RenderedContext) -> list:
        return to_openai_input(rendered)["input"]

    def _send(self, messages: list) -> Turn:
        headers = {"authorization": f"Bearer {self.environ.get(self.key_env, '')}"}
        status, data = self.transport(OPENAI_URL, headers, {"model": self.model, "input": messages, "store": False})
        if status != 200:
            return Turn(error=f"{status} {_error_text(data)}")
        text = ""
        for item in data.get("output") or []:
            if item.get("type") == "reasoning":
                self.thinking_seen += 1
            if item.get("type") == "message":
                text += "".join(c.get("text", "") for c in item.get("content") or [])
                messages.append({"type": "message", "role": "assistant",
                                 "content": [{"type": "output_text", "text": text}]})
        usage = data.get("usage") or {}
        cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        return Turn(text, CallUsage(int(usage.get("input_tokens") or 0) - cached, cached, 0, 0,
                                    int(usage.get("output_tokens") or 0), usage), data.get("id", ""))


class GeminiApiHarness(_Stateless):
    """generateContent, stateless. Caching is implicit on a prefix, and small additions get less of it."""

    name = "gemini-api"
    default_model = "gemini-3.8-flash"
    key_env = "EUNOE_GEMINI_API_KEY"
    apply_supported = True

    def _user(self, prompt: str):
        return {"role": "user", "parts": [{"text": prompt}]}

    def _wire(self, rendered: RenderedContext) -> list:
        return to_gemini(rendered)["contents"]

    def _send(self, messages: list) -> Turn:
        url = GEMINI_URL.format(model=self.model)
        status, data = self.transport(url, {"x-goog-api-key": self.environ.get(self.key_env, "")}, {"contents": messages})
        if status != 200:
            return Turn(error=f"{status} {_error_text(data)}")
        parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
        text = "".join(p.get("text", "") for p in parts)
        messages.append({"role": "model", "parts": parts or [{"text": text}]})
        meta = data.get("usageMetadata") or {}
        cached = int(meta.get("cachedContentTokenCount") or 0)
        self.thinking_seen += 1 if meta.get("thoughtsTokenCount") else 0
        return Turn(text, CallUsage(int(meta.get("promptTokenCount") or 0) - cached, cached, 0, 0,
                                    int(meta.get("candidatesTokenCount") or 0), meta), data.get("responseId", ""))


class OllamaHarness(_Stateless):
    """A local model through Ollama: nothing to spend. `prompt_eval_count` counts only what it evaluated."""

    name = "ollama"
    default_model = "qwen3.5:2b"
    key_env = ""
    paid = False
    thinking_supported = False
    apply_supported = True

    def __init__(self, model=None, gate=None, transport=http_post, environ=None, url: str | None = None) -> None:
        super().__init__(model, gate, transport, environ)
        host = self.environ.get("OLLAMA_HOST", "")
        if host and not host.startswith("http"):
            host = f"http://{host}"
        self.url = url or (f"{host.rstrip('/')}/api/chat" if host else OLLAMA_URL)
        self.model = model or self.environ.get("EUNOE_OLLAMA_MODEL") or self.default_model

    def available(self) -> tuple:
        if is_offline(self.environ):
            return False, "offline: EUNOE_OFFLINE is set"
        try:
            self.transport(self.url, {}, {"model": self.model, "stream": False, "messages": [
                {"role": "user", "content": "Reply with only OK."}], "options": {"num_predict": 4}})
        except OSError:
            return False, "no Ollama server answered; set OLLAMA_HOST if it is on another port"
        return True, ""

    def _user(self, prompt: str):
        return {"role": "user", "content": prompt}

    def _wire(self, rendered: RenderedContext) -> list:
        return to_chat_messages(rendered)

    def _send(self, messages: list) -> Turn:
        started = time.monotonic()
        status, data = self.transport(self.url, {}, {"model": self.model, "stream": False, "messages": messages,
                                                     "think": False, "options": {"num_predict": 128}})
        if status != 200:
            return Turn(error=f"{status} {_error_text(data)}")
        message = data.get("message") or {}
        messages.append({"role": "assistant", "content": message.get("content", "")})
        turn = Turn(message.get("content", ""), CallUsage(int(data.get("prompt_eval_count") or 0), 0, 0, 0,
                                                          int(data.get("eval_count") or 0), data), "local")
        turn.usage.raw = {"seconds": round(time.monotonic() - started, 3)}
        return turn


def _error_text(data) -> str:
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)[:300]
        if error:
            return str(error)[:300]
    return str(data)[:300]
