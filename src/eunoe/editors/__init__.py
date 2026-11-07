"""Per-harness context editors, plus the conformance suite that decides whether to trust each one.

`get_harness` knows the built-in ones by name. `package.module:ClassName` loads your own: subclass
`eunoe.editors.base.Harness` and implement `start`, `resume` and `replace`, and the same suite runs on it.
"""
from __future__ import annotations

import importlib

from eunoe.editors.base import Harness, Turn, Unsupported

_BUILTIN = {
    "scripted": ("eunoe.editors.scripted", "ScriptedHarness"),
    "claude-cli": ("eunoe.editors.claude_cli", "ClaudeCliHarness"),
    "claude-agent-sdk": ("eunoe.editors.agent_sdk", "AgentSdkHarness"),
    "codex": ("eunoe.editors.codex", "CodexHarness"),
    "agy": ("eunoe.editors.agy", "AgyHarness"),
    "ollama": ("eunoe.editors.stateless", "OllamaHarness"),
    "claude-api": ("eunoe.editors.stateless", "ClaudeApiHarness"),
    "claude-api-server": ("eunoe.editors.server_side", "ClaudeServerHarness"),
    "openai-api": ("eunoe.editors.stateless", "OpenAiResponsesHarness"),
    "gemini-api": ("eunoe.editors.stateless", "GeminiApiHarness"),
}
HARNESSES = tuple(_BUILTIN)
#: What `--harness all` runs: nothing that spends API credit without being asked.
SUBSCRIPTION_OR_FREE = ("scripted", "claude-cli", "claude-agent-sdk", "codex", "agy", "ollama")


def get_harness(name: str, **kwargs) -> Harness:
    if ":" in name:
        module, cls = name.split(":", 1)
    elif name in _BUILTIN:
        module, cls = _BUILTIN[name]
    else:
        raise ValueError(f"unknown harness {name!r}, one of {HARNESSES}, or package.module:ClassName")
    return getattr(importlib.import_module(module), cls)(**kwargs)


__all__ = ["HARNESSES", "SUBSCRIPTION_OR_FREE", "Harness", "Turn", "Unsupported", "get_harness"]
