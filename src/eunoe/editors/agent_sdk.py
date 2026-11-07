"""The Claude Agent SDK: Claude Code as a library, so the same session files and the same edits.

An optional dependency, so the package stays stdlib: `pip install eunoe[agent-sdk]`. It drives the same
session files as `claude_cli`, so an edit is the same edit. Metered keys are scrubbed from the child's
environment, and which login it then uses is the CLI's business, so the suite shows what really happens.
"""
from __future__ import annotations

import asyncio
import os
import uuid

from eunoe.drivers.base import is_offline, scrubbed_env
from eunoe.editors.base import Turn
from eunoe.editors.claude_cli import ClaudeCliHarness, claude_usage


class AgentSdkHarness(ClaudeCliHarness):
    name = "claude-agent-sdk"

    def _sdk(self):
        try:
            import claude_agent_sdk
        except ImportError:
            return None
        return claude_agent_sdk

    def available(self) -> tuple:
        if is_offline(self.environ):
            return False, "offline: EUNOE_OFFLINE is set"
        if self._sdk() is None:
            return False, "claude-agent-sdk is not installed: pip install eunoe[agent-sdk]"
        return True, ""

    def version(self) -> str:
        sdk = self._sdk()
        return getattr(sdk, "__version__", "") if sdk else ""

    def _query(self, prompt: str, **options) -> Turn:
        sdk = self._sdk()
        opts = sdk.ClaudeAgentOptions(model=self.model, tools=[], strict_mcp_config=True, cwd=self._dir(),
                                      setting_sources=["project", "local"], env=_env_overlay(self.environ), **options)

        async def collect():
            out = Turn()
            async for message in sdk.query(prompt=prompt, options=opts):
                if isinstance(message, sdk.ResultMessage):
                    out.text = str(message.result or "")
                    out.session_id = message.session_id
                    out.usage = claude_usage(message.usage or {})
                    out.provider_cost_usd = message.total_cost_usd
                    if message.is_error:
                        out.error = str(message.result or message.subtype)[:300]
            return out

        try:
            return asyncio.run(collect())
        except Exception as exc:  # the SDK raises its own process errors; a step reports them
            return Turn(error=f"{type(exc).__name__}: {' '.join(str(exc).split())[:240]}")

    def start(self, prompt: str, thinking: bool = False) -> Turn:
        session_id = str(uuid.uuid4())
        self.current = session_id
        self.sessions.append(session_id)
        options = {"session_id": session_id}
        if thinking:
            options["effort"] = "high"
        return self._query(prompt, **options)

    def resume(self, prompt: str) -> Turn:
        return self._query(prompt, resume=self.current)

    def fork(self, prompt: str) -> Turn:
        turn = self._query(prompt, resume=self.current, fork_session=True)
        if turn.session_id and turn.session_id != self.current:
            self.sessions.append(turn.session_id)
        elif not turn.error:
            turn.error = "fork_session kept the same session id"
        return turn


def _env_overlay(environ) -> dict:
    """The SDK merges this over the process's own environment, so a scrubbed key has to be blanked."""
    clean = scrubbed_env(environ)
    return {**{name: "" for name in environ if name not in clean}, **clean}
