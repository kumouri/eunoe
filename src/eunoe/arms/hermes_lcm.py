"""hermes-lcm as an arm: its real engine, driven in its host's own Python environment.

[hermes-lcm](https://github.com/stephenschoettler/hermes-lcm) is a plugin for Hermes Agent and imports the
host's packages, so it can't be a dependency of a stdlib kit. So this arm starts `hermes_lcm_worker.py` in
an environment that has both, set by two variables, and talks to it over JSON lines:

    EUNOE_HERMES_PYTHON   a Python that has `pip install hermes-agent`
    EUNOE_HERMES_LCM_DIR  a checkout of hermes-lcm

Its one model seam, the host's `call_llm`, is replaced in the worker by a code-built digest unless
`EUNOE_HERMES_REAL_LLM=1`, so this arm's decisions and its summary DAG are the engine's, and its summary text
is not. The manifest labels that `emulated`.
"""
from __future__ import annotations

import json
import os
import subprocess

from eunoe.arms.strategies import RECALL_TOOL, Strategy
from eunoe.interfaces import Block, CallUsage, RenderedContext
from eunoe.ledger import ASSISTANT, THINKING
from eunoe.levels import estimate_tokens

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hermes_lcm_worker.py")


class HermesLcmNotConfigured(RuntimeError):
    pass


class HermesLcmStrategy(Strategy):
    emulated = True

    def __init__(self, *, system="", tools=None, budget=None, thinking="strip_on_edit", name: str = "lcm",
                 environ=None, leaf_tokens: int | None = None) -> None:
        super().__init__(name, system=system, tools=tools, budget=budget, thinking=thinking)
        env = dict(environ if environ is not None else os.environ)
        python, plugin = env.get("EUNOE_HERMES_PYTHON"), env.get("EUNOE_HERMES_LCM_DIR")
        if not python or not plugin:
            raise HermesLcmNotConfigured("the lcm arm needs EUNOE_HERMES_PYTHON and EUNOE_HERMES_LCM_DIR")
        window = self.budget.target_tokens
        env["LCM_LEAF_CHUNK_TOKENS"] = str(leaf_tokens or max(500, window // 8))
        env["LCM_FRESH_TAIL_COUNT"] = str(self.budget.keep_recent_calls * 2)
        self.proc = subprocess.Popen([python, "-I", WORKER, plugin, str(window)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, text=True,
                                     encoding="utf-8")
        self._ask({"op": "ping"})
        self.messages = [{"role": "system", "content": system or "You are a test agent."}]

    def _ask(self, request: dict) -> dict:
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise HermesLcmNotConfigured("the hermes-lcm worker exited; check that its environment imports")
        return json.loads(line)

    @property
    def has_recall(self) -> bool:
        return True

    def observe(self, spec, at, replay_thinking=True) -> None:
        if spec.kind == THINKING:
            return
        role = "assistant" if spec.kind == ASSISTANT else "user"
        content = spec.text if not spec.input else f"[{spec.kind[5:]} {json.dumps(spec.input)}]\n{spec.text}"
        self.messages.append({"role": role, "content": content})

    def _size(self) -> int:
        return sum(estimate_tokens(str(m.get("content"))) for m in self.messages)

    def prepare(self, now: float, last: CallUsage | None, gap_s: float) -> RenderedContext:
        size = self._size()
        if size > int(self.budget.target_tokens * 1.5):
            out = self._ask({"op": "compress", "messages": self.messages, "tokens": size})
            if out.get("status") == "compacted":
                self.messages = out["messages"]
                self.edits += 1
                self.overhead.append(CallUsage(cache_read=size, output=int(out.get("summary_tokens") or 0)))
        system = next((str(m.get("content")) for m in self.messages if m.get("role") == "system"), self.system)
        blocks = [Block("assistant" if m.get("role") == "assistant" else "user", "text", text=str(m.get("content")))
                  for m in self.messages if m.get("role") != "system"]
        return RenderedContext(system, blocks, self.tools + [RECALL_TOOL], {"now": now, "arm": self.name})

    def recall(self, query: str) -> str:
        return self._ask({"op": "search", "query": query}).get("text") or "no match"

    def held_tokens(self) -> int:
        return self._size()

    def manifest(self) -> dict:
        self.close()
        return {"arm": self.name, "emulated": True, "budget": self.budget.target_tokens, "edits": self.edits,
                "note": "hermes-lcm's engine with its model call replaced by a code-built digest"}

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
