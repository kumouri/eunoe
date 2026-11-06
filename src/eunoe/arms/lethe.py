"""LETHE as an arm: its own engine deciding what stays, with no model call of its own.

[LETHE](https://github.com/JesusGarcia9009/lethe-engineering) scores blocks by relevance, folds cold runs
into notes and pages the rest out to a store, leaving a stub. This arm runs its real `ContextManager`,
`Session` and `Scheduler`, not a reimplementation. Its `Compactor` asks an adapter for a summary, and the
adapter here is a code-built extractive stand-in, so the shape and its own decisions are LETHE's while the
summary text is not. That difference is labelled `emulated` in the manifest. Point it at a real model by
passing your own `adapter`.

Needs `pip install lethe-llm-context`, Python 3.12 or later, and it pins nothing about LETHE's internals
beyond the names used here. `eunoe demo` leaves it out, because the package is not stdlib.
"""
from __future__ import annotations

from eunoe import levels
from eunoe.arms.strategies import RECALL_TOOL, Strategy
from eunoe.interfaces import Block as EunoeBlock
from eunoe.interfaces import CallUsage, RenderedContext
from eunoe.ledger import ASSISTANT, THINKING


class LetheNotInstalled(RuntimeError):
    pass


def _import_lethe():
    try:
        from lethe import ContextManager
        from lethe.adapters.base import Response
        from lethe.core.block import Block
        from lethe.core.scheduler import Thresholds
    except ImportError as exc:
        raise LetheNotInstalled("the lethe arm needs `pip install lethe-llm-context` (Python 3.12+)") from exc
    return ContextManager, Block, Response, Thresholds


class _TokenCounter:
    def count(self, blocks: list) -> int:
        return sum(levels.estimate_tokens(b.content) + 3 for b in blocks)


class _ExtractiveAdapter:
    """What LETHE's Compactor and Curator call. No network: a code-built digest of the prompt."""

    name = "eunoe-extractive"

    def __init__(self, response_cls, context_window: int) -> None:
        self.response_cls = response_cls
        self.context_window = context_window
        self.token_counter = _TokenCounter()

    def complete(self, messages: list, **kw):
        prompt = messages[-1].blocks[-1].content if messages and messages[-1].blocks else ""
        body = prompt.split("\n\n", 1)[-1]
        return self.response_cls(text=levels.digest(body)[: max(200, len(body) // 10)])


class LetheStrategy(Strategy):
    emulated = True

    def __init__(self, *, system="", tools=None, budget=None, thinking="strip_on_edit", name: str = "lethe",
                 adapter=None) -> None:
        super().__init__(name, system=system, tools=tools, budget=budget, thinking=thinking)
        ContextManager, self._block, response, thresholds = _import_lethe()
        window = max(budget.target_tokens if budget else 200_000, 1)
        adapter = adapter or _ExtractiveAdapter(response, window)
        self.manager = ContextManager(agent=adapter, curator=None, budget=1.0, triggers={"every_steps": 0},
                                      thresholds=thresholds(keep_last_n_steps=budget.keep_recent_calls if budget else 6))
        self.manager.scheduler.curator.adapter = None
        self.manager.scheduler.curator.w_model = 0.0
        self.session = self.manager.session(goal="Answer questions and finish tasks about this session.")
        self.step = 0
        self.counter = 0

    @property
    def has_recall(self) -> bool:
        return True

    def observe(self, spec, at, replay_thinking=True) -> None:
        if spec.kind == THINKING:
            return
        self.step += 1
        self.counter += 1
        role, kind = "user", "text"
        if spec.kind == ASSISTANT:
            role = "assistant"
        elif spec.kind.startswith("tool:"):
            role, kind = "tool", "tool_result"
        elif spec.kind == "file":
            kind = "file"
        content = spec.text if not spec.input else f"{spec.kind[5:]} {spec.input}\n{spec.text}"
        label = spec.kind if not spec.input else f"{spec.kind[5:]} {next(iter(spec.input.values()), '')}"
        self.session.add(self._block(id=f"b{self.counter:05d}", role=role, kind=kind, content=content,
                                     created_step=self.step, meta={"label": label}))

    def prepare(self, now: float, last: CallUsage | None, gap_s: float) -> RenderedContext:
        before = self.session.stats()["evicted"]
        messages = self.session.render()
        if self.session.stats()["evicted"] != before:
            self.edits += 1
        blocks = [EunoeBlock("assistant" if m.role == "assistant" else "user", "text", text=m.blocks[0].content,
                             event_id=m.blocks[0].id) for m in messages]
        return RenderedContext(self.system, blocks, self.tools + [RECALL_TOOL], {"now": now, "arm": self.name})

    def recall(self, query: str) -> str:
        block = self.session.recall(query)
        return block.content if block else "no match"

    def held_tokens(self) -> int:
        return self.session.stats()["tokens_with_lethe"]

    def manifest(self) -> dict:
        return {"arm": self.name, "emulated": True, "budget": self.budget.target_tokens,
                "note": "LETHE's engine with a code-built summary adapter", "stats": self.session.stats()}
