"""How an arm decides what the next call sees.

A `Strategy` takes events as they happen, and produces the next call's context. The contracting arms
are a policy, a signal and a schedule over a ledger. The controls are the simplest honest version of
each thing they stand for. Where a control is a provider's feature and the driver has it, the request
asks the provider; where the driver has none, the emulation is labelled in the manifest and gets the
overhead a real one would cost, so nothing is silently approximated.
"""
from __future__ import annotations

from eunoe import levels
from eunoe.interfaces import Block, Budget, CallUsage, RenderedContext, ToolSpec
from eunoe.ledger import ASSISTANT, THINKING, USER, Ledger, block_tokens
from eunoe.levels import Level, estimate_tokens
from eunoe.renderer import DefaultRenderer, render_context
from eunoe.schedules import Never
from eunoe.tasks.session import EventSpec

THINKING_POLICIES = ("allow", "strip_on_edit", "drop_block", "server_clear", "off")
KEEP_THINKING_TURNS = 2

RECALL_TOOL = ToolSpec(
    "recall",
    "Search everything that happened earlier in this session, verbatim, by an id or by words.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)


class Strategy:
    name = ""
    emulated = False

    def __init__(self, name: str, *, system: str = "", tools=None, budget: Budget | None = None,
                 thinking: str = "strip_on_edit", marker: str = "M0", renderer=None) -> None:
        if thinking not in THINKING_POLICIES:
            raise ValueError(f"unknown thinking policy {thinking!r}, one of {THINKING_POLICIES}")
        self.name = name
        self.system = system
        self.tools = list(tools or [])
        self.budget = budget or Budget(200_000)
        self.thinking = thinking
        self.ledger = Ledger(renderer=renderer or DefaultRenderer(marker=marker))
        self.prefix: list = []
        self.edits = 0
        self.overhead: list = []

    @property
    def has_recall(self) -> bool:
        return False

    def observe(self, spec: EventSpec, at: float, replay_thinking: bool = True) -> None:
        if spec.kind == THINKING and (self.thinking == "off" or not replay_thinking):
            return
        self.ledger.now = at
        self.ledger.append(spec.kind, spec.text, input=spec.input, source=spec.source, signature=spec.signature,
                           tokens_hint=spec.tokens_hint, at=at)

    def prepare(self, now: float, last: CallUsage | None, gap_s: float) -> RenderedContext:
        """Apply whatever is due, then render. The runner adds the prompt."""
        self.ledger.now = now
        self.ledger.call += 1
        self._maybe_edit(last or CallUsage(), gap_s)
        ctx = render_context(self.ledger, self.system, self._tools(),
                             {"now": now, "arm": self.name, "thinking_policy": self.thinking})
        if self.prefix:
            ctx.blocks[:0] = list(self.prefix)
        return ctx

    def _maybe_edit(self, last: CallUsage, gap_s: float) -> None:
        return None

    def _tools(self) -> list:
        return self.tools + ([RECALL_TOOL] if self.has_recall else [])

    def recall(self, query: str) -> str:
        """What a lookup returns: verbatim copies, an id first if the query is one."""
        event = self.ledger.archive_get(query.strip())
        hits = [event] if event else self.ledger.search(query)
        if not hits:
            return "no match"
        return "\n---\n".join(f"{e.id} {e.kind}: {e.text}" for e in hits)

    def held_tokens(self) -> int:
        return self.ledger.tokens_held() + sum(block_tokens(b) for b in self.prefix)

    def _strip_thinking(self, upto: int = -1) -> None:
        """Thinking is kept or dropped, never edited. `allow` keeps it and lets the account decide."""
        if self.thinking not in ("strip_on_edit", "drop_block", "server_clear"):
            return
        thinking = [e for e in self.ledger.events if e.kind == THINKING and e.index > upto]
        if self.thinking == "server_clear":
            thinking = thinking[:-KEEP_THINKING_TURNS]
        for event in thinking:
            self.ledger.set_level(event.id, Level.L4)

    def manifest(self) -> dict:
        return {"arm": self.name, "thinking": self.thinking, "budget": self.budget.target_tokens,
                "emulated": self.emulated, "edits": self.edits}


class NoManagement(Strategy):
    """C0. Nothing is ever edited. The baseline everything is compared with."""


class ContractingStrategy(Strategy):
    """T1 to T4. A ladder over a ledger: a schedule says when, a policy says what."""

    def __init__(self, name, *, policy, schedule, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.policy = policy
        self.schedule = schedule or Never()

    @property
    def has_recall(self) -> bool:
        return True

    def _maybe_edit(self, last: CallUsage, gap_s: float) -> None:
        sched = self.schedule.state
        sched.calls += 1
        sched.calls_since_edit += 1
        sched.expired = self.schedule.expired(gap_s, getattr(self.schedule, "ttl", "1h"))
        if not self.schedule.due(self.ledger, last, gap_s):
            return
        pre = self.ledger.tokens_held()
        target = self.schedule.target(self.ledger, self.budget)
        frozen_share = getattr(self.schedule, "frozen_share", 0.0)
        if frozen_share:
            sched.frozen_upto = max(sched.frozen_upto, self._frozen_boundary(frozen_share * target))
        self._strip_thinking(sched.frozen_upto)
        changes = self.policy.plan(self.ledger, Budget(target, self.budget.keep_recent_calls), sched)
        for change in changes:
            self.ledger.set_level(change.event_id, change.level)
        self.schedule.note_edit(self.ledger.tokens_held(), target, pre)
        self.edits += 1

    def prepare(self, now, last, gap_s):
        ctx = super().prepare(now, last, gap_s)
        frozen = self.schedule.state.frozen_upto
        if frozen >= 0:
            # Reads survive an edit only as far back as a cache entry. So the region's end needs a breakpoint.
            last_id = self.ledger.events[frozen].id
            ends = [i for i, block in enumerate(ctx.blocks) if block.event_id and block.event_id <= last_id]
            if ends:
                ctx.meta["breakpoints"] = [ends[-1]]
        return ctx

    def _frozen_boundary(self, want: float) -> int:
        total, boundary = 0, -1
        for event in self.ledger.events:
            total += self.ledger.tokens_at(event, self.ledger.levels[event.id])
            if total > want:
                break
            boundary = event.index
        return boundary

    def manifest(self) -> dict:
        return {**super().manifest(), "schedule": type(self.schedule).__name__, "policy": type(self.policy).__name__,
                "signal": type(self.policy.signal).__name__, "marker": self.ledger.renderer.marker}


class _PlaceholderRenderer(DefaultRenderer):
    """A tool event at L3 keeps its call and shows a placeholder for its result: what masking does."""

    def __init__(self, placeholder: str) -> None:
        super().__init__("M0")
        self.placeholder = placeholder

    def render(self, event, level):
        if event.is_tool and Level(level) == Level.L3:
            return [Block("assistant", "tool_use", name=event.tool_name, input=event.input or {},
                          tool_id=event.tool_id, event_id=event.id),
                    Block("user", "tool_result", text=self.placeholder.format(lines=len(event.text.splitlines())),
                          tool_id=event.tool_id, event_id=event.id)]
        return super().render(event, level)


class MaskingStrategy(ContractingStrategy):
    """C3. Old tool output masked, oldest first, the rest verbatim. No lookup tool."""

    def __init__(self, name, *, schedule, **kwargs) -> None:
        from eunoe.policies import MaskPolicy

        kwargs["renderer"] = _PlaceholderRenderer("[old output omitted: {lines} lines]")
        super().__init__(name, policy=MaskPolicy(), schedule=schedule, **kwargs)

    @property
    def has_recall(self) -> bool:
        return False

    def manifest(self) -> dict:
        return {**Strategy.manifest(self), "schedule": type(self.schedule).__name__, "policy": "MaskPolicy"}


class ClearToolUsesStrategy(Strategy):
    """C4. Whole tool results cleared, oldest first, at a trigger, keeping the newest few."""

    def __init__(self, name, *, trigger: int, keep_tools: int = 3, clear_at_least: int = 5000,
                 server_side: bool = False, **kwargs) -> None:
        kwargs["renderer"] = _PlaceholderRenderer("[cleared]")
        super().__init__(name, **kwargs)
        self.trigger, self.keep_tools, self.clear_at_least = trigger, keep_tools, clear_at_least
        self.server_side = server_side
        self.emulated = not server_side

    def context_management(self) -> dict:
        return {"edits": [{"type": "clear_tool_uses_20250919",
                           "trigger": {"type": "input_tokens", "value": self.trigger},
                           "keep": {"type": "tool_uses", "value": self.keep_tools},
                           "clear_at_least": {"type": "input_tokens", "value": self.clear_at_least}}]}

    def prepare(self, now, last, gap_s):
        ctx = super().prepare(now, last, gap_s)
        if self.server_side:
            ctx.meta["context_management"] = self.context_management()
        return ctx

    def _maybe_edit(self, last, gap_s) -> None:
        if self.server_side or self.ledger.tokens_held() <= self.trigger:
            return
        tools = [e for e in self.ledger.events if e.is_tool and self.ledger.levels[e.id] == Level.L0]
        cleared, now = 0, self.ledger.tokens_held()
        for event in tools[:-self.keep_tools] if self.keep_tools else tools:
            if now <= self.budget.target_tokens and cleared >= self.clear_at_least:
                break
            drop = self.ledger.tokens_at(event, Level.L0) - self.ledger.tokens_at(event, Level.L3)
            self.ledger.set_level(event.id, Level.L3)
            now -= drop
            cleared += drop
        if cleared:
            self.edits += 1
            self._strip_thinking()


class CompactionStrategy(Strategy):
    """C1. At a trigger the older history becomes one summary, and a model call pays for writing it.

    With `server_side` the request asks the provider and nothing is edited here. Otherwise a code-built
    extractive summary stands in, labelled `emulated`, so the shape of the cost shows up offline.
    """

    def __init__(self, name, *, trigger: int, summary_tokens: int = 2000, keep_events: int = 8,
                 server_side: bool = False, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.trigger, self.summary_tokens, self.keep_events = trigger, summary_tokens, keep_events
        self.server_side = server_side
        self.emulated = not server_side

    def context_management(self) -> dict:
        return {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": self.trigger}}]}

    def prepare(self, now, last, gap_s):
        ctx = super().prepare(now, last, gap_s)
        if self.server_side:
            ctx.meta["context_management"] = self.context_management()
        return ctx

    def _maybe_edit(self, last, gap_s) -> None:
        if self.server_side or self.held_tokens() <= self.trigger:
            return
        keep = set(e.id for e in self.ledger.events[-self.keep_events:])
        old = [e for e in self.ledger.events if e.id not in keep and self.ledger.levels[e.id] < Level.L4]
        summary = _summary(self.ledger, old, self.summary_tokens, self.prefix)
        before = self.held_tokens()
        for event in old:
            self.ledger.set_level(event.id, Level.L4)
        self.prefix = [Block("user", "text", text=f"Summary of the conversation so far:\n{summary}", event_id="summary")]
        self.overhead.append(CallUsage(cache_read=before, output=estimate_tokens(summary)))
        self.edits += 1


class HandoffStrategy(Strategy):
    """C2. A fresh session at a trigger, holding a code-built handoff: user words, then a tail."""

    def __init__(self, name, *, trigger: int, user_tokens: int = 1500, tail_tokens: int = 1000, keep_events: int = 6,
                 **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.trigger, self.user_tokens, self.tail_tokens, self.keep_events = trigger, user_tokens, tail_tokens, keep_events

    def _maybe_edit(self, last, gap_s) -> None:
        if self.held_tokens() <= self.trigger:
            return
        keep = set(e.id for e in self.ledger.events[-self.keep_events:])
        words, tail, spent_words, spent_tail = [], [], 0, 0
        for event in reversed(self.ledger.events):
            if event.id in keep or self.ledger.levels[event.id] >= Level.L4:
                continue
            if event.kind == USER and spent_words < self.user_tokens:
                words.append(event.text)
                spent_words += estimate_tokens(event.text)
            elif event.kind in (USER, ASSISTANT) and spent_tail < self.tail_tokens:
                tail.append(levels.first_sentence(event.text))
                spent_tail += estimate_tokens(tail[-1])
            elif event.is_tool and spent_tail < self.tail_tokens:
                tail.append(f"{event.tool_name}: {levels.headline(event.text)}")
                spent_tail += estimate_tokens(tail[-1])
        for event in self.ledger.events:
            if event.id not in keep:
                self.ledger.set_level(event.id, Level.L4)
        text = "Handoff from the previous session.\nWhat the user said:\n" + "\n".join(reversed(words))
        text += "\nWhat happened most recently:\n" + "\n".join(reversed(tail))
        self.prefix = [Block("user", "text", text=text, event_id="handoff")]
        self.edits += 1


def _summary(ledger: Ledger, events: list, budget_tokens: int, prefix: list) -> str:
    """An extractive stand-in for a model's summary: earlier summary, then each event's digest."""
    parts = [b.text for b in prefix]
    spent = sum(estimate_tokens(p) for p in parts)
    share = max(20, budget_tokens // max(len(events), 1))
    for event in events:
        if event.kind == THINKING:
            continue
        line = levels.first_sentence(event.text) if not event.is_tool else f"{event.tool_name}: {levels.digest(event.text)}"
        line = line[: share * 4]
        spent += estimate_tokens(line)
        if spent > budget_tokens:
            break
        parts.append(line)
    return "\n".join(parts)
