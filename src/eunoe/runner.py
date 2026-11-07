"""Drive one scripted session through one arm and one driver, and record what happened.

Probes are forks: a question and its answer never enter the ledger, so asking does not teach the
model the fact. A task's tool call does enter it, because a task really changes the environment.

A pacing turn stays in history, as it would in a real session. That matters for the cost as well as the shape: a cache
entry ends on the request's last block, so a request whose prompt is then thrown away leaves an entry nothing can match.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from eunoe.interfaces import Block, CallUsage, ModelTurn, RenderedContext
from eunoe.judge import score_probe
from eunoe.ledger import ASSISTANT, USER
from eunoe.tasks.session import EventSpec, ScriptedSession
from eunoe.wire import to_text

PACE_PROMPT = "Continue. Reply with only the word OK."
MAX_TOOL_ROUNDS = 3


@dataclass
class SessionResult:
    session: str
    arm: str
    seed: int
    domain: str
    calls: int = 0
    cost_usd: float = 0.0
    usage: CallUsage = field(default_factory=CallUsage)
    scores: list = field(default_factory=list)
    tasks_ok: int = 0
    tasks_total: int = 0
    lookups: int = 0
    lookup_tokens: int = 0
    errors: list = field(default_factory=list)
    context_sum: int = 0
    context_peak: int = 0
    edits: int = 0
    worst_latency_s: float = 0.0
    manifest: dict = field(default_factory=dict)

    @property
    def mean_context(self) -> float:
        return self.context_sum / self.calls if self.calls else 0.0

    def rate(self, predicate, pool=None) -> float:
        rows = [s for s in self.scores if pool is None or pool(s)]
        return sum(1 for s in rows if predicate(s)) / len(rows) if rows else 0.0


def _add(total: CallUsage, usage: CallUsage) -> None:
    total.input += usage.input
    total.cache_read += usage.cache_read
    total.cache_write_5m += usage.cache_write_5m
    total.cache_write_1h += usage.cache_write_1h
    total.output += usage.output


class SessionRun:
    def __init__(self, session: ScriptedSession, strategy, driver, cost, ttl: str = "5m", spend_log=None) -> None:
        self.session, self.strategy, self.driver, self.cost, self.ttl = session, strategy, driver, cost, ttl
        self.env = session.environment()
        self.now = 0.0
        self.last_call_at = 0.0
        self.last = CallUsage()
        self.last_history = 0
        self.spend_log = spend_log
        self.result = SessionResult(session.id, strategy.name, session.seed, session.domain)
        self.replay_thinking = driver.capabilities.replays_scripted_thinking

    def run(self) -> SessionResult:
        for step in self.session.steps:
            self.now += step.gap_s
            if step.kind == "event":
                self.strategy.observe(step.event, self.now, self.replay_thinking)
            elif step.kind == "turn":
                self._turn(PACE_PROMPT, {})
            elif step.kind == "probe":
                self._probe(step)
            elif step.kind == "task":
                self._task(step)
        self.result.edits = self.strategy.edits
        self.result.manifest = {**self.strategy.manifest(), "fixture": self.session.fingerprint(),
                                "driver": self.driver.capabilities.name, "ttl": self.ttl}
        return self.result

    def _context(self, prompt: str, meta: dict, extra=()) -> RenderedContext:
        gap = self.now - self.last_call_at
        ctx = self.strategy.prepare(self.now, self.last, gap)
        ctx.meta.update(meta)
        history = len(ctx.blocks)
        ctx.blocks = list(ctx.blocks) + list(extra) + [Block("user", "text", text=prompt, event_id="prompt")]
        # A breakpoint at the end of the history writes an entry a later request can read, whatever this prompt was,
        # and one where the last request's history ended still reaches back after a long gap. A harness that can't
        # place breakpoints ignores both.
        points = list(ctx.meta.get("breakpoints", ()))
        if history:
            points.append(history - 1)
        if self.last_history and self.last_history <= history:
            points.append(self.last_history - 1)
        ctx.meta["breakpoints"] = sorted(set(points))
        self.last_history = history
        return ctx

    def _call(self, ctx: RenderedContext) -> ModelTurn:
        started = time.monotonic()
        turn = self.driver.call(ctx)
        result = self.result
        result.worst_latency_s = max(result.worst_latency_s, turn.latency_s or time.monotonic() - started)
        if turn.error:
            result.errors.append(turn.error[:200])
            return turn
        for usage in self.strategy.overhead:
            self._bill(usage, count=False)
        self.strategy.overhead.clear()
        self._bill(turn.usage)
        self.last = turn.usage
        self.last_call_at = self.now
        return turn

    def _bill(self, usage: CallUsage, count: bool = True) -> None:
        result = self.result
        _add(result.usage, usage)
        result.cost_usd += self.cost.price(usage, self.ttl)
        if count:
            result.calls += 1
            result.context_sum += usage.context
            result.context_peak = max(result.context_peak, usage.context)
        if self.spend_log is not None:
            self.spend_log(usage, self.cost.price(usage, self.ttl))

    def _turn(self, prompt: str, meta: dict) -> ModelTurn:
        turn = self._call(self._context(prompt, meta))
        if not turn.error:
            self.strategy.observe(EventSpec(USER, prompt), self.now)
            self.strategy.observe(EventSpec(ASSISTANT, turn.text or " "), self.now)
        return turn

    def _lookup(self, call) -> list:
        """A lookup's call and result, kept only for this question."""
        text = self.strategy.recall(str(call.args.get("query", "")))
        self.result.lookups += 1
        self.result.lookup_tokens += len(text) // 4
        return [Block("assistant", "tool_use", name=call.name, input=call.args, tool_id=call.id),
                Block("user", "tool_result", text=text, tool_id=call.id)]

    def _probe(self, step) -> None:
        probe = step.probe
        meta = {"id": probe.id, "key": probe.key, "subject": probe.subject, "prefix": step.prefix, "mode": "probe"}
        ctx = self._context(probe.ask, meta)
        evicted = bool(probe.gold) and probe.gold not in to_text(ctx)
        extra, looked_up = [], False
        turn = self._call(ctx)
        for _ in range(MAX_TOOL_ROUNDS):
            lookups = [c for c in turn.tool_calls if c.name == "recall"]
            if turn.error or not lookups:
                break
            for call in lookups:
                extra += self._lookup(call)
            looked_up = True
            turn = self._call(self._context(probe.ask, {**meta, "looked_up": True}, extra))
        if not turn.error:
            self.result.scores.append(score_probe(turn.text, probe, looked_up, evicted))

    def _task(self, step) -> None:
        task = step.task
        meta = {"id": task.id, "key": task.key, "subject": task.subject, "prefix": step.prefix, "mode": "task"}
        self.result.tasks_total += 1
        extra, done = [], False
        turn = self._call(self._context(task.ask, meta))
        for _ in range(MAX_TOOL_ROUNDS):
            if turn.error or not turn.tool_calls:
                break
            for call in turn.tool_calls:
                if call.name == "recall":
                    extra += self._lookup(call)
                    continue
                output = self.env.execute(call.name, call.args)
                self.strategy.observe(EventSpec(f"tool:{call.name}", output, input=call.args), self.now)
                done = call.name == "apply"
            if done:
                break
            turn = self._call(self._context(task.ask, {**meta, "looked_up": True}, extra))
        if self.env.check(task):
            self.result.tasks_ok += 1
        self.strategy.observe(EventSpec(ASSISTANT, f"Finished: {task.ask}"), self.now)


def aggregate(results: list) -> dict:
    """Arm-level measures. Rates pool every probe in every session."""
    scores = [s for r in results for s in r.scores]
    answerable = [s for s in scores if s.kind != "unanswerable"]
    unanswerable = [s for s in scores if s.kind == "unanswerable"]
    evicted = [s for s in answerable if s.evicted]
    tasks_ok, tasks = sum(r.tasks_ok for r in results), sum(r.tasks_total for r in results)
    cost = sum(r.cost_usd for r in results)
    calls = sum(r.calls for r in results)

    def share(rows, field_name):
        return sum(1 for s in rows if getattr(s, field_name)) / len(rows) if rows else 0.0

    by_kind = {}
    for kind in ("recall", "artifact", "continuation", "decision"):
        rows = [s for s in answerable if s.kind == kind]
        if rows:
            by_kind[kind] = share(rows, "correct")
    return {
        "probe_recall": share(answerable, "correct"),
        "recall_by_kind": by_kind,
        "hallucination_on_evicted": share(evicted, "hallucinated"),
        "correct_abstention": share(unanswerable, "correct"),
        "staleness": share(answerable, "stale"),
        "task_success": tasks_ok / tasks if tasks else 0.0,
        "lookups": sum(r.lookups for r in results),
        "cost_usd": cost,
        "cost_per_success_usd": cost / tasks_ok if tasks_ok else None,
        "calls": calls,
        "mean_context": sum(r.context_sum for r in results) / calls if calls else 0.0,
        "peak_context": max((r.context_peak for r in results), default=0),
        "edits": sum(r.edits for r in results),
        "cache_read": sum(r.usage.cache_read for r in results),
        "cache_write": sum(r.usage.cache_write for r in results),
        "errors": sum(len(r.errors) for r in results),
    }
