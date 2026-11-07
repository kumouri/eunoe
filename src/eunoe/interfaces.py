"""The seams of the kit. Everything an experimenter swaps sits behind one of these protocols.

`Agent`, `Driver`, `ContextEditor`, `TaskSet`, `EvictionPolicy`, `ImportanceSignal`, `Schedule`,
`Renderer`, `Judge` and `CostModel` are the swappable parts. The default implementations know nothing
about any particular person, assistant or dataset. A private task set implements `TaskSet` and stays
wherever its owner keeps it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from eunoe.ledger import Event, Ledger
    from eunoe.levels import Level
    from eunoe.tasks.session import ScriptedSession


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict


@dataclass(frozen=True)
class Block:
    """One rendered piece of context. A harness-neutral shape; `eunoe.wire` turns it into a request."""

    role: str  # "user" or "assistant"
    type: str  # "text", "tool_use", "tool_result" or "thinking"
    text: str = ""
    name: str = ""
    input: dict | None = None
    tool_id: str = ""
    event_id: str = ""
    signature: str = ""
    tokens_hint: int = 0  # for a block whose text is opaque, such as thinking


@dataclass
class RenderedContext:
    """What the next call will see. `meta` carries run bookkeeping and is never sent to a model."""

    system: str
    blocks: list
    tools: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class CallUsage:
    """Usage as a provider reported it. `raw` keeps the original, because cost is recomputed from it."""

    input: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    output: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def cache_write(self) -> int:
        return self.cache_write_5m + self.cache_write_1h

    @property
    def context(self) -> int:
        return self.input + self.cache_read + self.cache_write


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    usage: CallUsage = field(default_factory=CallUsage)
    latency_s: float = 0.0
    error: str = ""
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DriverCaps:
    name: str
    #: "none": edited history passes. "bound": an edit before a thinking block fails the check.
    #: "account": depends on the account; run the conformance suite to find out.
    thinking_binding: str = "none"
    ttls: tuple = ("5m", "1h")
    server_edits: bool = False
    cache_granularity: str = "prefix"
    spends: bool = False
    #: Whether a fixture's scripted thinking blocks may be replayed. No real provider accepts them.
    replays_scripted_thinking: bool = False


@dataclass(frozen=True)
class Budget:
    target_tokens: int
    keep_recent_calls: int = 10


@dataclass(frozen=True)
class LevelChange:
    event_id: str
    level: int


@dataclass
class ScheduleState:
    calls: int = 0
    calls_since_edit: int = 0
    last_edit_tokens: int = 0
    hold_until_tokens: int = 0
    frozen_upto: int = -1
    expired: bool = False
    edits: int = 0


@dataclass(frozen=True)
class SessionRef:
    """A harness's name for a session. `path` is never written to a report."""

    harness: str
    id: str
    path: str = ""


@dataclass(frozen=True)
class Probe:
    id: str
    kind: str  # recall, artifact, continuation, decision or unanswerable
    ask: str
    key: str
    subject: str
    gold: str = ""  # empty for an unanswerable probe
    stale: tuple = ()
    at_step: int = 0


@dataclass
class ProbeScore:
    probe_id: str
    kind: str
    correct: bool = False
    abstained: bool = False
    hallucinated: bool = False
    stale: bool = False
    looked_up: bool = False
    answer: str = ""
    evicted: bool = False


@dataclass(frozen=True)
class BlindItem:
    """What a judge sees: a question and an answer. Never an arm, a schedule or a cost."""

    id: str
    ask: str
    answer: str
    gold: str = ""


@dataclass
class Grade:
    item_id: str
    score: float
    label: str = ""


@dataclass
class StepResult:
    step: str
    status: str  # PASS, FAIL, UNSUPPORTED, INCONCLUSIVE or SKIPPED
    note: str = ""
    evidence: dict = field(default_factory=dict)


@dataclass
class ConformanceReport:
    harness: str
    version: str = ""
    available: bool = True
    reason: str = ""
    steps: list = field(default_factory=list)
    arms: dict = field(default_factory=dict)
    list_cost_usd: float = 0.0
    spent_usd: float = 0.0

    @property
    def conforms(self) -> bool:
        """Steps 1 to 4 are what a contracting arm cannot run without."""
        required = [s for s in self.steps if s.step[:1] in "1234"]
        return bool(required) and all(s.status == "PASS" for s in required)


@runtime_checkable
class Agent(Protocol):
    """What runs: instructions, tools and loop style."""

    instructions: str

    def tools(self) -> list: ...


@runtime_checkable
class Driver(Protocol):
    """Model, provider and billing. The only thing in the kit that can spend."""

    capabilities: DriverCaps

    def call(self, request: RenderedContext) -> ModelTurn: ...


@runtime_checkable
class ContextEditor(Protocol):
    """How an edit really reaches the next call on this harness."""

    def apply(self, session: SessionRef, rendered: RenderedContext) -> SessionRef: ...

    def conformance(self) -> ConformanceReport: ...


@runtime_checkable
class TaskSet(Protocol):
    """Fixtures, probes and gold. A private set implements the same thing."""

    def sessions(self, seed: int) -> "Iterator[ScriptedSession]": ...

    def score(self, turn: ModelTurn, probe: Probe) -> ProbeScore: ...


@runtime_checkable
class EvictionPolicy(Protocol):
    """Given the ledger and a budget: which event goes down a level."""

    def plan(self, ledger: "Ledger", budget: Budget, sched: ScheduleState) -> list: ...


@runtime_checkable
class ImportanceSignal(Protocol):
    """How much an event matters. Never reads the schedule or the cache. Lower goes first."""

    def score(self, event: "Event", ledger: "Ledger", now: float) -> float: ...


@runtime_checkable
class Schedule(Protocol):
    """When edits land. Knows TTLs and prices, never importance."""

    def due(self, ledger: "Ledger", last: CallUsage, gap_s: float) -> bool: ...

    def target(self, ledger: "Ledger", budget: Budget) -> int: ...


@runtime_checkable
class Renderer(Protocol):
    """(event, level) to blocks. A pure function, so the bytes are stable across calls."""

    def render(self, event: "Event", level: "Level") -> list: ...


@runtime_checkable
class Judge(Protocol):
    """Deterministic gold first. A human or a model judges only where a run declares it."""

    def grade(self, item: BlindItem) -> Grade: ...


@runtime_checkable
class CostModel(Protocol):
    """Usage times a dated price file. A provider's own total is logged beside it, never instead."""

    def price(self, usage: CallUsage, ttl: str) -> float: ...


JSON = Any
