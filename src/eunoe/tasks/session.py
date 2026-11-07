"""A generated session: its steps, facts, probes, tasks and the environment its tools act on."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from eunoe.interfaces import Probe, ToolSpec


@dataclass(frozen=True)
class EventSpec:
    kind: str
    text: str = ""
    input: dict | None = None
    source: str = ""
    signature: str = ""
    tokens_hint: int = 0


@dataclass(frozen=True)
class Fact:
    key: str
    subject: str
    value: str
    kind: str  # recall, artifact, continuation or decision
    prefix: str  # what a value looks like, without saying which one
    stale: tuple = ()
    at_step: int = 0


@dataclass(frozen=True)
class Task:
    id: str
    ask: str
    subject: str
    key: str
    gold: str
    prefix: str = ""
    at_step: int = 0


@dataclass(frozen=True)
class Step:
    """`kind` is event, turn, probe or task. `gap_s` is simulated time since the step before."""

    kind: str
    gap_s: float = 0.0
    event: EventSpec | None = None
    probe: Probe | None = None
    task: Task | None = None
    prefix: str = ""


class Environment:
    """The simulated world behind a session's tools. A task is scored by checking its state."""

    def __init__(self, tool_specs: list, results: dict | None = None) -> None:
        self.tool_specs = tool_specs
        self.results = dict(results or {})
        self.applied: dict = {}

    def execute(self, name: str, args: dict) -> str:
        if name == "apply":
            subject, value = str(args.get("subject", "")), str(args.get("value", ""))
            self.applied[subject] = value
            return f"applied {value} to {subject}"
        key = _result_key(name, args)
        return self.results.get(key, f"{name}: no records for {json.dumps(args, sort_keys=True)}")

    def check(self, task: Task) -> bool:
        return _norm(self.applied.get(task.subject, "")) == _norm(task.gold)


def _result_key(name: str, args: dict) -> str:
    return f"{name}\x00{json.dumps(args or {}, sort_keys=True)}"


def _norm(value: str) -> str:
    return str(value).strip().strip(".,;:'\"`").casefold()


APPLY_TOOL = ToolSpec("apply", "Apply a value to a subject in the environment. The environment records it.", {
    "type": "object",
    "properties": {"subject": {"type": "string"}, "value": {"type": "string"}},
    "required": ["subject", "value"],
})


@dataclass
class ScriptedSession:
    id: str
    domain: str
    seed: int
    profile: str
    steps: list = field(default_factory=list)
    facts: list = field(default_factory=list)
    tool_specs: list = field(default_factory=list)
    results: dict = field(default_factory=dict)

    def environment(self) -> Environment:
        return Environment(self.tool_specs, self.results)

    @property
    def probes(self) -> list:
        return [s.probe for s in self.steps if s.kind == "probe"]

    @property
    def tasks(self) -> list:
        return [s.task for s in self.steps if s.kind == "task"]

    def to_json(self) -> str:
        return json.dumps({"id": self.id, "domain": self.domain, "seed": self.seed, "profile": self.profile,
                           "steps": [asdict(s) for s in self.steps], "facts": [asdict(f) for f in self.facts]},
                          sort_keys=True, ensure_ascii=False)

    def fingerprint(self) -> str:
        """Pinned in a run's manifest, so anyone can see a fixture did not change."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
