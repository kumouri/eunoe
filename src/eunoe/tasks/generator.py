"""A seeded procedural generator: long scripted sessions in invented domains, with gold by construction.

Stdlib, deterministic, no private data, and nothing derived from any real transcript. Each session has a
tool surface against a simulated environment that returns large outputs, plus:

* **planted facts** at known positions with known gold, some later superseded;
* **probes** at scheduled points, of four kinds: recall, artifact, continuation and decision;
* **unanswerable probes**, where the right answer is an admission or a lookup;
* **tasks** that need earlier detail and are scored by checking the environment's state.

Gold comes by construction, so scoring needs no judge, and new seeds avoid a benchmark's memorisation
problem. Everything a person might be named is built from syllables, so no name is anyone's.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from eunoe.interfaces import Probe, ToolSpec
from eunoe.ledger import ASSISTANT, FILE, SYSTEM_NOTE, THINKING, USER
from eunoe.rng import Rng
from eunoe.tasks.session import APPLY_TOOL, EventSpec, Fact, ScriptedSession, Step, Task, _result_key

_ONSETS = ("b", "br", "d", "dr", "f", "g", "gr", "k", "l", "m", "n", "p", "r", "s", "t", "th", "v", "z")
_VOWELS = ("a", "e", "i", "o", "u", "ae", "io")
_CODAS = ("", "l", "n", "r", "s", "th", "x")
_WORDS = ("amber", "basalt", "cobalt", "delta", "ember", "flint", "garnet", "harbor", "indigo", "juniper",
          "kestrel", "lumen", "marrow", "nimbus", "onyx", "pewter", "quartz", "russet", "sable", "tundra")


def invented_name(rng: Rng) -> str:
    def part() -> str:
        return "".join(rng.choice(s) for s in (_ONSETS, _VOWELS, _ONSETS, _VOWELS, _CODAS)).capitalize()
    return f"{part()} {part()}"


def _token(rng: Rng, length: int = 5) -> str:
    return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(length))


@dataclass(frozen=True)
class Profile:
    """The shape of a session. `shares` is the chance of each non-tool event per step."""

    name: str
    steps: int
    mean_result_chars: int
    probes: int
    tasks: int
    paced_every: int
    shares: dict = field(default_factory=dict)
    gap_s: float = 30.0
    long_gap_every: int = 0
    long_gap_s: float = 4000.0
    thinking_tokens: int = 200


#: A neutral composition, chosen for variety and tied to no measurement. Pass your own to match a
#: real workload; `eunoe replay` measures one from your own transcripts without keeping their text.
NEUTRAL_SHARES = {THINKING: 0.25, ASSISTANT: 0.25, USER: 0.10, SYSTEM_NOTE: 0.08, FILE: 0.07}

PROFILES = {
    "tiny": Profile("tiny", 24, 1600, 4, 1, 8, NEUTRAL_SHARES),
    "demo": Profile("demo", 160, 3200, 14, 3, 12, NEUTRAL_SHARES, long_gap_every=60),
    "standard": Profile("standard", 600, 6000, 25, 5, 5, NEUTRAL_SHARES, long_gap_every=120),
}


def get_profile(name: str, **overrides) -> Profile:
    try:
        base = PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}, one of {sorted(PROFILES)}") from None
    return Profile(**{**base.__dict__, **overrides}) if overrides else base


# -------------------------------------------------------------------------------------- domains

class Domain:
    name = ""
    tools: tuple = ()
    brief = ""

    def __init__(self, rng: Rng) -> None:
        self.rng = rng

    def subject(self) -> str:
        raise NotImplementedError

    def plant(self, kind: str, subject: str) -> tuple:
        """`(key, value, prefix)` for a fact of a kind about a subject."""
        raise NotImplementedError

    def filler_line(self) -> str:
        raise NotImplementedError

    def file_path(self) -> str:
        return f"docs/{self.rng.choice(_WORDS)}-{self.rng.randint(10, 99)}.md"

    def tool_specs(self) -> list:
        return [ToolSpec(name, desc, {"type": "object", "properties": {arg: {"type": "string"}}, "required": [arg]})
                for name, desc, arg in self.tools] + [APPLY_TOOL]


class OnCall(Domain):
    name = "oncall"
    brief = "You are helping an on-call engineer run a shift for an invented service fleet."
    tools = (("fetch_logs", "Fetch recent logs for a service.", "service"),
             ("list_deploys", "List recent deployments of a service.", "service"),
             ("read_runbook", "Read a runbook by path.", "path"))

    def subject(self) -> str:
        return self.rng.choice(("INC", "DEP", "SVC")) + f"-{self.rng.randint(1000, 9999)}"

    def plant(self, kind: str, subject: str) -> tuple:
        r = self.rng
        if kind == "artifact":
            return f"runbook path for {subject}", f"runbooks/{r.choice(_WORDS)}-{r.randint(100, 999)}.md", "runbooks/"
        if kind == "continuation":
            return f"next step for {subject}", f"STEP-{r.choice(_WORDS).upper()}-{r.randint(10, 99)}", "STEP-"
        if kind == "decision":
            return f"mitigation chosen for {subject}", f"OPTION-{_token(r, 4)}", "OPTION-"
        if r.random() < 0.5:
            return f"rollback token for {subject}", f"RB-{_token(r)}-{r.randint(10, 99)}", "RB-"
        return f"failing host for {subject}", f"host-{r.choice(_WORDS)}-{r.randint(10, 99)}", "host-"

    def filler_line(self) -> str:
        r = self.rng
        level = r.choice(("INFO", "INFO", "INFO", "WARN", "DEBUG", "ERROR"))
        word = r.choice(_WORDS)
        return (f"2031-0{r.randint(1, 9)}-1{r.randint(0, 9)}T0{r.randint(0, 9)}:{r.randint(10, 59)}:{r.randint(10, 59)} "
                f"{level} {word}-{r.randint(10, 99)} request {r.randint(100000, 999999)} took {r.randint(3, 900)}ms "
                f"shard={r.randint(0, 31)} status={r.choice((200, 200, 200, 204, 429, 503))}")


class SmallBusiness(Domain):
    name = "smallbiz"
    brief = "You are helping the chief of staff of an invented small business run its week."
    tools = (("list_invoices", "List open invoices for a vendor.", "vendor"),
             ("search_mail", "Search the shared inbox of the invented business.", "query"),
             ("read_contract", "Read a contract by path.", "path"))

    def subject(self) -> str:
        return self.rng.choice(("PO", "INV", "VND")) + f"-{self.rng.randint(1000, 9999)}"

    def plant(self, kind: str, subject: str) -> tuple:
        r = self.rng
        if kind == "artifact":
            return f"contract path for {subject}", f"contracts/{r.choice(_WORDS)}-{r.randint(100, 999)}.pdf", "contracts/"
        if kind == "continuation":
            return f"next step for {subject}", f"STEP-{r.choice(_WORDS).upper()}-{r.randint(10, 99)}", "STEP-"
        if kind == "decision":
            return f"supplier chosen for {subject}", f"SUP-{_token(r, 4)}", "SUP-"
        if r.random() < 0.5:
            return f"approver of {subject}", invented_name(r), ""
        return f"payment reference for {subject}", f"PAY-{_token(r)}-{r.randint(10, 99)}", "PAY-"

    def filler_line(self) -> str:
        r = self.rng
        return (f"{r.choice(('invoice', 'order', 'memo', 'receipt'))} {r.randint(10000, 99999)} from "
                f"{invented_name(r)} re {r.choice(_WORDS)} supplies, {r.randint(1, 40)} units, "
                f"terms net {r.choice((15, 30, 45))}, status {r.choice(('open', 'paid', 'held', 'queried'))}")


class Research(Domain):
    name = "research"
    brief = "You are helping a research assistant keep track of an invented literature review."
    tools = (("fetch_paper", "Fetch a paper by id.", "id"),
             ("query_dataset", "Query an invented dataset by name.", "name"),
             ("search_notes", "Search the reading notes.", "query"))

    def subject(self) -> str:
        return self.rng.choice(("PAP", "SET", "RUN")) + f"-{self.rng.randint(1000, 9999)}"

    def plant(self, kind: str, subject: str) -> tuple:
        r = self.rng
        if kind == "artifact":
            return f"notes path for {subject}", f"notes/{r.choice(_WORDS)}-{r.randint(100, 999)}.txt", "notes/"
        if kind == "continuation":
            return f"next step for {subject}", f"STEP-{r.choice(_WORDS).upper()}-{r.randint(10, 99)}", "STEP-"
        if kind == "decision":
            return f"baseline chosen for {subject}", f"BASE-{_token(r, 4)}", "BASE-"
        if r.random() < 0.5:
            return f"first author of {subject}", invented_name(r), ""
        return f"random seed for {subject}", f"SEED-{r.randint(100000, 999999)}", "SEED-"

    def filler_line(self) -> str:
        r = self.rng
        return (f"{invented_name(r)} et al. report {r.choice(_WORDS)} at {r.randint(40, 99)}.{r.randint(0, 9)} percent "
                f"on the {r.choice(_WORDS)} split, n={r.randint(50, 5000)}, with {r.choice(_WORDS)} controls; "
                f"see section {r.randint(1, 9)}.{r.randint(1, 9)}")


DOMAINS = {cls.name: cls for cls in (OnCall, SmallBusiness, Research)}
_FACT_KINDS = ("recall", "recall", "artifact", "continuation", "decision")
_STATEMENT = "The {key} is {value}."
_ASK = "What is the {key}? Reply on one line as ANSWER: <value>, or ANSWER: UNKNOWN if you cannot tell."


class Generator:
    """Implements `TaskSet.sessions`. One session per domain per seed."""

    def __init__(self, profile: str | Profile = "demo", domains=None) -> None:
        self.profile = get_profile(profile) if isinstance(profile, str) else profile
        self.domains = tuple(domains or DOMAINS)

    def sessions(self, seed: int):
        for name in self.domains:
            yield generate_session(name, seed, self.profile)

    def score(self, turn, probe):
        from eunoe.judge import score_probe
        return score_probe(turn.text, probe)


def generate_session(domain_name: str, seed: int, profile: Profile) -> ScriptedSession:
    if domain_name not in DOMAINS:
        raise ValueError(f"unknown domain {domain_name!r}, one of {sorted(DOMAINS)}")
    rng = Rng("eunoe-fixture-v1", domain_name, seed, profile.name)
    domain = DOMAINS[domain_name](rng.child("domain"))
    session = ScriptedSession(f"{domain_name}-{profile.name}-{seed}", domain_name, seed, profile.name,
                              tool_specs=domain.tool_specs())
    plan = _plan_facts(domain, profile, rng.child("plan"))
    session.facts = [f for f, _ in plan["facts"]]
    planted = {}
    for fact, where in plan["facts"]:
        planted.setdefault(where, []).append(("fact", fact, fact.stale[0] if fact.stale else fact.value))
    for fact, where in plan["supersede"]:
        planted.setdefault(where, []).append(("update", fact, fact.value))
    probes_at, tasks_at = plan["probes"], plan["tasks"]
    prng = rng.child("steps")
    steps = []
    for i in range(profile.steps):
        gap = profile.gap_s * (0.5 + prng.random())
        if profile.long_gap_every and i and i % profile.long_gap_every == 0:
            gap = profile.long_gap_s
        steps.append(Step("event", gap, _make_event(i, domain, profile, prng, planted.pop(i, []), session)))
        for probe in probes_at.get(i, ()):
            steps.append(Step("probe", 2.0, probe=probe, prefix=plan["prefixes"].get(probe.id, "")))
        for task in tasks_at.get(i, ()):
            steps.append(Step("task", 2.0, task=task, prefix=task.prefix))
        if profile.paced_every and i % profile.paced_every == profile.paced_every - 1:
            steps.append(Step("turn", 1.0))
    session.steps = steps
    return session


def _plan_facts(domain: Domain, profile: Profile, rng: Rng) -> dict:
    n_probes, n_tasks = profile.probes, profile.tasks
    unanswerable = max(1, n_probes // 6)
    needed = n_probes - unanswerable + n_tasks
    facts, supersede, used = [], [], set()
    horizon = max(1, int(profile.steps * 0.85))
    for k in range(needed):
        subject = domain.subject()
        while subject in used:
            subject = domain.subject()
        used.add(subject)
        kind = _FACT_KINDS[k % len(_FACT_KINDS)]
        key, value, prefix = domain.plant(kind, subject)
        where = rng.randint(0, horizon - 1)
        stale = ()
        if kind == "recall" and rng.random() < 0.25 and where + 4 < horizon:
            old = value
            _, value, _ = domain.plant(kind, subject)
            if value != old and prefix and value.startswith(prefix):
                stale = (old,)
            else:
                value = old
        fact = Fact(key, subject, value, kind, prefix, stale, where)
        facts.append((fact, where))
        if stale:
            supersede.append((fact, rng.randint(where + 1, horizon)))
    rng.shuffle(facts)
    probe_facts, task_facts = facts[:needed - n_tasks], facts[needed - n_tasks:]
    probes, tasks, prefixes = {}, {}, {}
    for j, (fact, where) in enumerate(probe_facts):
        at = rng.randint(max(where + 1, horizon), profile.steps - 1) if rng.random() < 0.7 else \
            rng.randint(where + 1, profile.steps - 1)
        probe = Probe(f"p{j:03d}", fact.kind, _ASK.format(key=fact.key), fact.key, fact.subject, fact.value, fact.stale, at)
        probes.setdefault(at, []).append(probe)
        prefixes[probe.id] = fact.prefix
    for j in range(unanswerable):
        subject = domain.subject()
        while subject in used:
            subject = domain.subject()
        used.add(subject)
        key, _, prefix = domain.plant("recall", subject)
        at = rng.randint(profile.steps // 2, profile.steps - 1)
        probe = Probe(f"u{j:03d}", "unanswerable", _ASK.format(key=key), key, subject, "", (), at)
        probes.setdefault(at, []).append(probe)
        prefixes[probe.id] = prefix
    for j, (fact, where) in enumerate(task_facts):
        at = rng.randint(max(where + 1, horizon), profile.steps - 1)
        ask = (f"Apply the {fact.key} to {fact.subject} in the environment with the apply tool, "
               f"subject {fact.subject}.")
        tasks.setdefault(at, []).append(Task(f"t{j:03d}", ask, fact.subject, fact.key, fact.value, fact.prefix, at))
    return dict(facts=facts, supersede=supersede, probes=probes, tasks=tasks, prefixes=prefixes)


def _body(domain: Domain, rng: Rng, chars: int, planted: list) -> str:
    lines, size = [], 0
    while size < chars:
        line = domain.filler_line()
        lines.append(line)
        size += len(line) + 1
    for _, fact, value in planted:
        at = rng.randint(len(lines) // 4, max(len(lines) // 4, len(lines) - 1 - len(lines) // 4)) if lines else 0
        lines.insert(at, _STATEMENT.format(key=fact.key, value=value))
    return "\n".join(lines)


def _heavy_tail(rng: Rng, mean: int) -> int:
    """Most outputs are short and a few are very long: a long tail, chosen for variety."""
    draw = rng.random()
    if draw < 0.6:
        return int(mean * (0.2 + 0.4 * rng.random()))
    if draw < 0.9:
        return int(mean * (0.8 + 1.0 * rng.random()))
    return int(mean * (3.0 + 4.0 * rng.random()))


def _make_event(i: int, domain: Domain, profile: Profile, rng: Rng, planted: list, session: ScriptedSession) -> EventSpec:
    draw, edge = rng.random(), 0.0
    kind = "tool"
    if not planted:
        for candidate, share in sorted(profile.shares.items()):
            edge += share
            if draw < edge:
                kind = candidate
                break
    prose = [p for p in planted if p[1].kind in ("continuation", "decision")]
    if prose:
        kind = ASSISTANT
    if kind == THINKING:
        return EventSpec(THINKING, "", signature=f"sig-{session.id}-{i}", tokens_hint=profile.thinking_tokens)
    if kind == ASSISTANT:
        lines = [f"Working through the {rng.choice(_WORDS)} queue now; nothing unusual so far."]
        lines += [_STATEMENT.format(key=f.key, value=v) for _, f, v in planted]
        lines += [domain.filler_line() for _ in range(rng.randint(1, 4))]
        return EventSpec(ASSISTANT, "\n\n".join(lines))
    if kind == USER:
        return EventSpec(USER, f"Please keep an eye on {domain.subject()} and carry on with the {rng.choice(_WORDS)} work.")
    if kind == SYSTEM_NOTE:
        return EventSpec(SYSTEM_NOTE, f"Reminder: the {rng.choice(_WORDS)} review is due at the end of the shift.")
    if kind == FILE:
        path = domain.file_path()
        return EventSpec(FILE, _body(domain, rng, _heavy_tail(rng, profile.mean_result_chars), []), source=path)
    name, _, arg = rng.choice(domain.tools)
    value = planted[0][1].subject if planted else f"{rng.choice(_WORDS)}-{rng.randint(10, 99)}"
    args = {arg: value}
    text = _body(domain, rng, _heavy_tail(rng, profile.mean_result_chars), planted)
    session.results[_result_key(name, args)] = text
    return EventSpec(f"tool:{name}", text, input=args)
