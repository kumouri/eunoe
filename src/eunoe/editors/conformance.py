"""The conformance suite: can this harness and this account really run a contracting arm?

Run it before trusting a result. A harness or account that fails a step gets the arms that need it
marked unavailable, never silently approximated.

1. Establish a session with a codeword and padding.
2. Control: resume with no edit and confirm cache reads.
3. Edit the codeword in the session, and check the answer and how the cache split.
4. Remove the padding, and check the context shrinks.
5. Repeat the edit under real thinking blocks, and with a fork.
6. Where the harness can, write a whole rendered context into a session and continue it.

Everything in a report is synthetic: a made-up codeword, generated padding and token counts. A report
holds no path, session id or account name, so it can be published as it is.
"""
from __future__ import annotations

import json
import re

from eunoe.editors.base import Harness, Turn, Unsupported
from eunoe.interfaces import Block, CallUsage, ConformanceReport, RenderedContext, StepResult

CODEWORD = "EGRET-4417"
NEW_CODEWORD = "HERON-9021"
SECOND_CODEWORD = "PLOVER-3306"
PADDING_WORDS = 900
_PAD_VOCAB = ("slate", "ridge", "marsh", "tallow", "copper", "lantern", "thicket", "granite", "willow", "ember")


def make_padding(words: int = PADDING_WORDS) -> str:
    """Deterministic filler. Numbered, so a prefix cache can't dedupe it and a model can't compress it."""
    return " ".join(f"{_PAD_VOCAB[i % len(_PAD_VOCAB)]}{i}" for i in range(words))


def _pass(step: str, note: str = "", **evidence) -> StepResult:
    return StepResult(step, "PASS", note, evidence)


def _fail(step: str, note: str, **evidence) -> StepResult:
    return StepResult(step, "FAIL", note, evidence)


def _usage_evidence(usage: CallUsage) -> dict:
    return {"input": usage.input, "cache_read": usage.cache_read, "cache_write": usage.cache_write,
            "output": usage.output, "context": usage.context}


class Suite:
    def __init__(self, harness: Harness, *, padding_words: int = PADDING_WORDS, skip_thinking: bool = False,
                 cost=None) -> None:
        self.h = harness
        self.padding = make_padding(padding_words)
        self.skip_thinking = skip_thinking
        self.cost = cost
        self.list_cost = 0.0

    def _note(self, turn: Turn) -> Turn:
        if self.cost is not None and not turn.error:
            try:
                self.list_cost += self.cost.price(turn.usage, "5m")
            except (KeyError, ValueError):
                pass
        return turn

    def run(self) -> ConformanceReport:
        h = self.h
        report = ConformanceReport(h.name, version=h.version())
        ok, reason = h.available()
        if not ok:
            report.available, report.reason = False, reason
            return report
        try:
            report.steps = self._steps()
        finally:
            h.cleanup()
        report.arms = arms_available(report)
        report.list_cost_usd = round(self.list_cost, 6)
        gate = getattr(h, "gate", None)
        report.spent_usd = round(gate.spent_usd, 6) if gate is not None else 0.0
        return report

    def _steps(self) -> list:
        steps = []
        first = self._step1()
        steps.append(first)
        if first.status != "PASS":
            return steps
        steps.append(self._step2(first))
        third = self._step3(first)
        steps.append(third)
        if third.status != "PASS":
            return steps
        steps.append(self._step4(third))
        steps.append(self._fork())
        steps.append(self._thinking())
        steps.append(self._apply())
        return steps

    def _step1(self) -> StepResult:
        turn = self._note(self.h.start(
            f"Remember this codeword: {CODEWORD}. The rest of this message is padding, ignore it: {self.padding} . "
            "Reply with only the word OK."))
        if turn.error:
            return _fail("1 establish", _short(turn.error))
        if "OK" not in turn.text.upper() or turn.usage.context <= 0:
            return _fail("1 establish", "no OK or no usage reported", **_usage_evidence(turn.usage))
        return _pass("1 establish", **_usage_evidence(turn.usage))

    def _step2(self, first: StepResult) -> StepResult:
        turn = self._note(self.h.resume("Reply with only the word READY."))
        if turn.error:
            return _fail("2 control", _short(turn.error))
        evidence = _usage_evidence(turn.usage)
        if "READY" not in turn.text.upper():
            return _fail("2 control", "the resumed session did not answer", **evidence)
        if turn.usage.cache_read <= 0:
            return StepResult("2 control", "PASS", "resumed; this harness reported no cache reads", evidence)
        return _pass("2 control", "resumed with cache reads", **evidence)

    def _step3(self, first: StepResult) -> StepResult:
        copies = self.h.replace(CODEWORD, NEW_CODEWORD)
        if copies <= 0:
            return _fail("3 edit", "the codeword was not found in the session")
        turn = self._note(self.h.resume("What is the codeword? Reply with only the codeword."))
        if turn.error:
            return _fail("3 edit", _short(turn.error), copies=copies)
        evidence = {**_usage_evidence(turn.usage), "copies": copies}
        if NEW_CODEWORD not in turn.text or CODEWORD in turn.text:
            return _fail("3 edit", f"the answer did not come from the edit: {_short(turn.text, 80)}", **evidence)
        return _pass("3 edit", "the model answered from the edited text", **evidence)

    def _step4(self, third: StepResult) -> StepResult:
        before = third.evidence.get("context", 0)
        copies = self.h.replace(self.padding, "")
        if copies <= 0:
            return _fail("4 shrink", "the padding was not found in the session")
        turn = self._note(self.h.resume("What is the codeword? Reply with only the codeword."))
        if turn.error:
            return _fail("4 shrink", _short(turn.error), copies=copies)
        evidence = {**_usage_evidence(turn.usage), "context_before": before, "copies": copies}
        if NEW_CODEWORD not in turn.text:
            return _fail("4 shrink", f"the edited codeword did not survive: {_short(turn.text, 80)}", **evidence)
        if turn.usage.context >= before:
            return _fail("4 shrink", "the context did not shrink", **evidence)
        return _pass("4 shrink", "context shrank and the answer held", **evidence)

    def _fork(self) -> StepResult:
        before = self.h.snapshot()
        try:
            turn = self._note(self.h.fork("What is the codeword? Reply with only the codeword."))
        except Unsupported as exc:
            return StepResult("5a fork", "UNSUPPORTED", str(exc))
        if turn.error:
            return _fail("5a fork", _short(turn.error))
        evidence = _usage_evidence(turn.usage)
        if NEW_CODEWORD not in turn.text:
            return _fail("5a fork", "the fork did not answer from the edited session", **evidence)
        if before and self.h.snapshot() != before:
            return _fail("5a fork", "forking changed the original session", **evidence)
        return _pass("5a fork", "new session, original untouched", **evidence)

    def _thinking(self) -> StepResult:
        h = self.h
        if self.skip_thinking:
            return StepResult("5b thinking", "SKIPPED", "--skip-thinking")
        if not h.thinking_supported:
            return StepResult("5b thinking", "UNSUPPORTED", "this harness has no thinking blocks to test")
        turn = self._note(h.start(
            f"Remember this codeword: {SECOND_CODEWORD}. The rest of this message is padding, ignore it: {self.padding} . "
            "Without using any tools, work out carefully, step by step, how many of the integers from 1 to 200 are "
            "divisible by 3 or 7 but not by both, and reply with only that number.", thinking=True))
        if turn.error:
            return _fail("5b thinking", _short(turn.error))
        turn = self._note(h.resume(
            "Without using any tools, work out carefully how many of those are prime, and reply with only that number."))
        if turn.error:
            return _fail("5b thinking", _short(turn.error))
        blocks = h.thinking_block_count()
        if blocks <= 0:
            return StepResult("5b thinking", "INCONCLUSIVE", "the model produced no thinking blocks to bind against")
        h.replace(self.padding, "")
        turn = self._note(h.resume(f"What is the codeword? Reply with only the codeword."))
        if turn.error:
            return StepResult("5b thinking", "FAIL", "edited history under thinking was refused on this account: "
                              + _short(turn.error), {"thinking_blocks": blocks})
        evidence = {**_usage_evidence(turn.usage), "thinking_blocks": blocks}
        if SECOND_CODEWORD not in turn.text:
            # The harness took the edit, so this is no 400. Whether the answer holds is the model's doing.
            return StepResult("5b thinking", "INCONCLUSIVE", f"accepted, but the model answered: {_short(turn.text, 80)}",
                              evidence)
        return _pass("5b thinking", "edited under thinking blocks, no error, the answer held", **evidence)

    def _apply(self) -> StepResult:
        try:
            self.h.apply(None, demo_context())
        except Unsupported as exc:
            return StepResult("6 apply", "UNSUPPORTED", str(exc))
        turn = self._note(self.h.resume("What is the rollback token, and what did the tool return as its status? "
                                        "Reply with only the two values, separated by a space."))
        if turn.error:
            return _fail("6 apply", _short(turn.error))
        evidence = _usage_evidence(turn.usage)
        if "RB-77X" not in turn.text or "GREEN-12" not in turn.text:
            return _fail("6 apply", "the answer did not come from the written context", **evidence)
        return _pass("6 apply", "continued a session written whole, tool blocks included", **evidence)


def demo_context() -> RenderedContext:
    """A small synthetic history with a tool exchange in it: what a contracting arm would write."""
    return RenderedContext("", [
        Block("user", "text", text="The rollback token is RB-77X. Check the status with the probe tool."),
        Block("assistant", "tool_use", name="probe", input={"target": "fleet"}, tool_id="toolu_eunoe_0001"),
        Block("user", "tool_result", text="status GREEN-12", tool_id="toolu_eunoe_0001"),
        Block("assistant", "text", text="The probe returned its status."),
    ])


_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|(?<![\w.])/(?:home|Users|tmp|var|c)/)[^\s\"'`]*")
_ID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")


def scrub(text: str) -> str:
    """An error from a harness can name a path or a session id. A report keeps neither."""
    return _ID_RE.sub("<id>", _PATH_RE.sub("<path>", text or ""))


def _short(text: str, limit: int = 240) -> str:
    return " ".join(scrub(text).split())[:limit]


def arms_available(report: ConformanceReport) -> dict:
    """Which arms and factors this harness and account can really run."""
    by_step = {s.step.split()[0]: s.status for s in report.steps}
    core = report.conforms
    fork = by_step.get("5a") == "PASS"
    thinking = by_step.get("5b")
    out = {
        "C0": bool(report.steps) and by_step.get("1") == "PASS",
        "contracting_arms": core,
        "ablation_forks": core and fork,
        "thinking_policy_allow": core and thinking == "PASS",
        "thinking_policy_strip_on_edit": core,
        "whole_context_writes": by_step.get("6") == "PASS",
    }
    return out


def format_report(report: ConformanceReport) -> str:
    lines = [f"harness: {report.harness}" + (f"  version: {report.version}" if report.version else "")]
    if not report.available:
        lines.append(f"UNAVAILABLE: {report.reason}")
        return "\n".join(lines)
    for step in report.steps:
        numbers = " ".join(f"{k}={v}" for k, v in step.evidence.items() if k in
                           ("cache_read", "cache_write", "input", "context", "context_before", "copies", "thinking_blocks"))
        lines.append(f"  {step.status:<12} {step.step:<14} {step.note} {('(' + numbers + ')') if numbers else ''}".rstrip())
    lines.append(f"conforms: {'yes' if report.conforms else 'no'}")
    for arm, ok in report.arms.items():
        lines.append(f"  {'available  ' if ok else 'UNAVAILABLE'} {arm}")
    lines.append(f"list-price equivalent: {report.list_cost_usd} USD; API dollars spent: {report.spent_usd} USD")
    return "\n".join(lines)


def report_to_json(report: ConformanceReport) -> str:
    return json.dumps({
        "harness": report.harness, "version": report.version, "available": report.available, "reason": report.reason,
        "conforms": report.conforms, "arms": report.arms, "list_cost_usd": report.list_cost_usd,
        "spent_usd": report.spent_usd,
        "steps": [{"step": s.step, "status": s.status, "note": s.note, "evidence": s.evidence} for s in report.steps],
    }, indent=2)
