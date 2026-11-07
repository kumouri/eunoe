"""The Claude API's server-side edits as a suite of their own: `context_management.edits`.

The client's history stays full and the server decides what the model reads, so the steps are not the
codeword steps. A tool exchange holds a codeword, the server clears it, and the check is that the model
can no longer read it while the usage says the context shrank. Compaction is the second door. A paid
harness, so it needs a funded key and `--allow-spend`, and it is tested offline against recorded shapes.
"""
from __future__ import annotations

from eunoe.editors.stateless import ANTHROPIC_URL, BETA_COMPACTION, BETA_CONTEXT_EDITING, ClaudeApiHarness
from eunoe.editors.claude_cli import claude_usage
from eunoe.interfaces import ConformanceReport, StepResult


def tool_history(codeword: str, padding: str) -> list:
    return [
        {"role": "user", "content": [{"type": "text", "text": "Look up the fleet status."}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_eunoe_a", "name": "probe", "input": {"t": "fleet"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_eunoe_a",
                                      "content": f"codeword {codeword}. {padding}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        {"role": "user", "content": [{"type": "text", "text": "Reply with only the word READY."}]},
    ]


PROBE_TOOL = [{"name": "probe", "description": "Look something up.",
               "input_schema": {"type": "object", "properties": {"t": {"type": "string"}}, "required": ["t"]}}]


class ClaudeServerHarness(ClaudeApiHarness):
    name = "claude-api-server"
    apply_supported = False

    def run_suite(self, padding: str, codeword: str = "EGRET-4417") -> ConformanceReport:
        report = ConformanceReport(self.name, version=self.model)
        ok, reason = self.available()
        if not ok:
            report.available, report.reason = False, reason
            return report
        steps = []
        history = tool_history(codeword, padding)
        status, full = self._post(history)
        steps.append(self._result("1 establish", status, full, lambda d: "READY" in _text(d)))
        edit = {"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": 1},
                "keep": {"type": "tool_uses", "value": 0}}
        ask = history[:-1] + [{"role": "user", "content": [{"type": "text", "text":
                               "What codeword did the tool return? Reply with only the codeword, or UNKNOWN."}]}]
        status, cleared = self._post(ask, {"edits": [edit]}, BETA_CONTEXT_EDITING)
        steps.append(self._result("2 clear_tool_uses", status, cleared, lambda d: codeword not in _text(d)))
        if status == 200:
            before = claude_usage(full.get("usage") or {}).context
            after = claude_usage(cleared.get("usage") or {}).context
            steps.append(StepResult("3 shrink", "PASS" if after < before else "FAIL", "server edit shrank the context",
                                    {"context_before": before, "context": after}))
        status, compacted = self._post(ask, {"edits": [{"type": "compact_20260112",
                                                        "trigger": {"type": "input_tokens", "value": 1}}]}, BETA_COMPACTION)
        steps.append(self._result("4 compaction", status, compacted, lambda d: bool(d.get("content"))))
        report.steps = steps
        report.spent_usd = round(self.gate.spent_usd, 6)
        return report

    def _post(self, messages: list, context_management=None, *betas) -> tuple:
        self.gate.check(self.name)
        body = {"model": self.model, "max_tokens": 512, "messages": messages, "tools": PROBE_TOOL}
        if context_management:
            body["context_management"] = context_management
        status, data = self.transport(ANTHROPIC_URL, self.headers(*betas), body)
        if status == 200:
            usage = claude_usage(data.get("usage") or {})
            self.gate.record(self.name, self.model, usage, self._price(usage))
        return status, data

    def _result(self, step: str, status: int, data: dict, check) -> StepResult:
        if status != 200:
            return StepResult(step, "FAIL", f"{status}: {str(data.get('error'))[:200]}")
        return StepResult(step, "PASS" if check(data) else "FAIL", "", {"context": claude_usage(data.get("usage") or {}).context})


def _text(data: dict) -> str:
    return "".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")

