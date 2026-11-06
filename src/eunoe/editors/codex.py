"""OpenAI's Codex CLI: an edit reaches the model by editing the rollout file, then `exec resume`.

Two catches. `turn.completed` usage is cumulative across a resume, so each turn is a difference. And a
compacted rollout holds encrypted items that can't be edited at that item.

`codex fork` is interactive, so a fork here is the rollout copied under a new thread id, and the suite
shows `exec resume` accepting it. Whole-context writes keep a real rollout's opening rows, because they
hold the harness's own instructions, and write the rendered context after them.

Runs read-only in an empty scratch directory with user config ignored, so no MCP server or rule loads,
and with every metered key scrubbed, so it bills the CLI's own login.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import tempfile
import uuid

from eunoe.drivers.base import is_offline, run_cli, scrubbed_env
from eunoe.editors.base import Harness, Turn, edit_jsonl
from eunoe.interfaces import CallUsage, RenderedContext, SessionRef
from eunoe.wire import to_openai_input

COMMON = ["--json", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules", "--disable", "plugins"]


class CodexHarness(Harness):
    name = "codex"
    default_model = "gpt-6-luna"
    apply_supported = True

    def __init__(self, model=None, gate=None, binary: str = "codex", runner=run_cli, environ=None) -> None:
        super().__init__(model, gate)
        self.binary, self.runner = binary, runner
        self.environ = environ if environ is not None else os.environ
        self.home = self.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
        self.scratch = ""
        self.thread = ""
        self.threads: list = []
        self.totals = (0, 0, 0)

    def available(self) -> tuple:
        if is_offline(self.environ):
            return False, "offline: EUNOE_OFFLINE is set"
        if not shutil.which(self.binary):
            return False, f"{self.binary} is not on PATH"
        return True, ""

    def version(self) -> str:
        if is_offline(self.environ) or not shutil.which(self.binary):
            return ""
        done = self.runner([self.binary, "--version"], env=scrubbed_env(self.environ), timeout=30)
        return done.stdout.strip() if done.returncode == 0 else ""

    def _dir(self) -> str:
        if not self.scratch:
            self.scratch = tempfile.mkdtemp(prefix="eunoe-conform-")
        return self.scratch

    def _run(self, argv: list, prompt: str) -> Turn:
        done = self.runner(argv, cwd=self._dir(), env=scrubbed_env(self.environ), input_text=prompt, timeout=240)
        text, usage, error = "", None, ""
        thread = self.thread
        for line in done.stdout.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            kind = row.get("type")
            if kind == "thread.started":
                thread = row.get("thread_id") or thread
            elif kind == "item.completed":
                item = row.get("item") or {}
                if item.get("type") == "agent_message":
                    text = item.get("text") or text
            elif kind == "turn.completed":
                usage = row.get("usage") or {}
            elif kind in ("turn.failed", "error"):
                error = str(row.get("message") or row.get("error") or kind)
        if usage is None:
            return Turn(error=error or done.stderr.strip()[:300] or f"exit {done.returncode}", session_id=thread)
        return Turn(text, self._delta(usage), thread)

    def _delta(self, usage: dict) -> CallUsage:
        """Each counter is cumulative across a resume, so this turn is the difference."""
        total_in = int(usage.get("input_tokens") or 0)
        total_cached = int(usage.get("cached_input_tokens") or 0)
        total_out = int(usage.get("output_tokens") or 0)
        prev_in, prev_cached, prev_out = self.totals
        self.totals = (total_in, total_cached, total_out)
        turn_in, cached = total_in - prev_in, total_cached - prev_cached
        return CallUsage(max(0, turn_in - cached), max(0, cached), 0, 0, max(0, total_out - prev_out), usage)

    def start(self, prompt: str, thinking: bool = False) -> Turn:
        self.totals = (0, 0, 0)
        argv = [self.binary, "exec", *COMMON, "--sandbox", "read-only", "--model", self.model]
        if thinking:
            argv += ["-c", 'model_reasoning_effort="high"']
        turn = self._run(argv + ["-"], prompt)
        self.thread = turn.session_id
        if self.thread:
            self.threads.append(self.thread)
        return turn

    def resume(self, prompt: str) -> Turn:
        argv = [self.binary, "exec", "resume", *COMMON, "--model", self.model, self.thread, "-"]
        return self._run(argv, prompt)

    def rollout(self, thread: str | None = None) -> str | None:
        hits = glob.glob(os.path.join(self.home, "sessions", "**", f"*{thread or self.thread}.jsonl"), recursive=True)
        return hits[0] if hits else None

    def replace(self, old: str, new: str) -> int:
        path = self.rollout()
        return edit_jsonl(path, old, new) if path else 0

    def _copy_rollout(self, rows_filter=None) -> str:
        """The current rollout under a new thread id. Returns the new id."""
        source = self.rollout()
        new_id = str(uuid.uuid4())
        with open(source, encoding="utf-8", newline="") as fh:
            lines = fh.readlines()
        if rows_filter:
            lines = rows_filter(lines)
        with open(source.replace(self.thread, new_id), "w", encoding="utf-8", newline="") as fh:
            fh.writelines(line.replace(self.thread, new_id) for line in lines)
        self.threads.append(new_id)
        return new_id

    def fork(self, prompt: str) -> Turn:
        keep = self.thread, self.totals
        self.thread = self._copy_rollout()
        turn = self.resume(prompt)
        self.thread, self.totals = keep
        return turn

    def apply(self, session, rendered: RenderedContext) -> SessionRef:
        def opening_then_rendered(lines: list) -> list:
            kept, ordinal = [], -1
            for line in lines:
                kept.append(line)
                row = json.loads(line)
                ordinal = max(ordinal, int(row.get("ordinal", ordinal)))
                if row.get("type") == "turn_context":
                    break
            return kept + [json.dumps(row, ensure_ascii=False) + "\n" for row in _rollout_rows(rendered, ordinal + 1)]

        self.thread = self._copy_rollout(opening_then_rendered)
        self.totals = (0, 0, 0)
        return SessionRef(self.name, self.thread)

    def snapshot(self) -> str:
        path = self.rollout()
        if not path:
            return ""
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    def thinking_block_count(self) -> int:
        path = self.rollout()
        if not path:
            return 0
        with open(path, encoding="utf-8") as fh:
            return sum(1 for line in fh if '"type":"reasoning"' in line.replace(" ", ""))

    def cleanup(self) -> None:
        for thread in self.threads:
            path = self.rollout(thread)
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
        if self.scratch:
            shutil.rmtree(self.scratch, ignore_errors=True)


def _rollout_rows(rendered: RenderedContext, first_ordinal: int = 0) -> list:
    """Rendered context as rollout rows. Resume refuses a final row with no ordinal."""
    stamp = "2030-01-01T00:00:00.000Z"
    return [{"timestamp": stamp, "ordinal": first_ordinal + i, "type": "response_item", "payload": item}
            for i, item in enumerate(to_openai_input(rendered)["input"])]
