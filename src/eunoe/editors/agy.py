"""Gemini through Google's Antigravity CLI: an edit reaches the model by editing the conversation store.

A conversation is a SQLite file whose `steps.step_payload` and `gen_metadata.data` columns hold protobuf
blobs. The kit does not parse protobuf. It changes text in place at the same byte length, because a length
prefix would otherwise go wrong. So:

* a codeword swaps for one of the same length;
* padding is removed by overwriting it with spaces. Its bytes stay in the file, and the model reads blanks.

A fork is the store copied under a new conversation id, with its `cascade_id` changed to match. There is
no door for a whole rendered context, so a contracting arm here shortens events in place instead.

Usage in the JSON result is cumulative across `--conversation`, so each turn is a difference. Sign-in has to
be the subscription's: metered keys are scrubbed from the environment, and the suite's report says so.
Runs in plan mode in an empty scratch directory, with the prompt on the command line, the CLI's only
non-experimental input.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid

from eunoe.drivers.base import is_offline, run_cli, scrubbed_env
from eunoe.editors.base import Harness, Turn, Unsupported
from eunoe.interfaces import CallUsage

_COLUMNS = (("steps", "step_payload", "idx"), ("gen_metadata", "data", "idx"))
#: agy takes its prompt on the command line, and Windows caps a command line.
MAX_PROMPT_CHARS = 28_000


class AgyHarness(Harness):
    name = "agy"
    default_model = "gemini-3.8-flash-low"

    def __init__(self, model=None, gate=None, binary: str = "agy", runner=run_cli, environ=None) -> None:
        super().__init__(model, gate)
        self.binary, self.runner = binary, runner
        self.environ = environ if environ is not None else os.environ
        self.home = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity-cli")
        self.scratch = ""
        self.conversation = ""
        self.conversations: list = []
        self.totals = (0, 0, 0)
        self.thinking_seen = 0
        self.active_model = self.model

    def available(self) -> tuple:
        if is_offline(self.environ):
            return False, "offline: EUNOE_OFFLINE is set"
        if not shutil.which(self.binary):
            return False, f"{self.binary} is not on PATH"
        return True, ""

    def version(self) -> str:
        if is_offline(self.environ) or not shutil.which(self.binary):
            return ""
        # `agy --version` can hang past any sensible wait, so the changelog's first line answers instead.
        done = self.runner([self.binary, "changelog"], env=scrubbed_env(self.environ), timeout=25)
        for line in done.stdout.splitlines():
            line = line.strip().lstrip("#").strip()
            if line and line[0].isdigit():
                return line.split()[0].rstrip(":")
        return "unknown"

    def _dir(self) -> str:
        if not self.scratch:
            self.scratch = tempfile.mkdtemp(prefix="eunoe-conform-")
        return self.scratch

    def _run(self, prompt: str, *extra) -> Turn:
        if len(prompt) > MAX_PROMPT_CHARS:
            return Turn(error=f"prompt over the {MAX_PROMPT_CHARS} character limit of a command line")
        argv = [self.binary, "-p", prompt, "--mode", "plan", "--model", self.active_model, "--output-format", "json", *extra]
        done = self.runner(argv, cwd=self._dir(), env=scrubbed_env(self.environ), timeout=240)
        data = None
        for start in (0, done.stdout.find('{"conversation_id"')):
            if start < 0:
                continue
            try:
                data = json.loads(done.stdout[start:])
                break
            except ValueError:
                continue
        if not isinstance(data, dict):
            return Turn(error=done.stderr.strip()[:300] or f"exit {done.returncode}, no JSON result")
        if data.get("status") != "SUCCESS":
            return Turn(error=str(data.get("error") or data.get("status") or "error")[:300],
                        session_id=data.get("conversation_id", ""))
        if data.get("denied_actions"):
            return Turn(error="a tool it tried was denied", session_id=data.get("conversation_id", ""))
        return Turn(str(data.get("response") or ""), self._delta(data.get("usage") or {}), data.get("conversation_id", ""))

    def _delta(self, usage: dict) -> CallUsage:
        total_in = int(usage.get("input_tokens") or 0)
        cached = int(usage.get("cache_read_tokens") or 0)
        total_out = int(usage.get("output_tokens") or 0) + int(usage.get("thinking_tokens") or 0)
        self.thinking_seen = max(self.thinking_seen, int(usage.get("thinking_tokens") or 0))
        prev_in, prev_cached, prev_out = self.totals
        self.totals = (total_in, cached, total_out)
        return CallUsage(max(0, total_in - prev_in - (cached - prev_cached)), max(0, cached - prev_cached), 0, 0,
                         max(0, total_out - prev_out), usage)

    def start(self, prompt: str, thinking: bool = False) -> Turn:
        self.totals = (0, 0, 0)
        self.thinking_seen = 0
        # A tier is part of the model's name here, and `--effort` conflicts with a low tier.
        self.active_model = self.model.replace("-low", "-high") if thinking else self.model
        turn = self._run(prompt)
        self.conversation = turn.session_id
        if self.conversation:
            self.conversations.append(self.conversation)
        return turn

    def resume(self, prompt: str) -> Turn:
        return self._run(prompt, "--conversation", self.conversation)

    def _db(self, conversation: str | None = None) -> str:
        return os.path.join(self.home, "conversations", f"{conversation or self.conversation}.db")

    def replace(self, old: str, new: str) -> int:
        """In place at equal byte length. Shorter text is padded with spaces, which is how a removal works."""
        old_b, new_b = old.encode("utf-8"), new.encode("utf-8")
        if len(new_b) > len(old_b):
            raise Unsupported("the store holds protobuf, so a replacement cannot be longer than the original")
        new_b = new_b.ljust(len(old_b), b" ")
        path = self._db()
        if not os.path.exists(path):
            return 0
        total = 0
        db = sqlite3.connect(path)
        try:
            for table, column, key in _COLUMNS:
                for row_id, value in db.execute(f"select {key}, {column} from {table}").fetchall():
                    if value and old_b in bytes(value):
                        total += bytes(value).count(old_b)
                        db.execute(f"update {table} set {column}=? where {key}=?",
                                   (bytes(value).replace(old_b, new_b), row_id))
            db.commit()
        finally:
            db.close()  # `with` only commits; an open handle blocks removing the file on Windows
        return total

    def fork(self, prompt: str) -> Turn:
        keep = self.conversation, self.totals
        new_id = str(uuid.uuid4())
        shutil.copy(self._db(), self._db(new_id))
        db = sqlite3.connect(self._db(new_id))
        try:
            db.execute("update trajectory_meta set cascade_id=?", (new_id,))
            db.commit()
        finally:
            db.close()
        self.conversations.append(new_id)
        self.conversation = new_id
        turn = self.resume(prompt)
        self.conversation, self.totals = keep
        return turn

    def snapshot(self) -> str:
        path = self._db()
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    def thinking_block_count(self) -> int:
        return 1 if self.thinking_seen else 0

    def cleanup(self) -> None:
        for conversation in self.conversations:
            for path in (self._db(conversation), os.path.join(self.home, "annotations", f"{conversation}.pbtxt"),
                         os.path.join(self.home, "presence", f"{conversation}.lock")):
                try:
                    os.remove(path)
                except OSError:
                    pass
            shutil.rmtree(os.path.join(self.home, "brain", conversation), ignore_errors=True)
        if self.scratch:
            shutil.rmtree(self.scratch, ignore_errors=True)
