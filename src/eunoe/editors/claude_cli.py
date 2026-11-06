"""Claude Code's CLI: an edit reaches the model by editing the session file, then resuming.

The format is undocumented, so this module assumes nothing it can't check: the suite runs per CLI version
and the version goes in every manifest. Three things to know:

* A running process holds the history in memory, so an edit lands at the next spawn.
* The file holds several copies of a message's text, and every copy has to change.
* Whether editing history under a thinking block is refused depends on the account, which is why the
  suite tests it per login.

Everything runs in an empty scratch directory with no tools, no MCP servers and no user-level settings,
and with every metered key scrubbed, so a run bills the CLI's own login. The session files it creates are
removed afterwards, and only those.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid

from eunoe.drivers.base import is_offline, run_cli, scrubbed_env
from eunoe.editors.base import Harness, Turn, Unsupported, edit_jsonl
from eunoe.interfaces import CallUsage, RenderedContext, SessionRef
from eunoe.wire import to_anthropic


def claude_usage(usage: dict) -> CallUsage:
    created = usage.get("cache_creation") or {}
    w5, w1 = int(created.get("ephemeral_5m_input_tokens") or 0), int(created.get("ephemeral_1h_input_tokens") or 0)
    total = int(usage.get("cache_creation_input_tokens") or 0)
    if not w5 and not w1:
        w5 = total
    return CallUsage(int(usage.get("input_tokens") or 0), int(usage.get("cache_read_input_tokens") or 0), w5, w1,
                     int(usage.get("output_tokens") or 0), usage)


class ClaudeCliHarness(Harness):
    name = "claude-cli"
    default_model = "claude-haiku-5-5"
    apply_supported = True

    def __init__(self, model=None, gate=None, binary: str = "claude", runner=run_cli, config_dir: str | None = None,
                 environ=None) -> None:
        super().__init__(model, gate)
        self.binary, self.runner = binary, runner
        self.environ = environ if environ is not None else os.environ
        self.config_dir = config_dir or self.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
        self.scratch = ""
        self.sessions: list = []
        self.current = ""
        self.extra_args: list = []
        self._version: str | None = None

    def available(self) -> tuple:
        if is_offline(self.environ):
            return False, "offline: EUNOE_OFFLINE is set"
        if not shutil.which(self.binary):
            return False, f"{self.binary} is not on PATH"
        done = self.runner([self.binary, "--version"], env=self._env(), timeout=30)
        if done.returncode != 0:
            return False, "the CLI did not report a version"
        return True, ""

    def version(self) -> str:
        if self._version is None:
            if is_offline(self.environ) or not shutil.which(self.binary):
                return ""
            done = self.runner([self.binary, "--version"], env=self._env(), timeout=30)
            self._version = done.stdout.strip().split("\n")[0] if done.returncode == 0 else ""
        return self._version

    def _env(self) -> dict:
        return scrubbed_env(self.environ)

    def _dir(self) -> str:
        if not self.scratch:
            self.scratch = tempfile.mkdtemp(prefix="eunoe-conform-")
        return self.scratch

    def _argv(self, *extra) -> list:
        return [self.binary, "-p", "--output-format", "json", "--model", self.model, "--tools", "",
                "--strict-mcp-config", "--setting-sources", "project,local", *self.extra_args, *extra]

    def _run(self, prompt: str, *extra) -> Turn:
        done = self.runner(self._argv(*extra), cwd=self._dir(), env=self._env(), input_text=prompt, timeout=240)
        try:
            data = json.loads(done.stdout)
        except ValueError:
            return Turn(error=done.stderr.strip()[:300] or done.stdout.strip()[:300] or f"exit {done.returncode}")
        if data.get("is_error"):
            return Turn(error=str(data.get("result") or data.get("subtype") or "error")[:300],
                        session_id=data.get("session_id", ""))
        return Turn(str(data.get("result") or ""), claude_usage(data.get("usage") or {}), data.get("session_id", ""),
                    provider_cost_usd=data.get("total_cost_usd"))

    def start(self, prompt: str, thinking: bool = False) -> Turn:
        session_id = str(uuid.uuid4())
        extra = ["--session-id", session_id]
        if thinking:
            extra += ["--effort", "high"]
        self.current = session_id
        self.sessions.append(session_id)
        return self._run(prompt, *extra)

    def resume(self, prompt: str) -> Turn:
        return self._run(prompt, "--resume", self.current)

    def session_path(self, session_id: str | None = None) -> str | None:
        hits = glob.glob(os.path.join(self.config_dir, "projects", "*", f"{session_id or self.current}.jsonl"))
        return hits[0] if hits else None

    def replace(self, old: str, new: str) -> int:
        path = self.session_path()
        return edit_jsonl(path, old, new) if path else 0

    def fork(self, prompt: str) -> Turn:
        turn = self._run(prompt, "--resume", self.current, "--fork-session")
        if turn.session_id and turn.session_id != self.current:
            self.sessions.append(turn.session_id)
        elif not turn.error:
            turn.error = "--fork-session kept the same session id"
        return turn

    def snapshot(self) -> str:
        path = self.session_path()
        if not path:
            return ""
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    def thinking_block_count(self) -> int:
        path = self.session_path()
        if not path:
            return 0
        count = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if '"thinking"' not in line:
                    continue
                row = json.loads(line)
                if row.get("type") != "assistant":
                    continue
                content = (row.get("message") or {}).get("content") or []
                count += sum(1 for b in content if isinstance(b, dict) and b.get("type") == "thinking")
        return count

    def apply(self, session, rendered: RenderedContext) -> SessionRef:
        session_id = str(uuid.uuid4())
        slug = _project_slug(self._dir())
        folder = os.path.join(self.config_dir, "projects", slug)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{session_id}.jsonl")
        rows = session_rows(rendered, session_id, self._dir(), self.version().split(" ")[0] or "0", self.model)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.sessions.append(session_id)
        self.current = session_id
        return SessionRef(self.name, session_id)

    def append_rows(self, rendered: RenderedContext) -> int:
        """New events on the end of the current session, chained to its newest row. Returns how many rows."""
        path = self.session_path()
        parent = None
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("uuid") and row.get("type") in ("user", "assistant", "attachment", "system"):
                    parent = row["uuid"]
        rows = session_rows(rendered, self.current, self._dir(), self.version().split(" ")[0] or "0", self.model,
                            parent=parent, stamp=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                            salt=str(os.path.getsize(path)), continuation=True)
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        return len(rows)

    def cleanup(self) -> None:
        for session_id in self.sessions:
            path = self.session_path(session_id)
            if path:
                folder = os.path.dirname(path)
                _remove_quietly(path)
                sidecar = os.path.join(folder, session_id)
                if os.path.isdir(sidecar):
                    shutil.rmtree(sidecar, ignore_errors=True)
        if self.scratch:
            # The project folder is named for our own scratch directory, so all of it is ours.
            slug = _project_slug(self.scratch)
            if "eunoe-conform-" in slug:
                shutil.rmtree(os.path.join(self.config_dir, "projects", slug), ignore_errors=True)
            shutil.rmtree(self.scratch, ignore_errors=True)


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _project_slug(cwd: str) -> str:
    """How the CLI names a project folder. Checked by the suite: step 6 fails if this ever drifts."""
    return "".join(c if c.isalnum() else "-" for c in os.path.realpath(cwd))


def session_rows(rendered: RenderedContext, session_id: str, cwd: str, version: str, model: str, *, parent=None,
                 stamp: str = "2030-01-01T00:00:00.000Z", salt: str = "", continuation: bool = False) -> list:
    """A rendered context as session rows: a chain of user and assistant messages."""
    wire = to_anthropic(rendered, thinking_replay=False)
    messages = wire["messages"]
    if continuation and rendered.blocks and rendered.blocks[0].role == "assistant":
        messages = messages[1:]  # to_anthropic opens a new conversation with a user turn; this one is already open
    rows = []
    for i, message in enumerate(messages):
        uid = str(uuid.UUID(hashlib.md5(f"{session_id}:{salt}:{i}".encode()).hexdigest()))
        row = {"parentUuid": parent, "isSidechain": False, "type": message["role"], "uuid": uid, "timestamp": stamp,
               "userType": "external", "entrypoint": "sdk-cli", "cwd": cwd, "sessionId": session_id, "version": version}
        if message["role"] == "user":
            row["message"] = {"role": "user", "content": message["content"]}
        else:
            row["message"] = {"model": model, "id": f"msg_eunoe_{i:06d}", "type": "message", "role": "assistant",
                              "content": message["content"], "stop_reason": "end_turn", "stop_sequence": None,
                              "usage": {"input_tokens": 0, "output_tokens": 0}}
        rows.append(row)
        parent = uid
    return rows


__all__ = ["ClaudeCliHarness", "claude_usage", "session_rows", "Unsupported"]
