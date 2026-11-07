"""What every driver shares: the spend gate, a scrubbed environment and injectable doors out.

CI can never spend. A subscription harness runs with every metered key scrubbed from its environment, so
a stray key can't switch its billing. A paid call needs an explicit `--allow-spend` and an explicitly
named key variable, and each one is written to a ledger with its list price.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

#: Any of these in a child's environment could turn a subscription run into a metered one.
_METERED_ENV_RE = re.compile(r"^(?:ANTHROPIC_|CLAUDE_API_|OPENAI_|GEMINI_|GOOGLE_API_|GOOGLE_GENAI_|VERTEX_|"
                             r"GOOGLE_CLOUD_|AWS_BEARER_TOKEN_BEDROCK)")
#: A harness started from inside another agent's session must not think it is nested.
_NESTING_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT")


class OfflineError(RuntimeError):
    """Something tried to leave the machine under `EUNOE_OFFLINE=1`."""


class SpendRefused(RuntimeError):
    """A paid call without permission, or past the cap."""


def is_offline(environ=None) -> bool:
    return (environ if environ is not None else os.environ).get("EUNOE_OFFLINE", "") not in ("", "0")


def scrubbed_env(environ=None) -> dict:
    env = dict(environ if environ is not None else os.environ)
    for name in list(env):
        if _METERED_ENV_RE.match(name) or name in _NESTING_ENV:
            del env[name]
    return env


@dataclass
class CliResult:
    returncode: int
    stdout: str
    stderr: str


def run_cli(argv: list, *, cwd: str | None = None, env: dict | None = None, input_text: str | None = None,
            timeout: float = 180.0) -> CliResult:
    """A subprocess, UTF-8, with a timeout. A hang is a failed step, never a hung suite."""
    if is_offline():
        raise OfflineError(f"{argv[0]}: EUNOE_OFFLINE is set")
    try:
        done = subprocess.run(argv, cwd=cwd, env=env, input=input_text, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return CliResult(124, "", f"timed out after {timeout:.0f} s")
    except OSError as exc:
        return CliResult(127, "", f"could not start: {type(exc).__name__}")
    return CliResult(done.returncode, done.stdout or "", done.stderr or "")


def http_post(url: str, headers: dict, body: dict, timeout: float = 300.0) -> tuple:
    """`(status, json_body)`. An error status comes back as data: a 400 is what a probe wants to see."""
    if is_offline():
        raise OfflineError(f"{url}: EUNOE_OFFLINE is set")
    request = urllib.request.Request(url, json.dumps(body).encode("utf-8"),
                                     {"content-type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"error": {"message": raw[:500]}}


@dataclass
class SpendGate:
    """Every metered call asks here first. It records dollars at list and refuses past the cap."""

    allow: bool = False
    cap_usd: float = 5.0
    ledger_path: str = ""
    spent_usd: float = 0.0
    rows: list = field(default_factory=list)

    def check(self, what: str) -> None:
        if not self.allow:
            raise SpendRefused(f"{what} spends API credit: pass --allow-spend once its funding is confirmed")
        if self.spent_usd >= self.cap_usd:
            raise SpendRefused(f"{what}: the spend cap of {self.cap_usd} USD is used up")

    def record(self, what: str, model: str, usage, usd: float) -> None:
        self.spent_usd += usd
        row = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "what": what, "model": model,
               "input": usage.input, "cache_read": usage.cache_read, "cache_write": usage.cache_write,
               "output": usage.output, "list_usd": round(usd, 6)}
        self.rows.append(row)
        if self.ledger_path:
            with open(self.ledger_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
