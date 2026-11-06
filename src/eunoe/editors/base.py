"""A harness, as the conformance suite sees it: start, resume, edit, fork."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from eunoe.interfaces import CallUsage, RenderedContext, SessionRef


class Unsupported(Exception):
    """This harness has no door for that. The step is marked, never approximated."""


@dataclass
class Turn:
    text: str = ""
    usage: CallUsage = field(default_factory=CallUsage)
    session_id: str = ""
    thinking_blocks: int = 0
    error: str = ""
    provider_cost_usd: float | None = None


class Harness:
    """Subclass this to run the suite on your own harness: `--harness package.module:ClassName`."""

    name = "harness"
    spends = False
    default_model = ""
    thinking_supported = True
    apply_supported = False

    def __init__(self, model: str | None = None, gate=None) -> None:
        self.model = model or self.default_model
        self.gate = gate

    def version(self) -> str:
        return ""

    def available(self) -> tuple:
        """`(ok, reason)`. The reason must never carry a path, an id or a secret."""
        return True, ""

    def start(self, prompt: str, thinking: bool = False) -> Turn:
        raise NotImplementedError

    def resume(self, prompt: str) -> Turn:
        raise NotImplementedError

    def replace(self, old: str, new: str) -> int:
        """Change every copy of `old` in the session. Returns how many it changed."""
        raise NotImplementedError

    def fork(self, prompt: str) -> Turn:
        """Resume into a new session, leaving the original untouched."""
        raise Unsupported(f"{self.name} has no fork")

    def apply(self, session: SessionRef | None, rendered: RenderedContext) -> SessionRef:
        """Write a whole rendered context into a new session, which the next `resume` continues."""
        raise Unsupported(f"{self.name} cannot take a whole rendered context")

    def conformance(self):
        from eunoe.editors.conformance import Suite

        return Suite(self).run()

    def snapshot(self) -> str:
        """A hash of the session as it stands, to show a fork left it alone."""
        return ""

    def thinking_block_count(self) -> int:
        return 0

    def cleanup(self) -> None:
        """Remove what this run created, and only that."""


def replace_in_json(value, old: str, new: str) -> tuple:
    """Every string in a parsed JSON value, keys included, with `old` replaced. Returns `(value, count)`."""
    if isinstance(value, str):
        return value.replace(old, new), value.count(old)
    if isinstance(value, list):
        out, total = [], 0
        for item in value:
            item, n = replace_in_json(item, old, new)
            out.append(item)
            total += n
        return out, total
    if isinstance(value, dict):
        out, total = {}, 0
        for key, item in value.items():
            item, n = replace_in_json(item, old, new)
            key, m = replace_in_json(key, old, new)
            out[key] = item
            total += n + m
        return out, total
    return value, 0


def edit_jsonl(path: str, old: str, new: str) -> int:
    """`old` to `new` through every row of a JSONL file, preserving the rows it does not touch byte for byte."""
    total = 0
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for line in fh:
            if old not in line and json.dumps(old)[1:-1] not in line:
                rows.append(line)
                continue
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            row, n = replace_in_json(json.loads(line), old, new)
            total += n
            rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + ending)
    tmp = path + ".eunoe-tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(rows)
    os.replace(tmp, path)
    return total
