"""Calls out of a CLI session file: deduped by `message.id`, with no text kept."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Call:
    id: str
    ts: float
    input: int = 0
    read: int = 0
    write: int = 0
    w5: int = 0
    out: int = 0
    model: str = ""

    @property
    def ctx(self) -> int:
        return self.input + self.read + self.write

    def update(self, usage: dict) -> None:
        self.input = int(usage.get("input_tokens") or 0)
        self.read = int(usage.get("cache_read_input_tokens") or 0)
        self.write = int(usage.get("cache_creation_input_tokens") or 0)
        self.out = int(usage.get("output_tokens") or 0)
        created = usage.get("cache_creation") or {}
        self.w5 = int(created.get("ephemeral_5m_input_tokens") or 0)


def parse_ts(value) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def read_usage(path: str) -> dict:
    """Usage only. Parses just the rows it needs, line by line, and holds no message text."""
    calls, by_id = [], {}
    rows = boundaries = 0
    with open(path, "rb") as fh:
        for line in fh:
            if b'"compact_boundary"' in line:
                try:
                    boundaries += json.loads(line).get("subtype") == "compact_boundary"
                except ValueError:
                    pass
                continue
            if b'"type":"assistant"' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("isSidechain") or row.get("type") != "assistant":
                continue
            message = row.get("message") or {}
            usage = message.get("usage") or {}
            rows += 1
            message_id = message.get("id")
            if not message_id:
                continue
            if message_id in by_id:
                if usage:
                    by_id[message_id].update(usage)
                continue
            call = Call(message_id, parse_ts(row.get("timestamp")) or 0.0, model=str(message.get("model") or ""))
            call.update(usage)
            by_id[message_id] = call
            calls.append(call)
    return {"calls": [c for c in calls if c.ctx > 0 and c.ts], "rows": rows, "boundaries": boundaries}
