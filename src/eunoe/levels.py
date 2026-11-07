"""The ladder: what of an event stays at each level, made by code with no model call.

| Level | What stays |
|---|---|
| L0 Verbatim | Everything |
| L1 Trimmed | Structure: head and tail, line counts, error lines, exit code, and whatever L2 would keep |
| L2 Digest | Facts: names, numbers, paths, decisions, outcome |
| L3 Headline | One line: what, and how it came out |
| L4 Gone | Nothing in context; the archive still holds it |

An event moves one level at a time and leaves only after every other option is used. A file read is
reconstructible, because the file is the archive, so it goes straight to path, hash and line count.
Thinking is opaque and has no levels: it is kept or dropped.
"""
from __future__ import annotations

import bisect
import hashlib
import re
from enum import IntEnum


class Level(IntEnum):
    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4


CHARS_PER_TOKEN = 4.0
HEADLINE_CHARS = 160
DIGEST_SHARE = 0.06


def estimate_tokens(text: str) -> int:
    """A declared estimate, and it runs low on number-heavy logs. Real usage comes from the provider."""
    return int(len(text) / CHARS_PER_TOKEN) + 1 if text else 0


_DETAIL_RE = re.compile(
    r"(?P<id>\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|\b\d{8}-\d{6}-[0-9a-f]{4}\b|\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b|#\d{2,6}\b"
    r"|\b[A-Z]{1,6}-[A-Z0-9]+(?:-[A-Z0-9]+)*\b)"
    r"|(?P<date>\b20\d\d-\d\d-\d\d\b)"
    r"|(?P<time>\b\d{1,2}:\d{2}(?::\d{2})?\b)"
    r"|(?P<path>(?<![\w.~-])(?:[A-Za-z]:)?(?:[\w.~-]+[\\/])+[\w.-]+)"
    r"|(?P<file>\b[\w-]+\.(?:py|md|jsonl?|tsx?|js|ps1|sh|toml|ya?ml|txt|csv|html|kt|lua|env|sql|db|log)\b)"
    r"|(?P<slug>\b[a-z0-9]+(?:-[a-z0-9]+){3,}\b)"
    r"|(?P<name>\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b|\b[a-z]+(?:[A-Z][a-z0-9]+)+\b|\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b)"
    r"|(?P<number>\b\d+(?:\.\d+)?%|\b\d[\d,]*\.\d+\b|\b\d{4,}\b|\b\d+(?:\.\d+)?[kKmMgG][bB]?\b)"
    r"|(?P<proper>(?<=[a-z,;:] )[A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})?)"
)
_ERROR_RE = re.compile(r"(?i)\b(?:error|failed|failure|exception|traceback|fatal|refused|exit code)\b")
_DECISION_RE = re.compile(r"(?i)\b(?:decided|decision|chose|chosen|will|plan|next step|agreed|because|selected)\b")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_EXIT_RE = re.compile(r"(?i)\bexit(?:ed)?(?: code)?[:= ]+(-?\d+)")


def find_details(text: str) -> list:
    """`(detail, class, position)` for each identifier, number, path or name in `text`, in order."""
    found = []
    for m in _DETAIL_RE.finditer(text or ""):
        found.append((m.group(0), m.lastgroup, m.start()))
    return found


def _window(text: str) -> int:
    return max(200, len(text) // 8)


def trim(text: str) -> str:
    """L1: head and tail of the text, its line count, error lines and exit code."""
    n = len(text)
    window = _window(text)
    if n <= 2 * window:
        return text
    lines = text.splitlines()
    head, tail = text[:window], text[-window:]
    errors = [ln.strip() for ln in text[window:-window].splitlines() if _ERROR_RE.search(ln)]
    parts = [head.rstrip(), f"[{len(lines)} lines, {n} characters, middle trimmed]"]
    if errors:
        parts.append("\n".join(errors[:20]))
    code = _EXIT_RE.search(text)
    if code and code.group(0) not in head and code.group(0) not in tail:
        parts.append(f"exit code {code.group(1)}")
    # L1 has to hold whatever L2 would, or stepping down a level could bring a fact back. Measured on forks: without this
    # line, a fact in the middle of an output was gone at L1 and still there at L2.
    kept = digest(text[window:-window], max(120, int(DIGEST_SHARE * n)))
    if kept:
        parts.append(f"[details in the trimmed part: {kept}]")
    parts.append(tail.lstrip())
    return "\n".join(parts)


def digest(text: str, budget: int | None = None) -> str:
    """L2: details from the text, those on error lines first, up to 6% of its size."""
    details = find_details(text)
    if not details:
        return headline(text)
    spans = []
    for m in _ERROR_RE.finditer(text):
        start = text.rfind("\n", 0, m.start()) + 1
        end = text.find("\n", m.end())
        spans.append((start, len(text) if end < 0 else end))
    starts = [s for s, _ in spans]

    def on_error(pos: int) -> bool:
        i = bisect.bisect_right(starts, pos) - 1
        return i >= 0 and pos <= spans[i][1]

    if budget is None:
        budget = max(120, int(DIGEST_SHARE * len(text)))
    kept, seen, spent = [], set(), 0
    for detail, _, _ in sorted(details, key=lambda d: (not on_error(d[2]), d[2])):
        if detail in seen:
            continue
        spent += len(detail) + 2
        if spent > budget:
            break
        seen.add(detail)
        kept.append(detail)
    return "; ".join(kept)


def headline(text: str) -> str:
    """L3: the first line of the text, cut to a headline's length."""
    first = (text or "").strip().split("\n", 1)[0].strip()
    return first[:HEADLINE_CHARS]


def first_sentence(text: str) -> str:
    parts = _SENTENCE_RE.split((text or "").strip(), maxsplit=1)
    return parts[0].split("\n", 1)[0][:HEADLINE_CHARS] if parts else ""


def first_last_paragraphs(text: str) -> str:
    """L1 for prose: the first and last paragraphs, and from the middle only the sentences L2 would keep."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if len(paragraphs) <= 2:
        return text
    middle = [s.strip() for p in paragraphs[1:-1] for s in _SENTENCE_RE.split(p) if _DECISION_RE.search(s) or find_details(s)]
    kept = f"\n\n{' '.join(middle)}" if middle else ""
    return f"{paragraphs[0]}\n\n[{len(paragraphs) - 2} paragraphs trimmed]{kept}\n\n{paragraphs[-1]}"


def key_sentences(text: str) -> str:
    """L2 for prose: the sentences that carry a detail or a decision."""
    sentences = _SENTENCE_RE.split((text or "").strip())
    kept = [s.strip() for s in sentences if _DECISION_RE.search(s) or find_details(s)]
    if not kept:
        return first_sentence(text)
    budget = max(120, int(0.25 * len(text)))
    out, spent = [], 0
    for sentence in kept:
        spent += len(sentence) + 1
        if spent > budget:
            break
        out.append(sentence)
    return " ".join(out) if out else first_sentence(text)


def reconstructible_stub(source: str, text: str) -> str:
    """A read of a file: the path, a hash to check staleness on a re-read, and its line count."""
    digest_hex = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]
    return f"{source} sha256:{digest_hex} {len((text or '').splitlines())} lines"


def trim_input(value, limit: int = 200):
    """Long string values in a tool's arguments, cut. The call stays valid."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"…[{len(value)} chars]"
    if isinstance(value, dict):
        return {k: trim_input(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [trim_input(v, limit) for v in value]
    return value
