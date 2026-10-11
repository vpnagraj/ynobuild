"""Reduce a log tail to the part that carries the failure signal.

Logs are long and the cause is usually near the first error line and at the very
end. Sentence encoders truncate at 256-512 tokens, so feeding the raw tail would
often cut off exactly the part that matters. Every model (baselines included)
sees the SAME excerpt, so comparisons measure the model, not the input.

ERROR_PATTERNS mirrors web/app.py (the web image does not install ynbtriage, so
the list is duplicated). If you change one, change the other.
"""
from __future__ import annotations

import re

ERROR_PATTERNS = [
    r"error:", r"\bE:\s", r"fatal error", r"fatal:", r"\bKilled\b", r"No space left",
    r"not found", r"parse error", r"unknown instruction", r"Traceback",
    r"CondaToSNonInteractiveError", r"Terms of Service", r"\b40[13]\b",
    r"exec format error", r"GLIBC", r"Connection timed out", r"curl: \(\d+\)",
    r"\*\*\* .*Error", r"returned error", r"was not found",
]
_ERR_RE = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)

# Light normalization: tokens that vary per build but carry no failure signal.
_NORMALIZERS = [
    (re.compile(r"\b(?:sha256:)?[0-9a-f]{12,}\b", re.IGNORECASE), "<HEX>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b"), "<TS>"),
    (re.compile(r"\b\d+(?:\.\d+)?\s?(?:kB|KB|MB|GB|B)/?s?\b"), "<SIZE>"),
]


def normalize(text: str) -> str:
    for pat, repl in _NORMALIZERS:
        text = pat.sub(repl, text)
    return text


def error_excerpt(log: str, before: int = 5, after: int = 15, tail: int = 30,
                  max_chars: int = 4000) -> str:
    """Lines around the first error match, plus the last `tail` lines.

    Overlapping windows are merged, order is preserved, and the result is
    normalized and capped at `max_chars` (keeping the end, where the cause
    usually is).
    """
    lines = (log or "").splitlines()
    if not lines:
        return ""
    keep: set[int] = set(range(max(0, len(lines) - tail), len(lines)))
    for i, ln in enumerate(lines):
        if _ERR_RE.search(ln):
            keep.update(range(max(0, i - before), min(len(lines), i + after + 1)))
            break
    text = "\n".join(lines[i] for i in sorted(keep))
    text = normalize(text)
    return text[-max_chars:]
