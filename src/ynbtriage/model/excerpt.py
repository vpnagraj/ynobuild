"""Reduce a log tail to the part the model reads.

Two rules are available. Every model (baselines included) in a run sees the SAME
excerpt, so comparisons measure the model, not the input. The rule a model was
trained with is recorded in its model.pt and metrics.json, and used again at
prediction time.

    tail          The last 30 lines. The default. A failed build stops right
                  after it fails, so the end of the log usually holds the real
                  error and its summary messages.
    error_window  The original rule: the 5 lines before and 15 after the first
                  line matching ERROR_PATTERNS, plus the last 30 lines. Models
                  trained before the rule was recorded used this one.

The experiments in experiments/ showed the error window adds nothing over the
tail alone (docs/experiments.md, experiments/02_tail_only.ipynb), which is why
`tail` is the default. `error_window` is kept so those experiments and older
models still work.

ERROR_PATTERNS mirrors web/app.py, which uses the list to highlight the failing
region for people reading a log (the web image does not install ynbtriage, so
the list is duplicated). If you change one, change the other.
"""
from __future__ import annotations

import re

EXCERPT_RULES = ("tail", "error_window")
DEFAULT_EXCERPT = "tail"
LEGACY_EXCERPT = "error_window"   # what a model.pt without a recorded rule was trained with

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


def check_rule(rule: str) -> str:
    if rule not in EXCERPT_RULES:
        raise ValueError(f"unknown excerpt rule {rule!r} ({' | '.join(EXCERPT_RULES)})")
    return rule


def log_excerpt(log: str, rule: str = DEFAULT_EXCERPT, tail: int = 30, before: int = 5,
                after: int = 15, max_chars: int = 4000) -> str:
    """Apply an excerpt rule (see the module docstring).

    The result is normalized and capped at `max_chars`, keeping the end, where
    the cause usually is. `before` and `after` only apply to `error_window`.
    """
    check_rule(rule)
    lines = (log or "").splitlines()
    if not lines:
        return ""
    keep: set[int] = set(range(max(0, len(lines) - tail), len(lines)))
    if rule == "error_window":
        for i, ln in enumerate(lines):
            if _ERR_RE.search(ln):
                keep.update(range(max(0, i - before), min(len(lines), i + after + 1)))
                break
    text = "\n".join(lines[i] for i in sorted(keep))
    return normalize(text)[-max_chars:]


def error_excerpt(log: str, **kwargs) -> str:
    """The original rule under its original name (same as rule="error_window")."""
    return log_excerpt(log, rule="error_window", **kwargs)
