"""Load a CSV of build failures into the `builds` table.

Typical input is your socr8s build-results export joined to a logs file, so a row
carries: an identifier, tool name, repo URL, the Dockerfile *path* within the
repo, the build context, a build date, and a (usually already tail-truncated) log.

Logs from the capture pipeline commonly arrive with newlines escaped as literal
``\\n`` (backslash-n) and may carry an upstream truncation marker such as
``[... truncated 27026/28026 lines, showing last 1000 ...]``. This module
un-escapes the newlines so the text renders and searches correctly, recovers the
true (pre-truncation) line count from that marker or a line-count column, and
applies a byte cap on pathological giants (keeping the tail, where the error is).

Column names vary, so each logical field is resolved against a list of common
header aliases (case-insensitive). Only `tool_name` and `log_tail` are required;
everything else is optional. Adjust DEFAULT_MAP if your CSV uses other headers.

If a row carries a predefined leaf-level label (e.g. a `label` column with a
value like `auth_required`), it is recorded as a `prefilled` annotation with
source `import` — a suggestion the annotator confirms in the UI. Prefilled
labels do NOT count as confirmed ground truth and never feed splits/gold until
confirmed. Unknown leaf values are reported and ignored (never fatal); existing
human or confirmed annotations are never overwritten; re-importing the same
label is a no-op.

If no identifier column is present (or its values aren't unique), a stable
`external_id` is synthesized from repo_url + dockerfile_path + built_at + tool so
that re-ingesting the same CSV updates rows in place rather than duplicating them.

Because the Dockerfile path is already known here, it is stored on the build and
the fetch step will pull that exact path instead of guessing conventional ones.
Alternatively, a `dockerfile_content` column may carry the Dockerfile body inline;
when present it is stored directly, marked `provided`, and `fetch` skips it.
"""
from __future__ import annotations

import hashlib
import re
import sys

import pandas as pd

from .config import LOG_MAX_BYTES
from .db import ensure_schema, get_conn
from .repository import add_annotation
from .taxonomy import seed_taxonomy

# annotations created from a predefined label column carry this source, so they
# are distinguishable from human work and can be applied idempotently. They land
# in a `prefilled` state (a suggestion), NOT `confirmed`: the annotator reviews
# and confirms them in the UI, and they never feed splits or the gold set until
# confirmed.
IMPORT_SOURCE = "import"
IMPORT_STATUS = "prefilled"

# logical field -> acceptable CSV header aliases (first match wins)
DEFAULT_MAP: dict[str, list[str]] = {
    "external_id":     ["id", "build_id", "external_id", "uuid", "run_id", "k8s_job_name", "job_name"],
    "tool_name":       ["tool", "tool_name", "name", "package", "recipe"],
    "tool_version":    ["version", "tool_version", "tag", "image_tag"],
    "image_ref":       ["image", "image_ref", "image_name", "container", "result_repo"],
    "source_repo_url": ["repo", "repo_url", "source_repo_url", "url", "git_url", "homepage", "source"],
    "dockerfile_path": ["dockerfile", "dockerfile_path", "df_path", "containerfile"],
    "dockerfile_content": ["dockerfile_content", "dockerfile_text", "dockerfile_body", "containerfile_content"],
    "build_context":   ["context", "build_context", "docker_context"],
    "log_tail":        ["log", "log_tail", "build_log", "truncated_log", "logs", "output", "log_excerpt"],
    "log_lines_src":   ["log_lines", "log_line_count", "n_lines", "line_count"],
    "label_leaf":      ["label", "leaf_id", "leaf", "predefined_label", "predicted_label", "gold_label", "class_leaf"],
    "build_status":    ["status", "build_status", "outcome", "result", "result_status"],
    "built_at":        ["built_at", "created_at", "timestamp", "date", "build_date", "time", "started_at"],
    "source_batch":    ["batch", "source_batch", "run", "socr8s_run", "build_type"],
}

# Upstream truncation banner, e.g. "[... truncated 27026/28026 lines, showing last 1000 ...]"
_TRUNC_MARKER = re.compile(r"truncated\s+\d+\s*/\s*(\d+)\s+lines", re.IGNORECASE)


def normalize_log(raw: str | None, source_line_count: int | None = None):
    """Un-escape newlines, recover the true line count, and cap pathological size.

    Returns (text, full_line_count, truncated_flag).

    * Literal ``\\r\\n`` / ``\\n`` / ``\\r`` / ``\\t`` (backslash-escaped, as the
      capture pipeline emits them) become real characters.
    * The full (pre-truncation) line count is taken from the truncation marker if
      present, else from a line-count column, else counted from the text.
    * If the text still exceeds LOG_MAX_BYTES, keep the last LOG_MAX_BYTES bytes
      (the failing tail) and mark it truncated.
    """
    if raw is None:
        return None, None, 0
    text = (
        raw.replace("\\r\\n", "\n")
           .replace("\\n", "\n")
           .replace("\\r", "\n")
           .replace("\\t", "\t")
    )

    truncated = 0
    full_lines = None
    m = _TRUNC_MARKER.search(text)
    if m:
        full_lines = int(m.group(1))
        truncated = 1
    elif source_line_count:
        full_lines = source_line_count
    if full_lines is None:
        full_lines = text.count("\n") + 1

    if LOG_MAX_BYTES and len(text) > LOG_MAX_BYTES:
        text = text[-LOG_MAX_BYTES:]
        truncated = 1

    # if a line-count column disagrees upward with what we found, trust the larger
    if source_line_count and source_line_count > full_lines:
        full_lines = source_line_count
        truncated = 1

    return text, full_lines, truncated



def resolve_columns(df: pd.DataFrame, mapping=DEFAULT_MAP) -> dict[str, str]:
    lower = {c.lower(): c for c in df.columns}
    resolved = {}
    for field, candidates in mapping.items():
        for cand in candidates:
            if cand.lower() in lower:
                resolved[field] = lower[cand.lower()]
                break
    return resolved


def _s(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s or None


def forge_of(url: str | None) -> str | None:
    if not url:
        return None
    u = url.lower()
    if "github.com" in u:
        return "github"
    if "gitlab" in u:
        return "gitlab"
    if "bitbucket" in u:
        return "bitbucket"
    return "other"


def _synth_id(vals: dict) -> str:
    """Deterministic id from the fields that identify a build, so re-ingest upserts."""
    key = "|".join(
        (vals.get(f) or "")
        for f in ("source_repo_url", "dockerfile_path", "built_at", "tool_name")
    )
    return "ynb-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def load_csv(csv_path: str, db_path=None, mapping=None):
    df = pd.read_csv(csv_path)
    resolved = resolve_columns(df, mapping or DEFAULT_MAP)

    missing = [f for f in ("tool_name", "log_tail") if f not in resolved]
    if missing:
        raise SystemExit(
            f"Could not find required column(s) {missing}.\n"
            f"CSV columns present: {list(df.columns)}\n"
            f"Edit DEFAULT_MAP in ynbtriage/ingest.py to add your header aliases."
        )

    # If an id column was found but its values aren't unique across the file,
    # fall back to synthesizing ids so upsert keys stay unique.
    id_col = resolved.get("external_id")
    id_usable = False
    if id_col is not None:
        col = df[id_col].map(_s)
        nonnull = col.dropna()
        id_usable = len(nonnull) == len(df) and nonnull.is_unique

    inserted = updated = skipped = labeled = 0
    unknown_labels: dict[str, int] = {}
    with get_conn(db_path) as conn:
        ensure_schema(conn)
        seed_taxonomy(conn)
        valid_leaves = {r["leaf_id"] for r in conn.execute("SELECT leaf_id FROM taxonomy_leaf")}
        for _, row in df.iterrows():
            vals = {f: _s(row[col]) for f, col in resolved.items()}
            tool_name = vals.get("tool_name")
            raw_log = vals.get("log_tail")
            if not tool_name or not raw_log:
                skipped += 1
                continue

            src_lines = None
            if vals.get("log_lines_src"):
                try:
                    src_lines = int(float(vals["log_lines_src"]))
                except (TypeError, ValueError):
                    src_lines = None
            log_tail, line_count, truncated = normalize_log(raw_log, src_lines)

            repo = vals.get("source_repo_url")
            forge = forge_of(repo)
            df_path = vals.get("dockerfile_path")
            df_content = vals.get("dockerfile_content")
            ext = vals.get("external_id") if id_usable else _synth_id(vals)

            # If the Dockerfile body is supplied inline, store it and mark it
            # 'provided' so `fetch` skips it. Otherwise queue for fetch when we can
            # reach the repo (fetch uses the exact dockerfile_path if we have one).
            if df_content:
                fetch_status = "provided"
            elif repo and forge in ("github", "gitlab"):
                fetch_status = "pending"
            else:
                fetch_status = None

            existing = conn.execute(
                "SELECT build_id FROM builds WHERE external_id=?", (ext,)
            ).fetchone() if ext else None

            if existing:
                bid = existing["build_id"]
                conn.execute(
                    """UPDATE builds SET
                         tool_name=?, tool_version=?, image_ref=?, source_repo_url=?, source_forge=?,
                         dockerfile_path=COALESCE(?, dockerfile_path), build_context=?,
                         dockerfile_content=COALESCE(?, dockerfile_content),
                         dockerfile_fetch_status=COALESCE(?, dockerfile_fetch_status),
                         log_tail=?, log_line_count=?, log_truncated=?,
                         build_status=COALESCE(?, build_status), built_at=?, source_batch=?
                       WHERE build_id=?""",
                    (
                        tool_name, vals.get("tool_version"), vals.get("image_ref"), repo, forge,
                        df_path, vals.get("build_context"),
                        df_content, ("provided" if df_content else None),
                        log_tail, line_count, truncated,
                        vals.get("build_status"), vals.get("built_at"), vals.get("source_batch"),
                        bid,
                    ),
                )
                updated += 1
            else:
                cur = conn.execute(
                    """INSERT INTO builds(
                         external_id, tool_name, tool_version, image_ref, source_repo_url, source_forge,
                         dockerfile_path, build_context, dockerfile_content, dockerfile_fetch_status,
                         log_tail, log_line_count, log_truncated, build_status, built_at, source_batch)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        ext, tool_name, vals.get("tool_version"), vals.get("image_ref"), repo, forge,
                        df_path, vals.get("build_context"), df_content, fetch_status,
                        log_tail, line_count, truncated, vals.get("build_status") or "failed",
                        vals.get("built_at"), vals.get("source_batch"),
                    ),
                )
                bid = cur.lastrowid
                inserted += 1

            # Optional predefined leaf-level label -> confirmed annotation.
            # Unknown leaves are counted and reported, never fatal. Existing human
            # annotations are never overwritten; re-importing the same label is a no-op.
            if _apply_label(conn, bid, vals.get("label_leaf"), valid_leaves, unknown_labels):
                labeled += 1

    if unknown_labels:
        top = ", ".join(f"{k} ({n})" for k, n in sorted(unknown_labels.items(), key=lambda kv: -kv[1])[:10])
        print(
            f"warning: {sum(unknown_labels.values())} row(s) had a label that is not a known leaf; "
            f"ignored. Unknown values: {top}",
            file=sys.stderr,
        )

    return inserted, updated, skipped, labeled


def _apply_label(conn, build_id, leaf, valid_leaves, unknown_labels) -> bool:
    """Record a predefined leaf label as a `prefilled` import annotation (a
    suggestion for the annotator to confirm).

    Returns True if an annotation was written. Skips unknown leaves (counting
    them), never overwrites an existing human or already-confirmed annotation,
    and is idempotent for a label already prefilled for this build.
    """
    if not leaf:
        return False
    if leaf not in valid_leaves:
        unknown_labels[leaf] = unknown_labels.get(leaf, 0) + 1
        return False
    ca = conn.execute(
        "SELECT leaf_id, status, source FROM current_annotation WHERE build_id=?", (build_id,)
    ).fetchone()
    if ca:
        # Don't clobber human work (confirmed/deferred), and don't clobber a
        # confirmation of any kind — only re-prefill when nothing better exists.
        human = ca["source"] != IMPORT_SOURCE and ca["status"] in ("confirmed", "deferred")
        confirmed = ca["status"] == "confirmed"
        same = ca["source"] == IMPORT_SOURCE and ca["leaf_id"] == leaf and ca["status"] == IMPORT_STATUS
        if human or confirmed or same:
            return False
    add_annotation(conn, build_id, leaf_id=leaf, status=IMPORT_STATUS,
                   annotator=IMPORT_SOURCE, source=IMPORT_SOURCE)
    return True