"""ynobuild — build-failure triage viewer + annotation UI.

Two screens:
  * Browse   — filterable table of builds; open one to see Dockerfile + log side
               by side, with the first failing region surfaced.
  * Annotate — walk unannotated builds; pick a coarse class, then reveal and pick
               a leaf within it (two-level, per the taxonomy). Writes to the DB.
"""
import os
import re

import streamlit as st

import api_client as api

st.set_page_config(page_title="ynobuild", layout="wide")

DEFAULT_ANNOTATOR = os.environ.get("YNB_ANNOTATOR", "unknown")

# Patterns used to locate the first "failing region" in a log tail.
ERROR_PATTERNS = [
    r"error:", r"\bE:\s", r"fatal error", r"fatal:", r"\bKilled\b", r"No space left",
    r"not found", r"parse error", r"unknown instruction", r"Traceback",
    r"CondaToSNonInteractiveError", r"Terms of Service", r"\b40[13]\b",
    r"exec format error", r"GLIBC", r"Connection timed out", r"curl: \(\d+\)",
    r"\*\*\* .*Error", r"returned error", r"was not found",
]
_ERR_RE = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)


def first_error_line(log: str):
    lines = log.splitlines()
    for i, ln in enumerate(lines):
        if _ERR_RE.search(ln):
            return i, lines
    return None, lines


def render_log(log: str, context: int = 6, full_lines=None, truncated=None):
    shown = (log or "").count("\n") + 1 if log else 0
    if truncated and full_lines and full_lines > shown:
        st.caption(f"⚠ truncated — showing {shown} lines of {full_lines} (tail)")
    idx, lines = first_error_line(log or "")
    if idx is not None:
        lo, hi = max(0, idx - context), min(len(lines), idx + context + 1)
        st.caption(f"Failing region (line {idx + 1} of {len(lines)} shown)")
        st.code("\n".join(lines[lo:hi]) or "(empty)", language="text")
        with st.expander("Full log tail", expanded=False):
            st.code(log or "(empty)", language="text")
    else:
        st.caption("No error pattern matched; showing full tail.")
        st.code(log or "(empty)", language="text")


@st.cache_data(ttl=30)
def taxonomy():
    return api.get_taxonomy()


def class_leaf_lookup(tax):
    """Return {class_id: {'display': ..., 'leaves': [(leaf_id, display), ...]}}."""
    out = {}
    for c in tax:
        out[c["class_id"]] = {
            "display": c["display_name"],
            "leaves": [(l["leaf_id"], l["display_name"]) for l in c["leaves"]],
        }
    return out


# ---------------------------------------------------------------- sidebar
if not api.health():
    st.error(f"Cannot reach the API at {api.API_URL}. Is the `api` service up?")
    st.stop()

st.sidebar.title("ynobuild")
screen = st.sidebar.radio("Screen", ["Browse", "Annotate"])
annotator = st.sidebar.text_input("Annotator", value=DEFAULT_ANNOTATOR)

stats = api.get_stats()
st.sidebar.markdown("**Corpus**")
st.sidebar.write(
    f"{stats['total_builds']} builds · {stats['annotated']} labelled · "
    f"{stats.get('prefilled', 0)} prefilled · {stats['deferred']} deferred · "
    f"{stats['unannotated']} to do"
)
if stats.get("by_class"):
    st.sidebar.markdown("**By class**")
    for cid, n in stats["by_class"].items():
        st.sidebar.write(f"- {cid}: {n}")
if stats.get("by_split"):
    st.sidebar.markdown("**Splits**")
    st.sidebar.write(", ".join(f"{k}={v}" for k, v in stats["by_split"].items()))

with st.sidebar.expander("Maintenance"):
    st.caption("Delete builds whose log is the `[no logs]` sentinel.")
    if st.button("Scan for [no logs] builds"):
        st.session_state.no_log_count = api.prune_no_logs(dry_run=True)["count"]
    n = st.session_state.get("no_log_count")
    if n is not None:
        if n == 0:
            st.success("No [no logs] builds found.")
        else:
            st.warning(f"{n} build(s) with no logs.")
            if st.checkbox("Confirm permanent deletion", key="prune_ok") and \
               st.button(f"Delete {n} build(s)", type="primary"):
                res = api.prune_no_logs(dry_run=False)
                st.session_state.no_log_count = None
                st.cache_data.clear()
                st.success(f"Deleted {res['count']} build(s).")
                st.rerun()

TAX = taxonomy()
LOOKUP = class_leaf_lookup(TAX)


# ---------------------------------------------------------------- annotate widget
def annotation_controls(build, key_prefix):
    """Two-level control: coarse class -> revealed leaves. Returns a submit dict
    or None. Writes happen in the caller."""
    current = build.get("current_annotation") or {}
    if current.get("status") == "prefilled":
        leaf = current.get("leaf_id") or "?"
        st.info(f"Suggested label (prefilled from import): **{leaf}** — review and Confirm, or change it.")
    class_ids = list(LOOKUP.keys())
    class_labels = [LOOKUP[c]["display"] for c in class_ids]

    default_class_idx = 0
    if current.get("class_id") in class_ids:
        default_class_idx = class_ids.index(current["class_id"])

    ci = st.radio(
        "Failure class",
        options=range(len(class_ids)),
        format_func=lambda i: class_labels[i],
        index=default_class_idx,
        key=f"{key_prefix}_class",
        horizontal=True,
    )
    chosen_class = class_ids[ci]

    leaves = LOOKUP[chosen_class]["leaves"]
    leaf_ids = [lid for lid, _ in leaves]
    leaf_labels = [lbl for _, lbl in leaves]
    default_leaf_idx = 0
    if current.get("leaf_id") in leaf_ids:
        default_leaf_idx = leaf_ids.index(current["leaf_id"])

    li = st.radio(
        f"Leaf within {LOOKUP[chosen_class]['display']}",
        options=range(len(leaf_ids)),
        format_func=lambda i: leaf_labels[i],
        index=default_leaf_idx,
        key=f"{key_prefix}_leaf",
    )
    chosen_leaf = leaf_ids[li]

    note = st.text_input("Note (optional)", value=current.get("note") or "", key=f"{key_prefix}_note")

    c1, c2, c3 = st.columns(3)
    if c1.button("Confirm", type="primary", key=f"{key_prefix}_confirm"):
        return {"class_id": chosen_class, "leaf_id": chosen_leaf, "status": "confirmed",
                "annotator": annotator, "note": note or None}
    if c2.button("Defer (low confidence)", key=f"{key_prefix}_defer"):
        return {"status": "deferred", "annotator": annotator, "note": note or None}
    if c3.button("Skip", key=f"{key_prefix}_skip"):
        return {"status": "skipped", "annotator": annotator, "note": note or None}
    return None


def build_detail(build):
    left, right = st.columns(2)
    with left:
        st.subheader("Dockerfile")
        if build.get("dockerfile_content"):
            st.caption(build.get("dockerfile_path") or "")
            st.code(build["dockerfile_content"], language="dockerfile")
        else:
            status = build.get("dockerfile_fetch_status") or "not fetched"
            st.info(f"No Dockerfile available ({status}). Run `ynbtriage fetch-dockerfiles`.")
            if build.get("source_repo_url"):
                st.write(f"Repo: {build['source_repo_url']}")
    with right:
        st.subheader("Build log (tail)")
        render_log(
            build.get("log_tail", ""),
            full_lines=build.get("log_line_count"),
            truncated=build.get("log_truncated"),
        )


# ---------------------------------------------------------------- BROWSE
if screen == "Browse":
    st.header("Browse builds")

    fc1, fc2, fc3, fc4 = st.columns(4)
    class_opts = ["All"] + list(LOOKUP.keys())
    f_class = fc1.selectbox("Class", class_opts)
    leaf_opts = ["All"] + ([lid for lid, _ in LOOKUP[f_class]["leaves"]] if f_class != "All" else [])
    f_leaf = fc2.selectbox("Leaf", leaf_opts)
    f_annot = fc3.selectbox("Annotated", ["All", "yes", "no", "deferred", "prefilled"])
    f_split = fc4.selectbox("Split", ["All", "train", "val", "test", "gold"])
    f_q = st.text_input("Search tool name or log text")

    page_size = 25
    if "browse_offset" not in st.session_state:
        st.session_state.browse_offset = 0

    resp = api.list_builds(
        class_id=None if f_class == "All" else f_class,
        leaf_id=None if f_leaf == "All" else f_leaf,
        annotated=None if f_annot == "All" else f_annot,
        split=None if f_split == "All" else f_split,
        q=f_q,
        limit=page_size,
        offset=st.session_state.browse_offset,
    )
    items, total = resp["items"], resp["total"]
    st.caption(f"{total} builds match. Showing {len(items)}.")

    if items:
        st.dataframe(
            [
                {
                    "id": r["build_id"],
                    "tool": r["tool_name"],
                    "version": r.get("tool_version"),
                    "class": r.get("class_id"),
                    "leaf": r.get("leaf_id"),
                    "annot": r.get("annotation_status"),
                    "split": r.get("split"),
                    "dockerfile": r.get("dockerfile_fetch_status"),
                }
                for r in items
            ],
            use_container_width=True,
            hide_index=True,
        )

    pcol1, pcol2, pcol3 = st.columns([1, 1, 6])
    if pcol1.button("← Prev", disabled=st.session_state.browse_offset == 0):
        st.session_state.browse_offset = max(0, st.session_state.browse_offset - page_size)
        st.rerun()
    if pcol2.button("Next →", disabled=st.session_state.browse_offset + page_size >= total):
        st.session_state.browse_offset += page_size
        st.rerun()

    st.divider()
    if items:
        ids = [r["build_id"] for r in items]
        chosen = st.selectbox("Open build", ids, format_func=lambda i: f"#{i}")
        build = api.get_build(chosen)
        if build:
            meta = f"**{build['tool_name']}** {build.get('tool_version') or ''} · " \
                   f"status={build.get('build_status')} · split={build.get('split') or '—'}"
            if build.get("current_annotation"):
                ca = build["current_annotation"]
                meta += f" · label={ca.get('leaf_id') or ca.get('class_id') or ca.get('status')}"
            st.markdown(meta)
            build_detail(build)

            with st.expander("⚠ Danger zone — delete this build"):
                st.caption(
                    "Permanently removes this build and its entire annotation "
                    "history. Re-ingesting the source CSV would re-create it."
                )
                ok = st.checkbox("I understand this is permanent", key=f"del_ok_{build['build_id']}")
                if st.button("Delete build", disabled=not ok, key=f"del_btn_{build['build_id']}"):
                    try:
                        api.delete_build(build["build_id"])
                        st.session_state.browse_offset = 0
                        st.cache_data.clear()
                        st.success(f"Deleted build #{build['build_id']}.")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Failed to delete: {e}")


# ---------------------------------------------------------------- ANNOTATE
elif screen == "Annotate":
    st.header("Annotate")

    mode = st.radio("Queue", ["Unannotated", "All"], horizontal=True)
    resp = api.list_builds(annotated=None if mode == "All" else "no", limit=500)
    queue = resp["items"]
    if not queue:
        st.success("Nothing left in this queue. 🎉")
        st.stop()

    if "annot_idx" not in st.session_state:
        st.session_state.annot_idx = 0
    st.session_state.annot_idx %= len(queue)

    nav1, nav2, nav3 = st.columns([1, 1, 6])
    if nav1.button("← Prev build"):
        st.session_state.annot_idx = (st.session_state.annot_idx - 1) % len(queue)
        st.rerun()
    if nav2.button("Next build →"):
        st.session_state.annot_idx = (st.session_state.annot_idx + 1) % len(queue)
        st.rerun()

    current_row = queue[st.session_state.annot_idx]
    build = api.get_build(current_row["build_id"])
    st.caption(f"Build {st.session_state.annot_idx + 1} of {len(queue)} in queue")
    st.markdown(f"### #{build['build_id']} · {build['tool_name']} {build.get('tool_version') or ''}")

    build_detail(build)
    st.divider()

    payload = annotation_controls(build, key_prefix=f"ann_{build['build_id']}")
    if payload:
        try:
            api.annotate(build["build_id"], payload)
            st.session_state.annot_idx = (st.session_state.annot_idx + 1) % len(queue)
            st.cache_data.clear()
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to save: {e}")