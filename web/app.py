"""ynobuild — build-failure triage viewer + annotation UI.

Four screens:
  * Browse   — filterable table of builds; open one to see Dockerfile + log side
               by side, with the first failing region surfaced.
  * Annotate — walk unannotated builds; pick a coarse class, then reveal and pick
               a leaf within it (two-level, per the taxonomy). Writes to the DB.
  * Predict  — upload or paste a build log; the trained MLP (model service)
               predicts its failure class. Writes nothing.
  * Train    — retrain the MLP with chosen hyperparameters (model service), watch
               the loss curves live, compare runs on validation data, and choose
               which run Predict serves. Never writes to the DB.
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
screen = st.sidebar.radio("Screen", ["Browse", "Annotate", "Predict", "Train"])
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
        st.success("Nothing left in the queue. 🍕")
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

# ---------------------------------------------------------------- PREDICT
elif screen == "Predict":
    import hashlib

    # Check the model service before drawing the header, so the card can sit beside it.
    status = api.model_status()
    info = api.model_info() if status and status.get("model_loaded") else None

    # The empty third column is a spacer that keeps the card in from the right edge.
    head, card, _ = st.columns([5, 2, 1], vertical_alignment="top")

    with head:
        st.header("Predict")
        st.caption("Classify a failed build from its log with the trained neural network (MLP) model.")

    # ---- compact model card, top right: the served MLP first, baselines tucked away
    if info:
        with card, st.container(border=True):
            arch = info.get("architecture", {})
            hidden = "→".join(str(h) for h in arch.get("hidden", []))
            mlp_test = info.get("scores", {}).get("mlp", {}).get("test")
            mlp_val = info.get("scores", {}).get("mlp", {}).get("val")
            lines = [f"**MLP** `{info['name']}`"]
            if mlp_test:
                lines.append(f"test macro-F1 **{mlp_test['macro_f1']:.3f}**")
            if mlp_val:
                lines.append(f"val macro-F1 {mlp_val['macro_f1']:.3f}")
            st.markdown("  \n".join(lines))
            st.caption(f"hidden {hidden} · {arch.get('activation', '?')} · "
                       f"{info.get('features', '?')}  \n"
                       f"trained {str(info.get('trained_at', '?'))[:10]}")

            names = {"mlp": "MLP (served)", "tfidf_logreg": "TF-IDF + logistic reg.",
                     "naive_bayes": "Naive Bayes (word counts)", "majority": "Majority class"}
            scores = {k: v["test"] for k, v in info.get("scores", {}).items() if v.get("test")}
            if len(scores) > 1:
                with st.popover("Baselines"):
                    st.caption("Held-out test scores from training. The baselines are for "
                               "comparison with the MLP.")
                    st.table([{"model": names.get(k, k),
                               "macro-F1": f"{t['macro_f1']:.3f}",
                               "accuracy": f"{t['accuracy']:.3f}"}
                              for k, t in sorted(scores.items(),
                                                 key=lambda kv: list(names).index(kv[0])
                                                 if kv[0] in names else 99)])

    if status is None:
        st.error(f"Cannot reach the model service at {api.MODEL_URL}. "
                 "Is the `model` service up? (`docker compose up -d model`)")
        st.stop()
    if not status.get("model_loaded"):
        st.warning("The model service is running but has no model loaded.")
        st.caption(status.get("error") or "")
        st.caption("Train one with `ynbtriage train --out models/final ...`, then "
                   "`docker compose restart model`.")
        st.stop()

    # ---- input
    source = st.radio("Log input", ["Upload a file", "Paste text"], horizontal=True)
    log_text = ""
    if source == "Upload a file":
        up = st.file_uploader("Build log", type=["log", "txt"])
        if up is not None:
            log_text = up.getvalue().decode("utf-8", errors="replace")
    else:
        log_text = st.text_area("Build log text", height=220,
                                placeholder="Paste the output of a failed build …")

    # Very large uploads: keep the tail (the service also caps it).
    if len(log_text) > 1_000_000:
        log_text = log_text[-1_000_000:]

    key = hashlib.sha1(log_text.encode("utf-8", errors="replace")).hexdigest()
    if st.button("Predict", type="primary", disabled=not log_text.strip()):
        try:
            st.session_state.prediction = (key, api.predict_log(log_text))
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")

    # ---- result (only if it belongs to the log currently entered)
    saved = st.session_state.get("prediction")
    if saved and saved[0] == key and log_text.strip():
        res = saved[1]
        probs = sorted(res["probabilities"].items(), key=lambda kv: -kv[1])
        top_label, top_p = probs[0]
        display = lambda cid: LOOKUP.get(cid, {}).get("display", cid)  # noqa: E731

        st.divider()
        c1, c2 = st.columns([2, 1])
        c1.metric("Predicted failure class", display(top_label))
        c2.metric("Model probability", f"{top_p:.1%}")

        if top_p < 0.6:
            st.warning("Low confidence: the model is unsure between classes. Read the log.")
        if not res.get("error_line_found"):
            st.info("No recognizable error line in this log, so the prediction rests on the "
                    "last lines only and may be unreliable.")
        if res.get("input_truncated"):
            st.caption("The log was long; only its tail was used.")

        st.markdown("**Probability by class**")
        for cid, p in probs:
            st.progress(p, text=f"{display(cid)}: {p:.1%}")

        render_log(res.get("excerpt", ""))

# ---------------------------------------------------------------- TRAIN
elif screen == "Train":
    import pandas as pd

    RUNNING = ("queued", "loading_data", "training")
    LOSS_COLORS = ["#2a78d6", "#eb6834"]  # train, val (same as loss_curve.png)
    FALLBACK = {"hidden": [64], "dropout": 0.2, "lr": 1e-3, "epochs": 200,
                "features": "tfidf", "activation": "relu"}

    st.header("Train")
    st.caption("Retrain the neural network (MLP) with different hyperparameters, compare runs "
               "on validation data, and choose which run the Predict screen uses.")

    if api.model_status() is None:
        st.error(f"Cannot reach the model service at {api.MODEL_URL}. "
                 "Is the `model` service up? (`docker compose up -d model`)")
        st.stop()

    runs = api.list_runs()
    served = next((r for r in runs if r["served"]), None)
    default = next((r for r in runs if r["default"]), None)
    latest = api.train_latest()
    running = latest if latest and latest["status"] in RUNNING else None

    def fmt_hidden(h):
        return "→".join(str(x) for x in (h or [])) or "?"

    def f1(scores):
        return scores["macro_f1"] if scores else None

    # ---- what's served, and the way back to the default run
    with st.container(border=True):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        if served:
            c1.markdown(f"Predict is currently using **`{served['name']}`** · hidden {fmt_hidden(served['hidden'])} · "
                        f"val macro-F1 **{f1(served['val']) or float('nan'):.3f}**")
        else:
            c1.markdown("Predict has no model loaded.")
        if default and (not served or served["name"] != default["name"]):
            if c2.button(f"Switch back to {default['name']}", use_container_width=True):
                try:
                    api.deploy_run(default["name"])
                    st.toast(f"Predict is now using {default['name']}.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not switch: {e}")
                st.rerun()
        elif default:
            c2.caption(f"`{default['name']}` is the default model.")

    # ---- hyperparameters (defaults = how the default run was trained)
    base = dict(FALLBACK)
    if default:
        base.update({k: default[k] for k in FALLBACK if default.get(k) is not None})
    lrs = sorted({1e-4, 3e-4, 1e-3, 3e-3, 1e-2, float(base["lr"])})
    feats = ["tfidf", "embed", "both"]
    acts = ["relu", "gelu", "tanh", "sigmoid"]

    st.subheader("New run")
    with st.form("train_form"):
        a, b = st.columns(2)
        hidden_txt = a.text_input(
            "Hidden layers", value=",".join(str(h) for h in base["hidden"]),
            help="Units per hidden layer, comma-separated. 64 is one layer of 64 units; "
                 "128,64 is two layers. Up to 4 layers.")
        epochs = b.number_input(
            "Max epochs", min_value=1, max_value=2000, value=int(base["epochs"]), step=10,
            help="Training stops early once validation loss has not improved for 25 epochs; "
                 "the weights from the best epoch are kept.")
        lr = a.select_slider("Learning rate", options=lrs, value=float(base["lr"]),
                             format_func=lambda v: f"{v:g}")
        dropout = b.slider("Dropout", 0.0, 0.6, float(base["dropout"]), 0.05)
        features = st.radio(
            "Input features", feats, horizontal=True,
            index=feats.index(base["features"]) if base["features"] in feats else 0,
            help="tfidf: TF-IDF of the log excerpt, reduced to 256 numbers. "
                 "embed: frozen sentence-encoder embeddings (the first embed run downloads the encoder). "
                 "both: the two combined.")
        with st.expander("Advanced"):
            activation = st.selectbox("Activation", acts,
                                      index=acts.index(base["activation"]) if base["activation"] in acts else 0)
        submitted = st.form_submit_button("Train", type="primary", disabled=running is not None)

    if running:
        st.caption("A run is in progress; the form unlocks when it finishes.")
    if submitted:
        try:
            hidden = [int(x) for x in hidden_txt.replace(" ", "").split(",") if x]
        except ValueError:
            st.error("Hidden layers must be whole numbers separated by commas, e.g. 128,64.")
        else:
            try:
                api.train_start({"hidden": hidden, "epochs": int(epochs), "lr": float(lr),
                                 "dropout": float(dropout), "features": features,
                                 "activation": activation})
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not start training: {e}")

    # ---- live monitor / latest result
    def loss_chart(history):
        h = pd.DataFrame(history)
        if h.empty:
            return
        st.line_chart(h.set_index("epoch")[["train_loss", "val_loss"]], color=LOSS_COLORS,
                      x_label="epoch", y_label="cross-entropy loss", height=260)

    if running:
        @st.fragment(run_every=1.0)
        def monitor(job_id):
            j = api.train_status(job_id)
            if j["status"] not in RUNNING:
                st.rerun()  # finished: redraw the whole page with the result and run list
            st.subheader(f"Training `{j['run']}`")
            hist = j["history"]
            if j["status"] == "loading_data":
                st.caption("Fetching labelled builds from the api…")
            else:
                done = len(hist)
                st.progress(min(done / max(j["epochs"], 1), 1.0),
                            text=f"epoch {done} of up to {j['epochs']}")
                if hist:
                    last = hist[-1]
                    best = min(hist, key=lambda r: r["val_loss"])
                    st.caption(f"val loss {last['val_loss']:.4f} · val accuracy {last['val_acc']:.3f} · "
                               f"best so far: epoch {best['epoch']} (val loss {best['val_loss']:.4f})")
                loss_chart(hist)
            if st.button("Cancel run"):
                api.train_cancel(job_id)

        monitor(running["id"])

    elif latest:
        st.subheader(f"Latest run: `{latest['run']}`")
        if latest["status"] == "failed":
            st.error(f"Training failed: {latest['error']}")
        elif latest["status"] == "cancelled":
            st.info("Cancelled; nothing was saved.")
        elif latest["status"] == "done" and latest.get("result"):
            res = latest["result"]
            this = next((r for r in runs if r["name"] == latest["run"]), None)
            st.caption(f"{res['epochs_run']} epochs (best epoch {res['best_epoch']}; its weights were kept)")
            compare = [{"model": f"this run ({latest['run']})",
                        "val macro-F1": f1(res["mlp"]["val"]), "val accuracy": res["mlp"]["val"]["accuracy"]}]
            if served and served["name"] != latest["run"]:
                compare.append({"model": f"currently served ({served['name']})",
                                "val macro-F1": f1(served["val"]),
                                "val accuracy": (served["val"] or {}).get("accuracy")})
            tfl = res["baselines"].get("tfidf_logreg", {}).get("val")
            if tfl:
                compare.append({"model": "TF-IDF + logistic reg. (baseline)",
                                "val macro-F1": tfl["macro_f1"], "val accuracy": tfl["accuracy"]})
            st.dataframe(pd.DataFrame(compare), hide_index=True, use_container_width=True,
                         column_config={"val macro-F1": st.column_config.NumberColumn(format="%.3f"),
                                        "val accuracy": st.column_config.NumberColumn(format="%.3f")})
            loss_chart(latest["history"])
            if this and not this["served"]:
                if st.button(f"Deploy {latest['run']}", type="primary"):
                    try:
                        api.deploy_run(latest["run"])
                        st.toast(f"Predict now uses {latest['run']}.")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Could not deploy: {e}")
                    st.rerun()
            elif this is None:
                st.caption("This run has since been deleted.")

    # ---- all runs
    st.subheader("Runs")
    if not runs:
        st.info("No trained runs yet.")
        st.stop()

    ref_hash = ((served or default or {}).get("data_fingerprint") or {}).get("hash")
    table = []
    for r in runs:
        h = (r.get("data_fingerprint") or {}).get("hash")
        tags = [t for t, on in (("served", r["served"]), ("default", r["default"])) if on]
        table.append({
            "run": r["name"],
            "": " · ".join(tags),
            "hidden": fmt_hidden(r["hidden"]),
            "activation": r["activation"],
            "dropout": r["dropout"],
            "lr": f"{r['lr']:g}" if r["lr"] is not None else None,
            "features": r["features"],
            "epochs (best)": f"{r['epochs_run']} ({r['best_epoch']})" if r["epochs_run"] else None,
            "val macro-F1": f1(r["val"]),
            "val accuracy": (r["val"] or {}).get("accuracy"),
            "data": "same" if h and h == ref_hash else ("different" if h else "unknown"),
        })
    table.sort(key=lambda x: -(x["val macro-F1"] or 0))
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True,
                 column_config={"val macro-F1": st.column_config.NumberColumn(format="%.3f"),
                                "val accuracy": st.column_config.NumberColumn(format="%.3f")})
    ref = served or default
    tfl_val = f1(((ref or {}).get("tfidf_logreg") or {}).get("val"))

    with st.expander("Test scores"):
        st.caption("Held-out test scores, for reporting the run you chose on validation. "
                   "Don't use them to choose between runs: picking by test score turns the "
                   "test set into a second validation set.")
        st.dataframe(pd.DataFrame([{
            "run": r["name"],
            "test macro-F1": f1(r["test"]),
            "test accuracy": (r["test"] or {}).get("accuracy"),
            "TF-IDF baseline test macro-F1": f1((r.get("tfidf_logreg") or {}).get("test")),
        } for r in runs]), hide_index=True, use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="%.3f")
                           for c in ("test macro-F1", "test accuracy", "TF-IDF baseline test macro-F1")})

    # ---- deploy / delete any run
    names = [r["name"] for r in runs]
    by_name = {r["name"]: r for r in runs}
    a, b, c = st.columns([3, 1, 1], vertical_alignment="bottom")
    pick = a.selectbox("Run", names, index=names.index(served["name"]) if served else 0)
    chosen = by_name[pick]
    if b.button("Deploy", disabled=chosen["served"], use_container_width=True):
        try:
            api.deploy_run(pick)
            st.toast(f"Predict now uses {pick}.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not deploy: {e}")
        st.rerun()
    with c.popover("Delete", disabled=not chosen["deletable"], use_container_width=True):
        st.write(f"Permanently delete `{pick}` and its files?")
        if st.button("Delete run", type="primary", key=f"del_{pick}"):
            try:
                api.delete_run(pick)
                st.toast(f"Deleted {pick}.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not delete: {e}")
            st.rerun()
    if not chosen["deletable"]:
        why = ("it is the default run" if chosen["default"] else
               "it is being served" if chosen["served"] else
               "it was not created from this screen")
        st.caption(f"`{pick}` can't be deleted here: {why}.")
