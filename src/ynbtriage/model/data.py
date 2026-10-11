"""Pull labelled builds out of the DB into a DataFrame for modeling.

Rules (they follow the DB's own semantics, see schema.sql and splits.py):

  * Ground truth is the latest annotation with status='confirmed'.
  * The split comes from dataset_split. Rows split as 'gold' are NEVER returned
    here: gold is held out from all model development.
  * Prefilled annotations (imported suggestions, e.g. from the build-failure-review
    skill) are not ground truth. With include_prefilled=True they are added as
    extra TRAINING rows only ("silver" labels); they never enter val/test.
  * target='class' predicts the 3 coarse classes; target='leaf' the 12 leaves.
"""
from __future__ import annotations

import pandas as pd

from ..db import connect
from .excerpt import error_excerpt

_QUERY = """
SELECT b.build_id, b.tool_name, b.log_tail,
       ca.class_id, ca.leaf_id, ca.status AS ann_status,
       ds.split
FROM builds b
JOIN current_annotation ca ON ca.build_id = b.build_id
LEFT JOIN dataset_split ds ON ds.build_id = b.build_id
WHERE ca.status IN ('confirmed', 'prefilled')
"""


def label_order(db_path=None, target: str = "class") -> list[str]:
    """Label names in taxonomy order (stable across runs, independent of data)."""
    conn = connect(db_path)
    try:
        if target == "class":
            rows = conn.execute("SELECT class_id AS id FROM taxonomy_class ORDER BY sort_order")
        else:
            rows = conn.execute(
                "SELECT l.leaf_id AS id FROM taxonomy_leaf l "
                "JOIN taxonomy_class c ON c.class_id = l.class_id "
                "ORDER BY c.sort_order, l.sort_order"
            )
        return [r["id"] for r in rows.fetchall()]
    finally:
        conn.close()


def load_dataset(db_path=None, target: str = "class", include_prefilled: bool = False,
                 excerpt_kwargs: dict | None = None) -> pd.DataFrame:
    if target not in ("class", "leaf"):
        raise ValueError("target must be 'class' or 'leaf'")
    conn = connect(db_path)
    try:
        df = pd.read_sql_query(_QUERY, conn)
    finally:
        conn.close()

    df["label"] = df["class_id"] if target == "class" else df["leaf_id"]
    df = df[df["label"].notna()]

    confirmed = df[(df["ann_status"] == "confirmed") & df["split"].isin(["train", "val", "test"])]
    parts = [confirmed]
    if include_prefilled:
        silver = df[(df["ann_status"] == "prefilled") & df["split"].isna()].copy()
        silver["split"] = "train"
        parts.append(silver)
    out = pd.concat(parts, ignore_index=True)

    out["text"] = out["log_tail"].map(lambda s: error_excerpt(s, **(excerpt_kwargs or {})))
    out["silver"] = out["ann_status"] == "prefilled"
    return out[["build_id", "tool_name", "text", "label", "split", "silver"]]


def leakage_report(df: pd.DataFrame) -> dict:
    """How many tools have builds in more than one split.

    splits.py stratifies by class but does not group by tool, so several builds
    of the same tool (different versions, near-identical logs) can land in both
    train and test. That inflates scores for every model. This reports the size
    of the problem; it does not fix it.
    """
    per_tool = df.groupby("tool_name")["split"].nunique()
    shared = per_tool[per_tool > 1].index
    test_rows = df[df["split"] == "test"]
    return {
        "tools_total": int(per_tool.size),
        "tools_in_multiple_splits": int(len(shared)),
        "test_rows_whose_tool_is_also_in_train": int(
            test_rows["tool_name"].isin(df.loc[df["split"] == "train", "tool_name"]).sum()
        ),
        "test_rows": int(len(test_rows)),
    }
