"""Data-access functions. The API is the only online caller of these; batch
jobs use them offline. Nothing here holds a connection open across calls.
"""
from __future__ import annotations


def get_taxonomy(conn) -> list[dict]:
    classes = conn.execute("SELECT * FROM taxonomy_class ORDER BY sort_order").fetchall()
    out = []
    for c in classes:
        leaves = conn.execute(
            "SELECT * FROM taxonomy_leaf WHERE class_id=? ORDER BY sort_order",
            (c["class_id"],),
        ).fetchall()
        cd = dict(c)
        cd["leaves"] = [dict(l) for l in leaves]
        out.append(cd)
    return out


_JOINS = """
    FROM builds b
    LEFT JOIN current_annotation ca ON ca.build_id = b.build_id
    LEFT JOIN dataset_split      ds ON ds.build_id = b.build_id
"""


def list_builds(
    conn,
    *,
    class_id=None,
    leaf_id=None,
    status=None,
    split=None,
    annotated=None,   # 'yes' | 'no' | 'deferred'
    q=None,
    limit=50,
    offset=0,
):
    where, params = [], []
    if class_id:
        where.append("ca.class_id = ?"); params.append(class_id)
    if leaf_id:
        where.append("ca.leaf_id = ?"); params.append(leaf_id)
    if status:
        where.append("b.build_status = ?"); params.append(status)
    if split:
        where.append("ds.split = ?"); params.append(split)
    if annotated == "yes":
        where.append("ca.annotation_id IS NOT NULL AND ca.status = 'confirmed'")
    elif annotated == "no":
        where.append("(ca.annotation_id IS NULL OR ca.status NOT IN ('confirmed','deferred'))")
    elif annotated == "deferred":
        where.append("ca.status = 'deferred'")
    elif annotated == "prefilled":
        where.append("ca.status = 'prefilled'")
    if q:
        where.append("(b.tool_name LIKE ? OR b.log_tail LIKE ?)")
        params += [f"%{q}%", f"%{q}%"]

    wsql = ("WHERE " + " AND ".join(where)) if where else ""

    total = conn.execute(f"SELECT COUNT(*) AS n {_JOINS} {wsql}", params).fetchone()["n"]
    rows = conn.execute(
        f"""SELECT b.build_id, b.tool_name, b.tool_version, b.image_ref, b.build_status,
                   b.source_repo_url, b.source_forge, b.dockerfile_fetch_status,
                   ca.class_id, ca.leaf_id, ca.status AS annotation_status,
                   ds.split
            {_JOINS} {wsql}
            ORDER BY b.build_id
            LIMIT ? OFFSET ?""",
        params + [limit, offset],
    ).fetchall()
    return [dict(r) for r in rows], total


def get_build(conn, build_id: int):
    b = conn.execute("SELECT * FROM builds WHERE build_id=?", (build_id,)).fetchone()
    if not b:
        return None
    ca = conn.execute("SELECT * FROM current_annotation WHERE build_id=?", (build_id,)).fetchone()
    ds = conn.execute("SELECT split FROM dataset_split WHERE build_id=?", (build_id,)).fetchone()
    d = dict(b)
    d["current_annotation"] = dict(ca) if ca else None
    d["split"] = ds["split"] if ds else None
    return d


def add_annotation(
    conn,
    build_id: int,
    *,
    class_id=None,
    leaf_id=None,
    status="confirmed",
    annotator="unknown",
    note=None,
    source="human",
):
    """Append an annotation. Validates that leaf belongs to class; a confirmed
    annotation requires at least a class."""
    if leaf_id:
        row = conn.execute("SELECT class_id FROM taxonomy_leaf WHERE leaf_id=?", (leaf_id,)).fetchone()
        if not row:
            raise ValueError(f"unknown leaf '{leaf_id}'")
        if class_id and row["class_id"] != class_id:
            raise ValueError(f"leaf '{leaf_id}' does not belong to class '{class_id}'")
        class_id = class_id or row["class_id"]
    if status in ("confirmed", "prefilled") and not class_id:
        raise ValueError(f"a {status} annotation requires a class")

    conn.execute(
        """INSERT INTO annotation(build_id, class_id, leaf_id, status, annotator, note, source)
           VALUES(?,?,?,?,?,?,?)""",
        (build_id, class_id, leaf_id, status, annotator, note, source),
    )
    row = conn.execute("SELECT * FROM current_annotation WHERE build_id=?", (build_id,)).fetchone()
    return dict(row) if row else None


def find_build_id(conn, external_id: str):
    """Resolve a build_id from an external_id, or None if not present."""
    row = conn.execute("SELECT build_id FROM builds WHERE external_id=?", (external_id,)).fetchone()
    return row["build_id"] if row else None


def delete_build(conn, build_id: int):
    """Delete a build with its annotation history and split row, in one transaction.

    Foreign keys are enforced and there is no ON DELETE CASCADE, so children are
    removed first. Returns a summary dict, or None if the build_id doesn't exist.
    This is permanent; re-ingesting the source CSV re-creates the build, since the
    upsert keys on external_id.
    """
    if not conn.execute("SELECT 1 FROM builds WHERE build_id=?", (build_id,)).fetchone():
        return None
    n_ann = conn.execute("SELECT COUNT(*) n FROM annotation WHERE build_id=?", (build_id,)).fetchone()["n"]
    n_spl = conn.execute("SELECT COUNT(*) n FROM dataset_split WHERE build_id=?", (build_id,)).fetchone()["n"]
    conn.execute("DELETE FROM annotation WHERE build_id=?", (build_id,))
    conn.execute("DELETE FROM dataset_split WHERE build_id=?", (build_id,))
    conn.execute("DELETE FROM builds WHERE build_id=?", (build_id,))
    return {"build_id": build_id, "annotations_deleted": n_ann, "split_deleted": n_spl}


def no_log_build_ids(conn):
    """build_ids whose log is the '[no logs]' sentinel (nothing to annotate from)."""
    rows = conn.execute("SELECT build_id FROM builds WHERE TRIM(log_tail)='[no logs]'").fetchall()
    return [r["build_id"] for r in rows]


def prune_no_logs(conn, dry_run=False):
    """Delete every build whose log is the '[no logs]' sentinel.

    With dry_run=True, only counts them (deletes nothing). Returns
    {count, build_ids, dry_run}.
    """
    ids = no_log_build_ids(conn)
    if not dry_run:
        for bid in ids:
            delete_build(conn, bid)
    return {"count": len(ids), "build_ids": ids, "dry_run": dry_run}


def get_stats(conn) -> dict:
    total = conn.execute("SELECT COUNT(*) n FROM builds").fetchone()["n"]
    confirmed = conn.execute(
        "SELECT COUNT(*) n FROM current_annotation WHERE status='confirmed'"
    ).fetchone()["n"]
    deferred = conn.execute(
        "SELECT COUNT(*) n FROM current_annotation WHERE status='deferred'"
    ).fetchone()["n"]
    prefilled = conn.execute(
        "SELECT COUNT(*) n FROM current_annotation WHERE status='prefilled'"
    ).fetchone()["n"]
    by_class = conn.execute(
        "SELECT class_id, COUNT(*) n FROM current_annotation WHERE status='confirmed' GROUP BY class_id"
    ).fetchall()
    by_leaf = conn.execute(
        "SELECT leaf_id, COUNT(*) n FROM current_annotation "
        "WHERE status='confirmed' AND leaf_id IS NOT NULL GROUP BY leaf_id"
    ).fetchall()
    by_split = conn.execute("SELECT split, COUNT(*) n FROM dataset_split GROUP BY split").fetchall()
    df_status = conn.execute(
        "SELECT COALESCE(dockerfile_fetch_status,'none') s, COUNT(*) n FROM builds GROUP BY 1"
    ).fetchall()
    return {
        "total_builds": total,
        "annotated": confirmed,
        "deferred": deferred,
        "prefilled": prefilled,
        "unannotated": total - confirmed - deferred - prefilled,
        "by_class": {r["class_id"]: r["n"] for r in by_class},
        "by_leaf": {r["leaf_id"]: r["n"] for r in by_leaf},
        "by_split": {r["split"]: r["n"] for r in by_split},
        "dockerfile_status": {r["s"]: r["n"] for r in df_status},
    }