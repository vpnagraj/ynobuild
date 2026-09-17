"""Assign train / val / test / gold splits over *labelled* builds.

Stratified by coarse class so each split sees the same class mix. Seeded, so the
assignment is reproducible. `gold` is STICKY: once a build is gold it stays gold
across re-runs, because the gold set must never be trained on and must never
silently change. Only builds with a confirmed class label are eligible.
"""
from __future__ import annotations

import random

from .db import ensure_schema, get_conn


def assign_splits(db_path=None, seed=7400, gold_frac=0.2, val_frac=0.15, test_frac=0.15):
    rng = random.Random(seed)
    with get_conn(db_path) as conn:
        ensure_schema(conn)

        existing = {
            r["build_id"]: r["split"]
            for r in conn.execute("SELECT build_id, split FROM dataset_split").fetchall()
        }
        rows = conn.execute(
            "SELECT build_id, class_id FROM current_annotation WHERE status='confirmed'"
        ).fetchall()

        by_class: dict[str, list[int]] = {}
        for r in rows:
            by_class.setdefault(r["class_id"], []).append(r["build_id"])

        assignments: dict[int, str] = {}
        for _cls, ids in by_class.items():
            n = len(ids)
            already_gold = [i for i in ids if existing.get(i) == "gold"]
            pool = [i for i in ids if existing.get(i) != "gold"]
            rng.shuffle(pool)

            need_gold = max(0, round(n * gold_frac) - len(already_gold))
            new_gold = pool[:need_gold]
            rest = pool[need_gold:]

            n_val = round(n * val_frac)
            n_test = round(n * test_frac)
            val = rest[:n_val]
            test = rest[n_val:n_val + n_test]
            train = rest[n_val + n_test:]

            for i in already_gold + new_gold:
                assignments[i] = "gold"
            for i in val:
                assignments[i] = "val"
            for i in test:
                assignments[i] = "test"
            for i in train:
                assignments[i] = "train"

        for bid, split in assignments.items():
            conn.execute(
                """INSERT INTO dataset_split(build_id, split, method, seed)
                   VALUES(?,?,?,?)
                   ON CONFLICT(build_id) DO UPDATE SET
                     split=excluded.split, method=excluded.method,
                     seed=excluded.seed, assigned_at=datetime('now')""",
                (bid, split, "stratified_by_class", seed),
            )

        counts: dict[str, int] = {}
        for s in assignments.values():
            counts[s] = counts.get(s, 0) + 1
    return counts
