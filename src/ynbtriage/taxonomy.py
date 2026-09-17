"""Load taxonomy_map.yaml and seed it into the DB (idempotent upsert)."""
import yaml

from .config import TAXONOMY_PATH


def load_taxonomy(path=None) -> dict:
    with open(path or TAXONOMY_PATH) as f:
        return yaml.safe_load(f)


def seed_taxonomy(conn, taxonomy=None) -> None:
    tax = taxonomy or load_taxonomy()
    for class_id, c in tax["classes"].items():
        conn.execute(
            """INSERT INTO taxonomy_class(class_id, display_name, description, sort_order)
               VALUES(?,?,?,?)
               ON CONFLICT(class_id) DO UPDATE SET
                 display_name=excluded.display_name,
                 description=excluded.description,
                 sort_order=excluded.sort_order""",
            (class_id, c["display_name"], (c.get("description") or "").strip(), c.get("sort_order", 0)),
        )
        for i, (leaf_id, leaf) in enumerate(c["leaves"].items()):
            conn.execute(
                """INSERT INTO taxonomy_leaf(leaf_id, class_id, display_name, description,
                                             rule_prefilled, caveat, sort_order)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(leaf_id) DO UPDATE SET
                     class_id=excluded.class_id,
                     display_name=excluded.display_name,
                     description=excluded.description,
                     rule_prefilled=excluded.rule_prefilled,
                     caveat=excluded.caveat,
                     sort_order=excluded.sort_order""",
                (
                    leaf_id,
                    class_id,
                    leaf["display_name"],
                    (leaf.get("description") or "").strip(),
                    1 if leaf.get("rule_prefilled") else 0,
                    leaf.get("caveat"),
                    i,
                ),
            )
