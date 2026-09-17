-- ynobuild schema (container build-failure triage).
-- Design notes:
--   * Leaf labels are the stored ground truth; coarse class is derived from the
--     leaf via taxonomy_map.yaml (seeded into taxonomy_class / taxonomy_leaf).
--   * `annotation` is append-only (an audit trail). The latest row per build is
--     exposed through the `current_annotation` view. Nothing is ever updated in
--     place, so annotation history is preserved and reversible.
--   * dataset_split is a separate table so splits can be re-derived cleanly.
--     'gold' is sticky (see splits.py) and must never be trained on.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS taxonomy_class (
    class_id     TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description  TEXT,
    sort_order   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS taxonomy_leaf (
    leaf_id        TEXT PRIMARY KEY,
    class_id       TEXT NOT NULL REFERENCES taxonomy_class(class_id),
    display_name   TEXT NOT NULL,
    description    TEXT,
    rule_prefilled INTEGER NOT NULL DEFAULT 0,   -- e.g. dockerfile_parse_error
    caveat         TEXT,                          -- e.g. unstable_ground_truth
    sort_order     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS builds (
    build_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id             TEXT UNIQUE,          -- id from socr8s / CSV, if present
    tool_name               TEXT NOT NULL,
    tool_version            TEXT,
    image_ref               TEXT,
    source_repo_url         TEXT,
    source_forge            TEXT,                 -- github / gitlab / bitbucket / other
    dockerfile_path         TEXT,                 -- path within repo (from CSV, or discovered)
    build_context           TEXT,                 -- docker build context dir (from CSV)
    dockerfile_content      TEXT,                 -- fetched Dockerfile body
    dockerfile_fetch_status TEXT,                 -- pending / ok / not_found / error / NULL
    dockerfile_fetched_at   TEXT,
    log_tail                TEXT NOT NULL,        -- truncated log (from CSV), newlines unescaped
    log_line_count          INTEGER,              -- true full line count (pre-truncation)
    log_truncated           INTEGER NOT NULL DEFAULT 0,
    build_status            TEXT DEFAULT 'failed',
    built_at                TEXT,
    source_batch            TEXT,                 -- which socr8s run / CSV batch
    created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_builds_tool   ON builds(tool_name);
CREATE INDEX IF NOT EXISTS idx_builds_status ON builds(build_status);

CREATE TABLE IF NOT EXISTS annotation (
    annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    build_id      INTEGER NOT NULL REFERENCES builds(build_id),
    class_id      TEXT REFERENCES taxonomy_class(class_id),  -- NULL if deferred/skipped
    leaf_id       TEXT REFERENCES taxonomy_leaf(leaf_id),    -- NULL if class-only / deferred
    status        TEXT NOT NULL DEFAULT 'confirmed',         -- confirmed / prefilled / deferred / skipped
    annotator     TEXT NOT NULL DEFAULT 'unknown',
    note          TEXT,
    source        TEXT NOT NULL DEFAULT 'human',             -- human / seed / teacher (later)
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_annotation_build ON annotation(build_id);

-- Latest annotation per build (monotonic annotation_id => latest wins).
CREATE VIEW IF NOT EXISTS current_annotation AS
SELECT a.*
FROM annotation a
JOIN (
    SELECT build_id, MAX(annotation_id) AS max_id
    FROM annotation
    GROUP BY build_id
) m ON a.annotation_id = m.max_id;

CREATE TABLE IF NOT EXISTS dataset_split (
    build_id    INTEGER PRIMARY KEY REFERENCES builds(build_id),
    split       TEXT NOT NULL,        -- train / val / test / gold
    method      TEXT,
    seed        INTEGER,
    assigned_at TEXT NOT NULL DEFAULT (datetime('now'))
);