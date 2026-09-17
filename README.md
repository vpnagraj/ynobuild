# ynobuild

Tooling to browse and annotate container build failures under a collapsed 3-class taxonomy. Backed by SQLite, the tool helps build a labelled corpus and "gold standard" data for downstream modeling.

The label space is defined once in [`taxonomy/taxonomy_map.yaml`](taxonomy/taxonomy_map.yaml) and seeded into the DB. Leaf labels are ground truth; the coarse class is derived
from the leaf. Three classes, twelve leaves, none excluded (five carry leaf-level caveats).
 
> **Note on the label space.** This repo implements the taxonomy v0.2 scheme
> (`environment_decay`, `recipe_error`, `build_execution`). An older simplified
> draft lists a different set (`access_gated`, with three leaves excluded). They
> disagree; v0.2 is treated as authoritative here. To switch, edit only
> `taxonomy_map.yaml` and re-run `initdb` — nothing else hard-codes the classes.
 
---
 
## Architecture
 
Four containers, one shared SQLite volume.
 
```
                 ┌──────────────┐        HTTP        ┌──────────────┐
   browser  ───▶ │  web         │  ───────────────▶  │  api         │
   :8501         │  Streamlit   │                    │  FastAPI     │
                 │ (thin client)│                    │  owns the DB │
                 └──────────────┘                    └──────┬───────┘
                                                            │ sqlite (WAL)
                              ┌─────────────────────────────┴───────────┐
                              │            dbdata volume                 │
                              │            /data/ynobuild.db             │
                              └─────────────────────────────────────────┘
        one-shot jobs (share the volume, run offline):
        ingest ── load CSV      fetch ── pull Dockerfiles      splits ── train/val/test/gold
```
 
**Why the API owns the DB.** SQLite is embedded, not a server — you can't have
several containers open the file for writing without contention. So exactly one
long-running process (`api`) opens it; the UI is a pure HTTP client and never
touches SQLite. This is also the shape that ports cleanly to Kubernetes: `web`
and `api` become Deployments, the jobs become `Job`/`CronJob`, and only the DB
needs a shared volume (or, later, a swap to Postgres to drop the volume
constraint entirely).
 
The batch jobs open the DB directly for bootstrap convenience. WAL mode + a 30s
busy-timeout make the occasional overlap wait rather than error, but the
discipline is **single writer**: run jobs while the app is idle. Routing all
writes through the API is the fully-correct version and a natural cleanup when
porting to k8s.
 
---
 
## Prerequisites
 
- Docker + Docker Compose v2
- (Optional) a GitHub token for Dockerfile fetching — see `.env.example`
```bash
cp .env.example .env      # optional; set YNB_ANNOTATOR / GITHUB_TOKEN
```
 
---
 
## Quickstart A — see the UI in 60 seconds (synthetic data)
 
```bash
make demo          # builds images, seeds ~12 labelled demo builds, assigns splits, starts up
# UI:  http://localhost:8501
# API: http://localhost:8000/docs
```
 
`make demo` is `docker compose build` → `run --rm ingest demo-seed` →
`run --rm splits` → `up`.
 
---
 
## Quickstart B — your real corpus
 
### 1. Prepare `./data/builds.csv`
 
One row per build, with the repo URL and the **Dockerfile path within the repo**
already known — that path is used to fetch the exact Dockerfile later, no
guessing. A truncated **log** and a **tool name** are the only required fields;
everything else is optional. Header names are resolved against common aliases
(first match wins); edit `DEFAULT_MAP` in `src/ynbtriage/ingest.py` to add your
own:
 
| Field             | Accepted headers (first match wins)                                    |
|-------------------|------------------------------------------------------------------------|
| `external_id`     | id, build_id, external_id, uuid, run_id, **k8s_job_name**, job_name     |
| `tool_name` *req* | tool, tool_name, name, package, recipe                                  |
| `tool_version`    | version, tool_version, tag, image_tag                                   |
| `image_ref`       | image, image_ref, image_name, container, result_repo                   |
| `source_repo_url` | repo, **repo_url**, source_repo_url, url, git_url, homepage, source     |
| `dockerfile_path` | **dockerfile**, dockerfile_path, df_path, containerfile                 |
| `dockerfile_content` | dockerfile_content, dockerfile_text, dockerfile_body                 |
| `build_context`   | **context**, build_context, docker_context                             |
| `log_tail` *req*  | log, log_tail, build_log, truncated_log, logs, output, log_excerpt      |
| `log_lines_src`   | **log_lines**, log_line_count, n_lines, line_count                      |
| `build_status`    | status, build_status, outcome, result, result_status                   |
| `built_at`        | built_at, **created_at**, timestamp, date, build_date, time, started_at |
| `source_batch`    | batch, source_batch, run, socr8s_run, build_type                       |
| `label_leaf`      | label, leaf_id, leaf, predefined_label, predicted_label, gold_label     |
 
This matches a socr8s build-results export joined to a logs file: the export
supplies `repo_url`, `dockerfile`, `context`, `tool`, `created_at`, and
`k8s_job_name`; the logs file supplies `log` and `log_lines`. Join the two on
`(tool, tool_idx)` — that pair is unique in the build-results export and every
log row matches one. Ready-to-run examples are in
[`data/builds.example.csv`](data/builds.example.csv) (real repos, fetchable
Dockerfiles, predefined labels) and
[`data/builds.with-dockerfile.example.csv`](data/builds.with-dockerfile.example.csv)
(Dockerfile bodies supplied inline).
 
**Dockerfiles: fetch or inline.** Normally you give the repo URL and the
Dockerfile path, and `fetch` pulls that exact file. Alternatively, if you already
have the Dockerfile text, put it in a `dockerfile_content` column: it's stored
directly, marked `provided`, and `fetch` skips that build. You can mix both in one
CSV — rows with inline content are left alone, rows without it are fetched.
 
**Logs: escaping, truncation, empties.** Captured logs often arrive with newlines
escaped as literal `\n` and may carry an upstream banner like
`[... truncated 27026/28026 lines, showing last 1000 ...]`. Ingest un-escapes the
newlines (so the text renders and searches correctly), recovers the true
pre-truncation line count from that banner or the `log_lines` column, and marks
the build truncated. A final safety cap (`YNB_LOG_MAX_BYTES`, default 256 KB)
trims pathological giants to their failing tail. Rows whose log is a `[no logs]`
sentinel are ingested but carry no signal to annotate from.
 
**Identifiers.** If an id column is present and its values are unique across the
file, it's used as-is. Otherwise a stable id is synthesized from
`repo_url + dockerfile_path + built_at + tool`, so re-ingesting the same CSV
updates rows in place instead of duplicating them.
 
**Predefined labels (optional).** If a row carries a leaf-level label (a `label`
column with a value like `auth_required`), ingest records it as a **`prefilled`**
annotation — a suggestion, not confirmed ground truth. Prefilled builds show up
in the Annotate queue with the suggested class/leaf pre-selected (one click to
Confirm) and can be filtered in Browse via *Annotated → prefilled*. They are
**excluded from splits and the gold set until confirmed**, and never counted as
`annotated` in stats. Unknown leaf values are reported and ignored (never fatal);
an existing human or confirmed annotation is never overwritten; re-importing the
same label is a no-op. `load-csv` prints a `labeled N` count of prefills written.
 
### 2. Build and populate the DB
 
```bash
docker compose build
 
# ingest the CSV (creates schema + seeds taxonomy on first run)
docker compose run --rm ingest                    # runs: load-csv /data-in/builds.csv
 
# fetch the exact Dockerfile for each build (uses the known path from the CSV)
docker compose run --rm fetch                      # needs network; GITHUB_TOKEN raises limits
 
# assign train/val/test/gold over the labelled builds
docker compose run --rm splits
```
 
Check what landed:
 
```bash
docker compose run --rm ingest stats
```
 
### 3. Start the app
 
```bash
docker compose up -d api web
# UI: http://localhost:8501
```
 
---
 
## The two screens
 
**Browse.** Filter by class, leaf, annotation state, split, or free-text search
over tool name and log. Open a build to see the Dockerfile and the log tail side
by side, with the first failing region surfaced automatically.
 
**Annotate.** Walk the unannotated queue. Pick a coarse class; the leaves for
that class are revealed; pick one. Confirm, **Defer** (low confidence — kept out
of training later), or **Skip**. Builds imported with a predefined label arrive
here as **prefilled** suggestions with the class/leaf pre-selected — review and
Confirm (or change) rather than starting from scratch. Every action appends to an
immutable annotation history; the latest row per build wins. This is where the
hand-curated **gold set** comes from, and the gold split never moves once
assigned.
 
---
 
## Populating the DB — reference
 
| Command                                              | Effect                                             |
|------------------------------------------------------|----------------------------------------------------|
| `docker compose run --rm ingest`                     | Load `data/builds.csv` → `builds`                  |
| `docker compose run --rm ingest demo-seed`           | Insert synthetic labelled builds                   |
| `docker compose run --rm fetch`                      | Fetch each build's Dockerfile by its known path    |
| `docker compose run --rm splits`                     | Assign train/val/test/gold (stratified, seeded)    |
| `docker compose run --rm ingest stats`               | Print corpus statistics                            |
| `docker compose run --rm ingest delete-build <id>`   | Delete one build + its annotation history          |
| `docker compose run --rm ingest prune-no-logs`       | Delete all `[no logs]` sentinel builds             |
 
The `initdb` step is implicit: every command and the API itself run
`ensure_schema` + `seed_taxonomy` on start, so ordering never breaks a fresh
volume.
 
---
 
## Deleting builds
 
Deletion is permanent and drops the build's entire annotation history. It's the
right tool for "this build shouldn't be in the corpus at all" — to reverse a
*label*, don't delete; just re-annotate (latest-per-build wins). Re-ingesting the
source CSV re-creates a deleted build, since the upsert keys on `external_id`.
 
Foreign keys are enforced with no cascade, so the tooling removes the
`annotation` and `dataset_split` rows before the build, in one transaction. Two
ways in:
 
**UI.** Open a build in Browse → *⚠ Danger zone* → tick the confirm box → Delete.
Bulk-remove `[no logs]` builds from the sidebar *Maintenance* → Scan → confirm.
 
**CLI** (jobs allocate a TTY, so the confirm prompt works; pass `--yes` for
non-interactive use):
 
```bash
docker compose run --rm ingest delete-build 42            # by build_id
docker compose run --rm ingest delete-build --external-id build-6a14…   # by external_id
docker compose run --rm ingest prune-no-logs --dry-run    # count sentinels, delete nothing
docker compose run --rm ingest prune-no-logs --yes        # delete them
```
 
Run deletions while the app is idle — the API is the DB's single writer. Deleting
via `docker compose run` opens the DB from a second process; WAL + the busy
timeout usually absorb it, but `docker compose stop api web` first is the clean
path. `make clean` (`docker compose down -v`) wipes the whole DB volume instead.
 
---
 
## Schema (essentials)
 
- **`builds`**: One row per build (tool, repo, `dockerfile_path`,
  `build_context`, log tail, and the fetched Dockerfile body filled by `fetch`).
- **`taxonomy_class` / `taxonomy_leaf`**: Seeded from the YAML; the leaf carries
  `rule_prefilled` and `caveat` flags.
- **`annotation`**: Append-only. `current_annotation` view = latest per build.
- **`dataset_split`**: The split of train/val/test/gold; gold is sticky.
Inspect it:
 
```bash
make shell         # opens a python sqlite3 REPL with `con` bound to the DB
```
 
---
 
## Ports
 
| Service | URL                          |
|---------|------------------------------|
| web     | http://localhost:8501        |
| api     | http://localhost:8000/docs   |
 
---
 
## Path to Kubernetes
 
Nothing here blocks the port. `web` and `api` are stateless HTTP services →
Deployments + Services. `ingest` / `fetch` / `splits` → `Job`s (or `fetch` and a
periodic rebuild as `CronJob`s). The one snag is the shared `dbdata` volume:
SQLite on a `ReadWriteMany` PVC works but is the weakest link. When it starts to
hurt, swap the `db.py` connection for Postgres — the repository layer is the only
thing that touches SQL, so the blast radius is one module.
 
---
 
## Troubleshooting
 
- **`database is locked`** — a job and the API wrote at the same time. Bring the
  app down (`docker compose stop api web`), run the job, bring it back up. WAL +
  busy-timeout usually absorb this, but heavy concurrent writes will still trip.
- **UI says "cannot reach the API"** — `docker compose ps`; wait for `api` to be
  healthy (it self-migrates on first boot), then reload.
- **Dockerfile fetch returns `not_found`** — the path in the CSV didn't resolve
  on the repo's default / `main` / `master` branch (repo moved, renamed, or the
  path is stale). The fetcher falls back to a few conventional paths; add more to
  `CANDIDATE_PATHS` in `src/ynbtriage/fetch.py` if needed.
- **Rebuild after code changes** — `docker compose build <service>` then `up -d`.
---
 
## Running without Docker (optional, for quick iteration)
 
```bash
pip install -e ".[api]"
export YNB_DB_PATH=./ynobuild.db
ynbtriage demo-seed && ynbtriage splits
uvicorn ynbtriage.api:app --reload --port 8000 &
YNB_API_URL=http://localhost:8000 streamlit run web/app.py
```