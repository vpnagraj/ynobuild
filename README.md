# ynobuild

`ynobuild` provides tooling to browse and annotate container build failures using a set taxonomy. Backed by a database, the web app visualizes container build recipes (i.e., Dockerfiles) and build log file contents. Each build can be annotated with failure categories, which will serve as labeled data for downstream modeling. The goal is to eventually use logs as text data inputs to fine-tune a deep learning language model.

The failure label space is specified via [`taxonomy/taxonomy_map.yaml`](taxonomy/taxonomy_map.yaml) and seeded into the DB. Leaf labels are ground truth; the coarse class is derived from the leaf. There are three classes and twelve leaves overall.

## Overview

- **What it is**: A web app for reviewing failed container builds and labeling why each one failed. You browse the failures, read their logs and Dockerfiles, and assign each a failure category. Labeled log text data can serve downstream objectives to develop a deep learning classifier for build failures.

- **The data**: Records of failed container (Docker/Apptainer) builds. Each record has a build log and its spec file contents (e.g., Dockerfile) contents. The logs are the main thing being read and labeled.

- **Running it**: Requires Docker and Docker Compose. Run `docker compose build`, load your data with `docker compose run --rm ingest`, then `docker compose up -d api web` and open http://localhost:8501 (or `make demo` to start with a small synthetic dataset).

- **Main libraries**: Streamlit for the interface, FastAPI for the backend, and SQLite for storage, with pandas, Typer, and httpx for loading data, the command-line tools, and HTTP requests. Everything runs in Docker containers via Docker Compose.

- **Features**: Search and view records of build failure details, including a side-by-side Dockerfile and logs with the first error line highlighted. The ingest utility requires build logs, and can either accept Dockerfile text provided for each build or pull contents from GitHub/GitLab. The splits utility assigns train/validation/test/gold data. and lets you delete builds if needed.

- **Annotation**: Yes. Builds are labeled under a three-class, twelve-leaf taxonomy, and labels can be pre-loaded from the input file as suggestions you confirm.

- **Backend / database**:  Yes. A FastAPI service owns a SQLite database (delivered as Docker volume), and the Streamlit interface talks to it over HTTP instead of touching the database directly.

---
 
## Architecture

`ynobuild` is containerized with a web app (Streamlit), API (FastAPI), and database (SQLite) as a volume. The tool includes a CLI (called `ynbtriage`) for managing the corpus in the DB. These actions operate as jobs that run separately to populate data and generate data splits for downstream modeling:
 
```
                 ┌──────────────┐        HTTP        ┌──────────────┐
   browser  ───▶ │  web         │  ───────────────▶  │  api         │
   :8501         │  Streamlit   │                    │  FastAPI     │
                 └──────────────┘                    └──────┬───────┘
                                                            │ sqlite
                              ┌─────────────────────────────┴───────────┐
                              │            dbdata volume                 │
                              │            /data/ynobuild.db             │
                              └─────────────────────────────────────────┘
        jobs (share the volume, run offline via containers with the ynbtriage CLI):
        ingest ── load CSV      fetch ── pull Dockerfiles      splits ── train/val/test/gold
```

**NOTE**: The API owns the DB because SQLite is not run on a server, and therefore can't handle multiple container connections. The API acts as the gateway for all DB transactions, and the Streamlit web app never touches SQLite. However, the batch jobs open the DB directly for convenience. WAL mode combined with a 30 second busy-timeout make the occasional overlap wait rather than error.
 
---
 
## Setup
 
The stack requires Docker and Docker Compose (v2). Docker actions are managed Docker Compose, and the repo includes a `Makefile` for convenience. So `make` is optional,  but plain `docker compose` commands will work as well.

Optionally, set a GitHub token to raise rate limits when fetching Dockerfiles:
`cp .env.example .env`, then edit `.env` (see relevant variable names provided and commented out in example file).
 
---
 
## Getting started

### Demo

The `Makefile` includes a `demo` command that wraps the `docker compose build`, `ingest` and `splits` jobs, and `docker compose up` with a demo dataset. The UI and API are launched to run at `localhost`.
 
```bash
## build images, ingest and split demo data, and launch
## UI:  http://localhost:8501
## API: http://localhost:8000/docs
make demo
```
---
 
### Basic usage
 
#### 1. Prepare `data/builds.csv`
 
To ingest build failure data, prepare a `builds.csv` file with one row per tool build. Each tool must have a name and log. Other elements such as a unique ID, Dockerfile contents, tool repo URL and path to Dockerfile, and build metadata can be provided. The table below presents all fields in the default schema. Header names are resolved against common aliases (first match wins). Edit `DEFAULT_MAP` in `src/ynbtriage/ingest.py` to modify:
 
| Field             | Accepted headers (first match wins)                                    |
|-------------------|------------------------------------------------------------------------|
| `external_id`     | id, build_id, external_id, uuid, run_id, **k8s_job_name**, job_name     |
| `tool_name`       | tool, tool_name, name, package, recipe                                  |
| `tool_version`    | version, tool_version, tag, image_tag                                   |
| `image_ref`       | image, image_ref, image_name, container, result_repo                   |
| `source_repo_url` | repo, **repo_url**, source_repo_url, url, git_url, homepage, source     |
| `dockerfile_path` | **dockerfile**, dockerfile_path, df_path, containerfile                 |
| `dockerfile_content` | dockerfile_content, dockerfile_text, dockerfile_body                 |
| `build_context`   | **context**, build_context, docker_context                             |
| `log_tail`        | log, log_tail, build_log, truncated_log, logs, output, log_excerpt      |
| `log_lines_src`   | **log_lines**, log_line_count, n_lines, line_count                      |
| `build_status`    | status, build_status, outcome, result, result_status                   |
| `built_at`        | built_at, **created_at**, timestamp, date, build_date, time, started_at |
| `source_batch`    | batch, source_batch, run, socr8s_run, build_type                       |
| `label_leaf`      | label, leaf_id, leaf, predefined_label, predicted_label, gold_label     |
 
#### 2. Build and populate the DB
 
```bash
## build images
docker compose build
 
## ingest the builds.csv (creates schema and seeds taxonomy on first run)
## NOTE: this uses the app CLI to run load-csv on /data-in/builds.csv
docker compose run --rm ingest
 
## fetch the exact Dockerfile for each build (uses the known path from the CSV)
## setting the GITHUB_TOKEN can help cut down rate limits to GitHub API
docker compose run --rm fetch
 
## assign train/val/test/gold over the labeled builds
docker compose run --rm splits
```
 
To see what was populated:
 
```bash
docker compose run --rm ingest stats
```
 
#### 4. Start the app

To launch the app (with UI running at `http://localhost:8501`):

```bash
docker compose up -d api web
```

#### 5. Use the viewer

With the web app running, connect to a browser to use the viewer. The viewer features two tabs:

- **Browse.** Filter by failure class (and/or leaf), annotation state, split, or free-text search over tool name and log. Open a build to see the Dockerfile and the log tail side by side, with the first failing region presented above.
- **Annotate.** Look through the queue to add new annotations or confirm existing ones. You can view annotation labels at the class or leaf levels of the failure taxonomy. Use the Confirm/Defer/Skip buttons. Note that builds imported with a predefined label arrive here as prefilled suggestions with the class/leaf pre-selected.
 
---
 

## Managing the DB

### Populating records

The table below provides a reference for basic commands to manage the corpus of text in the DB:
 
| Command                                              | Effect                                             |
|------------------------------------------------------|----------------------------------------------------|
| `docker compose run --rm ingest`                     | Load `data/builds.csv` into `builds`                  |
| `docker compose run --rm ingest demo-seed`           | Insert synthetic labeled into builds                   |
| `docker compose run --rm fetch`                      | Fetch each build's Dockerfile by its known path    |
| `docker compose run --rm splits`                     | Assign train/val/test/gold    |
| `docker compose run --rm ingest stats`               | Print corpus statistics                            |
| `docker compose run --rm ingest delete-build <id>`   | Delete one build and its annotation history          |
| `docker compose run --rm ingest prune-no-logs`       | Bulk-delete builds whose log is the "[no logs]" placeholder             |
  
### Deleting records
 
Deletion is permanent. The delete feature should not be used to "reverse a label". In that case, re-annotate to overwrite an existing label. If you do delete, you can repopulate with the `builds.csv`, but the annotation history will be gone.

You can delete in the UI (Open a build in Browse → *⚠ Danger zone* → tick the confirm box → Delete) or from the CLI:
 
```bash
## by build id
docker compose run --rm ingest delete-build 42 
## by external id
docker compose run --rm ingest delete-build --external-id build-6a14
## any records with "no logs" provided
docker compose run --rm ingest prune-no-logs --dry-run
docker compose run --rm ingest prune-no-logs --yes
```
**NOTE**: You should only run delete commands in the CLI while the app is idle. 

To wipe the whole DB volume:

```bash
make clean
```
---
 
### Inspecting the schema

`Makefile` includes a command to inspect the DB schema via Python SQLite REPL with the connection bound to the DB:

```bash
make shell
```

The main components of the schema are **builds** (one row per build), **taxonomy_class**/**taxonomy_leaf** (build failure categories seeded from YAML spec), **annotation** (stores the annotation information as append-only), and **dataset_split** (the split of train/test/val/gold).
 
---

## Troubleshooting
 
- **Database is locked**: The DB can become locked when a job and the API wrote at the same time. Bring the app down (`docker compose stop api web`), run the job, bring it back up.
- **UI says "cannot reach the API"**: `docker compose ps` and wait for `api` to be healthy, then reload.
- **Dockerfile fetch returns `not_found`**: The path in the CSV didn't resolve on the repo's default / `main` / `master` branch. This could be because the repo moved, was renamed, or the path is stale. See paths fetched in `src/ynbtriage/fetch.py` and adjust `CANDIDATE_PATHS` in that module if needed.
- **Rebuild after code changes**: `docker compose build <service>` then `up -d`.

---

## Next steps

- **Moving to Kubernetes**: Currently `ynobuild` is packaged/delivered with Docker Compose. The stack could be ported to Kubernetes (`web` and `api` as Deployment/Services; `ingest`, `fetch`, `splits` as Jobs). The SQLite DB (currently in `dbdata` volume) would need to be reconsidered, but a PV/PVC may work. 