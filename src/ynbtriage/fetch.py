"""Enrich builds with Dockerfile contents fetched from GitHub / GitLab.

For each build with a repo URL and no Dockerfile body yet: if the build already
carries a known Dockerfile path (from the CSV), fetch exactly that path across
the likely branches; otherwise fall back to a handful of conventional paths.
Marks each build ok / not_found / error so the UI can render gracefully either way.

Set GITHUB_TOKEN to raise the anonymous rate limit (60 req/hr -> 5000 req/hr).
"""
from __future__ import annotations

import time
from urllib.parse import urlparse

import httpx

from .config import GITHUB_TOKEN
from .db import ensure_schema, get_conn

CANDIDATE_PATHS = [
    "Dockerfile",
    "docker/Dockerfile",
    ".docker/Dockerfile",
    "Dockerfile.build",
    "build/Dockerfile",
    "containers/Dockerfile",
    "container/Dockerfile",
]


def parse_repo(url: str):
    p = urlparse(url)
    parts = [x for x in p.path.split("/") if x]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    host = p.netloc.lower()
    forge = "github" if "github" in host else "gitlab" if "gitlab" in host else "other"
    return forge, owner, repo, host


def _github_default_branch(client, owner, repo):
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    r = client.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers, timeout=20)
    if r.status_code == 200:
        return r.json().get("default_branch")
    return None


def _paths_for(known_path):
    """Prefer the known path; fall back to conventional ones (de-duplicated)."""
    if known_path:
        return [known_path] + [p for p in CANDIDATE_PATHS if p != known_path]
    return CANDIDATE_PATHS


def _try_raw(client, base_url_fn, branches, paths):
    for br in branches:
        for path in paths:
            r = client.get(base_url_fn(br, path), timeout=20)
            if r.status_code == 200 and r.text.strip():
                return r.text, path
    return None, None


def try_github(client, owner, repo, known_path=None):
    default = _github_default_branch(client, owner, repo) or "main"
    branches = [default] + [b for b in ("main", "master") if b != default]
    return _try_raw(
        client,
        lambda br, path: f"https://raw.githubusercontent.com/{owner}/{repo}/{br}/{path}",
        branches,
        _paths_for(known_path),
    )


def try_gitlab(client, host, owner, repo, known_path=None):
    return _try_raw(
        client,
        lambda br, path: f"https://{host}/{owner}/{repo}/-/raw/{br}/{path}",
        ("main", "master"),
        _paths_for(known_path),
    )


def fetch_dockerfiles(db_path=None, limit=None, sleep=0.5):
    ok = miss = err = 0
    with get_conn(db_path) as conn:
        ensure_schema(conn)
        rows = conn.execute(
            "SELECT build_id, source_repo_url, dockerfile_path FROM builds "
            "WHERE dockerfile_content IS NULL AND source_repo_url IS NOT NULL "
            "ORDER BY build_id"
        ).fetchall()
        if limit:
            rows = rows[:limit]

        with httpx.Client(follow_redirects=True, headers={"User-Agent": "ynbtriage/0.1"}) as client:
            for row in rows:
                bid = row["build_id"]
                parsed = parse_repo(row["source_repo_url"])
                if not parsed:
                    conn.execute("UPDATE builds SET dockerfile_fetch_status='error' WHERE build_id=?", (bid,))
                    err += 1
                    conn.commit()
                    continue
                forge, owner, repo, host = parsed
                known_path = row["dockerfile_path"]
                try:
                    if forge == "github":
                        content, path = try_github(client, owner, repo, known_path)
                    elif forge == "gitlab":
                        content, path = try_gitlab(client, host, owner, repo, known_path)
                    else:
                        content, path = (None, None)

                    if content:
                        conn.execute(
                            "UPDATE builds SET dockerfile_content=?, dockerfile_path=?, "
                            "dockerfile_fetch_status='ok', dockerfile_fetched_at=datetime('now') "
                            "WHERE build_id=?",
                            (content, path, bid),
                        )
                        ok += 1
                    else:
                        conn.execute(
                            "UPDATE builds SET dockerfile_fetch_status='not_found', "
                            "dockerfile_fetched_at=datetime('now') WHERE build_id=?",
                            (bid,),
                        )
                        miss += 1
                except Exception:
                    conn.execute("UPDATE builds SET dockerfile_fetch_status='error' WHERE build_id=?", (bid,))
                    err += 1
                conn.commit()
                time.sleep(sleep)
    return ok, miss, err
