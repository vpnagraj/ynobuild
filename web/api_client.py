"""Thin HTTP client for the Streamlit UI. No SQLite here — the UI only speaks
HTTP to the API service, which keeps it decoupled and k8s-friendly.
"""
import os

import httpx

API_URL = os.environ.get("YNB_API_URL", "http://api:8000")


def _client():
    return httpx.Client(base_url=API_URL, timeout=30)


def health():
    try:
        with _client() as c:
            return c.get("/health").json().get("status") == "ok"
    except Exception:
        return False


def get_taxonomy():
    with _client() as c:
        return c.get("/taxonomy").json()


def get_stats():
    with _client() as c:
        return c.get("/stats").json()


def list_builds(**params):
    clean = {k: v for k, v in params.items() if v not in (None, "", "All")}
    with _client() as c:
        return c.get("/builds", params=clean).json()


def get_build(build_id):
    with _client() as c:
        r = c.get(f"/builds/{build_id}")
        return r.json() if r.status_code == 200 else None


def annotate(build_id, payload):
    with _client() as c:
        r = c.post(f"/builds/{build_id}/annotations", json=payload)
        if r.status_code >= 400:
            raise RuntimeError(r.text)
        return r.json()


def delete_build(build_id):
    with _client() as c:
        r = c.delete(f"/builds/{build_id}")
        if r.status_code >= 400:
            raise RuntimeError(r.text)
        return r.json()


def prune_no_logs(dry_run=False):
    with _client() as c:
        r = c.post("/maintenance/prune-no-logs", params={"dry_run": dry_run})
        if r.status_code >= 400:
            raise RuntimeError(r.text)
        return r.json()