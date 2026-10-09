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


# ---------------------------------------------------------------- model service
# A separate service (ynbtriage.serve). The UI must keep working when it is down,
# so these never raise on connection failure where a status check is enough.

MODEL_URL = os.environ.get("YNB_MODEL_URL", "http://model:8001")


def _model_client():
    return httpx.Client(base_url=MODEL_URL, timeout=60)


def model_status():
    """{'status', 'model_loaded', 'error'} from the model service, or None if unreachable."""
    try:
        with _model_client() as c:
            return c.get("/health").json()
    except Exception:
        return None


def model_info():
    with _model_client() as c:
        r = c.get("/model")
        if r.status_code >= 400:
            raise RuntimeError(r.json().get("detail", r.text))
        return r.json()


def predict_log(log_text):
    with _model_client() as c:
        r = c.post("/predict", json={"log": log_text})
        if r.status_code >= 400:
            raise RuntimeError(r.json().get("detail", r.text))
        return r.json()


# ---------------------------------------------------------------- training (model service)

def _model_call(method, path, **kw):
    with _model_client() as c:
        r = c.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RuntimeError(detail if isinstance(detail, str) else str(detail))
        return r.json()


def list_runs():
    return _model_call("GET", "/runs")


def train_start(params):
    return _model_call("POST", "/train", json=params)


def train_status(job_id):
    return _model_call("GET", f"/train/{job_id}")


def train_latest():
    return _model_call("GET", "/train/latest")


def train_cancel(job_id):
    return _model_call("POST", f"/train/{job_id}/cancel")


def deploy_run(name):
    return _model_call("POST", "/deploy", json={"run": name})


def delete_run(name):
    return _model_call("DELETE", f"/runs/{name}")


def prune_no_logs(dry_run=False):
    with _client() as c:
        r = c.post("/maintenance/prune-no-logs", params={"dry_run": dry_run})
        if r.status_code >= 400:
            raise RuntimeError(r.text)
        return r.json()