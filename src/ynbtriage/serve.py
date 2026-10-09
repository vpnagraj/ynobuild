"""Model-serving API: classify a build log with the trained MLP.

A separate service from `api.py`. It never opens the SQLite database: it loads
one trained model directory at startup and answers prediction requests. That
keeps PyTorch out of the data API and the web image, and keeps `api` the only
process that touches the DB.

    uvicorn ynbtriage.serve:app --host 0.0.0.0 --port 8001

Which model is served at startup comes from YNB_MODEL_DIR (default
/models/final), a directory written by `ynbtriage train`. That startup model is
the "default run": it can never be deleted, and a restart always goes back to it.

Training (the Train screen): POST /train starts a background job that fetches
the labelled data from the api service (GET /training-data; this service still
never opens the DB), trains with the requested hyperparameters, and writes a new
run directory next to the default one (models/ui-<timestamp>). POST /deploy
swaps the served model to any run, in memory; nothing is overwritten.

If the directory is missing or unreadable the service still starts: /health
reports model_loaded=false and /predict returns 503, so the web app can say so
instead of the whole stack failing.
"""
from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import shutil

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .config import LOG_MAX_BYTES
from .jobs import Busy, JobManager
from .model.excerpt import ERROR_PATTERNS

_ERR_RE = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)

# The loaded model lives here for the life of the process.
STATE: dict = {"predictor": None, "info": None, "error": None}


JOBS = JobManager()
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def model_dir() -> Path:
    """The default run, served at startup."""
    return Path(os.environ.get("YNB_MODEL_DIR", "/models/final"))


def models_root() -> Path:
    """Folder holding all run directories (the default run's parent)."""
    return model_dir().parent


def api_url() -> str:
    return os.environ.get("YNB_API_URL", "http://api:8000")


def fetch_training_data(target: str = "class") -> dict:
    """Labelled train/val/test excerpts from the api service (never gold)."""
    import httpx

    r = httpx.get(f"{api_url()}/training-data", params={"target": target}, timeout=120)
    r.raise_for_status()
    return r.json()


def _model_info(d: Path, predictor) -> dict:
    """Describe the loaded model, using metrics.json written by `train`."""
    info = {
        "name": d.name,
        "model_dir": str(d),
        "target": predictor.target,
        "labels": predictor.labels,
        "trained_at": datetime.fromtimestamp((d / "model.pt").stat().st_mtime, timezone.utc)
                              .isoformat(timespec="seconds"),
        "architecture": predictor.model.config,
    }
    mpath = d / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        info["features"] = m.get("features")
        info["n"] = m.get("n")
        info["epochs_run"], info["best_epoch"] = m.get("epochs_run"), m.get("best_epoch")
        info["scores"] = {
            "mlp": {s: m["mlp"][s] for s in ("val", "test") if s in m.get("mlp", {})},
            **{name: b for name, b in m.get("baselines", {}).items()},
        }
        # keep only the headline numbers; per-class detail stays in metrics.json
        for model_scores in info["scores"].values():
            for s, v in model_scores.items():
                model_scores[s] = {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
    return info


def _load(d: Path):
    """Load a run directory; raises if it is missing or unreadable."""
    from .model.predict import Predictor  # imports torch; only this service needs it

    p = Predictor.load(d)
    return p, _model_info(d, p)


def load_model(d: Path | str | None = None) -> None:
    """Startup load. On failure, record why and serve no model."""
    d = Path(d) if d else model_dir()
    try:
        p, info = _load(d)
        STATE.update(predictor=p, info=info, error=None)
    except Exception as e:  # noqa: BLE001 - report any load failure, don't crash
        STATE.update(predictor=None, info=None, error=f"{type(e).__name__}: {e} ({d})")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(title="ynobuild model service", version="0.1.0", lifespan=lifespan)


class PredictIn(BaseModel):
    log: str


def _tail_bytes(text: str, limit: int) -> tuple[str, bool]:
    """Keep the last `limit` bytes, where the failure usually is (same rule as ingest)."""
    raw = text.encode("utf-8", errors="replace")
    if limit and len(raw) > limit:
        return raw[-limit:].decode("utf-8", errors="ignore"), True
    return text, False


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE["predictor"] is not None,
            "error": STATE["error"]}


@app.get("/model")
def model():
    if STATE["predictor"] is None:
        raise HTTPException(503, f"no model loaded: {STATE['error']}")
    return {**STATE["info"], "default_run": model_dir().name}


@app.post("/predict")
def predict(body: PredictIn):
    p = STATE["predictor"]
    if p is None:
        raise HTTPException(503, f"no model loaded: {STATE['error']}")
    if not body.log.strip():
        raise HTTPException(422, "log is empty")
    log, truncated = _tail_bytes(body.log, LOG_MAX_BYTES)
    result = p.predict(log)
    result["error_line_found"] = any(_ERR_RE.search(ln) for ln in log.splitlines())
    result["input_truncated"] = truncated
    result["model"] = STATE["info"]["name"]
    return result


# ====================================================================== training

class TrainIn(BaseModel):
    """Hyperparameters the Train screen exposes. Everything else uses TrainConfig
    defaults (Adam optimizer, batch size 32, early-stopping patience 25, seed 7400)."""
    hidden: list[int] = Field(default=[64], min_length=1, max_length=4)
    dropout: float = Field(default=0.2, ge=0.0, le=0.8)
    lr: float = Field(default=1e-3, gt=0.0, le=1.0)
    epochs: int = Field(default=200, ge=1, le=2000)
    features: str = Field(default="tfidf", pattern="^(tfidf|embed|both)$")
    activation: str = Field(default="relu", pattern="^(relu|tanh|gelu|sigmoid)$")

    @field_validator("hidden")
    @classmethod
    def _widths(cls, v):
        if any(w < 2 or w > 2048 for w in v):
            raise ValueError("each hidden layer needs between 2 and 2048 units")
        return v


class DeployIn(BaseModel):
    run: str


def _run_dir(name: str) -> Path:
    """Resolve a run name to a directory inside the models folder (no path tricks)."""
    if not RUN_NAME_RE.match(name):
        raise HTTPException(400, f"invalid run name: {name!r}")
    d = models_root() / name
    if d.parent.resolve() != models_root().resolve():
        raise HTTPException(400, f"invalid run name: {name!r}")
    if not (d / "model.pt").exists():
        raise HTTPException(404, f"no trained model in run {name!r}")
    return d


def _headline(scores: dict | None) -> dict | None:
    return {"accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"]} if scores else None


@app.post("/train", status_code=202)
def train(body: TrainIn):
    root = models_root()
    if not os.access(root, os.W_OK):
        raise HTTPException(500, f"the models folder {root} is not writable by the model service")
    try:
        job = JOBS.start(body.model_dump(), root, lambda t: fetch_training_data(t))
    except Busy as e:
        raise HTTPException(409, f"a training job is already running ({e})")
    return job.public()


@app.get("/train/latest")
def train_latest():
    """The most recent job (running or finished), or null. Lets the UI re-attach after a reload."""
    return JOBS.latest.public() if JOBS.latest else None


@app.get("/train/{job_id}")
def train_status(job_id: str):
    job = JOBS.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job.public()


@app.post("/train/{job_id}/cancel")
def train_cancel(job_id: str):
    job = JOBS.cancel(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job.public()


@app.get("/runs")
def runs():
    """Every run in the models folder, with its settings and headline scores."""
    served = STATE["info"]["name"] if STATE["info"] else None
    default = model_dir().name
    running = JOBS.running()
    out = []
    root = models_root()
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if (p / "model.pt").exists()):
        mpath = d / "metrics.json"
        m = json.loads(mpath.read_text()) if mpath.exists() else {}
        cfg = m.get("train_config", {})
        mlp = m.get("mlp", {})
        out.append({
            "name": d.name,
            "source": m.get("source", "cli"),
            "created": datetime.fromtimestamp((d / "model.pt").stat().st_mtime, timezone.utc)
                               .isoformat(timespec="seconds"),
            "served": d.name == served,
            "default": d.name == default,
            "deletable": d.name.startswith("ui-") and d.name not in (served, default)
                         and not (running and running.run == d.name),
            "features": m.get("features"),
            "hidden": cfg.get("hidden"),
            "activation": cfg.get("activation"),
            "dropout": cfg.get("dropout"),
            "lr": cfg.get("lr"),
            "epochs": cfg.get("epochs"),
            "epochs_run": m.get("epochs_run"),
            "best_epoch": m.get("best_epoch"),
            "val": _headline(mlp.get("val")),
            "test": _headline(mlp.get("test")),
            "tfidf_logreg": {s: _headline(m.get("baselines", {}).get("tfidf_logreg", {}).get(s))
                             for s in ("val", "test")},
            "data_fingerprint": m.get("data_fingerprint"),
        })
    return out


@app.post("/deploy")
def deploy(body: DeployIn):
    """Serve a different run. The swap happens only after the new run loads cleanly."""
    d = _run_dir(body.run)
    try:
        p, info = _load(d)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not load run {body.run!r}: {type(e).__name__}: {e}")
    STATE.update(predictor=p, info=info, error=None)
    return {**info, "default_run": model_dir().name}


@app.delete("/runs/{name}")
def delete_run(name: str):
    """Delete a run made from the Train screen. The default run and the served run are protected."""
    d = _run_dir(name)
    served = STATE["info"]["name"] if STATE["info"] else None
    running = JOBS.running()
    if name == model_dir().name:
        raise HTTPException(409, f"{name!r} is the default run and cannot be deleted")
    if name == served:
        raise HTTPException(409, f"{name!r} is being served; deploy another run first")
    if running and running.run == name:
        raise HTTPException(409, f"{name!r} is still training")
    if not name.startswith("ui-"):
        raise HTTPException(409, f"{name!r} was not created from the Train screen; "
                                 "delete it by hand if you really mean to")
    shutil.rmtree(d)
    return {"deleted": name}
