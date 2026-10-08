"""Model-serving API: classify a build log with the trained MLP.

A separate service from `api.py`. It never opens the SQLite database: it loads
one trained model directory at startup and answers prediction requests. That
keeps PyTorch out of the data API and the web image, and keeps `api` the only
process that touches the DB.

    uvicorn ynbtriage.serve:app --host 0.0.0.0 --port 8001

Which model is served comes from YNB_MODEL_DIR (default /models/final), a
directory written by `ynbtriage train`. To deploy a better model, point
YNB_MODEL_DIR at the new run and restart the service.

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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import LOG_MAX_BYTES
from .model.excerpt import ERROR_PATTERNS

_ERR_RE = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)

# The loaded model lives here for the life of the process.
STATE: dict = {"predictor": None, "info": None, "error": None}


def model_dir() -> Path:
    return Path(os.environ.get("YNB_MODEL_DIR", "/models/final"))


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


def load_model(d: Path | str | None = None) -> None:
    from .model.predict import Predictor  # imports torch; only this service needs it

    d = Path(d) if d else model_dir()
    try:
        p = Predictor.load(d)
        STATE.update(predictor=p, info=_model_info(d, p), error=None)
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
    return STATE["info"]


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
