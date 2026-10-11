"""Background training jobs for the model service.

One job at a time. A job:
  1. fetches the labelled dataset (from the api service; see serve.py),
  2. trains with model.pipeline.run_df, reporting every epoch back to the job,
  3. writes a new run directory under the models folder (never overwrites one).

Job state lives in memory. Finished runs are on disk, so nothing important is
lost if the service restarts; only the progress view of an in-flight job is.
"""
from __future__ import annotations

import shutil
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd


class Cancelled(Exception):
    """Raised from the per-epoch hook to stop training early at the user's request."""


class Busy(Exception):
    """A training job is already running."""


@dataclass
class Job:
    id: str
    run: str                       # name of the run directory this job writes
    params: dict
    status: str = "queued"         # queued | loading_data | training | done | failed | cancelled
    epochs: int = 0                # planned maximum (early stopping may end sooner)
    history: list = field(default_factory=list)
    summary: dict | None = None    # metrics.json content when done
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    finished_at: str | None = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        s = self.summary or {}
        return {
            "id": self.id, "run": self.run, "params": self.params, "status": self.status,
            "epochs": self.epochs, "history": self.history, "error": self.error,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "result": None if not s else {
                "mlp": {k: s["mlp"][k] for k in ("val", "test")},
                "baselines": s.get("baselines", {}),
                "epochs_run": s.get("epochs_run"), "best_epoch": s.get("best_epoch"),
                "data_fingerprint": s.get("data_fingerprint"),
            },
        }


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.latest: Job | None = None

    def running(self) -> Job | None:
        j = self.latest
        return j if j and j.status in ("queued", "loading_data", "training") else None

    def start(self, params: dict, models_root: Path,
              fetch_data: Callable[[str], dict]) -> Job:
        with self._lock:
            if self.running():
                raise Busy(self.latest.id)
            name = "ui-" + datetime.now().strftime("%Y%m%d-%H%M%S")
            while (models_root / name).exists():          # two runs in the same second
                name += "x"
            job = Job(id=uuid.uuid4().hex[:12], run=name, params=params,
                      epochs=int(params["epochs"]))
            self.jobs[job.id] = job
            self.latest = job
        threading.Thread(target=self._work, args=(job, models_root, fetch_data),
                         daemon=True, name=f"train-{job.id}").start()
        return job

    def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job:
            job._cancel.set()
        return job

    # ------------------------------------------------------------------ worker
    def _work(self, job: Job, models_root: Path, fetch_data):
        from .model.pipeline import run_df
        from .model.train import TrainConfig

        out = models_root / job.run
        try:
            job.status = "loading_data"
            payload = fetch_data("class")
            df = pd.DataFrame(payload["rows"])
            if df.empty:
                raise ValueError("no labelled train/val/test builds returned by the api")

            p = job.params
            cfg = TrainConfig(hidden=tuple(p["hidden"]), activation=p["activation"],
                              dropout=p["dropout"], lr=p["lr"], epochs=p["epochs"])

            def on_epoch(rec):
                job.history.append(rec)
                if job._cancel.is_set():
                    raise Cancelled()

            job.status = "training"
            job.summary = run_df(out, df, payload["labels"], target="class",
                                 features=p["features"], cfg=cfg, on_epoch=on_epoch,
                                 extra={"source": "ui", "job_id": job.id})
            job.status = "done"
        except Cancelled:
            job.status = "cancelled"
            shutil.rmtree(out, ignore_errors=True)
        except BaseException as e:  # noqa: BLE001 - report everything, never kill the service
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            shutil.rmtree(out, ignore_errors=True)
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
