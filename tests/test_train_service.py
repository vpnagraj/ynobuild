"""Tests for training from the model service, and the api's /training-data endpoint.
Run: pytest -q (needs the [api,model] extras)."""
import time

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def synth_db(tmp_path_factory):
    from ynbtriage.model.synth import synth_db as make

    db = tmp_path_factory.mktemp("db") / "synth.db"
    make(db, n=300)
    return db


@pytest.fixture()
def service(synth_db, tmp_path, monkeypatch):
    """Model service with a 'final' default run and data served straight from the synth DB."""
    from ynbtriage import serve
    from ynbtriage.jobs import JobManager
    from ynbtriage.model.data import label_order, load_dataset
    from ynbtriage.model.pipeline import run
    from ynbtriage.model.train import TrainConfig

    models = tmp_path / "models"
    run(models / "final", db_path=synth_db, features="tfidf",
        cfg=TrainConfig(epochs=10, patience=0))

    def fetch(target="class"):
        df = load_dataset(synth_db, target=target)
        return {"target": target, "labels": label_order(synth_db, target),
                "rows": df.to_dict(orient="records")}

    monkeypatch.setenv("YNB_MODEL_DIR", str(models / "final"))
    monkeypatch.setattr(serve, "fetch_training_data", fetch)
    monkeypatch.setattr(serve, "JOBS", JobManager())
    with TestClient(serve.app) as c:
        yield c, models


def _wait(c, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/train/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("training did not finish")


def test_train_deploy_switch_back_delete(service):
    c, models = service
    r = c.post("/train", json={"hidden": [32, 16], "dropout": 0.1, "lr": 3e-3, "epochs": 15,
                               "features": "tfidf", "activation": "tanh"})
    assert r.status_code == 202
    job = _wait(c, r.json()["id"])
    assert job["status"] == "done", job["error"]
    assert job["history"] and job["result"]["mlp"]["val"]["macro_f1"] >= 0
    name = job["run"]
    assert name.startswith("ui-") and (models / name / "model.pt").exists()

    runs = {x["name"]: x for x in c.get("/runs").json()}
    assert runs[name]["hidden"] == [32, 16] and runs[name]["activation"] == "tanh"
    assert runs[name]["source"] == "ui" and runs[name]["deletable"] is True
    assert runs["final"]["served"] and runs["final"]["default"] and not runs["final"]["deletable"]
    assert runs[name]["data_fingerprint"]["hash"] == runs["final"]["data_fingerprint"]["hash"]

    # deploy the new run; predict now uses it; it can no longer be deleted
    assert c.post("/deploy", json={"run": name}).json()["name"] == name
    assert c.post("/predict", json={"log": "curl: (28) Connection timed out"}).json()["model"] == name
    assert c.delete(f"/runs/{name}").status_code == 409

    # switch back to the default run, then delete the new one
    m = c.post("/deploy", json={"run": "final"}).json()
    assert m["name"] == "final" and m["default_run"] == "final"
    assert c.delete(f"/runs/{name}").status_code == 200
    assert not (models / name).exists()


def test_default_run_is_protected(service):
    c, _ = service
    assert c.delete("/runs/final").status_code == 409
    assert c.delete("/runs/..").status_code in (400, 404)
    assert c.post("/deploy", json={"run": "does-not-exist"}).status_code == 404


def test_one_job_at_a_time_and_cancel(service):
    c, models = service
    first = c.post("/train", json={"epochs": 2000, "features": "tfidf"}).json()
    assert c.post("/train", json={"epochs": 5}).status_code == 409
    c.post(f"/train/{first['id']}/cancel")
    job = _wait(c, first["id"])
    assert job["status"] == "cancelled"
    assert not (models / first["run"]).exists()
    assert c.get("/train/latest").json()["id"] == first["id"]


def test_bad_hyperparameters_rejected(service):
    c, _ = service
    assert c.post("/train", json={"hidden": []}).status_code == 422
    assert c.post("/train", json={"hidden": [64, 64, 64, 64, 64]}).status_code == 422
    assert c.post("/train", json={"activation": "softmax"}).status_code == 422
    assert c.post("/train", json={"dropout": 0.95}).status_code == 422


def test_api_training_data_excludes_gold(synth_db, monkeypatch):
    import ynbtriage.db
    from ynbtriage import api

    monkeypatch.setattr(ynbtriage.db, "DB_PATH", synth_db)
    with TestClient(api.app) as c:
        body = c.get("/training-data", params={"target": "class"}).json()
    splits = {r["split"] for r in body["rows"]}
    assert splits == {"train", "val", "test"}
    assert body["labels"][0] == "environment_decay"
    assert all(r["text"] for r in body["rows"])
