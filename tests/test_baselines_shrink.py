"""Tests for the Naive Bayes baseline, the baselines-only command, and featurizer shrinking.
Run: pytest -q (needs the [api,model] extras)."""
import json

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from ynbtriage.model.pipeline import run
    from ynbtriage.model.synth import synth_db
    from ynbtriage.model.train import TrainConfig

    tmp = tmp_path_factory.mktemp("bs")
    synth_db(tmp / "synth.db", n=300)
    out = tmp / "final"
    s = run(out, db_path=tmp / "synth.db", features="tfidf", cfg=TrainConfig(epochs=10, patience=0))
    return tmp / "synth.db", out, s


def test_training_reports_naive_bayes(trained):
    _, _, s = trained
    assert {"majority", "naive_bayes", "tfidf_logreg"} <= set(s["baselines"])


def test_baselines_command_matches_training(trained):
    from ynbtriage.model.pipeline import run_baselines

    db, out, s = trained
    b = run_baselines(out, db_path=db)
    saved = json.loads((out / "baselines.json").read_text())
    assert saved["data_fingerprint"] == s["data_fingerprint"]
    for name in ("majority", "naive_bayes", "tfidf_logreg"):
        assert b["baselines"][name]["test"]["macro_f1"] == pytest.approx(
            s["baselines"][name]["test"]["macro_f1"])


def test_new_runs_store_float32_projection(trained):
    import joblib
    import numpy as np

    _, out, _ = trained
    feat = joblib.load(out / "featurizer.joblib")
    assert feat.svd.components_.dtype == np.float32


def test_shrink_model_keeps_predictions(tmp_path, trained):
    import shutil

    import joblib
    import numpy as np
    from typer.testing import CliRunner

    from ynbtriage.cli import app
    from ynbtriage.model.predict import Predictor

    db, out, _ = trained
    old = tmp_path / "old"
    shutil.copytree(out, old)
    # make it look like a run saved before this change: float64, uncompressed
    feat = joblib.load(old / "featurizer.joblib")
    feat.svd.components_ = feat.svd.components_.astype(np.float64)
    joblib.dump(feat, old / "featurizer.joblib")
    before_size = (old / "featurizer.joblib").stat().st_size
    logs = ["E: The repository 'http://deb.debian.org/debian buster Release' does not have a Release file.",
            "curl: (28) Connection timed out after 30001 milliseconds"]
    before = Predictor.load(old).predict_many(logs)

    r = CliRunner().invoke(app, ["shrink-model", str(old), "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert "label changes 0" in r.output
    assert (old / "featurizer.joblib").stat().st_size < before_size
    after = Predictor.load(old).predict_many(logs)
    assert [x["label"] for x in before] == [x["label"] for x in after]


def test_service_shows_baselines_json(trained, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ynbtriage import serve
    from ynbtriage.model.pipeline import run_baselines

    db, out, _ = trained
    run_baselines(out, db_path=db)
    monkeypatch.setenv("YNB_MODEL_DIR", str(out))
    with TestClient(serve.app) as c:
        scores = c.get("/model").json()["scores"]
    assert {"mlp", "majority", "naive_bayes", "tfidf_logreg"} <= set(scores)
