"""Tests for the model-serving API. Run: pytest -q (needs the [api,model] extras)."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    from ynbtriage.model.pipeline import run
    from ynbtriage.model.synth import synth_db
    from ynbtriage.model.train import TrainConfig

    tmp = tmp_path_factory.mktemp("serve")
    synth_db(tmp / "synth.db", n=300)
    out = tmp / "final"
    run(out, db_path=tmp / "synth.db", features="tfidf", cfg=TrainConfig(epochs=20, patience=0))
    return out


def test_predict_with_model(trained_model, monkeypatch):
    from ynbtriage import serve

    monkeypatch.setenv("YNB_MODEL_DIR", str(trained_model))
    with TestClient(serve.app) as c:
        h = c.get("/health").json()
        assert h["model_loaded"] is True

        info = c.get("/model").json()
        assert info["name"] == "final" and info["target"] == "class"
        assert "mlp" in info["scores"] and "tfidf_logreg" in info["scores"]

        log = ("Step 3/9 : RUN apt-get update\n"
               "E: The repository 'http://deb.debian.org/debian buster Release' does not have a Release file.\n")
        r = c.post("/predict", json={"log": log})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in info["labels"]
        assert abs(sum(body["probabilities"].values()) - 1) < 1e-5
        assert "Release file" in body["excerpt"]
        assert body["error_line_found"] is True

        assert c.post("/predict", json={"log": "   "}).status_code == 422


def test_no_model_degrades_gracefully(tmp_path, monkeypatch):
    from ynbtriage import serve

    monkeypatch.setenv("YNB_MODEL_DIR", str(tmp_path / "does-not-exist"))
    with TestClient(serve.app) as c:
        h = c.get("/health")
        assert h.status_code == 200 and h.json()["model_loaded"] is False
        assert c.get("/model").status_code == 503
        assert c.post("/predict", json={"log": "error: x"}).status_code == 503


def test_long_log_is_tail_truncated(trained_model, monkeypatch):
    from ynbtriage import serve

    monkeypatch.setenv("YNB_MODEL_DIR", str(trained_model))
    monkeypatch.setattr(serve, "LOG_MAX_BYTES", 1000)
    with TestClient(serve.app) as c:
        log = "noise line\n" * 5000 + "curl: (28) Connection timed out after 30001 milliseconds\n"
        body = c.post("/predict", json={"log": log}).json()
        assert body["input_truncated"] is True
        assert "Connection timed out" in body["excerpt"]
