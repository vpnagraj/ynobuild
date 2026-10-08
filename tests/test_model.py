"""Smoke tests for the modeling package. Run: pytest -q (needs the [model] extra)."""
import json

import pytest

torch = pytest.importorskip("torch")


def test_manual_backprop_matches_autograd():
    from ynbtriage.model.backprop import autograd_grads, init_params, manual_grads, toy_data

    X, y = toy_data(N=50)
    params = init_params(X.shape[1], 8, 3)
    _, g_man = manual_grads(X, y, params)
    _, g_auto = autograd_grads(X, y, params)
    for a, b in zip(g_man, g_auto):
        assert torch.allclose(a, b, atol=1e-10)


def test_excerpt_keeps_error_and_tail():
    from ynbtriage.model.excerpt import error_excerpt

    log = "\n".join([f"noise {i}" for i in range(200)] + ["E: Unable to locate package foo"]
                    + [f"after {i}" for i in range(100)])
    ex = error_excerpt(log, before=2, after=2, tail=5)
    assert "Unable to locate package" in ex
    assert "after 99" in ex
    assert "noise 0" not in ex


def test_pipeline_end_to_end(tmp_path):
    from ynbtriage.model.pipeline import run
    from ynbtriage.model.predict import Predictor
    from ynbtriage.model.synth import synth_db
    from ynbtriage.model.train import TrainConfig

    db = tmp_path / "synth.db"
    synth_db(db, n=300)
    out = tmp_path / "run"
    s = run(out, db_path=db, features="tfidf", cfg=TrainConfig(epochs=30, patience=0))

    for f in ("model.pt", "featurizer.joblib", "metrics.json", "history.csv", "loss_curve.png"):
        assert (out / f).exists()
    hist = json.loads((out / "metrics.json").read_text())
    assert hist["epochs_run"] == 30
    # training actually happens: loss goes down
    import pandas as pd
    h = pd.read_csv(out / "history.csv")
    assert h["train_loss"].iloc[-1] < h["train_loss"].iloc[0]
    # beats the majority-class floor on synthetic data
    assert s["mlp"]["test"]["macro_f1"] > s["baselines"]["majority"]["test"]["macro_f1"]

    pred = Predictor.load(out).predict("standard_init_linux.go:228: exec format error")
    assert pred["label"] in s["labels"]
    assert abs(sum(pred["probabilities"].values()) - 1) < 1e-5


def test_synth_refuses_real_db(tmp_path):
    from ynbtriage.db import ensure_schema, get_conn
    from ynbtriage.model.synth import synth_db

    db = tmp_path / "real.db"
    with get_conn(db) as conn:
        ensure_schema(conn)
        conn.execute("INSERT INTO builds(tool_name, log_tail, source_batch) VALUES('x','y','socr8s')")
    with pytest.raises(SystemExit):
        synth_db(db, n=5)
