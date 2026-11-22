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


def test_default_excerpt_keeps_only_the_tail():
    from ynbtriage.model.excerpt import log_excerpt

    log = "\n".join([f"early {i}" for i in range(200)] + ["E: Unable to locate package foo"]
                    + [f"after {i}" for i in range(5)])
    ex = log_excerpt(log, tail=6)
    assert "Unable to locate package" in ex          # within the last 6 lines
    assert "after 4" in ex
    assert "early 199" not in ex
    assert log_excerpt("") == ""


def test_excerpt_normalizes_and_caps():
    from ynbtriage.model.excerpt import log_excerpt

    ex = log_excerpt("pulled sha256:" + "a" * 64 + " at 2024-01-02 03:04:05 (12.5 MB)")
    assert "<HEX>" in ex and "<TS>" in ex and "<SIZE>" in ex
    assert len(log_excerpt("x" * 10_000, max_chars=100)) == 100


def test_error_window_excerpt_keeps_error_and_tail():
    from ynbtriage.model.excerpt import error_excerpt, log_excerpt

    log = "\n".join([f"noise {i}" for i in range(200)] + ["E: Unable to locate package foo"]
                    + [f"after {i}" for i in range(100)])
    ex = log_excerpt(log, "error_window", before=2, after=2, tail=5)
    assert "Unable to locate package" in ex
    assert "after 99" in ex
    assert "noise 0" not in ex
    assert error_excerpt(log, before=2, after=2, tail=5) == ex
    assert "Unable to locate package" not in log_excerpt(log, "tail", tail=5)


def test_unknown_excerpt_rule_is_rejected():
    import pytest
    from ynbtriage.model.excerpt import log_excerpt

    with pytest.raises(ValueError, match="unknown excerpt rule"):
        log_excerpt("x", "first_line")


def test_rule_is_recorded_and_reused_at_prediction(tmp_path):
    import torch
    from ynbtriage.model.pipeline import run
    from ynbtriage.model.predict import Predictor
    from ynbtriage.model.synth import synth_db
    from ynbtriage.model.train import TrainConfig

    synth_db(tmp_path / "s.db", n=200)
    cfg = TrainConfig(epochs=3, patience=0)
    for rule in ("tail", "error_window"):
        out = tmp_path / rule
        run(out, db_path=tmp_path / "s.db", features="tfidf", cfg=cfg, excerpt=rule)
        assert Predictor.load(out).excerpt == rule

    # a model saved before the rule was recorded falls back to the original rule
    ckpt = torch.load(tmp_path / "tail" / "model.pt", weights_only=False)
    del ckpt["excerpt"]
    torch.save(ckpt, tmp_path / "tail" / "model.pt")
    assert Predictor.load(tmp_path / "tail").excerpt == "error_window"


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


def test_train_mlp_accepts_custom_model():
    import numpy as np
    from torch import nn
    from ynbtriage.model.train import TrainConfig, train_mlp

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8)).astype("float32")
    y = (X[:, 0] > 0).astype(int)
    built = []

    def model_fn(in_dim, n_classes):
        built.append((in_dim, n_classes))
        return nn.Linear(in_dim, n_classes)

    model, hist = train_mlp(X, y, X, y, 2, TrainConfig(epochs=3, patience=0), model_fn=model_fn)
    assert built == [(8, 2)] and isinstance(model, nn.Linear) and len(hist) == 3


def test_synth_refuses_real_db(tmp_path):
    from ynbtriage.db import ensure_schema, get_conn
    from ynbtriage.model.synth import synth_db

    db = tmp_path / "real.db"
    with get_conn(db) as conn:
        ensure_schema(conn)
        conn.execute("INSERT INTO builds(tool_name, log_tail, source_batch) VALUES('x','y','socr8s')")
    with pytest.raises(SystemExit):
        synth_db(db, n=5)
