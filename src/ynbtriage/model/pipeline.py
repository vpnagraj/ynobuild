"""End to end: DB -> features -> baselines + MLP -> metrics -> artifact directory.

An artifact directory holds everything needed to reproduce and serve a run:

    model.pt          MLP weights + architecture config + label names
    featurizer.joblib fitted featurizer (TF-IDF vocab/SVD; encoder name only)
    metrics.json      config, data summary, leakage report, val/test scores
                      for the MLP and every baseline
    history.csv       per-epoch train/val loss and accuracy
    loss_curve.png    loss-vs-epoch figure
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import baselines
from .data import label_order, leakage_report, load_dataset
from .features import make_featurizer
from .metrics import predict_labels, score
from .train import TrainConfig, train_mlp

TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"


def run(out_dir, db_path=None, target="class", features="embed", include_prefilled=False,
        cfg: TrainConfig | None = None, tfidf_dim=256, on_epoch=None) -> dict:
    """Train from the database (used by the `ynbtriage train` CLI)."""
    df = load_dataset(db_path, target=target, include_prefilled=include_prefilled)
    all_labels = label_order(db_path, target)
    try:
        return run_df(out_dir, df, all_labels, target=target, features=features,
                      include_prefilled=include_prefilled, cfg=cfg, tfidf_dim=tfidf_dim,
                      on_epoch=on_epoch)
    except ValueError as e:
        raise SystemExit(str(e)) from e


def data_fingerprint(df: pd.DataFrame) -> dict:
    """Identify exactly which labelled rows a run was trained and scored on.

    Two runs with the same hash saw the same builds, splits and labels, so their
    scores are directly comparable. Re-annotating or re-splitting changes it.
    """
    keys = sorted(f"{b}:{s}:{l}" for b, s, l in zip(df["build_id"], df["split"], df["label"]))
    return {"hash": hashlib.sha1("\n".join(keys).encode()).hexdigest()[:12],
            "n": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")}}


def prepare(df: pd.DataFrame, all_labels: list[str]):
    """Keep labels that have training rows; split; encode labels as integers.

    Shared by MLP training and the baselines-only command, so both always use
    exactly the same rows and label order.
    """
    train_df = df[df["split"] == "train"]
    labels = [l for l in all_labels if l in set(train_df["label"])]  # drop labels with no training rows
    dropped = sorted(set(df["label"]) - set(labels))
    df = df[df["label"].isin(labels)]
    lab2idx = {l: i for i, l in enumerate(labels)}
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    for s, d in parts.items():
        if len(d) == 0:
            raise ValueError(f"no '{s}' rows. Annotate more builds and re-run assign-splits.")
    y = {s: d["label"].map(lab2idx).to_numpy() for s, d in parts.items()}
    texts = {s: d["text"].tolist() for s, d in parts.items()}
    return df, labels, dropped, parts, y, texts


def fit_baselines(texts: dict, y: dict, labels: list[str], seed: int = 7400) -> dict:
    """Fit every shallow baseline on train; score on val and test."""
    models = {
        "majority": baselines.Majority().fit(texts["train"], y["train"]),
        "naive_bayes": baselines.CountNB().fit(texts["train"], y["train"]),
        "tfidf_logreg": baselines.TfidfLogReg(seed=seed).fit(texts["train"], y["train"]),
    }
    return {name: {s: score(y[s], m.predict(texts[s]), labels) for s in ("val", "test")}
            for name, m in models.items()}


def run_baselines(out_dir, db_path=None, target="class", seed: int = 7400) -> dict:
    """Score only the shallow baselines and write <out_dir>/baselines.json.

    For adding baselines to an existing run without retraining (or touching) its
    MLP. The model service shows these alongside the run's own metrics.
    """
    df = load_dataset(db_path, target=target)
    all_labels = label_order(db_path, target)
    try:
        df, labels, _, parts, y, texts = prepare(df, all_labels)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    out = {
        "target": target, "labels": labels, "seed": seed,
        "n": {s: int(len(d)) for s, d in parts.items()},
        "data_fingerprint": data_fingerprint(df),
        "baselines": fit_baselines(texts, y, labels, seed=seed),
    }
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "baselines.json").write_text(json.dumps(out, indent=2))
    return out


def run_df(out_dir, df: pd.DataFrame, all_labels: list[str], target="class", features="embed",
           include_prefilled=False, cfg: TrainConfig | None = None, tfidf_dim=256,
           on_epoch=None, extra: dict | None = None) -> dict:
    """Train from an already-loaded dataset (columns as returned by data.load_dataset).

    Used directly by the model service, which gets its data from the api service
    rather than opening the database. `extra` is merged into metrics.json.
    """
    cfg = cfg or TrainConfig()
    out = Path(out_dir)

    # ---- data --------------------------------------------------------------
    df, labels, dropped, parts, y, texts = prepare(df, all_labels)

    # ---- features (fit on train only) --------------------------------------
    feat = make_featurizer(features, tfidf_dim=tfidf_dim)
    feat.fit(texts["train"])
    X = {s: feat.transform(t) for s, t in texts.items()}

    # ---- baselines ---------------------------------------------------------
    results: dict = {"baselines": fit_baselines(texts, y, labels, seed=cfg.seed)}

    # ---- MLP ---------------------------------------------------------------
    model, history = train_mlp(X["train"], y["train"], X["val"], y["val"], len(labels), cfg,
                               on_epoch=on_epoch)
    with torch.no_grad():
        results["mlp"] = {
            s: score(y[s], predict_labels(model.predict_proba(torch.tensor(X[s])).numpy()), labels)
            for s in ("train", "val", "test")
        }

    # ---- write artifact ----------------------------------------------------
    import joblib
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config, "labels": labels,
                "target": target, "features": features}, out / "model.pt")
    joblib.dump(feat, out / "featurizer.joblib", compress=3)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    plot_history(history, out / "loss_curve.png")

    summary = {
        "target": target, "features": features, "include_prefilled": include_prefilled,
        "labels": labels, "labels_dropped_no_train_rows": dropped,
        "train_config": cfg.to_dict(), "epochs_run": len(history),
        "best_epoch": int(np.argmin([h["val_loss"] for h in history]) + 1),
        "n": {s: int(len(d)) for s, d in parts.items()},
        "n_silver_in_train": int(parts["train"]["silver"].sum()),
        "train_label_counts": parts["train"]["label"].value_counts().to_dict(),
        "leakage": leakage_report(df),
        "data_fingerprint": data_fingerprint(df),
        **results,
        **(extra or {}),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot_history(history: list[dict], path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, title in ((axes[0], "loss", "Cross-entropy loss"), (axes[1], "acc", "Accuracy")):
        for split, color in (("train", TRAIN_COLOR), ("val", VAL_COLOR)):
            ax.plot(h["epoch"], h[f"{split}_{metric}"], color=color, lw=2, label=split)
            ax.annotate(split, (h["epoch"].iloc[-1], h[f"{split}_{metric}"].iloc[-1]),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=9,
                        color="#333333")
        best = int(h["val_loss"].idxmin())
        ax.axvline(h["epoch"].iloc[best], color="#999999", lw=1, ls="--")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("epoch")
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_ylim(0, 1.02)
    fig.text(0.99, 0.01, "dashed line: best val-loss epoch (weights kept)", ha="right",
             fontsize=8, color="#666666")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
