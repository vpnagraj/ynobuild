"""A two-stream MLP with a learned per-log gate.

The original model merges the first-error window and the log tail into one excerpt.
This experiment keeps them apart, gives each its own branch, and lets a learned
gate decide, log by log, how much each region counts:

    log_tail -> split_excerpt() -> (first-error window, tail)
             -> TfidfFeaturizer per region (REUSED from ynbtriage)
             -> branch per region: Linear -> ReLU -> Dropout     (new)
             -> gate: w = softmax(Linear([z_err, z_tail]))      (new)
             -> fused = w_err * z_err + w_tail * z_tail
             -> Linear -> 3 logits

Everything else is reused from ynbtriage: data loading and splits (data.py),
the TF-IDF featurizer (features.py), the training loop with early stopping
(train.py, via its `model_fn` hook), metrics (metrics.py) and the shallow
baselines (pipeline.fit_baselines).

The two region feature blocks are concatenated into one (N, d_err + d_tail)
array. The model slices it back apart. That keeps the input a plain 2-D array,
so ynbtriage's train_mlp can train this model unchanged.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from ynbtriage.db import connect
from ynbtriage.model.data import label_order, load_dataset
from ynbtriage.model.excerpt import _ERR_RE, normalize
from ynbtriage.model.features import TfidfFeaturizer
from ynbtriage.model.metrics import score
from ynbtriage.model.pipeline import prepare
from ynbtriage.model.train import TrainConfig, train_mlp

NO_WINDOW = "<NO_ERROR_WINDOW>"


# ============================================================ preprocessing
def split_excerpt(log: str, before: int = 5, after: int = 15, tail: int = 30,
                  max_chars: int = 4000, fallback_lines: int = 20) -> tuple[str, str, bool]:
    """Same rule as ynbtriage's error_excerpt(), but return the two regions separately.

    Returns (window, tail_text, error_line_found).

      window    `before` lines before the first error match to `after` lines after it.
                If no line matches, the `fallback_lines` lines just above the tail
                (so the branch still sees *something* from earlier in the log).
                If even that is empty (very short logs), the token NO_WINDOW.
      tail      the last `tail` lines.

    Each region gets half of `max_chars`, so the total text matches the baseline's
    cap. In short logs the window and the tail can overlap; both branches then see
    the shared lines. That's deliberate: it mirrors what the merged excerpt contains.
    """
    lines = (log or "").splitlines()
    if not lines:
        return NO_WINDOW, "", False
    n = len(lines)
    tail_start = max(0, n - tail)
    found = False
    for i, ln in enumerate(lines):
        if _ERR_RE.search(ln):
            win = lines[max(0, i - before): min(n, i + after + 1)]
            found = True
            break
    else:
        win = lines[max(0, tail_start - fallback_lines): tail_start]
    half = max_chars // 2
    window = normalize("\n".join(win))[:half] or NO_WINDOW      # keep the start (the error line)
    tail_text = normalize("\n".join(lines[tail_start:]))[-half:]  # keep the end
    return window, tail_text, found


def load_two_stream(db_path=None, target: str = "class") -> tuple[pd.DataFrame, list[str]]:
    """Exactly the rows/splits/labels the original model used, plus the two regions.

    `text` is the original merged excerpt (for the baselines); `err_text` and
    `tail_text` are the split regions for the two-stream model.
    """
    df = load_dataset(db_path, target=target)
    conn = connect(db_path)
    try:
        logs = pd.read_sql_query("SELECT build_id, log_tail FROM builds", conn)
    finally:
        conn.close()
    df = df.merge(logs, on="build_id", how="left")
    parts = df["log_tail"].map(split_excerpt)
    df["err_text"] = parts.map(lambda p: p[0])
    df["tail_text"] = parts.map(lambda p: p[1])
    df["error_line_found"] = parts.map(lambda p: p[2])
    return df.drop(columns="log_tail"), label_order(db_path, target)


class TwoStreamFeaturizer:
    """One REUSED TfidfFeaturizer per region, fit on train only, outputs side by side."""

    def __init__(self, dim_per_region: int = 128, seed: int = 7400):
        self.err = TfidfFeaturizer(dim=dim_per_region, seed=seed)
        self.tail = TfidfFeaturizer(dim=dim_per_region, seed=seed)

    def fit(self, err_texts, tail_texts):
        self.err.fit(err_texts)
        self.tail.fit(tail_texts)
        return self

    def transform(self, err_texts, tail_texts) -> np.ndarray:
        a, b = self.err.transform(err_texts), self.tail.transform(tail_texts)
        self.d_err = a.shape[1]          # where the model should slice
        return np.hstack([a, b]).astype(np.float32)


# ===================================================================== model
class TwoStreamGated(nn.Module):
    """Two branches + a per-log softmax gate. Input: (B, d_err + d_tail)."""

    def __init__(self, d_err: int, d_tail: int, n_classes: int, branch_dim: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        self.d_err = d_err

        def branch(d_in):
            return nn.Sequential(nn.Linear(d_in, branch_dim), nn.ReLU(), nn.Dropout(dropout))

        self.branch_err = branch(d_err)
        self.branch_tail = branch(d_tail)
        # The gate sees both branch summaries and outputs 2 scores -> softmax -> 2 weights.
        # Zero init: every log starts at w = [0.5, 0.5], so neither region is favored
        # until training gives a reason.
        self.gate = nn.Linear(2 * branch_dim, 2)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.head = nn.Linear(branch_dim, n_classes)
        self.config = dict(arch="two_stream_gated", d_err=d_err, d_tail=d_tail,
                           n_classes=n_classes, branch_dim=branch_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, return_parts: bool = False, knockout: str | None = None):
        x_err, x_tail = x[:, :self.d_err], x[:, self.d_err:]   # (B, d_err), (B, d_tail)
        z_err = self.branch_err(x_err)                         # (B, branch_dim)
        z_tail = self.branch_tail(x_tail)                      # (B, branch_dim)
        w = torch.softmax(self.gate(torch.cat([z_err, z_tail], dim=1)), dim=1)  # (B, 2)
        if knockout == "err":        # force all weight onto the tail
            w = torch.stack([torch.zeros_like(w[:, 0]), torch.ones_like(w[:, 1])], dim=1)
        elif knockout == "tail":     # force all weight onto the error window
            w = torch.stack([torch.ones_like(w[:, 0]), torch.zeros_like(w[:, 1])], dim=1)
        elif knockout is not None:
            raise ValueError("knockout must be None, 'err' or 'tail'")
        part_err, part_tail = w[:, 0:1] * z_err, w[:, 1:2] * z_tail
        logits = self.head(part_err + part_tail)               # (B, n_classes)
        if return_parts:
            return logits, dict(w=w, z_err=z_err, z_tail=z_tail,
                                part_err=part_err, part_tail=part_tail)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, knockout: str | None = None) -> torch.Tensor:
        self.eval()
        return torch.softmax(self.forward(x, knockout=knockout), dim=1)


# ================================================================== training
def train_two_stream(X: dict, y: dict, d_err: int, n_classes: int, cfg: TrainConfig,
                     branch_dim: int = 32):
    """Train with ynbtriage's train_mlp; also log the mean val gate weight per epoch."""
    holder = {}

    def model_fn(in_dim, n_cls):
        holder["m"] = TwoStreamGated(d_err, in_dim - d_err, n_cls, branch_dim, cfg.dropout)
        return holder["m"]

    Xva = torch.tensor(X["val"])

    def log_gate(rec):  # rec is the same dict train_mlp appends to history
        m = holder["m"]
        was_training = m.training
        m.eval()
        with torch.no_grad():
            _, p = m(Xva, return_parts=True)
        rec["val_w_err"] = p["w"][:, 0].mean().item()
        m.train(was_training)

    return train_mlp(X["train"], y["train"], X["val"], y["val"], n_classes, cfg,
                     on_epoch=log_gate, model_fn=model_fn)


def run_seeds(X_merged: dict, X_two: dict, y: dict, d_err: int, labels: list[str],
              seeds=(7400, 7401, 7402, 7403, 7404), cfg: TrainConfig | None = None,
              verbose: bool = True) -> list[dict]:
    """Baseline MLP (original config, merged excerpt) and two-stream model, same seeds."""
    base = cfg or TrainConfig()
    runs = []
    for s in seeds:
        c = _with_seed(base, s)
        mlp, h_mlp = train_mlp(X_merged["train"], y["train"], X_merged["val"], y["val"],
                               len(labels), c)
        two, h_two = train_two_stream(X_two, y, d_err, len(labels), c)
        rec = {"seed": s, "mlp": mlp, "two": two,
               "hist_mlp": pd.DataFrame(h_mlp), "hist_two": pd.DataFrame(h_two)}
        for name, m, Xs in (("mlp", mlp, X_merged), ("two", two, X_two)):
            for split in ("val", "test"):
                pred = m.predict_proba(torch.tensor(Xs[split])).numpy().argmax(1)
                rec[f"{name}_{split}"] = score(y[split], pred, labels)
        runs.append(rec)
        if verbose:
            print(f"seed {s}: macro-F1 val  MLP {rec['mlp_val']['macro_f1']:.3f}  "
                  f"two-stream {rec['two_val']['macro_f1']:.3f}   "
                  f"(epochs {len(h_mlp)}, {len(h_two)})")
    return runs


def _with_seed(cfg: TrainConfig, seed: int) -> TrainConfig:
    c = copy.deepcopy(cfg)
    c.seed = seed
    return c


# ============================================================== measurements
@torch.no_grad()
def per_log_measurements(model: TwoStreamGated, X: np.ndarray, y: np.ndarray,
                         labels: list[str]) -> pd.DataFrame:
    """One row per log: gate weight, contribution share, and knockout predictions."""
    model.eval()
    Xt = torch.tensor(X)
    logits, p = model(Xt, return_parts=True)
    n_err = p["part_err"].norm(dim=1)
    n_tail = p["part_tail"].norm(dim=1)
    return pd.DataFrame({
        "true": [labels[i] for i in y],
        "pred": [labels[i] for i in logits.argmax(1).numpy()],
        "w_err": p["w"][:, 0].numpy(),
        # fraction of the fused vector's size that came from the error window
        "contrib_err": (n_err / (n_err + n_tail).clamp_min(1e-8)).numpy(),
        "pred_ko_err": [labels[i] for i in model(Xt, knockout="err").argmax(1).numpy()],
        "pred_ko_tail": [labels[i] for i in model(Xt, knockout="tail").argmax(1).numpy()],
    })


def knockout_summary(m: pd.DataFrame) -> pd.DataFrame:
    """Accuracy with both branches vs. with one removed, and how many predictions flip."""
    rows = []
    for name, col in (("full model", "pred"), ("error window removed", "pred_ko_err"),
                      ("tail removed", "pred_ko_tail")):
        rows.append({"variant": name,
                     "accuracy": (m[col] == m["true"]).mean(),
                     "flip_rate_vs_full": (m[col] != m["pred"]).mean()})
    return pd.DataFrame(rows)


def summarize_runs(runs: list[dict], split: str = "test") -> pd.DataFrame:
    """Mean ± sd across seeds for each model and metric."""
    rows = []
    for name, label in (("mlp", "Original MLP (merged excerpt)"), ("two", "Two-stream gated")):
        acc = [r[f"{name}_{split}"]["accuracy"] for r in runs]
        f1 = [r[f"{name}_{split}"]["macro_f1"] for r in runs]
        rows.append({"model": label, "accuracy_mean": np.mean(acc), "accuracy_sd": np.std(acc, ddof=1),
                     "macro_f1_mean": np.mean(f1), "macro_f1_sd": np.std(f1, ddof=1)})
    return pd.DataFrame(rows)


__all__ = ["split_excerpt", "load_two_stream", "TwoStreamFeaturizer", "TwoStreamGated",
           "train_two_stream", "run_seeds", "per_log_measurements", "knockout_summary",
           "summarize_runs", "prepare", "NO_WINDOW"]
