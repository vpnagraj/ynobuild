"""Evaluation metrics. Macro-F1 is the headline number: the classes are
imbalanced (environment_decay is ~60-70% of the corpus), so accuracy alone
rewards predicting the majority class."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def score(y_true, y_pred, labels: list[str]) -> dict:
    idx = list(range(len(labels)))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)),
        "per_class_f1": dict(zip(labels, map(float, f1_score(
            y_true, y_pred, labels=idx, average=None, zero_division=0)))),
        "confusion": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }


def predict_labels(proba: np.ndarray) -> np.ndarray:
    return np.asarray(proba).argmax(axis=1)
