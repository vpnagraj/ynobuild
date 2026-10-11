"""Load a trained artifact directory and classify log text.

This is the piece a FastAPI endpoint (or the Streamlit app) will call:

    p = Predictor.load("models/run-001")
    p.predict(log_tail)   # -> {"label": ..., "probabilities": {label: p, ...}}
"""
from __future__ import annotations

from pathlib import Path

import torch

from .excerpt import error_excerpt
from .mlp import MLP


class Predictor:
    def __init__(self, model: MLP, featurizer, labels: list[str], target: str):
        self.model, self.featurizer, self.labels, self.target = model, featurizer, labels, target

    @classmethod
    def load(cls, model_dir):
        import joblib
        d = Path(model_dir)
        ckpt = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        c = ckpt["config"]
        model = MLP(c["in_dim"], c["n_classes"], tuple(c["hidden"]), c["activation"], c["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return cls(model, joblib.load(d / "featurizer.joblib"), ckpt["labels"], ckpt["target"])

    def predict_many(self, logs: list[str]) -> list[dict]:
        excerpts = [error_excerpt(l) for l in logs]
        X = torch.tensor(self.featurizer.transform(excerpts))
        proba = self.model.predict_proba(X).numpy()
        out = []
        for row, ex in zip(proba, excerpts):
            probs = {l: float(p) for l, p in zip(self.labels, row)}
            out.append({"label": max(probs, key=probs.get), "target": self.target,
                        "probabilities": probs,
                        "excerpt": ex})  # the text the model actually saw
        return out

    def predict(self, log: str) -> dict:
        return self.predict_many([log])[0]
