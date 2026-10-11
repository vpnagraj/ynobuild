"""The neural network: a small feed-forward multi-layer perceptron.

    input x (d features)
      -> Linear(d, h1) -> activation -> Dropout
      -> ... (one block per hidden layer)
      -> Linear(h_last, K)  = logits, one score per class

There is no softmax at the end on purpose: nn.CrossEntropyLoss takes raw logits
and applies log-softmax internally, which is numerically more stable. Use
`predict_proba` when you want probabilities.
"""
from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "sigmoid": nn.Sigmoid}


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: tuple[int, ...] = (64,),
                 activation: str = "relu", dropout: float = 0.2):
        super().__init__()
        if not hidden:
            raise ValueError("an MLP needs at least one hidden layer")
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), ACTIVATIONS[activation](), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
        # Saved with the weights so the model can be rebuilt from the checkpoint alone.
        self.config = dict(in_dim=in_dim, n_classes=n_classes, hidden=list(hidden),
                           activation=activation, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.softmax(self.forward(x), dim=1)
