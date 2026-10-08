"""The training loop, written out step by step.

This is the file to be able to explain. The numbered comments match the prescribed
outline:

  1. Initialize network parameters
  2. Forward propagation
  3. Calculate the loss
  4. Backpropagation
  5. (gradients are now in each parameter's .grad)
  6. Update parameters with gradient descent
  7. Repeat 2-6 until convergence

What PyTorch does for you in step 4: during the forward pass it records every
operation on tensors that require gradients. `loss.backward()` walks that record
in reverse, applying the chain rule, and leaves dLoss/dParameter in `p.grad` for
every weight and bias. backprop.py does the same thing by hand for a one-hidden-
layer network and checks the result against autograd.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn

from .mlp import MLP


@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = (64,)
    activation: str = "relu"
    dropout: float = 0.2
    optimizer: str = "adam"        # "sgd" is plain (minibatch) gradient descent
    lr: float = 1e-3
    momentum: float = 0.0          # sgd only
    weight_decay: float = 1e-4     # L2 penalty on weights
    epochs: int = 200
    batch_size: int = 32
    patience: int = 25             # early stopping on val loss; 0 disables
    class_weighted: bool = True    # up-weight rare classes in the loss
    seed: int = 7400
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalized so the average weight is 1."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def _evaluate(model, X, y, loss_fn):
    model.eval()                                   # dropout off
    with torch.no_grad():                          # no graph needed, just numbers
        logits = model(X)
        loss = loss_fn(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              n_classes: int, cfg: TrainConfig | None = None,
              on_epoch: Callable[[dict], None] | None = None):
    """Train an MLP; return (best model, history list of per-epoch dicts).

    `on_epoch` is called after every epoch with that epoch's metrics. The CLI uses
    it to print; a Streamlit training monitor can use it to update a live chart.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.long)

    # ---- 1. Initialize parameters ------------------------------------------
    # nn.Linear initializes its weights randomly (Kaiming-uniform) and biases
    # small. Random init breaks symmetry: if all weights started equal, every
    # hidden unit would compute the same thing and receive the same gradient.
    model = MLP(Xtr.shape[1], n_classes, tuple(cfg.hidden), cfg.activation, cfg.dropout)

    # Loss: cross-entropy = -log(softmax(logits)[true class]), averaged over the
    # batch. With class weights, mistakes on rare classes cost more.
    weight = class_weights(y_train, n_classes) if cfg.class_weighted else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    # Validation loss is reported unweighted so it is comparable across settings.
    eval_loss_fn = nn.CrossEntropyLoss()

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        # Adam is gradient descent with a per-parameter step size adapted from
        # running averages of the gradient and its square. Same loop, same grads.
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError("optimizer must be 'sgd' or 'adam'")

    history: list[dict] = []
    best_val, best_state, since_best = float("inf"), None, 0
    n = len(Xtr)

    # ---- 7. Repeat ---------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        model.train()                              # dropout on
        order = rng.permutation(n)                 # reshuffle every epoch
        running, seen = 0.0, 0
        for start in range(0, n, cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            xb, yb = Xtr[idx], ytr[idx]

            # ---- 2. Forward propagation --------------------------------------
            logits = model(xb)                     # shape (batch, n_classes)

            # ---- 3. Loss ------------------------------------------------------
            loss = loss_fn(logits, yb)             # a single scalar

            # ---- 4. Backpropagation ------------------------------------------
            opt.zero_grad()                        # .grad accumulates; clear last step's
            loss.backward()                        # chain rule, output -> input

            # ---- 5. Gradients now live in p.grad for every parameter p -------
            # ---- 6. Update: p <- p - lr * p.grad (for plain SGD) -------------
            opt.step()

            running += loss.item() * len(idx)
            seen += len(idx)

        # Train metrics are re-measured in eval mode (no dropout) so train and
        # val curves are measured the same way.
        tr_loss, tr_acc = _evaluate(model, Xtr, ytr, eval_loss_fn)
        va_loss, va_acc = _evaluate(model, Xva, yva, eval_loss_fn)
        rec = dict(epoch=epoch, batch_loss=running / seen, train_loss=tr_loss, train_acc=tr_acc,
                   val_loss=va_loss, val_acc=va_acc)
        history.append(rec)
        if on_epoch:
            on_epoch(rec)

        # Early stopping: keep the weights from the epoch with the lowest val loss,
        # stop once it hasn't improved for `patience` epochs.
        if va_loss < best_val - 1e-6:
            best_val, best_state, since_best = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
            if cfg.patience and since_best >= cfg.patience:
                break

    if best_state is not None:        # stop once it hasn't improved for `patience` epochs.

        model.load_state_dict(best_state)
    model.eval()
    return model, history
