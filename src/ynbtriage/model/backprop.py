"""Backpropagation by hand for a one-hidden-layer classifier, checked against autograd.

Run it:  python -m ynbtriage.model.backprop

Not used by the pipeline. It exists so that `loss.backward()` in train.py is not
a black box. Shapes: N examples, D input features, H hidden units, K classes.

Forward
    Z1 = X @ W1.T + b1            (N, H)   pre-activation
    A1 = relu(Z1)                 (N, H)   hidden layer output
    Z2 = A1 @ W2.T + b2           (N, K)   logits
    P  = softmax(Z2)              (N, K)   class probabilities
    L  = -mean_i log P[i, y_i]             cross-entropy loss (a scalar)

Backward (chain rule, from the loss back toward the input)
    dZ2 = (P - Y) / N             (N, K)   Y is one-hot. Softmax + cross-entropy
                                           combine into this simple form.
    dW2 = dZ2.T @ A1              (K, H)
    db2 = dZ2.sum(0)              (K,)
    dA1 = dZ2 @ W2                (N, H)   error sent back to the hidden layer
    dZ1 = dA1 * (Z1 > 0)          (N, H)   relu passes gradient only where it was active
    dW1 = dZ1.T @ X               (H, D)
    db1 = dZ1.sum(0)              (H,)

Update (gradient descent)
    W <- W - lr * dW   for every parameter
"""
from __future__ import annotations

import torch


def forward(X, params):
    W1, b1, W2, b2 = params
    Z1 = X @ W1.T + b1
    A1 = torch.relu(Z1)
    Z2 = A1 @ W2.T + b2
    P = torch.softmax(Z2, dim=1)
    return Z1, A1, Z2, P


def loss_fn(P, y):
    return -torch.log(P[torch.arange(len(y)), y]).mean()


def manual_grads(X, y, params):
    W1, b1, W2, b2 = params
    Z1, A1, Z2, P = forward(X, params)
    N, K = P.shape
    Y = torch.zeros_like(P)
    Y[torch.arange(N), y] = 1.0
    dZ2 = (P - Y) / N
    dW2 = dZ2.T @ A1
    db2 = dZ2.sum(0)
    dA1 = dZ2 @ W2
    dZ1 = dA1 * (Z1 > 0).to(dA1.dtype)
    dW1 = dZ1.T @ X
    db1 = dZ1.sum(0)
    return loss_fn(P, y), [dW1, db1, dW2, db2]


def autograd_grads(X, y, params):
    ps = [p.clone().requires_grad_(True) for p in params]
    loss = loss_fn(forward(X, ps)[3], y)
    loss.backward()
    return loss.detach(), [p.grad for p in ps]


def toy_data(N=200, D=10, K=3, seed=0):
    """K Gaussian blobs in D dimensions: a problem a small MLP can learn."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(K, D, generator=g, dtype=torch.float64) * 0.6
    y = torch.randint(0, K, (N,), generator=g)
    X = centers[y] + torch.randn(N, D, generator=g, dtype=torch.float64) * 1.2
    return X, y


def init_params(D, H, K, seed=0):
    """Small random weights (breaks symmetry between hidden units), zero biases."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(H, D, generator=g, dtype=torch.float64) * (2 / D) ** 0.5,
            torch.zeros(H, dtype=torch.float64),
            torch.randn(K, H, generator=g, dtype=torch.float64) * (2 / H) ** 0.5,
            torch.zeros(K, dtype=torch.float64)]


def main(epochs: int = 100, lr: float = 0.1):
    X, y = toy_data()
    params = init_params(X.shape[1], 16, 3)

    # 1) Do the hand-derived gradients match what autograd computes?
    _, g_man = manual_grads(X, y, params)
    _, g_auto = autograd_grads(X, y, params)
    print("gradient check (max |manual - autograd|):")
    for name, a, b in zip(["W1", "b1", "W2", "b2"], g_man, g_auto):
        print(f"  {name:3s} {float((a - b).abs().max()):.2e}")
    assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(g_man, g_auto))

    # 2) Train with nothing but those hand-written gradients.
    print("\nfull-batch gradient descent with manual gradients:")
    for epoch in range(1, epochs + 1):
        loss, grads = manual_grads(X, y, params)
        params = [p - lr * g for p, g in zip(params, grads)]
        if epoch in (1, 2, 5, 10, 25, 50, epochs):
            acc = (forward(X, params)[3].argmax(1) == y).double().mean()
            print(f"  epoch {epoch:3d}  loss {float(loss):.4f}  acc {float(acc):.3f}")


if __name__ == "__main__":
    main()
