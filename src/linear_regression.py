"""
Multivariate linear regression trained with batch gradient descent,
implemented from scratch in numpy (no sklearn).

Model:      f(x) = w . x + b
Cost:       J(w,b) = (1 / 2m) * sum((f(x_i) - y_i)^2)
Gradients:  dJ/dw_j = (1/m) * sum((f(x_i) - y_i) * x_i_j)
            dJ/db   = (1/m) * sum(f(x_i) - y_i)
Update:     w_j := w_j - alpha * dJ/dw_j
            b   := b   - alpha * dJ/db
"""

import numpy as np


def zscore_normalize_features(X, mu=None, sigma=None):
    """Normalize each feature column to zero mean, unit variance.

    If mu/sigma are given (e.g. computed on the training set), reuse
    them instead of recomputing, so the test set is scaled the same
    way the model was trained on -- avoids leaking test statistics
    into the transform.
    """
    if mu is None:
        mu = np.mean(X, axis=0)
    if sigma is None:
        sigma = np.std(X, axis=0)
    return (X - mu) / sigma, mu, sigma


def predict(X, w, b):
    return X @ w + b


def compute_cost(X, y, w, b):
    m = X.shape[0]
    errors = predict(X, w, b) - y
    return np.sum(errors ** 2) / (2 * m)


def compute_gradient(X, y, w, b):
    m = X.shape[0]
    errors = predict(X, w, b) - y          # shape (m,)
    dj_dw = (X.T @ errors) / m             # shape (n,)
    dj_db = np.sum(errors) / m
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """Runs batch gradient descent, returns final w, b and the cost
    history (useful for plotting convergence / picking a learning rate).
    """
    w = w_in.copy()
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        J_history.append(compute_cost(X, y, w, b))

        if i % max(1, num_iters // 10) == 0 or i == num_iters - 1:
            print(f"iter {i:5d}: cost {J_history[-1]:.4f}")

    return w, b, J_history
