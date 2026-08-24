"""
Trains the from-scratch linear regression model on the auto imports
dataset, compares it against sklearn's LinearRegression, and saves
plots showing convergence and prediction quality.

Run from the project root:  python src/train.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from data_prep import load_clean_data, FEATURES, TARGET
from linear_regression import (
    zscore_normalize_features,
    gradient_descent,
    compute_cost,
    predict,
)

RNG_SEED = 42
TEST_FRACTION = 0.2
ALPHA = 0.1
NUM_ITERS = 1000


def train_test_split(X, y, test_fraction, seed):
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    idx = rng.permutation(m)
    n_test = int(m * test_fraction)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def main():
    df = load_clean_data()
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, TEST_FRACTION, RNG_SEED)
    print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows, "
          f"Features: {X_train.shape[1]}")

    # Normalize using train-set statistics only, then apply the same
    # transform to the test set (no leakage).
    X_train_norm, mu, sigma = zscore_normalize_features(X_train)
    X_test_norm, _, _ = zscore_normalize_features(X_test, mu, sigma)

    # --- Train from-scratch model on normalized features ---
    print("\nTraining from-scratch gradient descent on normalized features...")
    w_init = np.zeros(X_train.shape[1])
    b_init = 0.0
    w, b, J_hist_norm = gradient_descent(
        X_train_norm, y_train, w_init, b_init, ALPHA, NUM_ITERS
    )

    # --- Same run on raw (unscaled) features, much smaller alpha so it
    # doesn't diverge, to demonstrate why normalization matters. ---
    print("\nTraining from-scratch gradient descent on RAW (unnormalized) "
          "features for comparison...")
    alpha_raw = 1e-8
    w_raw, b_raw, J_hist_raw = gradient_descent(
        X_train, y_train, np.zeros(X_train.shape[1]), 0.0, alpha_raw, NUM_ITERS
    )

    # --- Learning rate sweep on normalized features: too small barely
    # moves, a good rate converges quickly, too large overshoots and
    # diverges. Run for fewer iterations so the diverging runs stay
    # finite (they blow up to inf/nan if left running too long). ---
    print("\nSweeping learning rates on normalized features...")
    sweep_alphas = [0.001, 0.01, 0.03, 0.1, 0.3]
    sweep_iters = 60
    sweep_histories = {}
    for a in sweep_alphas:
        _, _, J_hist = gradient_descent(
            X_train_norm, y_train, np.zeros(X_train.shape[1]), 0.0,
            a, sweep_iters, verbose=False,
        )
        sweep_histories[a] = J_hist
        print(f"  alpha={a:<6} final cost after {sweep_iters} iters: {J_hist[-1]:,.1f}")

    # --- sklearn baseline, trained on the same normalized data ---
    sk_model = LinearRegression()
    sk_model.fit(X_train_norm, y_train)

    # --- Evaluate both models on the held-out test set ---
    y_pred_scratch = predict(X_test_norm, w, b)
    y_pred_sklearn = sk_model.predict(X_test_norm)

    def report(name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"{name:18s} RMSE = ${rmse:,.2f}   R^2 = {r2:.4f}")
        return rmse, r2

    print("\n--- Test set performance ---")
    report("From-scratch GD", y_test, y_pred_scratch)
    report("sklearn", y_test, y_pred_sklearn)

    # --- Plot 1: cost vs iterations for the from-scratch model ---
    plt.figure(figsize=(7, 5))
    plt.plot(J_hist_norm)
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(w,b)")
    plt.title("Convergence of Gradient Descent (normalized features)")
    plt.tight_layout()
    plt.savefig("plots/cost_vs_iterations.png", dpi=150)
    plt.close()

    # --- Plot 2: normalized vs raw convergence ---
    plt.figure(figsize=(7, 5))
    plt.plot(J_hist_norm, label=f"Z-score normalized (alpha={ALPHA})")
    plt.plot(J_hist_raw, label=f"Raw features (alpha={alpha_raw})")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(w,b)")
    plt.title("Effect of Feature Scaling on Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/normalization_comparison.png", dpi=150)
    plt.close()

    # --- Plot 3: predicted vs actual price, both models ---
    plt.figure(figsize=(7, 7))
    lims = [min(y_test.min(), y_pred_scratch.min()), max(y_test.max(), y_pred_scratch.max())]
    plt.scatter(y_test, y_pred_scratch, alpha=0.7, label="From-scratch GD")
    plt.scatter(y_test, y_pred_sklearn, alpha=0.7, label="sklearn", marker="x")
    plt.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    plt.xlabel("Actual price ($)")
    plt.ylabel("Predicted price ($)")
    plt.title("Predicted vs Actual Car Price (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/predictions_vs_actual.png", dpi=150)
    plt.close()

    # --- Plot 4: learning rate sweep, log-scale so both the slow-moving
    # small-alpha runs and the diverging large-alpha run are visible ---
    plt.figure(figsize=(7, 5))
    for a in sweep_alphas:
        plt.plot(sweep_histories[a], label=f"alpha={a}")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(w,b)  (log scale)")
    plt.title("Learning Rate Sweep (normalized features)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/learning_rate_sweep.png", dpi=150)
    plt.close()

    print("\nSaved plots to plots/")


if __name__ == "__main__":
    main()
