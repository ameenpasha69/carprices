"""
Streamlit demo for the car price predictor.

Trains the from-scratch gradient descent model (and an sklearn baseline)
once on startup, then lets the user tweak car specs with sliders and see
both models' predicted price update live.

Run with:  streamlit run app.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression

from data_prep import load_clean_data, FEATURES, TARGET
from linear_regression import gradient_descent, predict, zscore_normalize_features

ALPHA = 0.1
NUM_ITERS = 1000

FEATURE_LABELS = {
    "wheel_base": "Wheel base (in)",
    "length": "Length (in)",
    "width": "Width (in)",
    "height": "Height (in)",
    "curb_weight": "Curb weight (lb)",
    "engine_size": "Engine size (cu in)",
    "bore": "Bore (in)",
    "stroke": "Stroke (in)",
    "compression_ratio": "Compression ratio",
    "horsepower": "Horsepower",
    "peak_rpm": "Peak RPM",
    "city_mpg": "City MPG",
    "highway_mpg": "Highway MPG",
}

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def train_models():
    df = load_clean_data(os.path.join(os.path.dirname(__file__), "data", "imports-85.data"))
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    X_norm, mu, sigma = zscore_normalize_features(X)
    w, b, _ = gradient_descent(
        X_norm, y, np.zeros(X.shape[1]), 0.0, ALPHA, NUM_ITERS, verbose=False
    )
    sk_model = LinearRegression().fit(X_norm, y)

    return {"df": df, "mu": mu, "sigma": sigma, "w": w, "b": b, "sk_model": sk_model}


models = train_models()
df = models["df"]

st.title("🚗 Car Price Predictor")
st.write(
    "Predicts a used car's price from its specs, using **multivariate linear "
    "regression trained with batch gradient descent, implemented from scratch "
    "in numpy** (z-score normalized features, no sklearn in the training loop). "
    "Trained on the [UCI 1985 Auto Imports dataset]"
    "(https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data) "
    "-- [source code](https://github.com/ameenpasha69/carprices)."
)

st.subheader("Enter car specs")

col1, col2 = st.columns(2)
inputs = {}
for i, feat in enumerate(FEATURES):
    col = col1 if i % 2 == 0 else col2
    lo, hi, med = float(df[feat].min()), float(df[feat].max()), float(df[feat].median())
    step = round((hi - lo) / 100, 2) or 1.0
    inputs[feat] = col.slider(FEATURE_LABELS.get(feat, feat), lo, hi, med, step=step)

x = np.array([inputs[f] for f in FEATURES], dtype=float).reshape(1, -1)
x_norm = (x - models["mu"]) / models["sigma"]

price_scratch = float(predict(x_norm, models["w"], models["b"])[0])
price_sklearn = float(models["sk_model"].predict(x_norm)[0])

st.subheader("Predicted price")
c1, c2 = st.columns(2)
c1.metric("From-scratch gradient descent", f"${price_scratch:,.0f}")
c2.metric("sklearn LinearRegression", f"${price_sklearn:,.0f}")
st.caption(
    "Both models are trained on the same normalized features, so they should "
    "land close together -- that's the sanity check that the from-scratch "
    "implementation is correct."
)

with st.expander("Model details"):
    st.write(
        "- Held-out test set: RMSE ≈ \\$3,136, R² ≈ 0.87 (from-scratch); "
        "nearly identical for sklearn.\n"
        "- 13 continuous numeric features, z-score normalized "
        "(fit on train, applied to test/inference).\n"
        "- See the [full writeup](https://github.com/ameenpasha69/carprices#readme) "
        "for the math, convergence plots, and a learning-rate sweep."
    )
