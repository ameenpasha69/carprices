"""
Loads the raw UCI "1985 Auto Imports" dataset, keeps the continuous
numeric features, drops rows with missing values, and writes a clean
CSV to data/clean_auto.csv.

Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
"""

import pandas as pd

COLUMN_NAMES = [
    "symboling", "normalized_losses", "make", "fuel_type", "aspiration",
    "num_of_doors", "body_style", "drive_wheels", "engine_location",
    "wheel_base", "length", "width", "height", "curb_weight",
    "engine_type", "num_of_cylinders", "engine_size", "fuel_system",
    "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
    "city_mpg", "highway_mpg", "price",
]

# Continuous numeric features used as predictors for this project.
# (Categorical features like `make` or `body_style` are left out for now
# since one-hot encoding hasn't been covered yet.)
FEATURES = [
    "wheel_base", "length", "width", "height", "curb_weight",
    "engine_size", "bore", "stroke", "compression_ratio",
    "horsepower", "peak_rpm", "city_mpg", "highway_mpg",
]
TARGET = "price"


def load_clean_data(raw_path="data/imports-85.data"):
    df = pd.read_csv(raw_path, names=COLUMN_NAMES, na_values="?")
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    print(f"Loaded {before} rows, kept {after} after dropping missing values.")
    return df


if __name__ == "__main__":
    df = load_clean_data()
    df.to_csv("data/clean_auto.csv", index=False)
    print(f"Wrote data/clean_auto.csv with {len(df)} rows, {len(FEATURES)} features.")
    print(df.describe().T[["mean", "std", "min", "max"]])
