"""Minimal training script that creates a toy dataset if none exists, trains a baseline model, and saves it."""
import os
import numpy as np
import pandas as pd

from .models import train_baseline, save_model


def make_toy_dataset(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "age_months": rng.randint(1, 120, size=n),
        "founders": rng.randint(1, 5, size=n),
        "team_size": rng.randint(1, 200, size=n),
        "raised_usd": rng.exponential(1e6, size=n),
    })
    # target: synthetic valuation
    df["valuation_usd"] = 50000 + df["raised_usd"] * 2 + df["team_size"] * 1000 + rng.randn(n) * 1e5
    df.to_csv(path, index=False)


def main():
    data_path = os.path.join("data", "toy_startups.csv")
    model_path = os.path.join("models", "baseline.joblib")
    if not os.path.exists(data_path):
        print("Creating toy dataset at", data_path)
        make_toy_dataset(data_path)

    df = pd.read_csv(data_path)
    X = df[["age_months", "founders", "team_size", "raised_usd"]]
    y = df["valuation_usd"]

    model, metrics = train_baseline(X, y)
    print("Trained baseline model; metrics:", metrics)
    save_model(model, model_path)
    print("Saved model to", model_path)


if __name__ == "__main__":
    main()
