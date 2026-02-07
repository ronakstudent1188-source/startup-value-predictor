from typing import Tuple
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def train_baseline(X, y, test_size: float = 0.2, random_state: int = 42) -> Tuple[LinearRegression, dict]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, {"r2": score}


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
