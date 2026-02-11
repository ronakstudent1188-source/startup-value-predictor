"""
Production-grade training script.
Loads real data, does feature engineering, trains best models, saves them.
Run: python src/train.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import xgboost as xgb
import joblib


def load_data(path='data/startup_funding.csv'):
    """Load startup data"""
    return pd.read_csv(path)


def engineer_features(df):
    """Transform raw data into engineered features"""
    df = df.copy()
    
    # Remove non-predictive columns
    df = df.drop('startup_name', axis=1)
    
    # Create derived features
    df['funding_per_team_member'] = df['total_funding_usd'] / (df['team_size'] + 1)
    df['rounds_per_year'] = df['funding_rounds'] / ((df['months_since_founding'] / 12) + 1)
    df['years_since_founding'] = df['months_since_founding'] / 12
    
    return df


def preprocess_data(df):
    """Scale and encode data"""
    df = df.copy()
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = ['valuation_usd', 'is_successful']
    feature_numeric_cols = [col for col in numeric_cols if col not in target_cols]
    categorical_cols = ['industry', 'country']
    
    # Scale numeric features
    scaler = StandardScaler()
    df[feature_numeric_cols] = scaler.fit_transform(df[feature_numeric_cols])
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df, scaler


def prepare_features_targets(df):
    """Separate features and targets"""
    X = df.drop(['valuation_usd', 'is_successful'], axis=1)
    y_regression = df['valuation_usd']
    y_classification = df['is_successful']
    
    return X, y_regression, y_classification


def train_models(X, y_reg, y_clf):
    """Train regression and classification models"""
    
    # Split data
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    _, _, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )
    
    print("[Training] Regression Model (XGBoost)...")
    xgb_reg = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    xgb_reg.fit(X_train, y_reg_train)
    r2 = r2_score(y_reg_test, xgb_reg.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_reg_test, xgb_reg.predict(X_test)))
    print(f"  ✓ R² = {r2:.4f}, RMSE = ${rmse:,.0f}")
    
    print("[Training] Classification Model (XGBoost)...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    xgb_clf.fit(X_train, y_clf_train)
    acc = accuracy_score(y_clf_test, xgb_clf.predict(X_test))
    print(f"  ✓ Accuracy = {acc:.4f}")
    
    return xgb_reg, xgb_clf


def main():
    """Full pipeline"""
    print("\n" + "=" * 60)
    print("PRODUCTION TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load
    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"  ✓ Loaded {len(df)} startups, {len(df.columns)} columns")
    
    # 2. Engineer features
    print("\n[2/6] Engineering features...")
    df = engineer_features(df)
    print(f"  ✓ Created derived features")
    
    # 3. Preprocess
    print("\n[3/6] Preprocessing (scale & encode)...")
    df, scaler = preprocess_data(df)
    print(f"  ✓ Scaled & encoded. Final shape: {df.shape}")
    
    # 4. Prepare
    print("\n[4/6] Preparing features & targets...")
    X, y_reg, y_clf = prepare_features_targets(df)
    print(f"  ✓ Features: {X.shape}, Targets: {y_reg.shape}, {y_clf.shape}")
    
    # 5. Train
    print("\n[5/6] Training models...")
    xgb_reg, xgb_clf = train_models(X, y_reg, y_clf)
    
    # 6. Save
    print("\n[6/6] Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_reg, 'models/xgboost_regressor.joblib')
    joblib.dump(xgb_clf, 'models/xgboost_classifier.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(X.columns.tolist(), 'models/feature_names.joblib')
    print("  ✓ Saved all models")
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
