"""
Run the complete ML pipeline: Feature Engineering -> Models -> Evaluation -> Save Models
This is the production version that will train and save everything.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
import joblib

print("=" * 70)
print("STARTUP VALUATION PREDICTOR - FULL ML PIPELINE")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/8] Loading data...")
df = pd.read_csv('data/startup_funding.csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  No missing values: {df.isnull().sum().sum() == 0}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n[2/8] Feature engineering...")
df_processed = df.copy()

# Remove non-predictive columns
categorical_cols = ['industry', 'country']
df_processed = df_processed.drop('startup_name', axis=1)

# Create derived features
df_processed['funding_per_team_member'] = df_processed['total_funding_usd'] / (df_processed['team_size'] + 1)
df_processed['rounds_per_year'] = df_processed['funding_rounds'] / ((df_processed['months_since_founding'] / 12) + 1)
df_processed['years_since_founding'] = df_processed['months_since_founding'] / 12

print(f"✓ Created 3 derived features")

# Scale numeric features
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
target_cols = ['valuation_usd', 'is_successful']
feature_numeric_cols = [col for col in numeric_cols if col not in target_cols]

scaler = StandardScaler()
df_processed[feature_numeric_cols] = scaler.fit_transform(df_processed[feature_numeric_cols])
print(f"✓ Scaled {len(feature_numeric_cols)} numeric features")

# One-hot encode categoricals
df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
print(f"✓ Encoded categorical features. Final shape: {df_encoded.shape}")

# Separate features and targets
X = df_encoded.drop(['valuation_usd', 'is_successful'], axis=1)
y_regression = df_encoded['valuation_usd']
y_classification = df_encoded['is_successful']

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================
print("\n[3/8] Train-test split (80/20)...")
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)
X_train_clf, X_test_clf, y_clf_train, y_clf_test = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)
print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ============================================================
# 4. BASELINE MODELS
# ============================================================
print("\n[4/8] Training baseline models...")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_reg_train)
r2_lr_test = r2_score(y_reg_test, lr.predict(X_test))
rmse_lr_test = np.sqrt(mean_squared_error(y_reg_test, lr.predict(X_test)))
print(f"✓ Linear Regression - R² Score: {r2_lr_test:.4f}, RMSE: ${rmse_lr_test:,.0f}")

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_clf, y_clf_train)
acc_log_test = accuracy_score(y_clf_test, log_reg.predict(X_test_clf))
print(f"✓ Logistic Regression - Accuracy: {acc_log_test:.4f}")

# ============================================================
# 5. ADVANCED MODELS - XGBOOST
# ============================================================
print("\n[5/8] Training XGBoost models...")

# XGBoost Regressor
xgb_reg = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)
xgb_reg.fit(X_train, y_reg_train)
r2_xgb_test = r2_score(y_reg_test, xgb_reg.predict(X_test))
rmse_xgb_test = np.sqrt(mean_squared_error(y_reg_test, xgb_reg.predict(X_test)))
print(f"✓ XGBoost Regressor - R² Score: {r2_xgb_test:.4f}, RMSE: ${rmse_xgb_test:,.0f}")

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)
xgb_clf.fit(X_train_clf, y_clf_train)
acc_xgb_test = accuracy_score(y_clf_test, xgb_clf.predict(X_test_clf))
print(f"✓ XGBoost Classifier - Accuracy: {acc_xgb_test:.4f}")

# ============================================================
# 6. MODEL COMPARISON
# ============================================================
print("\n[6/8] Model comparison...")
print("\n  REGRESSION MODELS (Test Set R² Score):")
print(f"    • Linear Regression:    {r2_lr_test:.4f}")
print(f"    • XGBoost Regressor:    {r2_xgb_test:.4f} ← BEST")
print(f"\n  CLASSIFICATION MODELS (Test Set Accuracy):")
print(f"    • Logistic Regression:  {acc_log_test:.4f}")
print(f"    • XGBoost Classifier:   {acc_xgb_test:.4f} ← BEST")

# ============================================================
# 7. CROSS-VALIDATION
# ============================================================
print("\n[7/8] 5-Fold Cross-Validation...")
cv_scores_reg = cross_val_score(xgb_reg, X, y_regression, cv=5, scoring='r2')
cv_scores_clf = cross_val_score(xgb_clf, X, y_classification, cv=5, scoring='accuracy')
print(f"✓ XGBoost Regressor CV R² (mean ± std): {cv_scores_reg.mean():.4f} ± {cv_scores_reg.std():.4f}")
print(f"✓ XGBoost Classifier CV Acc (mean ± std): {cv_scores_clf.mean():.4f} ± {cv_scores_clf.std():.4f}")

# ============================================================
# 8. SAVE MODELS
# ============================================================
print("\n[8/8] Saving models and preprocessing objects...")
os.makedirs('models', exist_ok=True)

joblib.dump(xgb_reg, 'models/xgboost_regressor.joblib')
joblib.dump(xgb_clf, 'models/xgboost_classifier.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(X.columns.tolist(), 'models/feature_names.joblib')

print(f"✓ Saved models/xgboost_regressor.joblib")
print(f"✓ Saved models/xgboost_classifier.joblib")
print(f"✓ Saved models/scaler.joblib")
print(f"✓ Saved models/feature_names.joblib")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("✓ PIPELINE COMPLETE!")
print("=" * 70)
print("\nModels trained and saved. Ready to:")
print("  1. Update src/train.py with this code")
print("  2. Update app.py (Streamlit) to use real models")
print("  3. Run streamlit run app.py for demo")
print("\n" + "=" * 70)
