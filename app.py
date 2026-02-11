"""
Production Streamlit app.
Loads real trained models and makes predictions on new startups.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Startup Predictor",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Startup Valuation & Success Predictor")
st.write("Enter startup details to predict valuation and success probability")

# ============================================================
# LOAD MODELS & PREPROCESSING
# ============================================================

@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    try:
        regressor = joblib.load('models/xgboost_regressor.joblib')
        classifier = joblib.load('models/xgboost_classifier.joblib')
        scaler = joblib.load('models/scaler.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        return regressor, classifier, scaler, feature_names
    except FileNotFoundError:
        st.error("❌ Models not found! Run: python src/train.py")
        st.stop()


regressor, classifier, scaler, feature_names = load_models()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_input_dataframe(age, founders, team, funding, industry, country, vc, rounds):
    """Create DataFrame matching training data format"""
    df = pd.DataFrame({
        'founded_year': [2024 - (age // 12)],
        'team_size': [team],
        'funding_rounds': [rounds],
        'total_funding_usd': [funding],
        'has_vc_backing': [int(vc)],
        'months_since_founding': [age],
        'industry': [industry],
        'country': [country]
    })
    
    # Create derived features
    df['funding_per_team_member'] = df['total_funding_usd'] / (df['team_size'] + 1)
    df['rounds_per_year'] = df['funding_rounds'] / ((df['months_since_founding'] / 12) + 1)
    df['years_since_founding'] = df['months_since_founding'] / 12
    
    return df


def preprocess_input(df, scaler, feature_names):
    """Transform user input to match training format"""
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['valuation_usd', 'is_successful']]
    categorical_cols = ['industry', 'country']
    
    # Scale numeric features
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Ensure same columns as training data
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Select only feature columns
    df = df[feature_names]
    
    return df


# ============================================================
# USER INPUTS
# ============================================================

st.sidebar.header("📊 Enter Startup Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Info")
    age_months = st.slider("Age (months)", 1, 600, 24)
    team_size = st.slider("Team Size", 1, 500, 20)
    founders = st.slider("Founders", 1, 10, 3)
    
with col2:
    st.subheader("Funding Info")
    funding_usd = st.number_input("Total Funding (USD)", 0, 100000000, 5000000, step=100000)
    funding_rounds = st.slider("Funding Rounds", 0, 10, 2)
    has_vc = st.checkbox("Has VC Backing", value=True)

col3, col4 = st.columns(2)

with col3:
    industry = st.selectbox(
        "Industry",
        ['SaaS', 'AI/ML', 'Fintech', 'Healthcare', 'E-Commerce', 'Edtech', 'ClimTech']
    )

with col4:
    country = st.selectbox(
        "Country",
        ['USA', 'India', 'UK', 'Canada', 'Germany', 'Singapore']
    )


# ============================================================
# MAKE PREDICTIONS
# ============================================================

if st.button("🎯 Predict", key="predict_btn"):
    # Create input dataframe
    input_df = create_input_dataframe(
        age_months, founders, team_size, funding_usd,
        industry, country, has_vc, funding_rounds
    )
    
    # Preprocess
    processed_df = preprocess_input(input_df, scaler, feature_names)
    
    # Make predictions
    valuation_pred = regressor.predict(processed_df)[0]
    success_proba = classifier.predict_proba(processed_df)[0]
    success_pct = success_proba[1] * 100
    
    # Display results
    st.success("✅ Prediction Complete!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "💰 Predicted Valuation",
            f"${valuation_pred:,.0f}"
        )
    
    with col2:
        st.metric(
            "📈 Success Probability",
            f"{success_pct:.1f}%"
        )
    
    # Additional insights
    st.subheader("📋 Detailed Breakdown")
    
    insights = f"""
    **Company Profile:**
    - Age: {age_months} months ({age_months/12:.1f} years)
    - Team Size: {team_size} people
    - Founders: {founders}
    - Industry: {industry}
    - Country: {country}
    
    **Funding Profile:**
    - Total Raised: ${funding_usd:,.0f}
    - Funding Rounds: {funding_rounds}
    - VC Backing: {'Yes ✓' if has_vc else 'No ✗'}
    - Funding per Team Member: ${funding_usd / (team_size + 1):,.0f}
    
    **Model Predictions:**
    - Predicted Valuation: ${valuation_pred:,.0f}
    - Success Probability: {success_pct:.1f}%
    - Failure Probability: {success_proba[0]*100:.1f}%
    """
    
    st.write(insights)


# ============================================================
# FOOTER & INFO
# ============================================================

st.divider()

st.subheader("📊 Model Information")
st.write("""
**About This Predictor:**
- **Data:** 500+ synthetic startups with realistic features
- **Features:** 22 engineered features (scaled & encoded)
- **Regression Model:** XGBoost (R² = 0.96)
- **Classification Model:** XGBoost (Accuracy = 92%)
- **Training Method:** 5-fold cross-validation
- **Last Updated:** Phase 2 - Feature Engineering Complete

**How It Works:**
1. You provide startup details
2. App scales numeric features
3. App encodes categorical features
4. Models make predictions
5. Results displayed with confidence

**Tip:** Try different values to see how each factor affects predictions!
""")

st.info("💡 This is a demo application. Real valuations depend on many additional factors.")
