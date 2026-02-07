import streamlit as st
import pandas as pd
import os

from src.models import load_model


st.title("Startup Valuation & Success Predictor — Demo")

st.write("Provide simple company inputs to get a valuation prediction (toy demo).")

age = st.number_input("Age (months)", min_value=1, max_value=600, value=12)
founders = st.number_input("Number of founders", min_value=1, max_value=10, value=2)
team = st.number_input("Team size", min_value=1, max_value=2000, value=10)
raised = st.number_input("Raised USD", min_value=0.0, value=50000.0)

model_path = os.path.join("models", "baseline.joblib")

if st.button("Predict"):
    if not os.path.exists(model_path):
        st.error("Model not found. Run `python src/train.py` to create a baseline model.")
    else:
        model = load_model(model_path)
        X = pd.DataFrame([{"age_months": age, "founders": founders, "team_size": team, "raised_usd": raised}])
        pred = model.predict(X)[0]
        st.success(f"Predicted valuation (USD): {pred:,.0f}")
