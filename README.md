# Startup Valuation & Success Predictor

This repository is a scaffold for a project to predict startup valuation and success probability from company features.

Tech stack (what each does):
- Python: primary language for data science and backend logic.
- pandas / numpy: data loading, cleaning, and numeric operations.
- scikit-learn: preprocessing, baseline models, model selection, and evaluation.
- XGBoost: high-performance gradient boosting for tabular data (strong baseline/production model).
- SHAP: model-interpretability (feature importance and explanations).
- joblib / pickle: save/load trained models.
- Streamlit: quick interactive UI for demos and simple deployment.
- matplotlib / seaborn: visualization during EDA.

Setup (Windows):
1. Create virtualenv and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install deps:

```bash
pip install -r requirements.txt
```

3. Train a quick baseline model:

```bash
python src/train.py
```

4. Run the demo app:

```bash
streamlit run app.py
```

Next recommended steps:
- Collect a dataset (Crunchbase, AngelList, PitchBook or public datasets).
- Run EDA in `src/eda.py` or a notebook under `notebooks/`.
- Implement feature engineering in `src/features.py`.
