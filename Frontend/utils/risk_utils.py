
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

DECISION_THRESHOLD = 0.31
RISK_BANDS = {
    "Low Risk": (0.00, 0.20),
    "Medium Risk": (0.20, 0.31),
    "High Risk": (0.31, 0.50),
    "Very High Risk": (0.50, 1.01),
}

FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "xgboost_pipeline.joblib"
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_customers.csv"


@st.cache_resource(show_spinner="Loading trained XGBoost pipeline...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_sample_customers():
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Sample data not found: {SAMPLE_PATH}")
    df = pd.read_csv(SAMPLE_PATH)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Sample data is missing model features: {missing}")
    if "Customer_ID" not in df.columns:
        df.insert(0, "Customer_ID", [f"R{i + 1:04d}" for i in range(len(df))])
    return df


def validate_features(df: pd.DataFrame):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")
    return df[FEATURES].copy()


def predict_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    X = validate_features(df)
    probabilities = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["Default Probability"] = probabilities
    out["Risk Category"] = [risk_category(p) for p in probabilities]
    out["Decision"] = [
        "High-Risk Flag" if p >= DECISION_THRESHOLD else "Lower-Risk Flag"
        for p in probabilities
    ]
    return out


def risk_category(probability: float) -> str:
    p = float(probability)
    if p < 0.20:
        return "Low Risk"
    if p < 0.31:
        return "Medium Risk"
    if p < 0.50:
        return "High Risk"
    return "Very High Risk"


def decision(probability: float) -> str:
    return "High-Risk Flag" if float(probability) >= DECISION_THRESHOLD else "Lower-Risk Flag"


def risk_recommendations(risk: str):
    return {
        "Low Risk": [
            "Maintain standard account monitoring.",
            "Continue normal credit-management procedures.",
            "Review changes in repayment behavior over time.",
        ],
        "Medium Risk": [
            "Increase monitoring of recent payment behavior.",
            "Review changes in repayment patterns before increasing exposure.",
            "Consider conservative credit-limit decisions according to bank policy.",
        ],
        "High Risk": [
            "Flag the account for additional review.",
            "Review recent and historical payment behavior.",
            "Consider tighter exposure controls according to bank policy.",
            "Increase monitoring frequency.",
        ],
        "Very High Risk": [
            "Flag for immediate manual review.",
            "Review repayment history and outstanding balances.",
            "Consider restricting additional exposure according to bank policy.",
            "Apply institutional risk-management procedures.",
        ],
    }[risk]


def format_feature_value(feature, value):
    if feature == "LIMIT_BAL":
        return f"{value:,.0f}"
    if feature.startswith("BILL_AMT") or feature.startswith("PAY_AMT"):
        return f"{value:,.0f}"
    return f"{value:g}"


def _aggregate_shap_features(transformed_names, shap_values):
    rows = []
    for name, value in zip(transformed_names, shap_values):
        original = name.split("__", 1)[1] if "__" in name else name
        # One-hot features look like SEX_1.0, EDUCATION_2.0, etc.
        for feature in FEATURES:
            if original == feature or original.startswith(feature + "_"):
                original = feature
                break
        rows.append((original, float(value)))
    return pd.DataFrame(rows, columns=["Feature", "SHAP"])


def explain_prediction(model, input_df, top_n=5):
    """
    Try local SHAP explanation first. If SHAP is unavailable/incompatible,
    fall back to the trained XGBoost model's global feature importance.
    """
    try:
        import shap

        pipeline = model
        preprocessor = pipeline.named_steps["preprocessor"]
        xgb = pipeline.named_steps["model"]

        transformed = preprocessor.transform(validate_features(input_df))
        names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(xgb)
        values = explainer.shap_values(transformed)

        if isinstance(values, list):
            values = values[1]

        values = np.asarray(values)
        if values.ndim == 2:
            values = values[0]

        agg = _aggregate_shap_features(names, values)
        agg = (
            agg.groupby("Feature", as_index=False)["SHAP"]
            .sum()
            .assign(Abs_SHAP=lambda d: d["SHAP"].abs())
            .sort_values("Abs_SHAP", ascending=False)
            .head(top_n)
        )
        return agg, "SHAP local explanation"

    except Exception:
        # Compatibility-safe fallback using the model's trained feature importance.
        try:
            pipeline = model
            preprocessor = pipeline.named_steps["preprocessor"]
            xgb = pipeline.named_steps["model"]
            names = preprocessor.get_feature_names_out()
            importances = np.asarray(xgb.feature_importances_)

            rows = []
            for name, importance in zip(names, importances):
                original = name.split("__", 1)[1] if "__" in name else name
                for feature in FEATURES:
                    if original == feature or original.startswith(feature + "_"):
                        original = feature
                        break
                rows.append((original, float(importance)))

            agg = pd.DataFrame(rows, columns=["Feature", "Importance"])
            agg = (
                agg.groupby("Feature", as_index=False)["Importance"]
                .sum()
                .sort_values("Importance", ascending=False)
                .head(top_n)
            )
            return agg, "Global model feature importance (SHAP fallback)"
        except Exception as exc:
            return pd.DataFrame(), f"Explanation unavailable: {exc}"


# ---------------------------------------------------------
# Reference data shared by the pages
# ---------------------------------------------------------

BAND_ORDER = list(RISK_BANDS.keys())

# Oldest month first, the way a statement history reads.
# (month, repayment status, statement balance, payment made)
HISTORY_MONTHS = [
    ("April", "PAY_6", "BILL_AMT6", "PAY_AMT6"),
    ("May", "PAY_5", "BILL_AMT5", "PAY_AMT5"),
    ("June", "PAY_4", "BILL_AMT4", "PAY_AMT4"),
    ("July", "PAY_3", "BILL_AMT3", "PAY_AMT3"),
    ("August", "PAY_2", "BILL_AMT2", "PAY_AMT2"),
    ("September", "PAY_0", "BILL_AMT1", "PAY_AMT1"),
]

PAY_STATUS = {
    -2: "No usage",
    -1: "Paid in full",
    0: "Minimum paid (revolving)",
    1: "1 month late",
    2: "2 months late",
    3: "3 months late",
    4: "4 months late",
    5: "5 months late",
    6: "6 months late",
    7: "7 months late",
    8: "8 months late",
    9: "9+ months late",
}
SEX = {1: "Male", 2: "Female"}
EDUCATION = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
MARRIAGE = {1: "Married", 2: "Single", 3: "Other"}

FEATURE_LABELS = {
    "LIMIT_BAL": "Credit limit",
    "SEX": "Gender",
    "EDUCATION": "Education",
    "MARRIAGE": "Marital status",
    "AGE": "Age",
}
for _month, _pay, _bill, _amt in HISTORY_MONTHS:
    FEATURE_LABELS[_pay] = f"{_month} repayment status"
    FEATURE_LABELS[_bill] = f"{_month} statement balance"
    FEATURE_LABELS[_amt] = f"{_month} payment made"


def status_label(code) -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return str(code)
    return PAY_STATUS.get(code, f"Code {code}")


@st.cache_data(show_spinner="Scoring accounts...")
def get_scored() -> pd.DataFrame:
    """Sample customers with probability, band and decision columns."""
    return predict_dataframe(load_model(), load_sample_customers())


def _original_feature(transformed_name: str) -> str:
    original = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    for feature in FEATURES:
        if original == feature or original.startswith(feature + "_"):
            return feature
    return original


@st.cache_data(show_spinner=False)
def global_importance() -> pd.DataFrame:
    """Model-wide XGBoost importance, summed back to the 23 raw inputs."""
    model = load_model()
    pre = model.named_steps["preprocessor"]
    xgb = model.named_steps["model"]
    names = pre.get_feature_names_out()
    rows = [(_original_feature(n), float(v)) for n, v in zip(names, xgb.feature_importances_)]
    out = (
        pd.DataFrame(rows, columns=["Feature", "Importance"])
        .groupby("Feature", as_index=False)["Importance"].sum()
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    out["Importance"] = out["Importance"] / out["Importance"].sum()
    return out
