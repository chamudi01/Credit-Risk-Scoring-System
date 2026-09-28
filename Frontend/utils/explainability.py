import os
import joblib
import pandas as pd


# ---------------------------------------------------------
# Human-readable feature names
# ---------------------------------------------------------

FEATURE_LABELS = {
    "LIMIT_BAL": "Credit Limit",
    "SEX": "Gender",
    "EDUCATION": "Education Level",
    "MARRIAGE": "Marital Status",
    "AGE": "Age",

    "PAY_0": "September Repayment Status",
    "PAY_2": "August Repayment Status",
    "PAY_3": "July Repayment Status",
    "PAY_4": "June Repayment Status",
    "PAY_5": "May Repayment Status",
    "PAY_6": "April Repayment Status",

    "BILL_AMT1": "September Bill Amount",
    "BILL_AMT2": "August Bill Amount",
    "BILL_AMT3": "July Bill Amount",
    "BILL_AMT4": "June Bill Amount",
    "BILL_AMT5": "May Bill Amount",
    "BILL_AMT6": "April Bill Amount",

    "PAY_AMT1": "September Payment Amount",
    "PAY_AMT2": "August Payment Amount",
    "PAY_AMT3": "July Payment Amount",
    "PAY_AMT4": "June Payment Amount",
    "PAY_AMT5": "May Payment Amount",
    "PAY_AMT6": "April Payment Amount",
}


# ---------------------------------------------------------
# Search for saved importance files
# ---------------------------------------------------------

def find_file(filename):
    """
    Search common project locations for an artifact.
    """

    possible_paths = [
        os.path.join(
            "notebook11_final_outputs",
            filename
        ),
        os.path.join(
            "notebook8_outputs",
            "final_artifacts",
            filename
        ),
        os.path.join(
            "final_artifacts",
            filename
        ),
        os.path.join(
            "outputs",
            "final_artifacts",
            filename
        ),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


# ---------------------------------------------------------
# Load SHAP importance
# ---------------------------------------------------------

def load_shap_importance():
    """
    Load the global SHAP feature-importance results
    produced by the modelling notebooks.
    """

    path = find_file("final_shap_importance.csv")

    if path is None:
        return None

    df = pd.read_csv(path)

    return df


# ---------------------------------------------------------
# Load permutation importance
# ---------------------------------------------------------

def load_permutation_importance():
    """
    Load the global permutation-importance results.
    """

    path = find_file("final_permutation_importance.csv")

    if path is None:
        return None

    df = pd.read_csv(path)

    return df


# ---------------------------------------------------------
# Convert raw feature names to readable names
# ---------------------------------------------------------

def readable_feature_name(feature):
    return FEATURE_LABELS.get(
        feature,
        feature.replace("_", " ").title()
    )


# ---------------------------------------------------------
# Prepare global risk explanation
# ---------------------------------------------------------

def get_global_risk_factors(top_n=5):
    """
    Return the most important model features.

    This is GLOBAL model explanation, not a causal explanation
    of an individual customer.
    """

    shap_df = load_shap_importance()

    if shap_df is None or shap_df.empty:
        return None

    # Find the feature column
    feature_column = None

    for candidate in [
        "Original Feature",
        "Feature",
        "feature",
    ]:
        if candidate in shap_df.columns:
            feature_column = candidate
            break

    if feature_column is None:
        return None

    # Find SHAP importance column
    importance_column = None

    for candidate in [
        "Mean Absolute SHAP",
        "mean_abs_shap",
        "Importance",
    ]:
        if candidate in shap_df.columns:
            importance_column = candidate
            break

    if importance_column is None:
        return None

    result = shap_df.copy()

    result["Readable Feature"] = result[
        feature_column
    ].apply(readable_feature_name)

    result = result.sort_values(
        importance_column,
        ascending=False
    ).head(top_n)

    return result