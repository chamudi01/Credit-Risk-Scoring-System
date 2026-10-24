import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import ui
from utils.risk_utils import (
    DECISION_THRESHOLD, FEATURE_LABELS, FEATURES, MODEL_PATH, PAY_STATUS, RISK_BANDS, SAMPLE_PATH,
    decision, global_importance, load_model, load_sample_customers,
)

# Optional file with training-data details and evaluation metrics (format in README.md).
# Nothing is shown from it unless it exists, and nothing is estimated when it does not.
CARD_PATH = MODEL_PATH.parent / "model_card.json"

CONFIG_LABELS = [
    ("n_estimators", "Number of trees"),
    ("max_depth", "Maximum tree depth"),
    ("learning_rate", "Learning rate"),
    ("subsample", "Row subsample"),
    ("colsample_bytree", "Column subsample per tree"),
    ("min_child_weight", "Minimum child weight"),
    ("gamma", "Minimum loss reduction (gamma)"),
    ("reg_alpha", "L1 regularisation"),
    ("reg_lambda", "L2 regularisation"),
    ("scale_pos_weight", "Positive class weight"),
    ("objective", "Objective"),
]


def load_model_card():
    try:
        with open(CARD_PATH, encoding="utf-8") as fh:
            card = json.load(fh)
        return card if isinstance(card, dict) else None
    except (OSError, ValueError):
        return None


def model_configuration():
    """Hyperparameters read from the saved XGBoost model, or None if they cannot be read."""
    try:
        params = load_model().named_steps["model"].get_params()
    except Exception:
        return None
    rows = [(label, str(params[key])) for key, label in CONFIG_LABELS if params.get(key) is not None]
    return pd.DataFrame(rows, columns=["Setting", "Value"]) if rows else None


def fmt_value(key, value):
    if key == "default_rate" and isinstance(value, (int, float)) and 0 <= value <= 1:
        return f"{value:.1%}"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) <= 1 else f"{value:,.2f}"
    return str(value)


card = load_model_card()

ui.page_header(
    "Model documentation",
    "Purpose, inputs, methodology and limitations of the credit risk scoring model.",
)

t_overview, t_inputs, t_eval, t_bands, t_limits = st.tabs(
    ["Overview", "Inputs and data", "Evaluation", "Risk classification", "Limitations"]
)

# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------
with t_overview:
    ui.section("Purpose")
    st.markdown(
        "The model estimates the probability that a customer account will default. It uses the customer's profile, "
        "credit limit and six months of repayment status, statement balances and payments. Scores support portfolio "
        "monitoring and manual credit review. They are not automated credit decisions."
    )

    ui.section("About XGBoost")
    st.markdown(
        "XGBoost (extreme gradient boosting) builds many small decision trees one after another. Each new tree is "
        "trained to correct the errors left by the trees before it, and the combined output is converted into a "
        "probability between 0 and 1. It can capture non-linear patterns and interactions between inputs, such as a "
        "rising balance combined with late payments, without those relationships being specified by hand."
    )

    ui.section("Scoring flow")
    ui.flow([
        ("Customer inputs", f"{len(FEATURES)} raw fields: profile, repayment status, balances and payments."),
        ("Saved pipeline", "Preprocessing and an XGBoost classifier, loaded from disk. Nothing is retrained at runtime."),
        ("Probability of default", "A value between 0% and 100% for each account."),
        ("Band and decision", f"Four risk bands, with a binary flag at {DECISION_THRESHOLD:.0%}."),
    ])

    config = model_configuration()
    ui.section("Model configuration", "Settings read directly from the saved model file.")
    if config is not None:
        ui.show_table(config)
    else:
        st.caption("Model settings could not be read from the saved file.")

# ---------------------------------------------------------
# Inputs and data
# ---------------------------------------------------------
with t_inputs:
    ui.section("Dataset", f"Sample records bundled with the app ({SAMPLE_PATH.name}).")
    try:
        sample = load_sample_customers()
    except Exception:
        sample = None
    if sample is not None:
        ui.kpi_row([
            ("Records", f"{len(sample):,}", "Sample customers scored in the app"),
            ("Model input fields", f"{len(FEATURES)}", "Raw columns passed to the pipeline"),
            ("Missing values", f"{int(sample[FEATURES].isna().sum().sum()):,}", "Across all model input fields"),
            ("Credit limit range", f"{sample['LIMIT_BAL'].min():,.0f} to {sample['LIMIT_BAL'].max():,.0f}", "NT$"),
            ("Age range", f"{int(sample['AGE'].min())} to {int(sample['AGE'].max())}", "Years"),
        ])
        ui.callout(
            "The sample contains no confirmed default outcomes, so it can be scored but not used to measure "
            "the model's accuracy.",
            lead="Scope of the sample.",
        )
    else:
        st.caption("The sample data file could not be read.")

    training = (card or {}).get("training_data")
    if isinstance(training, dict) and training:
        ui.section("Training data", "Recorded in model_card.json.")
        labels = [("name", "Dataset"), ("source", "Source"), ("records", "Records"),
                  ("default_rate", "Default rate"), ("period", "Period"), ("notes", "Notes")]
        rows = [(label, fmt_value(k, training[k])) for k, label in labels if training.get(k) not in (None, "")]
        ui.show_table(pd.DataFrame(rows, columns=["Item", "Detail"]))
    else:
        st.caption("Training-data details are not stored with the saved pipeline, so they are not shown here.")

    ui.section("Input fields")
    fields = pd.DataFrame([
        ("Credit limit", "LIMIT_BAL", "Credit line in NT$"),
        ("Gender", "SEX", "1 male, 2 female"),
        ("Education", "EDUCATION", "1 graduate school, 2 university, 3 high school, 4 other"),
        ("Marital status", "MARRIAGE", "1 married, 2 single, 3 other"),
        ("Age", "AGE", "Years"),
        ("Repayment status, six months", "PAY_0, PAY_2 to PAY_6", "Status code per month, see below"),
        ("Statement balance, six months", "BILL_AMT1 to BILL_AMT6", "NT$, September back to April"),
        ("Payment made, six months", "PAY_AMT1 to PAY_AMT6", "NT$, September back to April"),
    ], columns=["Field", "Model column", "Values"])
    ui.show_table(fields)

    ui.section("Repayment status codes")
    codes = pd.DataFrame({"Code": [str(c) if c < 9 else "9" for c in PAY_STATUS], "Meaning": list(PAY_STATUS.values())})
    ui.show_table(codes, height=460)

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
with t_eval:
    ui.section("Evaluation metrics")
    evaluation = (card or {}).get("evaluation")
    metrics = evaluation.get("metrics") if isinstance(evaluation, dict) else None
    if isinstance(metrics, dict) and metrics:
        ui.kpi_row([(name, fmt_value(name, value), None) for name, value in metrics.items()])
        basis = [str(evaluation[k]) for k in ("dataset", "records") if evaluation.get(k) not in (None, "")]
        if basis:
            st.caption("Evaluated on: " + ", ".join(basis) + ". Recorded in model_card.json.")
    else:
        ui.callout(
            "Evaluation metrics are not stored with the saved pipeline, so none are shown. To display them, add "
            "model/model_card.json (format in README.md). No figures are estimated here.",
            kind="caution",
            lead="Metrics not available.",
        )

    ui.section("Feature importance", "The ten inputs the trained model relies on most.")
    importance = global_importance()
    imp = importance.head(10).sort_values("Importance")
    with ui.panel("Share of total XGBoost importance", "Importance is summed back to the original input fields."):
        fig = go.Figure(go.Bar(
            x=imp["Importance"], y=[FEATURE_LABELS.get(f, f) for f in imp["Feature"]], orientation="h",
            marker_color=ui.PRIMARY, hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ))
        ui.style(fig, height=360)
        fig.update_xaxes(tickformat=".0%", title_text="Share of total XGBoost importance")
        fig.update_yaxes(showgrid=False)
        ui.show_chart(fig)
        st.caption("Model-wide importance shows what the model uses, not what causes default.")
    with st.expander("All input fields"):
        full = importance.assign(Field=importance["Feature"].map(lambda f: FEATURE_LABELS.get(f, f)))
        ui.show_table(
            pd.DataFrame({"Field": full["Field"], "Model column": full["Feature"], "Importance": full["Importance"] * 100}),
            column_config={"Importance": st.column_config.NumberColumn(format="%.1f%%")},
        )

# ---------------------------------------------------------
# Risk classification
# ---------------------------------------------------------
with t_bands:
    ui.section(
        "Risk bands",
        f"The {DECISION_THRESHOLD:.0%} cut-off was selected during model evaluation as the binary classification "
        "threshold. It does not mean a customer has a 31% chance of default.",
    )
    handling = {
        "Low Risk": "Standard monitoring",
        "Medium Risk": "Increased monitoring of payment behaviour",
        "High Risk": "Flag for additional review",
        "Very High Risk": "Flag for immediate manual review",
    }
    ui.show_table(pd.DataFrame({
        "Band": list(RISK_BANDS.keys()),
        "Predicted probability": [ui.band_range_label(b) for b in RISK_BANDS],
        "Decision flag": [decision(lo) for lo, _ in RISK_BANDS.values()],
        "Typical handling": [handling[b] for b in RISK_BANDS],
    }))
    st.caption(
        "Bands are fixed probability ranges applied to the model's predicted probability. The decision flag is "
        "High-Risk at or above the threshold and Lower-Risk below it."
    )

# ---------------------------------------------------------
# Limitations
# ---------------------------------------------------------
with t_limits:
    ui.section("Scope and limitations")
    for lead, text in [
        ("Prototype.", "The app scores sample records and manual entries. It is not connected to a bank customer database."),
        ("Predictions only.", "Outputs are model estimates. The sample contains no confirmed outcomes against which to check them."),
        ("Sample size.", f"Group-level figures are based on the {len(sample):,} bundled records and can move noticeably "
                         "with a different sample." if sample is not None else
                         "Group-level figures are based on a small sample and can vary noticeably."),
        ("Population shift.", "Scores depend on the training data and can shift on customer populations that differ from it."),
        ("Personal characteristics.", "Gender, marital status, education and age are model inputs. Any real use would "
                                      "need fairness testing and review under applicable regulation."),
        ("Human oversight.", "Credit decisions need review under institutional policy. The score supports that review."),
        ("Production requirements.", "Real use would need validation, monitoring, privacy and security controls, and governance."),
    ]:
        st.markdown(f"**{lead}** {text}")

ui.footnote("Academic demonstration. Not a production banking credit-decision system.")
