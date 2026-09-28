import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Risk Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

DECISION_THRESHOLD = 0.31

FEATURE_ORDER = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


# ============================================================
# FRIENDLY FEATURE NAMES
# ============================================================

FEATURE_NAMES = {

    "LIMIT_BAL":
        "Credit Limit (NT$)",

    "SEX":
        "Customer Gender",

    "EDUCATION":
        "Highest Education Level",

    "MARRIAGE":
        "Marital Status",

    "AGE":
        "Customer Age",

    "PAY_0":
        "Most Recent Payment Status — September",

    "PAY_2":
        "Payment Status — August",

    "PAY_3":
        "Payment Status — July",

    "PAY_4":
        "Payment Status — June",

    "PAY_5":
        "Payment Status — May",

    "PAY_6":
        "Payment Status — April",

    "BILL_AMT1":
        "Latest Monthly Bill — September (NT$)",

    "BILL_AMT2":
        "Monthly Bill — August (NT$)",

    "BILL_AMT3":
        "Monthly Bill — July (NT$)",

    "BILL_AMT4":
        "Monthly Bill — June (NT$)",

    "BILL_AMT5":
        "Monthly Bill — May (NT$)",

    "BILL_AMT6":
        "Monthly Bill — April (NT$)",

    "PAY_AMT1":
        "Latest Payment Amount — September (NT$)",

    "PAY_AMT2":
        "Payment Amount — August (NT$)",

    "PAY_AMT3":
        "Payment Amount — July (NT$)",

    "PAY_AMT4":
        "Payment Amount — June (NT$)",

    "PAY_AMT5":
        "Payment Amount — May (NT$)",

    "PAY_AMT6":
        "Payment Amount — April (NT$)",
}


# ============================================================
# PAYMENT STATUS OPTIONS
# ============================================================

PAYMENT_STATUS_OPTIONS = {
    "No consumption / zero balance": -2,
    "Paid on time / paid in full": -1,
    "Minimum payment / revolving balance": 0,
    "Payment delayed by 1 month": 1,
    "Payment delayed by 2 months": 2,
    "Payment delayed by 3 months": 3,
    "Payment delayed by 4 months": 4,
    "Payment delayed by 5 months": 5,
    "Payment delayed by 6 months": 6,
    "Payment delayed by 7 months": 7,
    "Payment delayed by 8 months": 8,
    "Payment delayed by 9 months or more": 9,
}


# ============================================================
# EDUCATION OPTIONS
# ============================================================

EDUCATION_OPTIONS = {
    "Graduate school": 1,
    "University": 2,
    "High school": 3,
    "Other / Not specified": 4,
}


# ============================================================
# MARRIAGE OPTIONS
# ============================================================

MARRIAGE_OPTIONS = {
    "Married": 1,
    "Single": 2,
    "Other / Not specified": 3,
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    possible_paths = [

        os.path.join(
            "model",
            "xgboost_pipeline.pkl"
        ),

        os.path.join(
            "model",
            "xgboost_pipeline.joblib"
        ),

        os.path.join(
            "model",
            "final_model_pipeline.joblib"
        ),

        os.path.join(
            "models",
            "xgboost_pipeline.pkl"
        ),

        os.path.join(
            "models",
            "final_model_pipeline.joblib"
        ),

        os.path.join(
            "final_artifacts",
            "final_model_pipeline.joblib"
        ),

    ]

    for path in possible_paths:

        if os.path.exists(path):

            return joblib.load(path)

    raise FileNotFoundError(
        "Model file was not found. "
        "Please place the trained model inside the "
        "'model' folder."
    )


try:

    model = load_model()

except Exception as e:

    st.error(
        "The trained model could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(probability):

    if probability < 0.20:

        return "Low Risk"

    elif probability < 0.31:

        return "Medium Risk"

    elif probability < 0.50:

        return "High Risk"

    else:

        return "Very High Risk"


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📊 Credit Risk Prediction")

st.markdown(
    """
    Enter the customer's financial and repayment information
    to estimate the probability of credit-card default.
    """
)

st.info(
    "The model uses historical credit and repayment information. "
    "The prediction is a machine-learning risk estimate and "
    "should be used as decision support rather than as an "
    "automatic credit decision."
)


# ============================================================
# SECTION 1 — CUSTOMER INFORMATION
# ============================================================

st.header("1. Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    credit_limit = st.number_input(
        "Credit Limit (NT$)",
        min_value=0.0,
        value=50000.0,
        step=5000.0,
        help=(
            "Total credit limit available to the customer."
        )
    )


with col2:

    gender_label = st.selectbox(
        "Customer Gender",
        list([
            "Male",
            "Female"
        ]),
        help="Customer gender."
    )

    gender = {
        "Male": 1,
        "Female": 2
    }[gender_label]


with col3:

    age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
        help="Customer's age in years."
    )


education_label = st.selectbox(
    "Highest Education Level",
    list(EDUCATION_OPTIONS.keys()),
    help="Highest reported education level."
)

education = EDUCATION_OPTIONS[
    education_label
]


marriage_label = st.selectbox(
    "Marital Status",
    list(MARRIAGE_OPTIONS.keys()),
    help="Customer's marital status."
)

marriage = MARRIAGE_OPTIONS[
    marriage_label
]


# ============================================================
# SECTION 2 — PAYMENT HISTORY
# ============================================================

st.header("2. Recent Payment History")

st.markdown(
    """
    Select the customer's repayment status for each month.
    
    **The most recent month is September.**
    
    A higher delay value means a longer payment delay.
    """
)


pay_values = {}


pay_columns = [
    ("PAY_0", "September"),
    ("PAY_2", "August"),
    ("PAY_3", "July"),
    ("PAY_4", "June"),
    ("PAY_5", "May"),
    ("PAY_6", "April"),
]


for row_start in range(0, 6, 2):

    col1, col2 = st.columns(2)

    feature1, month1 = pay_columns[row_start]

    with col1:

        selected_label = st.selectbox(
            f"{month1} Payment Status",
            list(PAYMENT_STATUS_OPTIONS.keys()),
            key=feature1,
            help=(
                "Repayment status recorded for "
                f"{month1}."
            )
        )

        pay_values[feature1] = PAYMENT_STATUS_OPTIONS[
            selected_label
        ]

    if row_start + 1 < len(pay_columns):

        feature2, month2 = pay_columns[row_start + 1]

        with col2:

            selected_label = st.selectbox(
                f"{month2} Payment Status",
                list(PAYMENT_STATUS_OPTIONS.keys()),
                key=feature2,
                help=(
                    "Repayment status recorded for "
                    f"{month2}."
                )
            )

            pay_values[feature2] = PAYMENT_STATUS_OPTIONS[
                selected_label
            ]


# ============================================================
# SECTION 3 — MONTHLY BILL AMOUNTS
# ============================================================

st.header("3. Monthly Bill Amounts")

st.markdown(
    "Enter the total amount shown on each monthly statement."
)


bill_values = {}


bill_columns = [
    ("BILL_AMT1", "September"),
    ("BILL_AMT2", "August"),
    ("BILL_AMT3", "July"),
    ("BILL_AMT4", "June"),
    ("BILL_AMT5", "May"),
    ("BILL_AMT6", "April"),
]


for row_start in range(0, 6, 2):

    col1, col2 = st.columns(2)

    feature1, month1 = bill_columns[row_start]

    with col1:

        bill_values[feature1] = st.number_input(
            f"{month1} Monthly Bill (NT$)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            key=f"bill_{feature1}",
            help=(
                f"Total billed amount for {month1}."
            )
        )

    if row_start + 1 < len(bill_columns):

        feature2, month2 = bill_columns[row_start + 1]

        with col2:

            bill_values[feature2] = st.number_input(
                f"{month2} Monthly Bill (NT$)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                key=f"bill_{feature2}",
                help=(
                    f"Total billed amount for {month2}."
                )
            )


# ============================================================
# SECTION 4 — PREVIOUS PAYMENT AMOUNTS
# ============================================================

st.header("4. Previous Payment Amounts")

st.markdown(
    "Enter the amount the customer actually paid each month."
)


payment_amount_values = {}


payment_amount_columns = [
    ("PAY_AMT1", "September"),
    ("PAY_AMT2", "August"),
    ("PAY_AMT3", "July"),
    ("PAY_AMT4", "June"),
    ("PAY_AMT5", "May"),
    ("PAY_AMT6", "April"),
]


for row_start in range(0, 6, 2):

    col1, col2 = st.columns(2)

    feature1, month1 = payment_amount_columns[row_start]

    with col1:

        payment_amount_values[feature1] = st.number_input(
            f"{month1} Payment Amount (NT$)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            key=f"payment_{feature1}",
            help=(
                f"Amount actually paid during {month1}."
            )
        )

    if row_start + 1 < len(payment_amount_columns):

        feature2, month2 = payment_amount_columns[row_start + 1]

        with col2:

            payment_amount_values[feature2] = st.number_input(
                f"{month2} Payment Amount (NT$)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                key=f"payment_{feature2}",
                help=(
                    f"Amount actually paid during {month2}."
                )
            )


# ============================================================
# CREATE MODEL INPUT
# ============================================================

input_data = {

    "LIMIT_BAL": credit_limit,

    "SEX": gender,

    "EDUCATION": education,

    "MARRIAGE": marriage,

    "AGE": age,

    "PAY_0": pay_values["PAY_0"],

    "PAY_2": pay_values["PAY_2"],

    "PAY_3": pay_values["PAY_3"],

    "PAY_4": pay_values["PAY_4"],

    "PAY_5": pay_values["PAY_5"],

    "PAY_6": pay_values["PAY_6"],

    "BILL_AMT1": bill_values["BILL_AMT1"],

    "BILL_AMT2": bill_values["BILL_AMT2"],

    "BILL_AMT3": bill_values["BILL_AMT3"],

    "BILL_AMT4": bill_values["BILL_AMT4"],

    "BILL_AMT5": bill_values["BILL_AMT5"],

    "BILL_AMT6": bill_values["BILL_AMT6"],

    "PAY_AMT1": payment_amount_values["PAY_AMT1"],

    "PAY_AMT2": payment_amount_values["PAY_AMT2"],

    "PAY_AMT3": payment_amount_values["PAY_AMT3"],

    "PAY_AMT4": payment_amount_values["PAY_AMT4"],

    "PAY_AMT5": payment_amount_values["PAY_AMT5"],

    "PAY_AMT6": payment_amount_values["PAY_AMT6"],

}


input_df = pd.DataFrame(
    [input_data],
    columns=FEATURE_ORDER
)


# ============================================================
# VERIFY FEATURE ORDER
# ============================================================

assert list(input_df.columns) == FEATURE_ORDER

assert input_df.shape == (1, 23)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Assess Credit Risk",
    type="primary",
    use_container_width=True
)


if predict_button:

    try:

        # --------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        probability = np.clip(
            probability,
            0.0,
            1.0
        )


        # --------------------------------------------
        # RISK CLASSIFICATION
        # --------------------------------------------

        risk_level = classify_risk(
            probability
        )


        # --------------------------------------------
        # DECISION FLAG
        # --------------------------------------------

        if probability >= DECISION_THRESHOLD:

            decision = "High-Risk Review Required"

        else:

            decision = "Below Decision Threshold"


        # --------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------

        st.divider()

        st.header("Credit Risk Assessment")


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Predicted Default Probability",
                f"{probability:.1%}"
            )


        with col2:

            st.metric(
                "Risk Level",
                risk_level
            )


        with col3:

            st.metric(
                "Decision Threshold",
                f"{DECISION_THRESHOLD:.0%}"
            )


        # --------------------------------------------
        # RESULT MESSAGE
        # --------------------------------------------

        if probability >= 0.50:

            st.error(
                f"⚠️ {risk_level}"
            )

        elif probability >= DECISION_THRESHOLD:

            st.warning(
                f"⚠️ {risk_level}"
            )

        elif probability >= 0.20:

            st.info(
                f"ℹ️ {risk_level}"
            )

        else:

            st.success(
                f"✓ {risk_level}"
            )


        st.subheader("Assessment Status")

        if probability >= DECISION_THRESHOLD:

            st.write(
                "The predicted probability is at or above "
                "the configured 31% decision threshold. "
                "The customer should be considered for "
                "additional review according to institutional policy."
            )

        else:

            st.write(
                "The predicted probability is below the "
                "configured 31% decision threshold."
            )


        # --------------------------------------------
        # PROBABILITY BAR
        # --------------------------------------------

        st.subheader("Default Probability")

        st.progress(
            probability
        )

        st.caption(
            f"Model probability: {probability:.2%}"
        )


        # --------------------------------------------
        # INPUT SUMMARY
        # --------------------------------------------

        with st.expander(
            "View submitted customer information"
        ):

            friendly_input = pd.DataFrame({

                "Information": [
                    FEATURE_NAMES[f]
                    for f in FEATURE_ORDER
                ],

                "Value": [
                    input_data[f]
                    for f in FEATURE_ORDER
                ]

            })

            # Create user-friendly values for display.
            # Keep the original numeric model input unchanged.

            display_values = []

            for feature in FEATURE_ORDER:

                value = input_data[feature]

                if feature.startswith("PAY_") and feature not in [
                    "PAY_AMT1",
                    "PAY_AMT2",
                    "PAY_AMT3",
                    "PAY_AMT4",
                    "PAY_AMT5",
                    "PAY_AMT6",
                ]:

                    for label, numeric_value in PAYMENT_STATUS_OPTIONS.items():

                        if numeric_value == value:

                            display_values.append(label)
                            break

                elif feature == "SEX":

                    display_values.append(
                        gender_label
                    )

                elif feature == "EDUCATION":

                    display_values.append(
                        education_label
                    )

                elif feature == "MARRIAGE":

                    display_values.append(
                        marriage_label
                    )

                else:

                    display_values.append(
                        value
                    )

            friendly_input["Value"] = display_values

            st.dataframe(
                friendly_input,
                use_container_width=True,
                hide_index=True
            )


        # --------------------------------------------
        # IMPORTANT DISCLAIMER
        # --------------------------------------------

        st.caption(
            "This prediction is generated by a machine-learning "
            "model and is intended for decision support. "
            "It should not be treated as an automatic banking "
            "approval or rejection decision."
        )


    except Exception as e:

        st.error(
            "Prediction could not be generated."
        )

        st.exception(e)