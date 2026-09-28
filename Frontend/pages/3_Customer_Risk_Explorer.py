import streamlit as st

from utils.risk_utils import load_model, load_sample_customers, predict_dataframe


# ---------------------------------------------------------
# Page title
# ---------------------------------------------------------
st.title("🔎 Customer Risk Explorer")

st.caption(
    "Explore demo customer records and view their predicted credit-default risk "
    "in an easy-to-understand format."
)


# ---------------------------------------------------------
# Load model and demo customer data
# ---------------------------------------------------------
try:
    model = load_model()
    customers = load_sample_customers()
    scored = predict_dataframe(model, customers)

except Exception as exc:
    st.error(f"Could not load the model or demo data: {exc}")
    st.stop()


# ---------------------------------------------------------
# Risk category filter
# ---------------------------------------------------------
risk_options = [
    "All Customers",
    "Low Risk",
    "Medium Risk",
    "High Risk",
    "Very High Risk",
]

selected = st.selectbox(
    "🎯 Filter Customers by Risk Level",
    risk_options,
)

if selected == "All Customers":
    filtered = scored
else:
    filtered = scored[
        scored["Risk Category"] == selected
    ]


# ---------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------
st.metric(
    "👥 Matching Customers",
    f"{len(filtered):,}"
)


# ---------------------------------------------------------
# Human-readable column names
#
# IMPORTANT:
# These are ONLY display names.
# The original model columns remain unchanged internally.
# ---------------------------------------------------------
display_column_names = {
    "Customer_ID": "Customer ID",

    "LIMIT_BAL": "Credit Limit (NT$)",

    "AGE": "Age",

    "PAY_0": "September Repayment Status",
    "PAY_2": "August Repayment Status",

    "Default Probability": "Default Probability",

    "Risk Category": "Risk Level",

    "Decision": "Model Decision",
}


# ---------------------------------------------------------
# Select columns to display
# ---------------------------------------------------------
display_cols = [
    "Customer_ID",
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "Default Probability",
    "Risk Category",
    "Decision",
]

available = [
    column
    for column in display_cols
    if column in filtered.columns
]


table = filtered[available].copy()


# ---------------------------------------------------------
# Convert repayment status values into understandable text
# ---------------------------------------------------------
def repayment_status_label(value):
    """
    Convert the original UCI repayment-status codes
    into human-readable descriptions.
    """

    try:
        value = int(value)
    except (ValueError, TypeError):
        return str(value)

    repayment_labels = {
        -2: "No consumption / zero balance",
        -1: "Paid in full / paid on time",
        0: "Minimum payment / revolving credit",
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

    return repayment_labels.get(
        value,
        f"Code {value}"
    )


# ---------------------------------------------------------
# Make repayment columns understandable
# ---------------------------------------------------------
for column in [
    "PAY_0",
    "PAY_2",
]:
    if column in table.columns:
        table[column] = table[column].apply(
            repayment_status_label
        )


# ---------------------------------------------------------
# Format credit limit
# ---------------------------------------------------------
if "LIMIT_BAL" in table.columns:
    table["LIMIT_BAL"] = table["LIMIT_BAL"].apply(
        lambda x: f"NT$ {x:,.0f}"
    )


# ---------------------------------------------------------
# Format default probability
# ---------------------------------------------------------
if "Default Probability" in table.columns:
    table["Default Probability"] = table[
        "Default Probability"
    ].apply(
        lambda x: f"{x:.1%}"
    )


# ---------------------------------------------------------
# Rename columns for display
# ---------------------------------------------------------
table = table.rename(
    columns=display_column_names
)


# ---------------------------------------------------------
# Display table
# ---------------------------------------------------------
st.subheader("📋 Customer Risk Details")

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Explanation of the displayed information
# ---------------------------------------------------------
st.subheader("ℹ️ What the Columns Mean")

st.markdown(
    """
**Customer ID**  
Identifier for the demo customer record.

**Credit Limit (NT$)**  
The amount of credit assigned to the customer in New Taiwan dollars.

**Age**  
Customer's age in years.

**September Repayment Status**  
Repayment behaviour recorded for September 2005.

**August Repayment Status**  
Repayment behaviour recorded for August 2005.

**Default Probability**  
The model's estimated probability that the customer will default on the credit payment.

**Risk Level**  
A risk category assigned from the model's predicted probability.

**Model Decision**  
The final decision generated from the project's selected prediction threshold.
"""
)


# ---------------------------------------------------------
# Explain repayment status codes
# ---------------------------------------------------------
with st.expander("📖 Repayment Status Meaning"):

    repayment_meaning = {
        "No consumption / zero balance": "-2",
        "Paid in full / paid on time": "-1",
        "Minimum payment / revolving credit": "0",
        "1 month late": "1",
        "2 months late": "2",
        "3 months late": "3",
        "4 months late": "4",
        "5 months late": "5",
        "6 months late": "6",
        "7 months late": "7",
        "8 months late": "8",
        "9+ months late": "9",
    }

    st.table(
        [
            {
                "Repayment Status": description,
                "Original Dataset Code": code,
            }
            for description, code
            in repayment_meaning.items()
        ]
    )


# ---------------------------------------------------------
# Disclaimer / project context
# ---------------------------------------------------------
st.info(
    "These are demo records from the project dataset. "
    "The predictions are generated by the project's trained model "
    "and are intended for demonstration and evaluation purposes. "
    "A real banking deployment would require an approved customer "
    "database, appropriate validation, monitoring, and governance."
)