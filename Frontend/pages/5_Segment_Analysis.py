import streamlit as st
import pandas as pd
import plotly.express as px

from utils.risk_utils import (
    load_model,
    load_sample_customers,
    predict_dataframe,
)


st.title("⚖️ Customer Segment Risk Analysis")

st.caption(
    "Compare model-predicted default risk across different "
    "customer segments in the demonstration dataset."
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

try:

    model = load_model()

    customers = load_sample_customers()

    scored = predict_dataframe(
        model,
        customers
    )

except Exception as exc:

    st.error(
        f"Could not load the model or customer data: {exc}"
    )

    st.stop()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def education_label(value):

    mapping = {
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Other",
    }

    return mapping.get(
        int(value),
        "Other"
    )


def marriage_label(value):

    mapping = {
        1: "Married",
        2: "Single",
        3: "Other",
    }

    return mapping.get(
        int(value),
        "Other"
    )


# ---------------------------------------------------------
# Create readable segmentation columns
# ---------------------------------------------------------

analysis = scored.copy()


analysis["Education"] = analysis[
    "EDUCATION"
].apply(education_label)


analysis["Marital Status"] = analysis[
    "MARRIAGE"
].apply(marriage_label)


analysis["Age Group"] = pd.cut(
    analysis["AGE"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=[
        "18–25",
        "26–35",
        "36–45",
        "46–55",
        "56+",
    ],
    include_lowest=True,
)


analysis["Credit Limit Group"] = pd.qcut(
    analysis["LIMIT_BAL"],
    q=4,
    labels=[
        "Lower",
        "Lower-Middle",
        "Upper-Middle",
        "Higher",
    ],
    duplicates="drop",
)


# ---------------------------------------------------------
# Segment selection
# ---------------------------------------------------------

segment_options = {
    "Age Group": "Age Group",
    "Education": "Education",
    "Marital Status": "Marital Status",
    "Credit Limit Group": "Credit Limit Group",
    "Risk Level": "Risk Category",
}


selected_segment = st.selectbox(
    "Select Customer Segment",
    list(segment_options.keys())
)


segment_column = segment_options[
    selected_segment
]


# ---------------------------------------------------------
# Calculate segment summary
# ---------------------------------------------------------

summary = (
    analysis
    .groupby(
        segment_column,
        observed=False
    )
    .agg(
        Customers=(
            "Default Probability",
            "size"
        ),
        Average_PD=(
            "Default Probability",
            "mean"
        ),
        High_Risk_Percentage=(
            "Default Probability",
            lambda x: (
                x >= 0.31
            ).mean()
        ),
    )
    .reset_index()
)


# ---------------------------------------------------------
# Display summary table
# ---------------------------------------------------------

st.subheader(
    f"📊 Risk by {selected_segment}"
)


display_summary = summary.copy()


display_summary["Average_PD"] = (
    display_summary["Average_PD"]
    .map(lambda x: f"{x:.1%}")
)


display_summary["High_Risk_Percentage"] = (
    display_summary[
        "High_Risk_Percentage"
    ]
    .map(lambda x: f"{x:.1%}")
)


display_summary = display_summary.rename(
    columns={
        segment_column: selected_segment,
        "Average_PD": "Average Default Probability",
        "High_Risk_Percentage": "At / Above 31% Threshold",
    }
)


st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Average PD chart
# ---------------------------------------------------------

st.subheader(
    "📈 Average Predicted Default Probability"
)


chart_summary = summary.copy()


chart_summary["Average PD"] = (
    chart_summary["Average_PD"] * 100
)


fig = px.bar(
    chart_summary,
    x=segment_column,
    y="Average PD",
    text="Average PD",
)


fig.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
)


fig.update_layout(
    yaxis_title="Average Predicted Default Probability (%)",
    xaxis_title=selected_segment,
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------------------------
# High-risk percentage chart
# ---------------------------------------------------------

st.subheader(
    "⚠️ Customers At or Above the 31% Threshold"
)


chart_summary["High-Risk %"] = (
    chart_summary[
        "High_Risk_Percentage"
    ] * 100
)


fig2 = px.bar(
    chart_summary,
    x=segment_column,
    y="High-Risk %",
    text="High-Risk %",
)


fig2.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
)


fig2.update_layout(
    yaxis_title="Customers at / above 31% (%)",
    xaxis_title=selected_segment,
)


st.plotly_chart(
    fig2,
    use_container_width=True
)


# ---------------------------------------------------------
# Explanation
# ---------------------------------------------------------

st.info(
    "This page describes differences in model-predicted risk "
    "between customer segments. A difference between groups "
    "does not by itself establish that the segment characteristic "
    "causes higher or lower default risk."
)