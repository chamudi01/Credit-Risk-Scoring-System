
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.risk_utils import load_model, load_sample_customers, predict_dataframe

st.title("📊 Dashboard")
st.caption("Demo portfolio overview using the supplied sample customer records.")

try:
    model = load_model()
    customers = load_sample_customers()
    scored = predict_dataframe(model, customers)
except Exception as exc:
    st.error(f"Could not load the model or demo data: {exc}")
    st.stop()

total = len(scored)
avg_pd = scored["Default Probability"].mean()
high_count = (scored["Default Probability"] >= 0.31).sum()
very_high_count = (scored["Default Probability"] >= 0.50).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Demo Customers", f"{total:,}")
c2.metric("Average PD", f"{avg_pd:.1%}")
c3.metric("High-Risk +", f"{high_count:,}", f"{high_count/total:.1%}")
c4.metric("Very High Risk", f"{very_high_count:,}", f"{very_high_count/total:.1%}")

st.markdown("### Risk Distribution")
risk_order = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
counts = scored["Risk Category"].value_counts().reindex(risk_order, fill_value=0)
risk_df = counts.rename_axis("Risk Category").reset_index(name="Customers")
fig = px.bar(risk_df, x="Risk Category", y="Customers", text="Customers")
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Default Probability Distribution")
fig2 = px.histogram(
    scored,
    x="Default Probability",
    nbins=20,
    labels={"Default Probability": "Predicted Default Probability"},
)
fig2.add_vline(x=0.31, line_dash="dash", annotation_text="31% threshold")
st.plotly_chart(fig2, use_container_width=True)

st.info(
    "The dashboard uses demo records because this prototype has no live bank database. "
    "The application recalculates predictions from the saved model."
)
