
import streamlit as st

st.set_page_config(
    page_title="Bank Credit Risk Management System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏦 Bank Credit Risk Management System")
st.caption("XGBoost-based credit-card default risk decision-support prototype")

st.markdown("""
## Welcome

This application demonstrates how a bank could use a trained XGBoost credit-risk
pipeline to estimate a customer's probability of default, classify risk, explore
demo customers, and review portfolio-level risk.

**Current prototype:** manual input + demo CSV. No live bank database is required.
""")

st.info(
    "Decision threshold: 31%. Risk bands: Low <20%, Medium 20–<31%, "
    "High 31–<50%, Very High ≥50%."
)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Model", "XGBoost")
with c2:
    st.metric("Decision Threshold", "31%")
with c3:
    st.metric("Input Features", "23")

st.markdown("### Use the sidebar")
st.write(
    "Start with **Risk Prediction** for the main demonstration. "
    "Use **Customer Risk Explorer** and **Portfolio Risk** for the demo portfolio."
)

st.markdown("---")
st.caption(
    "Academic demonstration only. Predictions are probabilistic decision-support "
    "outputs and should not be treated as automatic real-world banking decisions."
)
