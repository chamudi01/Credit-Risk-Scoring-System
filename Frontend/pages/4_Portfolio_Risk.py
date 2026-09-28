
import streamlit as st
import plotly.express as px

from utils.risk_utils import load_model, load_sample_customers, predict_dataframe

st.title("📈 Portfolio Risk & Recommendations")
st.caption("Portfolio-level demonstration using the sample customer records.")

try:
    model = load_model()
    customers = load_sample_customers()
    scored = predict_dataframe(model, customers)
except Exception as exc:
    st.error(f"Could not load the model or demo data: {exc}")
    st.stop()

lgd = st.slider("LGD assumption", 0.0, 1.0, 0.45, 0.05)
ead_factor = st.slider(
    "EAD assumption as a fraction of credit limit",
    0.0, 1.0, 1.0, 0.05
)

ead = scored["LIMIT_BAL"].clip(lower=0) * ead_factor
expected_loss = (scored["Default Probability"] * lgd * ead).sum()

high = (scored["Default Probability"] >= 0.31).mean()
very_high = (scored["Default Probability"] >= 0.50).mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(scored):,}")
c2.metric("Average PD", f"{scored['Default Probability'].mean():.1%}")
c3.metric("High-Risk +", f"{high:.1%}")
c4.metric("Estimated Expected Loss", f"{expected_loss:,.0f}")

st.markdown("### Risk Distribution")
risk_order = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
risk_df = (
    scored["Risk Category"]
    .value_counts()
    .reindex(risk_order, fill_value=0)
    .rename_axis("Risk Category")
    .reset_index(name="Customers")
)
fig = px.pie(risk_df, names="Risk Category", values="Customers", hole=0.45)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Default Probability by Risk Group")
fig2 = px.box(scored, x="Risk Category", y="Default Probability", category_orders={"Risk Category": risk_order})
fig2.add_hline(y=0.31, line_dash="dash", annotation_text="31% threshold")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### Risk-management recommendations")
if very_high >= 0.10:
    st.warning("The demo portfolio has at least 10% Very High Risk customers. Review high-risk exposure.")
else:
    st.info("The demo portfolio has less than 10% Very High Risk customers.")

for item in [
    "Monitor the proportion of high- and very-high-risk customers.",
    "Review customers with repeated payment difficulties.",
    "Monitor changes in portfolio-level predicted default probability.",
    "Use expected loss as a demonstration metric, not as a regulatory calculation.",
]:
    st.write(f"• {item}")

st.caption(
    "Expected Loss demonstration: EL = PD × LGD × EAD. "
    "Here EAD is an adjustable fraction of the customer's credit limit."
)
