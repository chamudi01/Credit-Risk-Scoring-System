
import streamlit as st
from utils.risk_utils import FEATURES, DECISION_THRESHOLD

st.title("🧠 Model & Methodology")

st.markdown("""
### Model

**XGBoost binary classification pipeline**

The application loads the already-trained preprocessing + XGBoost pipeline.
It does **not** retrain the model when Streamlit starts.

### Prediction workflow

```text
Customer information
        ↓
Saved preprocessing + XGBoost pipeline
        ↓
Probability of Default (PD)
        ↓
31% decision threshold
        ↓
Risk classification
        ↓
Risk explanation and recommendations
```
""")

st.metric("Decision Threshold", f"{DECISION_THRESHOLD:.0%}")

st.markdown("### Risk classification")
st.table({
    "Probability": ["< 20%", "20% – <31%", "31% – <50%", "≥ 50%"],
    "Risk": ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"],
})

st.markdown("### Model input features")
st.write(", ".join(FEATURES))

st.markdown("### Important interpretation")
st.info(
    "The 31% value is the binary classification threshold selected during the "
    "project's model-evaluation workflow. It is not a statement that a customer "
    "has a guaranteed 31% chance of default."
)

st.markdown("### Prototype limitations")
for item in [
    "This is a machine-learning decision-support prototype.",
    "It does not connect to a live bank customer database.",
    "Predictions are probabilistic and depend on the training data.",
    "Model performance can change on new customer populations.",
    "Real banking decisions require human oversight and institutional policies.",
    "Production use would require validation, monitoring, privacy, security, fairness, and governance controls.",
]:
    st.write(f"• {item}")
