from pathlib import Path

import streamlit as st

from utils import ui

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Bank Credit Risk Management System",
    page_icon=str(ROOT / "assets" / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.apply_theme()

navigation = st.navigation(
    {
        "Portfolio": [
            st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
            st.Page("views/portfolio.py", title="Portfolio analytics", icon=":material/account_balance:"),
            st.Page("views/segments.py", title="Customer segmentation", icon=":material/stacked_bar_chart:"),
        ],
        "Accounts": [
            st.Page("views/assessment.py", title="Risk assessment", icon=":material/calculate:"),
            st.Page("views/explorer.py", title="Account explorer", icon=":material/manage_search:"),
        ],
        "Reference": [
            st.Page("views/model_info.py", title="Model documentation", icon=":material/description:"),
        ],
    }
)

with st.sidebar:
    st.caption(
        "Academic prototype. Scores support credit review and are not automated credit decisions."
    )

navigation.run()
