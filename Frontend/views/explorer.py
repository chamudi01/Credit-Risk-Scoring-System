import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import ui
from utils.risk_utils import BAND_ORDER, FEATURES, HISTORY_MONTHS, explain_prediction, load_model, status_label

scored = ui.load_scored()
try:
    model = load_model()
except Exception as exc:
    st.error(f"The trained model could not be loaded: {exc}")
    st.stop()

LATEST_MONTH = HISTORY_MONTHS[-1][0]

ui.page_header(
    "Account explorer",
    "Search and filter the scored accounts, then select one to review its repayment history and score drivers.",
)

# ---------------------------------------------------------
# Filters
# ---------------------------------------------------------


def reset_filters():
    st.session_state["ex_query"] = ""
    st.session_state["ex_bands"] = list(BAND_ORDER)
    st.session_state["ex_pd"] = (0, 100)


with ui.panel("Filters"):
    f1, f2, f3 = st.columns([1.1, 2.9, 1.7], gap="large")
    query = f1.text_input("Customer ID", placeholder="e.g. C0042", key="ex_query",
                          help="Matches any part of the ID, ignoring case.").strip()
    bands = f2.pills("Risk band", BAND_ORDER, selection_mode="multi", default=BAND_ORDER, key="ex_bands") or []
    pd_lo, pd_hi = f3.slider("Default probability", 0, 100, (0, 100), format="%d%%", key="ex_pd",
                             help="Show accounts whose predicted probability of default is within this range.")
    st.button("Clear filters", on_click=reset_filters)

mask = scored["Risk Category"].isin(bands)
mask &= scored["Default Probability"].between(pd_lo / 100, pd_hi / 100)
if query:
    mask &= scored["Customer_ID"].astype(str).str.contains(query, case=False, regex=False)
filtered = scored[mask].sort_values("Default Probability", ascending=False).reset_index(drop=True)

if filtered.empty:
    ui.empty_state(
        "No accounts match the current filters",
        "Widen the probability range, select more risk bands, or clear the Customer ID search.",
    )
    st.stop()

# ---------------------------------------------------------
# Account table
# ---------------------------------------------------------

st.caption(
    f"{len(filtered):,} of {len(scored):,} accounts, highest risk first. Select a row for detail. "
    "Click a column header to sort."
)
ui.band_counts(filtered["Risk Category"].value_counts().reindex(BAND_ORDER, fill_value=0))

view = pd.DataFrame({
    "Customer ID": filtered["Customer_ID"],
    "Credit limit (NT$)": filtered["LIMIT_BAL"],
    "Age": filtered["AGE"],
    "Latest repayment status": filtered["PAY_0"].map(status_label),
    "Default probability": filtered["Default Probability"] * 100,
    "Risk band": filtered["Risk Category"],
    "Decision": filtered["Decision"],
})

event = ui.show_table(
    view,
    column_config={
        "Credit limit (NT$)": st.column_config.NumberColumn(format="localized"),
        "Default probability": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
    },
    on_select="rerun",
    selection_mode="single-row",
    key="explorer_table",
    height=330,
)

export = view.rename(columns={"Default probability": "Default probability (%)"}).round({"Default probability (%)": 2})
ui.download_csv(export, "account_explorer_export.csv", "Export filtered accounts (CSV)")

selected = event.selection.rows if event is not None else []
if not selected or selected[0] >= len(filtered):
    ui.footnote("Select an account in the table to see its detail. Scores are recalculated from the saved model.")
    st.stop()

# ---------------------------------------------------------
# Account detail
# ---------------------------------------------------------

acct = filtered.iloc[selected[0]]
p = float(acct["Default Probability"])
limit = float(acct["LIMIT_BAL"])
latest_balance = float(acct["BILL_AMT1"])
months = [m[0] for m in HISTORY_MONTHS]
codes = [int(acct[m[1]]) for m in HISTORY_MONTHS]
months_late = sum(1 for c in codes if c >= 1)

ui.rule()
ui.subheader(f"Account {acct['Customer_ID']}")
st.markdown(ui.badge_html(acct["Risk Category"]), unsafe_allow_html=True)

ui.kpi_row([
    ("Predicted default probability", f"{p:.1%}", acct["Decision"],
     "The model's estimate for this account. It is not a confirmed outcome."),
    ("Credit limit", f"NT$ {limit:,.0f}", f"{LATEST_MONTH} balance NT$ {latest_balance:,.0f}", None),
    (f"{LATEST_MONTH} utilisation", f"{latest_balance / limit:.0%}" if limit > 0 else "n/a",
     "Latest balance / credit limit", "Share of the credit limit in use on the most recent statement."),
    ("Latest repayment status", status_label(acct["PAY_0"]), f"{months_late} of 6 months late", None),
])

t_history, t_drivers = st.tabs(["Repayment history", "Score drivers"])

with t_history:
    with ui.panel("Statement balance and payments", "Monthly statement balance against payments made, oldest month first."):
        balance = [acct[m[2]] for m in HISTORY_MONTHS]
        paid = [acct[m[3]] for m in HISTORY_MONTHS]
        fig = go.Figure()
        fig.add_bar(x=months, y=paid, name="Payment made", marker_color=ui.PRIMARY, opacity=0.85,
                    hovertemplate="%{x}<br>Payment made: NT$ %{y:,.0f}<extra></extra>")
        fig.add_scatter(x=months, y=balance, name="Statement balance", mode="lines+markers",
                        line=dict(color=ui.INK, width=2), marker=dict(size=6),
                        hovertemplate="%{x}<br>Statement balance: NT$ %{y:,.0f}<extra></extra>")
        fig.add_hline(y=limit, line_dash="dot", line_color=ui.MUTED, line_width=1,
                      annotation_text="Credit limit", annotation_position="top left",
                      annotation_font=dict(size=11, color=ui.MUTED))
        ui.style(fig, height=300, legend=True)
        fig.update_yaxes(title_text="NT$", tickformat=",")
        ui.show_chart(fig)
    ui.section("Monthly detail")
    ui.show_table(
        pd.DataFrame({
            "Month": months,
            "Repayment status": [status_label(c) for c in codes],
            "Statement balance (NT$)": balance,
            "Payment made (NT$)": paid,
        }),
        column_config={
            "Statement balance (NT$)": st.column_config.NumberColumn(format="localized"),
            "Payment made (NT$)": st.column_config.NumberColumn(format="localized"),
        },
        height=248,
    )

with t_drivers:
    ui.section("What drove this score", "The inputs with the largest influence on this account's estimate.")
    drivers, method = explain_prediction(model, filtered.iloc[[selected[0]]][FEATURES], top_n=6)
    ui.driver_chart(drivers, method)

ui.footnote(
    "Scores are model estimates recalculated from the saved pipeline. They are not confirmed customer outcomes."
)
