import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import ui
from utils.risk_utils import BAND_ORDER, DECISION_THRESHOLD

scored = ui.load_scored()

ui.page_header(
    "Portfolio risk analytics",
    "Expected loss for the scored book under adjustable loss-given-default and exposure assumptions.",
)

if scored.empty:
    ui.empty_state("No accounts to analyse", "The sample data file contains no records.")
    st.stop()

# ---------------------------------------------------------
# Assumptions
# ---------------------------------------------------------

LGD_DEFAULT, EAD_DEFAULT = 45, 100


def reset_assumptions():
    st.session_state["lgd_pct"] = LGD_DEFAULT
    st.session_state["ead_pct"] = EAD_DEFAULT


with ui.panel("Loss assumptions", "Applied uniformly to every account. Change them to see how expected loss responds."):
    a1, a2 = st.columns(2, gap="large")
    lgd = a1.slider(
        "Loss given default (LGD)", 0, 100, LGD_DEFAULT, 5, format="%d%%", key="lgd_pct",
        help="Share of the exposure expected to be lost if an account defaults.",
    ) / 100
    ead_share = a2.slider(
        "Exposure at default, as share of credit limit", 0, 100, EAD_DEFAULT, 5, format="%d%%", key="ead_pct",
        help="Share of each account's credit limit assumed to be drawn at the time of default.",
    ) / 100
    st.button("Reset assumptions", on_click=reset_assumptions)

df = scored.copy()
df["EAD"] = df["LIMIT_BAL"].clip(lower=0) * ead_share
df["EL"] = df["Default Probability"] * lgd * df["EAD"]

total_ead, total_el = df["EAD"].sum(), df["EL"].sum()
flagged = df["Default Probability"] >= DECISION_THRESHOLD
flagged_ead_share = df.loc[flagged, "EAD"].sum() / total_ead if total_ead else 0

ui.kpi_row([
    ("Total credit limit", f"NT$ {df['LIMIT_BAL'].sum() / 1e6:,.1f}M", f"{len(df):,} accounts",
     "Sum of approved credit limits across all scored accounts."),
    ("Exposure at default", f"NT$ {total_ead / 1e6:,.1f}M", f"{ead_share:.0%} of limits",
     "Credit limit multiplied by the exposure assumption, summed over accounts."),
    ("Expected loss", f"NT$ {total_el / 1e6:,.2f}M",
     f"{total_el / total_ead:.2%} of exposure" if total_ead else None,
     "Predicted PD x LGD x exposure at default, summed over all accounts."),
    (f"Exposure at or above {DECISION_THRESHOLD:.0%}", f"{flagged_ead_share:.0%}", f"{int(flagged.sum())} accounts",
     "Share of modelled exposure held by accounts at or above the decision threshold."),
])
st.caption("Expected loss = PD x LGD x EAD, summed over accounts. An illustration, not a regulatory calculation.")

by_band = (
    df.groupby("Risk Category")
    .agg(Accounts=("EL", "size"), MeanPD=("Default Probability", "mean"), EAD=("EAD", "sum"), EL=("EL", "sum"))
    .reindex(BAND_ORDER)
    .fillna(0)
)
by_band["AccountShare"] = by_band["Accounts"] / by_band["Accounts"].sum()
by_band["ELShare"] = by_band["EL"] / total_el if total_el else 0.0

# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------

ui.section("Expected loss analysis")
left, right = st.columns(2, gap="large")
with left:
    with ui.panel("Accounts versus expected loss", "Share of the book that each risk band represents."):
        fig = go.Figure()
        fig.add_bar(y=BAND_ORDER, x=by_band["AccountShare"], orientation="h", name="Share of accounts",
                    marker_color="#B8C2CF", text=[f"{v:.0%}" for v in by_band["AccountShare"]], textposition="outside",
                    customdata=by_band["Accounts"].astype(int),
                    hovertemplate="%{y}<br>%{x:.1%} of accounts (%{customdata:,} accounts)<extra></extra>")
        fig.add_bar(y=BAND_ORDER, x=by_band["ELShare"], orientation="h", name="Share of expected loss",
                    marker_color=ui.PRIMARY, text=[f"{v:.0%}" for v in by_band["ELShare"]], textposition="outside",
                    customdata=by_band["EL"].round(0),
                    hovertemplate="%{y}<br>%{x:.1%} of expected loss (NT$ %{customdata:,.0f})<extra></extra>")
        ui.style(fig, height=320, legend=True)
        fig.update_layout(barmode="group", margin=dict(l=0, r=30, t=40, b=0))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(tickformat=".0%", title_text="Share of portfolio",
                         range=[0, max(by_band["AccountShare"].max(), by_band["ELShare"].max(), 0.01) * 1.2])
        ui.show_chart(fig)
        st.caption("Where a band's share of expected loss is larger than its share of accounts, it contributes more "
                   "loss than its size alone suggests.")

with right:
    with ui.panel("Loss concentration", "Cumulative expected loss when accounts are ranked from highest to lowest PD."):
        if total_el > 0:
            ranked = df.sort_values("Default Probability", ascending=False)
            x = np.arange(1, len(ranked) + 1) / len(ranked)
            y = ranked["EL"].cumsum().values / total_el
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.RULE_STRONG, dash="dash"),
                            name="Even spread", hoverinfo="skip")
            fig.add_scatter(x=np.r_[0, x], y=np.r_[0, y], mode="lines", line=dict(color=ui.PRIMARY, width=2.5),
                            name="Cumulative expected loss",
                            hovertemplate="Top %{x:.0%} of accounts: %{y:.0%} of loss<extra></extra>")
            ui.style(fig, height=320, legend=True)
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            fig.update_xaxes(tickformat=".0%", title_text="Accounts, ranked by predicted PD", range=[0, 1])
            fig.update_yaxes(tickformat=".0%", title_text="Share of expected loss", range=[0, 1.02])
            ui.show_chart(fig)
            st.caption("The further the curve rises above the dashed line, the more concentrated expected loss is "
                       "among the highest-PD accounts.")
        else:
            ui.empty_state("No expected loss to chart", "Expected loss is zero under the current assumptions.")

# ---------------------------------------------------------
# Detail
# ---------------------------------------------------------

ui.section("Risk band detail", "Exposure and expected loss by band under the assumptions above.")
table = pd.DataFrame({
    "Risk band": BAND_ORDER,
    "Accounts": by_band["Accounts"].astype(int).values,
    "Mean PD": by_band["MeanPD"].values * 100,
    "Exposure (NT$)": by_band["EAD"].round(0).values,
    "Expected loss (NT$)": by_band["EL"].round(0).values,
    "Share of expected loss": by_band["ELShare"].values * 100,
})
ui.show_table(table, column_config={
    "Mean PD": st.column_config.NumberColumn(format="%.1f%%"),
    "Exposure (NT$)": st.column_config.NumberColumn(format="localized"),
    "Expected loss (NT$)": st.column_config.NumberColumn(format="localized"),
    "Share of expected loss": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
})
ui.download_csv(table, "expected_loss_by_risk_band.csv", "Download band detail (CSV)")

if total_el > 0:
    ui.section("Observations", "Calculated from the current assumptions.")
    top_n = max(1, int(round(len(df) * 0.2)))
    top_share = df.sort_values("Default Probability", ascending=False)["EL"].head(top_n).sum() / total_el
    vh = by_band.loc["Very High Risk"]
    obs = [
        f"The riskiest 20% of accounts ({top_n}) carry {top_share:.0%} of expected loss.",
        f"Very High Risk accounts are {vh['AccountShare']:.0%} of the book and {vh['ELShare']:.0%} of expected loss.",
        f"Accounts at or above the {DECISION_THRESHOLD:.0%} threshold hold {flagged_ead_share:.0%} of modelled exposure.",
    ]
    ui.plain_list(obs)

ui.footnote(
    "PD comes from the saved model. LGD and exposure are user-set assumptions, applied uniformly to every account."
)
