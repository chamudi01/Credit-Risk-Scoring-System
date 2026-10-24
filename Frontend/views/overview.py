import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import ui
from utils.risk_utils import BAND_ORDER, DECISION_THRESHOLD, RISK_BANDS, status_label

scored = ui.load_scored()
n = len(scored)

ui.page_header(
    "Credit risk portfolio overview",
    "Predicted probability of default across the scored customer book, grouped into risk bands.",
    meta=ui.data_source_line(n),
)

if n == 0:
    ui.empty_state("No accounts to display", "The sample data file contains no records.")
    st.stop()

pdv = scored["Default Probability"]
thr = DECISION_THRESHOLD
very_high_floor = RISK_BANDS["Very High Risk"][0]
flagged_mask = pdv >= thr
flagged = int(flagged_mask.sum())
very_high = int((pdv >= very_high_floor).sum())
counts = scored["Risk Category"].value_counts().reindex(BAND_ORDER, fill_value=0)
largest = counts.idxmax()

# ---------------------------------------------------------
# Executive summary and key metrics
# ---------------------------------------------------------

ui.callout(
    f"{flagged / n:.0%} of the {n:,} scored accounts are at or above the {thr:.0%} decision threshold, and the "
    f"mean predicted probability of default is {pdv.mean():.1%}. {largest} is the largest band, with "
    f"{counts[largest]:,} accounts ({counts[largest] / n:.0%}).",
    lead="Summary.",
)

ui.kpi_row([
    ("Accounts scored", f"{n:,}", "Records in the sample book",
     "Number of customer records scored by the saved XGBoost pipeline."),
    ("Mean predicted PD", f"{pdv.mean():.1%}", f"Median {pdv.median():.1%}",
     "Average of the model's predicted probability of default across all accounts. "
     "It is an estimate, not an observed default rate."),
    (f"At or above {thr:.0%}", f"{flagged:,}", f"{flagged / n:.1%} of accounts",
     "Accounts whose predicted probability meets or exceeds the binary decision threshold."),
    (f"Very High Risk ({very_high_floor:.0%}+)", f"{very_high:,}", f"{very_high / n:.1%} of accounts",
     "Accounts in the highest risk band."),
])

with st.expander("How these figures are defined"):
    ui.plain_list([
        "Predicted probability of default (PD): the model's estimate, from 0% to 100%, that an account will default.",
        f"Decision threshold ({thr:.0%}): the cut-off used for the binary flag. It is a classification setting, "
        f"not the probability that a flagged customer defaults.",
        "Risk bands: Low, Medium, High and Very High, defined by fixed PD ranges (see Model documentation).",
        "Predictions versus outcomes: the sample contains no confirmed default outcomes, so every figure here "
        "describes model predictions only.",
    ])

# ---------------------------------------------------------
# Visual analysis
# ---------------------------------------------------------

ui.section("Risk distribution")
left, right = st.columns([3, 2], gap="large")

with left:
    with ui.panel("Distribution of predicted default probability", "Number of accounts at each predicted PD level."):
        fig = go.Figure()
        for band, (lo, hi) in RISK_BANDS.items():
            fig.add_vrect(x0=lo, x1=min(hi, 1.0), fillcolor=ui.BAND_COLORS[band], opacity=0.09,
                          line_width=0, layer="below")
        fig.add_histogram(x=pdv, xbins=dict(start=0, end=1, size=0.025), marker_color=ui.PRIMARY,
                          marker_line=dict(color="white", width=1),
                          hovertemplate="Predicted PD from %{x:.1%}: %{y} accounts<extra></extra>")
        fig.add_vline(x=thr, line_dash="dash", line_color=ui.INK, line_width=1,
                      annotation_text=f"Threshold {thr:.0%}", annotation_position="top right",
                      annotation_font=dict(size=11, color=ui.INK))
        ui.style(fig, height=310)
        fig.update_xaxes(range=[0, 1], tickformat=".0%", title_text="Predicted probability of default")
        fig.update_yaxes(title_text="Accounts")
        ui.show_chart(fig)
        st.caption("Shaded areas mark the four risk bands. The dashed line is the decision threshold.")

with right:
    with ui.panel("Accounts by risk band", "Count and share of all scored accounts."):
        fig = go.Figure(go.Bar(
            x=counts.values, y=BAND_ORDER, orientation="h",
            marker_color=[ui.BAND_COLORS[b] for b in BAND_ORDER],
            text=[f"{c:,}  ({c / n:.0%})" for c in counts.values], textposition="outside", cliponaxis=False,
            customdata=[[c, c / n, ui.band_range_label(b)] for b, c in zip(BAND_ORDER, counts.values)],
            hovertemplate="%{y} (%{customdata[2]})<br>%{customdata[0]:,} accounts, %{customdata[1]:.1%}<extra></extra>",
        ))
        ui.style(fig, height=310)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=False, showticklabels=False, showline=False, range=[0, max(counts.max(), 1) * 1.35])
        fig.update_layout(margin=dict(l=0, r=40, t=10, b=0))
        ui.show_chart(fig)
        st.caption("Band colours are used consistently on every page.")

by_band = (
    scored.groupby("Risk Category")
    .agg(Accounts=("Default Probability", "size"), MeanPD=("Default Probability", "mean"), Limit=("LIMIT_BAL", "sum"))
    .reindex(BAND_ORDER)
)
by_band["Accounts"] = by_band["Accounts"].fillna(0)
by_band["Limit"] = by_band["Limit"].fillna(0)
total_limit = by_band["Limit"].sum()

ui.section("Risk band summary", "Accounts, average predicted PD and credit limit held in each band.")
band_table = pd.DataFrame({
    "Risk band": BAND_ORDER,
    "Predicted PD range": [ui.band_range_label(b) for b in BAND_ORDER],
    "Accounts": by_band["Accounts"].astype(int).values,
    "Share of accounts": by_band["Accounts"].values / n * 100,
    "Mean predicted PD": by_band["MeanPD"].values * 100,
    "Share of credit limit": (by_band["Limit"].values / total_limit * 100) if total_limit else 0.0,
})
ui.show_table(band_table, column_config={
    "Share of accounts": st.column_config.NumberColumn(format="%.1f%%"),
    "Mean predicted PD": st.column_config.NumberColumn(format="%.1f%%"),
    "Share of credit limit": st.column_config.NumberColumn(format="%.1f%%"),
})

# ---------------------------------------------------------
# Supporting insights (all figures calculated from the scored data)
# ---------------------------------------------------------

insights = []
flagged_limit_share = scored.loc[flagged_mask, "LIMIT_BAL"].sum() / total_limit if total_limit else 0.0
insights.append(
    f"Accounts at or above the {thr:.0%} threshold are {flagged / n:.0%} of accounts and hold "
    f"{flagged_limit_share:.0%} of total credit limit."
)
medium = int(counts["Medium Risk"])
if medium:
    insights.append(
        f"{medium:,} accounts sit in the Medium Risk band, below the threshold but in the range where "
        f"closer monitoring of payment behaviour applies."
    )
late = scored["PAY_0"] >= 1
if late.any() and (~late).any():
    insights.append(
        f"{int(late.sum()):,} accounts were at least one month late in the latest statement. Their mean predicted PD is "
        f"{pdv[late].mean():.1%}, compared with {pdv[~late].mean():.1%} for the other accounts."
    )
if pdv.mean() - pdv.median() > 0.01:
    insights.append(
        f"The mean predicted PD ({pdv.mean():.1%}) is above the median ({pdv.median():.1%}), "
        f"so a minority of higher-risk accounts raises the portfolio average."
    )
ui.section("Portfolio insights", "Calculated from the scored accounts above.")
ui.plain_list(insights)

ui.section("Accounts with the highest predicted risk", "The ten accounts with the highest predicted PD in the scored book.")
top = scored.nlargest(10, "Default Probability")
view = pd.DataFrame({
    "Customer ID": top["Customer_ID"].values,
    "Credit limit (NT$)": top["LIMIT_BAL"].values,
    "Latest repayment status": top["PAY_0"].map(status_label).values,
    "Default probability": top["Default Probability"].values * 100,
    "Risk band": top["Risk Category"].values,
})
ui.show_table(view, column_config={
    "Credit limit (NT$)": st.column_config.NumberColumn(format="localized"),
    "Default probability": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
})

# ---------------------------------------------------------
# Data and refresh
# ---------------------------------------------------------

ui.section("Data and refresh")
st.caption(
    "Probabilities are calculated by the saved XGBoost pipeline when the app loads and are cached for the session. "
    "After replacing the data file, recalculate to refresh every page."
)
if st.button("Recalculate scores", help="Clears cached results and rescores the data with the saved model."):
    st.cache_data.clear()
    st.rerun()

ui.footnote(
    "The book is a set of sample customer records, not a live portfolio. Figures describe model predictions, "
    "not confirmed customer outcomes."
)
