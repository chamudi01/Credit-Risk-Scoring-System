import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import ui
from utils.risk_utils import DECISION_THRESHOLD, EDUCATION, MARRIAGE, SEX

MIN_GROUP = 20  # groups smaller than this are shown in a lighter colour and left out of the written reading
SMALL_GROUP_COLOR = "#9AA7B8"

scored = ui.load_scored()

ui.page_header(
    "Customer segmentation",
    "Compare predicted default risk across customer groups against the portfolio average.",
)

if scored.empty:
    ui.empty_state("No accounts to segment", "The sample data file contains no records.")
    st.stop()

df = scored.copy()
df["Age group"] = pd.cut(df["AGE"], [0, 25, 35, 45, 55, 120], labels=["Up to 25", "26 to 35", "36 to 45", "46 to 55", "56 and over"])
df["Education"] = pd.Categorical(df["EDUCATION"].map(EDUCATION).fillna("Other"), list(EDUCATION.values()))
df["Marital status"] = pd.Categorical(df["MARRIAGE"].map(MARRIAGE).fillna("Other"), list(MARRIAGE.values()))
df["Gender"] = pd.Categorical(df["SEX"].map(SEX), list(SEX.values()))

edges = df["LIMIT_BAL"].quantile([0, 0.25, 0.5, 0.75, 1]).round(-3).tolist()
q_labels = [f"NT$ {edges[i] / 1000:,.0f}k to {edges[i + 1] / 1000:,.0f}k" for i in range(4)]
try:
    df["Credit limit quartile"] = pd.qcut(df["LIMIT_BAL"], 4, labels=q_labels, duplicates="drop")
except ValueError:
    df["Credit limit quartile"] = pd.qcut(df["LIMIT_BAL"], 4, duplicates="drop").astype(str)

repay_bins = [-3, -1, 0, 1, 2, 99]
repay_labels = ["Paid in full or no usage", "Revolving", "1 month late", "2 months late", "3+ months late"]
df["Latest repayment status"] = pd.cut(df["PAY_0"], repay_bins, labels=repay_labels)

SEGMENT_NOTES = {
    "Age group": "Customers grouped by age in years.",
    "Education": "Highest education level recorded for the customer.",
    "Marital status": "Marital status as recorded for the customer.",
    "Credit limit quartile": (
        "Accounts split into four equally sized groups by credit limit. Income is not recorded in this dataset, "
        "so credit limit is the closest available measure of credit capacity. It is not income."
    ),
    "Latest repayment status": "Repayment behaviour in the most recent statement month.",
    "Gender": "Shown for fairness monitoring only.",
}
SEGMENTS = list(SEGMENT_NOTES)

choice = st.segmented_control("Group accounts by", SEGMENTS, default="Age group")
if choice is None:
    choice = "Age group"
st.caption(SEGMENT_NOTES[choice])

g = (
    df.groupby(choice, observed=True)["Default Probability"]
    .agg(Accounts="size", MeanPD="mean", Flagged=lambda s: (s >= DECISION_THRESHOLD).mean())
    .reset_index()
)
overall_pd = df["Default Probability"].mean()
overall_flag = (df["Default Probability"] >= DECISION_THRESHOLD).mean()
names = g[choice].astype(str).tolist()
counts = g["Accounts"].tolist()
labels = [f"{s}  (n={n})" for s, n in zip(names, counts)]
has_small = any(c < MIN_GROUP for c in counts)


def bars(values, ref, measure):
    colors = [ui.PRIMARY if c >= MIN_GROUP else SMALL_GROUP_COLOR for c in counts]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.1%}" for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(color=ui.INK, size=12), customdata=counts,
        hovertemplate=f"%{{y}}<br>{measure}: %{{x:.1%}}<br>%{{customdata}} accounts<extra></extra>",
    ))
    fig.add_vline(x=ref, line_dash="dash", line_color=ui.INK, line_width=1,
                  annotation_text=f"Portfolio {ref:.1%}", annotation_position="top",
                  annotation_font=dict(size=11, color=ui.INK))
    ui.style(fig, height=max(230, 52 * len(labels) + 60))
    fig.update_layout(margin=dict(l=0, r=40, t=24, b=0))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickformat=".0%", range=[0, max(max(values), ref) * 1.3])
    return fig


# ---------------------------------------------------------
# Written reading, calculated from the selected grouping
# ---------------------------------------------------------

eligible = g[g["Accounts"] >= MIN_GROUP]
if len(eligible) >= 2:
    hi = eligible.loc[eligible["MeanPD"].idxmax()]
    lo = eligible.loc[eligible["MeanPD"].idxmin()]
    reading = (
        f"Among groups with at least {MIN_GROUP} accounts, mean predicted PD is highest for {hi[choice]} "
        f"({hi['MeanPD']:.1%}) and lowest for {lo[choice]} ({lo['MeanPD']:.1%}), against a portfolio mean of "
        f"{overall_pd:.1%}."
    )
else:
    reading = (
        f"Fewer than two groups have at least {MIN_GROUP} accounts, so a reliable comparison is not possible "
        f"for this grouping."
    )
ui.callout(reading, lead="Reading the comparison.")

left, right = st.columns(2, gap="large")
with left:
    with ui.panel("Mean predicted default probability", "Average model PD for each group."):
        ui.show_chart(bars(g["MeanPD"], overall_pd, "Mean predicted PD"))
with right:
    with ui.panel(f"Share at or above the {DECISION_THRESHOLD:.0%} threshold", "Proportion of each group flagged by the decision threshold."):
        ui.show_chart(bars(g["Flagged"], overall_flag, "At or above threshold"))
caption = "The dashed line marks the portfolio-wide figure."
if has_small:
    caption += f" Lighter bars are groups with fewer than {MIN_GROUP} accounts, which can vary widely from sample to sample."
st.caption(caption)

ui.section("Group detail")
detail = pd.DataFrame({
    choice: names,
    "Accounts": g["Accounts"],
    "Mean PD": g["MeanPD"] * 100,
    "Difference from portfolio (pts)": (g["MeanPD"] - overall_pd) * 100,
    f"At or above {DECISION_THRESHOLD:.0%}": g["Flagged"] * 100,
    "Note": [f"Fewer than {MIN_GROUP} accounts, interpret with care" if c < MIN_GROUP else "" for c in counts],
})
ui.show_table(detail, column_config={
    "Mean PD": st.column_config.NumberColumn(format="%.1f%%"),
    "Difference from portfolio (pts)": st.column_config.NumberColumn(format="%+.1f"),
    f"At or above {DECISION_THRESHOLD:.0%}": st.column_config.NumberColumn(format="%.1f%%"),
})
ui.download_csv(detail, "segment_detail.csv", "Download group detail (CSV)")

ui.footnote(
    "Differences between groups describe the model's predictions and do not show that a characteristic "
    "causes higher or lower default risk. Gender is shown for fairness monitoring only. Income and employment "
    "are not recorded in this dataset, so they are not available as groupings."
)
