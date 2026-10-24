"""Shared presentation layer: design tokens, theme CSS, reusable components and chart styling.

Every page imports this module, so spacing, colours and component behaviour are defined once.
Streamlit-native widgets are used wherever they are reliable; custom HTML is limited to
read-only furniture (headings, KPI cards, badges, call-outs, empty states).
"""

import html
from contextlib import contextmanager
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from utils.risk_utils import (
    FEATURE_LABELS, MODEL_PATH, RISK_BANDS, SAMPLE_PATH, get_scored,
)

# ---------------------------------------------------------
# Design tokens
# ---------------------------------------------------------

INK = "#1B2430"
MUTED = "#5E6978"      # 5.6:1 on white, 5.1:1 on SURFACE
RULE = "#DCE1E8"
RULE_STRONG = "#B8C2CF"
GRID = "#EDF0F4"
SURFACE = "#F3F5F8"
PRIMARY = "#24456B"
CAUTION = "#9A6F00"

# Every band colour has at least 4.5:1 contrast against white text, so labels drawn on bars stay legible.
BAND_COLORS = {
    "Low Risk": "#2F7D5B",
    "Medium Risk": "#9A6F00",
    "High Risk": "#B8561A",
    "Very High Risk": "#A32D2D",
}
RAISES = "#A32D2D"
LOWERS = "#2F7D5B"

FONT = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea,
.stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stSidebar"] * {{
    font-family: {FONT};
}}
.stApp [data-testid="stIconMaterial"], .stApp .material-icons {{ font-family: "Material Symbols Rounded" !important; }}

.block-container {{ max-width: 1180px; padding-top: 2.4rem; padding-bottom: 3rem; }}
footer, [data-testid="stDecoration"] {{ display: none !important; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {RULE}; }}
[data-testid="stSidebarNav"]::before {{
    content: "Credit Risk Management";
    display: block; padding: 0.2rem 0.75rem 0.9rem; margin-bottom: 0.2rem; white-space: nowrap;
    font-size: 0.98rem; font-weight: 600; color: {INK}; border-bottom: 1px solid {RULE};
}}
[data-testid="stSidebarNav"] a {{ border-radius: 4px; transition: background-color 0.12s ease; }}
[data-testid="stSidebarNav"] a:hover {{ background: #E6EAF0; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: #E1E7EF; font-weight: 600; box-shadow: inset 3px 0 0 {PRIMARY};
}}

/* Page furniture */
.stApp h1.pg-title, .stApp h2.pg-title {{ padding: 0; font-size: 1.65rem; font-weight: 600; letter-spacing: -0.01em; color: {INK}; margin: 0; line-height: 1.25; }}
.stApp h2.pg-sub {{ padding: 0; font-size: 1.2rem; font-weight: 600; color: {INK}; margin: 1.4rem 0 0.3rem; line-height: 1.3; }}
.stApp p.pg-desc {{ color: {MUTED}; margin: 0.35rem 0 0.2rem; max-width: 74ch; font-size: 0.98rem; }}
.stApp p.pg-meta {{ color: {MUTED}; margin: 0.2rem 0 0; font-size: 0.8rem; }}
.pg-rule {{ border: 0; border-top: 1px solid {RULE}; margin: 1.1rem 0 0; }}

.stApp h3.sec {{ padding: 0; font-size: 1.02rem; font-weight: 600; letter-spacing: 0; color: {INK}; margin: 2rem 0 0.15rem; line-height: 1.4; }}
.stApp p.sec-note {{ color: {MUTED}; font-size: 0.86rem; margin: 0 0 0.6rem; max-width: 80ch; }}
.stApp p.panel-t {{ font-size: 0.95rem; font-weight: 600; color: {INK}; margin: 0; }}
.stApp p.panel-n {{ font-size: 0.82rem; color: {MUTED}; margin: 0.1rem 0 0.4rem; }}

/* KPI cards */
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; margin: 1.1rem 0 0.4rem; }}
.kpi {{ background: #FFFFFF; border: 1px solid {RULE}; border-radius: 6px; padding: 0.8rem 1rem 0.85rem; transition: border-color 0.15s ease; }}
.kpi:hover {{ border-color: {RULE_STRONG}; }}
.kpi .l {{ font-size: 0.8rem; color: {MUTED}; font-weight: 500; }}
.kpi .l[title] {{ cursor: help; text-decoration: underline dotted {RULE_STRONG}; text-underline-offset: 3px; }}
.kpi .v {{ font-size: 1.6rem; font-weight: 600; color: {INK}; line-height: 1.35; font-variant-numeric: tabular-nums; }}
.kpi .n {{ font-size: 0.8rem; color: {MUTED}; font-variant-numeric: tabular-nums; }}

/* Call-outs and empty states */
.callout {{ border: 1px solid {RULE}; border-left: 3px solid {PRIMARY}; background: {SURFACE}; border-radius: 4px;
    padding: 0.7rem 1rem; margin: 1rem 0 0.2rem; font-size: 0.93rem; color: {INK}; max-width: 100ch; }}
.callout.caution {{ border-left-color: {CAUTION}; }}
.callout b {{ font-weight: 600; }}
.empty {{ border: 1px dashed {RULE_STRONG}; border-radius: 6px; background: {SURFACE}; padding: 1.8rem 1.2rem; text-align: center; margin: 1rem 0; }}
.empty b {{ display: block; font-size: 1rem; font-weight: 600; color: {INK}; margin-bottom: 0.25rem; }}
.empty span {{ color: {MUTED}; font-size: 0.9rem; }}

.badge {{ display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.86rem; font-weight: 500; color: {INK}; margin-right: 1rem; }}
.badge .dot {{ width: 0.62rem; height: 0.62rem; border-radius: 50%; display: inline-block; }}
.badge .ct {{ color: {MUTED}; font-variant-numeric: tabular-nums; font-weight: 400; }}

.verdict .l {{ font-size: 0.85rem; color: {MUTED}; }}
.verdict .big {{ font-size: 3.2rem; font-weight: 500; line-height: 1.1; color: {INK}; font-variant-numeric: tabular-nums; margin: 0.1rem 0 0.5rem; }}
.verdict .sub {{ font-size: 0.82rem; color: {MUTED}; margin-top: 0.45rem; }}

.stApp ul.plain-list {{ list-style: none; margin: 0.3rem 0 0; padding: 0; }}
.plain-list li {{ padding: 0.55rem 0; border-top: 1px solid {RULE}; font-size: 0.95rem; }}
.plain-list li:last-child {{ border-bottom: 1px solid {RULE}; }}

.flow {{ display: flex; flex-wrap: wrap; margin: 0.6rem 0 0; border-top: 1px solid {RULE}; border-bottom: 1px solid {RULE}; }}
.flow > div {{ flex: 1 1 200px; padding: 0.85rem 1.1rem; border-left: 1px solid {RULE}; }}
.flow > div:first-child {{ border-left: 0; padding-left: 0; }}
.flow b {{ display: block; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.15rem; }}
.flow span {{ color: {MUTED}; font-size: 0.86rem; }}

.foot {{ color: {MUTED}; font-size: 0.82rem; border-top: 1px solid {RULE}; margin-top: 2.6rem; padding-top: 0.8rem; max-width: 90ch; }}

/* Controls */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{ border-radius: 4px; font-weight: 500; }}
[data-testid="stForm"] {{ border: 1px solid {RULE}; border-radius: 6px; padding: 1.1rem 1.25rem 1.25rem; }}
[data-testid="stDataFrame"] {{ border: 1px solid {RULE}; border-radius: 4px; }}
</style>
"""


def apply_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Data access shared by every page
# ---------------------------------------------------------

def load_scored():
    """Scored sample customers, or a clear error and a clean stop if the model or data are missing."""
    try:
        return get_scored()
    except Exception as exc:
        st.error(f"The model or sample data could not be loaded: {exc}")
        st.stop()


def _stamp(path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %Y, %H:%M")
    except OSError:
        return "unavailable"


def data_source_line(n_records: int) -> str:
    """One line naming the data and model files, with their last-modified times."""
    return (
        f"Data: {SAMPLE_PATH.name}, {n_records:,} records, modified {_stamp(SAMPLE_PATH)}  \u00b7  "
        f"Model: {MODEL_PATH.name}, modified {_stamp(MODEL_PATH)}"
    )


def band_range_label(band: str) -> str:
    """Human-readable probability range for a band, derived from RISK_BANDS so it never drifts."""
    lo, hi = RISK_BANDS[band]
    if lo <= 0:
        return f"Below {hi:.0%}"
    if hi > 1:
        return f"{lo:.0%} and above"
    return f"{lo:.0%} to below {hi:.0%}"


# ---------------------------------------------------------
# Page furniture
# ---------------------------------------------------------

def page_header(title: str, description: str = "", meta: str = ""):
    desc = f'<p class="pg-desc">{html.escape(description)}</p>' if description else ""
    meta_html = f'<p class="pg-meta">{html.escape(meta)}</p>' if meta else ""
    st.markdown(f'<h1 class="pg-title">{html.escape(title)}</h1>{desc}{meta_html}', unsafe_allow_html=True)


def subheader(title: str):
    st.markdown(f'<h2 class="pg-sub">{html.escape(title)}</h2>', unsafe_allow_html=True)


def section(title: str, note: str = ""):
    out = f'<h3 class="sec">{html.escape(title)}</h3>'
    if note:
        out += f'<p class="sec-note">{html.escape(note)}</p>'
    st.markdown(out, unsafe_allow_html=True)


def rule(top_margin: str = "2rem"):
    st.markdown(f'<hr class="pg-rule" style="margin-top:{top_margin}">', unsafe_allow_html=True)


def kpi_row(items):
    """items: iterable of (label, value, note or None) or (label, value, note, help)."""
    cells = []
    for item in items:
        label, value, note = item[0], item[1], item[2]
        tip = item[3] if len(item) > 3 else None
        title = f' title="{html.escape(tip, quote=True)}"' if tip else ""
        n = f'<div class="n">{html.escape(note)}</div>' if note else ""
        cells.append(
            f'<div class="kpi"><div class="l"{title}>{html.escape(label)}</div>'
            f'<div class="v">{html.escape(value)}</div>{n}</div>'
        )
    st.markdown(f'<div class="kpi-row">{"".join(cells)}</div>', unsafe_allow_html=True)


def callout(text: str, kind: str = "info", lead: str = ""):
    """Neutral information box. kind: 'info' (navy rule) or 'caution' (amber rule)."""
    cls = "callout caution" if kind == "caution" else "callout"
    lead_html = f"<b>{html.escape(lead)}</b> " if lead else ""
    st.markdown(f'<div class="{cls}">{lead_html}{html.escape(text)}</div>', unsafe_allow_html=True)


def empty_state(title: str, message: str):
    st.markdown(
        f'<div class="empty"><b>{html.escape(title)}</b><span>{html.escape(message)}</span></div>',
        unsafe_allow_html=True,
    )


@contextmanager
def panel(title: str, note: str = ""):
    """Bordered container with a title: the standard wrapper for a chart or table."""
    with st.container(border=True):
        out = f'<p class="panel-t">{html.escape(title)}</p>'
        if note:
            out += f'<p class="panel-n">{html.escape(note)}</p>'
        st.markdown(out, unsafe_allow_html=True)
        yield


def badge_html(band: str, count=None) -> str:
    color = BAND_COLORS.get(band, MUTED)
    ct = f' <span class="ct">{count:,}</span>' if count is not None else ""
    return (
        f'<span class="badge"><span class="dot" style="background:{color}"></span>'
        f'{html.escape(band)}{ct}</span>'
    )


def band_counts(counts):
    """Inline row of band badges with counts. counts: mapping band -> number."""
    st.markdown("".join(badge_html(b, int(c)) for b, c in counts.items()), unsafe_allow_html=True)


def plain_list(items):
    lis = "".join(f"<li>{html.escape(i)}</li>" for i in items)
    st.markdown(f'<ul class="plain-list">{lis}</ul>', unsafe_allow_html=True)


def flow(steps):
    cells = "".join(f"<div><b>{html.escape(t)}</b><span>{html.escape(d)}</span></div>" for t, d in steps)
    st.markdown(f'<div class="flow">{cells}</div>', unsafe_allow_html=True)


def footnote(text: str):
    st.markdown(f'<div class="foot">{html.escape(text)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Streamlit calls that changed signature across versions
# ---------------------------------------------------------

def show_chart(fig, key=None):
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def show_table(df, **kwargs):
    try:
        return st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        return st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def download_csv(df, filename: str, label: str = "Download CSV", key=None):
    st.download_button(
        label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", key=key,
    )


# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------

def style(fig, height=320, legend=False):
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family=FONT, size=12, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(font_family=FONT, bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(showline=True, linecolor=RULE, gridcolor=GRID, zeroline=False,
                     ticks="outside", tickcolor=RULE, title_font=dict(size=12, color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=RULE,
                     title_font=dict(size=12, color=MUTED))
    return fig


def risk_scale(probability: float, threshold: float):
    """A single horizontal scale showing the four bands with the account's score on it."""
    edges = [round(lo * 100) for lo, _ in RISK_BANDS.values()] + [100]
    fig = go.Figure()
    for (band, color), lo, hi in zip(BAND_COLORS.items(), edges[:-1], edges[1:]):
        fig.add_bar(
            x=[hi - lo], y=[""], base=lo, orientation="h", width=0.55,
            marker=dict(color=color, line=dict(width=0)), opacity=0.92,
            text=band.replace(" Risk", ""), textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white", size=11), hoverinfo="skip", showlegend=False,
        )
    x = probability * 100
    fig.add_shape(type="line", x0=x, x1=x, y0=-0.5, y1=0.5, line=dict(color=INK, width=3))
    fig.add_annotation(x=x, y=0.5, text=f"<b>{probability:.1%}</b>", showarrow=False,
                       yshift=12, font=dict(size=13, color=INK))
    fig.update_layout(barmode="overlay")
    style(fig, height=120)
    fig.update_layout(margin=dict(l=0, r=0, t=22, b=26))
    fig.update_xaxes(range=[0, 100], tickvals=edges, ticktext=[f"{e}%" for e in edges], showgrid=False)
    fig.update_yaxes(visible=False, range=[-0.6, 0.75])
    return fig


def driver_chart(drivers, method: str):
    """Render the output of explain_prediction as a chart with an honest caption."""
    if drivers is None or drivers.empty:
        st.caption(method)
        return
    d = drivers.copy()
    d["Label"] = d["Feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    local = "SHAP" in d.columns
    if local:
        d = d.sort_values("SHAP")
        colors = [RAISES if v > 0 else LOWERS for v in d["SHAP"]]
        fig = go.Figure(go.Bar(x=d["SHAP"], y=d["Label"], orientation="h", marker_color=colors,
                               hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        fig.update_xaxes(title_text="Effect on predicted risk (log-odds)", zeroline=True,
                         zerolinecolor=INK, zerolinewidth=1)
        caption = "Bars to the right raise this account's predicted risk; bars to the left lower it."
    else:
        d = d.sort_values("Importance")
        fig = go.Figure(go.Bar(x=d["Importance"], y=d["Label"], orientation="h", marker_color=PRIMARY,
                               hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.update_xaxes(title_text="Share of model importance", tickformat=".0%")
        caption = ("Account-level explanation was unavailable, so this shows model-wide importance. "
                   "It does not describe this account specifically.")
    style(fig, height=max(220, 38 * len(d) + 50))
    fig.update_yaxes(showgrid=False)
    show_chart(fig)
    st.caption(caption)
