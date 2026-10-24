import numpy as np
import pandas as pd
import streamlit as st

from utils import ui
from utils.risk_utils import (
    DECISION_THRESHOLD, EDUCATION, FEATURE_LABELS, FEATURES, HISTORY_MONTHS, MARRIAGE, PAY_STATUS, SEX,
    decision, explain_prediction, load_model, risk_category, risk_recommendations, status_label,
)

STATUS_TO_CODE = {label: code for code, label in PAY_STATUS.items()}
COL_STATUS, COL_BILL, COL_PAID = "Repayment status", "Statement balance (NT$)", "Payment made (NT$)"
LATEST_MONTH = HISTORY_MONTHS[-1][0]

try:
    model = load_model()
except Exception as exc:
    st.error(f"The trained model could not be loaded: {exc}")
    st.stop()

ui.page_header(
    "Customer default risk assessment",
    "Estimate the probability of default for one customer from their profile, credit limit and six months "
    "of statement history.",
)

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
# Widget keys carry a version number. "Reset inputs" bumps it, which gives every widget a fresh default.

st.session_state.setdefault("assess_v", 0)
V = st.session_state["assess_v"]


def reset_form():
    st.session_state["assess_v"] += 1
    st.session_state.pop("assessment", None)


default_history = pd.DataFrame({
    "Month": [m[0] for m in HISTORY_MONTHS],
    COL_STATUS: [PAY_STATUS[0]] * 6,
    COL_BILL: [38000.0, 39500.0, 41000.0, 42500.0, 44000.0, 45500.0],
    COL_PAID: [2000.0] * 6,
})

with st.form("assessment_form", border=False):
    ui.section("Customer information", "Details recorded for the account holder.")
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age", min_value=18, max_value=100, value=35, step=1, key=f"age_{V}",
                          help="Age in completed years. Values from 18 to 100 are accepted.")
    sex_label = c2.selectbox("Gender", list(SEX.values()), key=f"sex_{V}",
                             help="As recorded for the account. The model uses it as an input.")
    edu_label = c3.selectbox("Education", list(EDUCATION.values()), index=1, key=f"edu_{V}",
                             help="Highest education level recorded for the customer.")
    mar_label = c4.selectbox("Marital status", list(MARRIAGE.values()), index=1, key=f"mar_{V}",
                             help="Marital status as recorded for the customer.")

    ui.section("Financial details", "The credit facility extended to the customer.")
    f1, _spacer = st.columns([1.3, 2.7])
    limit = f1.number_input("Credit limit (NT$)", min_value=0, value=100000, step=5000, key=f"limit_{V}",
                            help="Total credit line approved for the account, in New Taiwan dollars. Must be above zero.")

    ui.section(
        "Credit history",
        "Oldest month first. Edit the cells directly. Repayment status describes how that month's payment compared "
        "with the due date. A negative statement balance means the account was in credit.",
    )
    history = st.data_editor(
        default_history,
        hide_index=True,
        num_rows="fixed",
        disabled=["Month"],
        key=f"history_{V}",
        column_config={
            "Month": st.column_config.TextColumn(width="small"),
            COL_STATUS: st.column_config.SelectboxColumn(
                options=list(PAY_STATUS.values()), required=True, width="medium",
                help="Payment behaviour in that month. 'Minimum paid (revolving)' means only the minimum "
                     "was paid and a balance carried over."),
            COL_BILL: st.column_config.NumberColumn(
                format="localized", step=1000, required=True,
                help="Balance on that month's statement. Negative values mean the account was in credit."),
            COL_PAID: st.column_config.NumberColumn(
                format="localized", min_value=0, step=500, required=True,
                help="Total amount the customer paid in that month. Cannot be negative."),
        },
    )

    b1, b2, _fill = st.columns([1.6, 1, 4])
    with b1:
        submitted = st.form_submit_button("Calculate default risk", type="primary")
    with b2:
        st.form_submit_button("Reset inputs", on_click=reset_form)


def build_inputs(history_df: pd.DataFrame) -> pd.DataFrame:
    row = {
        "LIMIT_BAL": float(limit),
        "SEX": {v: k for k, v in SEX.items()}[sex_label],
        "EDUCATION": {v: k for k, v in EDUCATION.items()}[edu_label],
        "MARRIAGE": {v: k for k, v in MARRIAGE.items()}[mar_label],
        "AGE": int(age),
    }
    by_month = history_df.set_index("Month")
    for month, pay, bill, paid in HISTORY_MONTHS:
        row[pay] = STATUS_TO_CODE[by_month.loc[month, COL_STATUS]]
        row[bill] = float(by_month.loc[month, COL_BILL])
        row[paid] = float(by_month.loc[month, COL_PAID])
    return pd.DataFrame([row], columns=FEATURES)


if submitted:
    if limit <= 0:
        st.error("Enter a credit limit above zero.")
        st.session_state.pop("assessment", None)
    elif history.isna().any().any():
        st.error("Every month needs a repayment status, a statement balance and a payment amount.")
        st.session_state.pop("assessment", None)
    else:
        X = build_inputs(history)
        probability = float(np.clip(model.predict_proba(X)[0, 1], 0, 1))
        drivers, method = explain_prediction(model, X, top_n=6)
        data_notes = []
        if (history[COL_BILL].astype(float) > float(limit)).any():
            data_notes.append("One or more statement balances are above the credit limit. Confirm the figures are correct.")
        st.session_state["assessment"] = dict(
            X=X, history=history.copy(), probability=probability, drivers=drivers, method=method, notes=data_notes,
        )

result = st.session_state.get("assessment")

# ---------------------------------------------------------
# Result
# ---------------------------------------------------------

if result is None:
    ui.callout(
        "Complete the form and select Calculate default risk. The estimate, risk band and score drivers appear here.",
        lead="No assessment yet.",
    )
else:
    X, hist, p = result["X"], result["history"], result["probability"]
    band = risk_category(p)
    thr = DECISION_THRESHOLD

    interpretation = {
        "Very High Risk": f"The estimated probability is in the highest band and well above the {thr:.0%} decision "
                          f"threshold. Referral for manual review is recommended.",
        "High Risk": f"The estimated probability is at or above the {thr:.0%} decision threshold. Referral for "
                     f"additional review is recommended.",
        "Medium Risk": f"The estimated probability is below the {thr:.0%} decision threshold but in the "
                       f"elevated-monitoring band.",
        "Low Risk": f"The estimated probability is below the {thr:.0%} decision threshold. Standard monitoring applies.",
    }[band]

    ui.rule()
    ui.section("Assessment result")
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown(
            f'<div class="verdict"><div class="l">Predicted probability of default</div>'
            f'<div class="big">{p:.1%}</div>{ui.badge_html(band)}'
            f'<div class="sub">Decision flag: {decision(p)}</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(f"**{interpretation}**")
        ui.show_chart(ui.risk_scale(p, thr))
        st.caption(f"The marker shows the estimate on the four risk bands. The {thr:.0%} mark is the binary decision "
                   f"threshold, not a default probability.")

    for note in result.get("notes", []):
        ui.callout(note, kind="caution", lead="Data check.")
    ui.callout(
        "This probability is a statistical estimate produced from the information entered. It is not a record of a "
        "confirmed default or non-default, and it should be read alongside bank policy and manual review.",
        lead="Model estimate, not a confirmed outcome.",
    )

    # Indicators derived directly from the submitted inputs (not model outputs)
    limit_used = float(X.iloc[0]["LIMIT_BAL"])
    bills = hist[COL_BILL].astype(float)
    paid = hist[COL_PAID].astype(float)
    codes = hist[COL_STATUS].map(STATUS_TO_CODE)
    latest_util = bills.iloc[-1] / limit_used
    positive_bills = bills.clip(lower=0).sum()
    pay_ratio = paid.sum() / positive_bills if positive_bills > 0 else None
    months_late = int((codes >= 1).sum())

    ui.section("Account indicators", "Calculated directly from the submitted inputs. They are not model outputs.")
    ui.kpi_row([
        (f"{LATEST_MONTH} utilisation", f"{latest_util:.0%}", "Latest balance / credit limit",
         "Share of the credit limit in use on the most recent statement."),
        ("Six-month payment ratio", f"{pay_ratio:.0%}" if pay_ratio is not None else "n/a", "Total paid / total billed",
         "Total payments made over six months divided by total positive statement balances."),
        ("Months in arrears", f"{months_late} of 6", "Repayment status of 1 or more",
         "Number of months in which the payment was one or more months late."),
        ("Longest delay", status_label(codes.max()) if codes.max() >= 1 else "None", None,
         "Most severe repayment status across the six months."),
    ])

    t_drivers, t_handling, t_inputs = st.tabs(["Score drivers", "Suggested handling", "Submitted inputs"])

    with t_drivers:
        ui.section("What drove this score", "The inputs with the largest influence on this account's estimate.")
        ui.driver_chart(result["drivers"], result["method"])

    with t_handling:
        ui.section("Suggested handling", f"General guidance for the {band} band. Apply institutional policy.")
        ui.plain_list(risk_recommendations(band))

    with t_inputs:
        row = X.iloc[0]
        ui.section("Customer and credit facility")
        ui.show_table(pd.DataFrame([
            ("Age", f"{int(row['AGE'])}"),
            ("Gender", SEX.get(int(row["SEX"]), "Other")),
            ("Education", EDUCATION.get(int(row["EDUCATION"]), "Other")),
            ("Marital status", MARRIAGE.get(int(row["MARRIAGE"]), "Other")),
            ("Credit limit (NT$)", f"{row['LIMIT_BAL']:,.0f}"),
        ], columns=["Field", "Value"]))
        ui.section("Credit history")
        ui.show_table(hist, column_config={
            COL_BILL: st.column_config.NumberColumn(format="localized"),
            COL_PAID: st.column_config.NumberColumn(format="localized"),
        })
        with st.expander("Model input vector (23 fields)"):
            shown = []
            for f in FEATURES:
                v = row[f]
                if f == "SEX":
                    v = SEX.get(int(v), v)
                elif f == "EDUCATION":
                    v = EDUCATION.get(int(v), "Other")
                elif f == "MARRIAGE":
                    v = MARRIAGE.get(int(v), "Other")
                elif f.startswith("PAY_") and not f.startswith("PAY_AMT"):
                    v = status_label(v)
                elif f == "AGE":
                    v = f"{int(v)}"
                else:
                    v = f"{v:,.0f}"
                shown.append((FEATURE_LABELS[f], f, v))
            ui.show_table(pd.DataFrame(shown, columns=["Field", "Model column", "Value"]))

ui.footnote(
    "The score is a machine-learning estimate for decision support. It should not be used as an "
    "automatic approval or rejection."
)
