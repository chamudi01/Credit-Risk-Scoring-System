# Bank Credit Risk Management System (Streamlit)

Scores customers with the saved preprocessing + XGBoost pipeline and presents portfolio, segment and
account-level views. No live bank database is required.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Requires Streamlit 1.40 or newer (`st.navigation`, row selection, pills, segmented control).

## Structure

```text
app.py                  Page config, theme, navigation
.streamlit/config.toml  Colours, font, minimal toolbar
assets/favicon.png      Browser-tab icon
views/
  overview.py           Credit risk portfolio overview
  portfolio.py          Portfolio analytics: expected loss with adjustable LGD / EAD
  segments.py           Customer segmentation: risk by customer group
  assessment.py         Customer default risk assessment (form, result, inputs)
  explorer.py           Filterable account table with drill-down and CSV export
  model_info.py         Model documentation (tabbed)
utils/
  risk_utils.py         Model loading, scoring, bands, labels, SHAP (unchanged)
  ui.py                 Design tokens, theme CSS, reusable components, chart styling
model/xgboost_pipeline.joblib
model/model_card.json   Optional: training-data details and evaluation metrics (see below)
data/sample_customers.csv
```

## Design notes

- All colours and the font live in `utils/ui.py` (tokens at the top) and `.streamlit/config.toml`.
- Risk-band colours are defined once in `ui.BAND_COLORS` and reused by every chart. Each has at least 4.5:1
  contrast against white, so labels drawn on the bars stay readable.
- Reusable components in `utils/ui.py`: `page_header`, `section`, `kpi_row`, `panel` (bordered chart/table
  container), `callout`, `empty_state`, `badge_html` / `band_counts`, `plain_list`, `footnote`, `show_chart`,
  `show_table`, `download_csv`, `style` (Plotly), `risk_scale`, `driver_chart`.
- Band ranges shown in the interface come from `RISK_BANDS` (`ui.band_range_label`), so they follow the thresholds
  in `risk_utils.py`.
- The sidebar highlights the active page using Streamlit's `aria-current` attribute. If a future Streamlit version
  changes it, the page still works and Streamlit's own highlighting applies.
- The font (IBM Plex Sans) loads from Google Fonts; the app falls back to system fonts offline.
- `utils/explainability.py` is the original notebook-artifact loader. The app no longer uses it.

## Model

The pipeline expects 23 raw input columns (`LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`,
`PAY_0`, `PAY_2` to `PAY_6`, `BILL_AMT1` to `BILL_AMT6`, `PAY_AMT1` to `PAY_AMT6`). It performs its own
preprocessing, so do not add engineered columns.

Decision threshold 31%. Bands: Low below 20%, Medium 20% to below 31%, High 31% to below 50%,
Very High 50% and above.

To use other data, replace `data/sample_customers.csv` with the same 23 columns (`Customer_ID` optional).
To use another model, replace the joblib file with a compatible pipeline whose steps are named
`preprocessor` and `model`.

## Optional model card

The Model documentation page never invents figures. Training-data details and evaluation metrics appear only if
`model/model_card.json` exists. Include only what you have recorded; every key is optional:

```json
{
  "training_data": {
    "name": "<dataset name>",
    "source": "<where it came from>",
    "records": "<number of rows>",
    "default_rate": "<share between 0 and 1>",
    "period": "<time period covered>",
    "notes": "<anything worth stating>"
  },
  "evaluation": {
    "dataset": "<e.g. hold-out test set>",
    "records": "<number of rows evaluated>",
    "metrics": { "<metric name>": "<value>" }
  }
}
```

## Note

Academic demonstration only. Not a production credit-decision system.
