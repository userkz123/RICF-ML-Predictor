# RICF-ML — consolidated research prototype (final handover)

## What is delivered
One Streamlit application combining the historical DSLIB-derived project library, the trained Random Forest baseline, experimental EVM/ETC components, exploratory prediction intervals, the synthetic company/portfolio demonstration, contingency decision analysis, and the guarded risk integration page.

This is **the final consolidated prototype**, not a newly trained risk-aware ML model and not evidence that ML universally outperforms EVM.

## Scientific status and findings
- 102 historical project candidates / 1,258 checkpoints in the previous extraction.
- Only 13 explicitly marked Closure; 89 completion endpoints are proxies and need primary-source confirmation.
- Historical project risk / ESG snapshot data are not available in this training table.
- 26 previously examined development projects do not qualify as an untouched independent external test.
- Earlier experimental ML-assisted ETC showed only a marginal difference versus EVM; its significance was not established.
- The four-company-project portfolio and risk register are synthetic. Never label them as real companies.
- Intervals and reserve recommendations are exploratory, not externally calibrated guarantees.

## Run on Mac / Windows / Linux
1. Install Python 3.10–3.12 (or another compatible release with the requirements).
2. In a terminal, `cd` into this extracted folder.
3. `python3 -m venv .venv` then `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
4. `python -m pip install -r requirements.txt`
5. `python -m streamlit run app.py`

Navigate through the pages in Streamlit's left sidebar; `Risk Integration Final` is the newest module.

## Suggested dissertation statement
We implemented a reproducible prototype for longitudinal EVM-based cost forecasting, experimental ML corrections, probabilistic risk and contingency decision support. Improvements in external forecasting accuracy remain unconfirmed; a future validation dataset with verified actual completion costs and dated risk indicators is required.

## How to make this a validated research tool
1. Confirm actual final cost in the DSLIB source project records.
2. Collect dated historical risk snapshots on the same projects.
3. Freeze training pipeline, hyperparameters and all selection criteria.
4. Test on projects not previously accessed for model selection; include EVM/BAC benchmarks, project-level CI and uncertainty coverage.
5. Document whether baseline forecast includes contingency and realized risk costs.

## Files
`app.py`, `pages/` — UI; `historical_projects.sqlite` — database; `trained_model.joblib` and `*_model.joblib` — experimental pretrained models; `risk_accounting_v13.py` — accounting safeguard; `portfolio_*.csv` — **synthetic** scenario; `V12_RESEARCH_REPORT.md`, `V13_RESEARCH_REPORT.md` — methodological context; `FINAL_METHODS.md` — summary.
