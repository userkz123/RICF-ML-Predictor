# RICF-ML v8: experimental Estimate-to-Complete forecasting

Run: `pip install -r requirements.txt` then `streamlit run app.py`. In the sidebar open **ETC Predictor v8**.

The v8 page accepts a CSV containing project_id, checkpoint, bac, ev, pv, ac, planned_duration, n_activities, sector and generates EVM versus ML-assisted EAC. Example template is in the page. The correction model is a Ridge regressor trained on 76 development projects; it is not shown to outperform EVM significantly.

The 26 formerly inspected holdout projects were excluded from model tuning but are no longer considered an untouched independent test set because they were studied in versions v1–v6. End-cost labels for most projects are proxies requiring financial verification.
