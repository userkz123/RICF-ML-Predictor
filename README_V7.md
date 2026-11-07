# RICF-ML v7 — Historical database + exploratory temporal correction

Run `pip install -r requirements.txt`, then `streamlit run app.py`. The app now has a separate page **Experimental temporal ML correction (v7)** accessible through the Streamlit sidebar. Upload a CSV of project checkpoints (template provided in the page).

**Scientific status:** the strongest ML-adjusted candidate did **not** beat the EVM baseline on the 76-project training-only grouped CV (EVM 6.845% vs ML correction 7.000% project-macro MAPE). The model is provided for research, not as a validated operational replacement. The earlier 26-project development set has already been analyzed, therefore cannot serve as independent final evidence. Only 13/102 project closure labels are explicit.

The predictor uses only current and prior checkpoints; the dataset's final cost, completion evidence and number of future checkpoints are excluded from the predictor. Full experiment and OOF details are in `research_v7/`.
