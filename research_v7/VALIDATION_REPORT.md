# RICF-ML v7 — Temporal feature & EVM residual-correction experiment

Training: **76 projects**, **915 checkpoint observations**. All 26 previously studied development-holdout projects excluded from model selection.

## Model selection — GroupKFold out-of-fold error (project-balanced)

| method                        |   project_macro_MAPE_pct |   median_project_MAPE_pct |   sd_project_MAPE_pct |
|:------------------------------|-------------------------:|--------------------------:|----------------------:|
| EVM                           |                    6.845 |                     3.544 |                 9.322 |
| Ridge_residual_shrink25_early |                    7.000 |                     3.300 |                 9.507 |
| RF_residual_shrink25_early    |                    7.180 |                     3.386 |                 9.926 |
| Ridge_residual_shrink25       |                    7.243 |                     3.554 |                 9.875 |
| RF_residual_shrink25          |                    7.575 |                     3.127 |                10.843 |
| Ridge_residual_shrink50       |                    8.025 |                     3.345 |                11.534 |
| RF_residual_shrink50          |                    8.549 |                     3.170 |                13.988 |
| Ridge_residual                |                   10.236 |                     4.497 |                16.644 |
| RF_full                       |                   10.585 |                     3.945 |                16.636 |
| Ridge_full                    |                   11.105 |                     4.487 |                18.021 |
| RF_residual                   |                   11.105 |                     3.429 |                22.196 |
| BAC                           |                   13.957 |                     4.912 |                23.927 |

Best non-EVM candidate: **Ridge_residual_shrink25_early**. Paired project error advantage over EVM (positive = model better): **-0.155 pp**, 95% project bootstrap **[-0.710, 0.241] pp**.

## Caveats
- 89 of 102 project end-points use inferred closure, not verified independent financial close-out.
- More than one checkpoint per project: group splitting prevents a project appearing in both training and validation.
- No future observations, final cost, final number of checkpoints, or closure evidence are features.
- Current and lagged cost performance only; risk and ESG historical data are not represented.
- Model comparisons are exploratory and do not constitute an untouched confirmatory test.
- An experimental residual model artifact is supplied regardless of rank; it is not automatically recommended for deployment.
- Stage analyses and comparisons across many variants are exploratory; multiple comparisons can overstate improvements.

## Technical approach
Residual model predicts final_cost / BAC minus current EVM EAC / BAC. At inference the predicted residual is added to the conventional EVM estimate. Shrunk and stage-gated variants are assessed solely through training-fold predictions.