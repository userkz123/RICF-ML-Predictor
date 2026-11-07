# RICF-ML v10 — Probabilistic Forecasting and Contingency

## What was actually evaluated
- Data: 76 previously used DSLIB development projects; 5-fold group-out-of-fold point predictions from v8.
- Methods: Traditional EVM and previously selected experimental Ridge ETC correction.
- For each target stage (25%, 50%, 75%), one nearest-progress checkpoint per project was evaluated. For every fold, interval half-width was calculated using project-level absolute errors from the other four folds only.
- Evaluation uses prior model-selection data, so coverage is **exploratory**, not independent prospective confirmation. Calibration set contains around 60 projects per fold.

## Coverage and interval width
| method          | stage   |   nominal_coverage |   n_projects |   empirical_coverage |   mean_width_pct_BAC |   mae_pct_BAC |
|:----------------|:--------|-------------------:|-------------:|---------------------:|---------------------:|--------------:|
| EVM             | 25%     |              0.800 |           76 |                0.816 |               29.194 |         9.367 |
| EVM             | 25%     |              0.900 |           76 |                0.908 |               43.956 |         9.367 |
| EVM             | 50%     |              0.800 |           76 |                0.829 |               19.052 |         7.382 |
| EVM             | 50%     |              0.900 |           76 |                0.921 |               40.550 |         7.382 |
| EVM             | 75%     |              0.800 |           76 |                0.816 |               11.300 |         5.118 |
| EVM             | 75%     |              0.900 |           76 |                0.908 |               19.669 |         5.118 |
| ML-assisted ETC | 25%     |              0.800 |           76 |                0.803 |               28.155 |         9.194 |
| ML-assisted ETC | 25%     |              0.900 |           76 |                0.908 |               49.473 |         9.194 |
| ML-assisted ETC | 50%     |              0.800 |           76 |                0.816 |               19.681 |         7.277 |
| ML-assisted ETC | 50%     |              0.900 |           76 |                0.908 |               36.452 |         7.277 |
| ML-assisted ETC | 75%     |              0.800 |           76 |                0.803 |               11.011 |         5.057 |
| ML-assisted ETC | 75%     |              0.900 |           76 |                0.882 |               19.868 |         5.057 |

## Caveats
- Finite sample quantile thresholds were used. Standard conformal coverage assumptions are not guaranteed because the model was selected using these same development projects and because the five cross-validation training sets overlap.
- Interval half-width is measured relative to BAC, so use of the model on fundamentally different industries or project scales may not preserve coverage.
- 89 of 102 project completion labels in the larger dataset require verification; this test uses only 76 development projects and does not resolve that issue.
- Differences in risk, ESG and contingency are NOT model training features. The interactive scenario is independently simulated and explicitly conditional on entered assumptions.
- No risk costs are automatically added to ML EAC to avoid double counting.

## Next scientific validation
- Freeze model architecture and calibration parameters before acquiring independent completed projects.
- Confirm project closeout documentation and actual final project costs.
- Evaluate coverage, interval width, point errors, and contingency adequacy on untouched projects, reporting project-level uncertainty.
