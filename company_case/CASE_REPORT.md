# ABC Infrastructure Ltd. — Synthetic Project Case / RICF-ML v6

**All company, project and risk data are synthetic. No real company supplied these inputs.** This case is for software demonstration and scenario sensitivity only; it is NOT validation of out-of-sample ML accuracy.

## Case baseline
- Infrastructure construction project `ABC-ROAD-001`, 18-month planned duration; BAC = 1,000 million KZT.
- 18 monthly cumulative PV/EV/AC values were deliberately designed as a coherent illustrative schedule, with PV and EV reaching BAC at month 18.
- The *assumed final actual cost* = 1,216 million KZT, defined by the synthetic narrative rather than ML model predictions.
- **Predictions are made at months 2–17**, prior to final completion. The final outcome is used only for descriptive comparison.
- ML estimates were produced using the saved v5 Random Forest trained on historical DSLIB candidate projects. No retraining was performed on the company case.

## Illustrative errors (NOT scientific validation)
- ML mean checkpoint APE: **8.34%**; traditional EVM mean checkpoint APE: **4.29%**.
- This is one hand-constructed company trajectory, so the comparative errors cannot support statistical inference or claims of general superiority.
- Source ML training used mostly proxy-completion projects and planned duration is expressed in DSLIB source units; 18 months here may not be comparable to source duration units. The synthetic project can be out-of-distribution.

## Risk module
- 12 separate synthetic risk scenarios with assumed occurrence probabilities and triangular severity distributions.
- The simulation uses an **independently specified risk-exclusive baseline of 1,100 million KZT**, not the ML prediction. This prevents presenting an unverified ML+risk sum as a scientifically calibrated outcome.
- Risks are sampled as independent Bernoulli occurrences, an oversimplification; correlated risks and time-dependent losses are future work.
- 20,000 reproducible Monte Carlo draws, random seed 2026; initial remaining reserve 75 million KZT.
- Illustrative P50: **1194.0**, P80: **1245.7**, P90: **1274.5** million KZT; reserve gap at P80 **70.7** million KZT.

## Research interpretation
1. **Historical backtesting** on credible real closed projects evaluates forecast accuracy.
2. **Synthetic case** demonstrates what the user sees, what the software calculates, and how risk assumptions change decisions.
3. The two strands should never be pooled as a single empirical sample.

## Files
- `synthetic_project_checkpoints.csv`: a complete demonstrative trajectory and ML/EVM predictions.
- `synthetic_risk_register.csv`: 12 synthetic risk inputs.
- `synthetic_monte_carlo_draws.csv`: independently simulated scenarios.
- `synthetic_simulation_summary.json`: fixed-seed simulation summary.
- `app.py`: now includes a fifth tab to browse the complete case.
