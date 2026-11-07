# RICF-ML v12 — Integrated Decision Framework (research prototype)

## Objective
Connect an already-trained ML-assisted ETC forecasting display (from v8) with the v11 independent risk-only Monte Carlo experiment **without double counting**. This release adds an explicit economic loss function for selecting a contingency reserve.

## Scope and scientific status
- The four projects and every risk distribution are synthetic. 20,000 risk draws per project were generated in v11 from assumed independent Bernoulli risk events and triangular conditional losses.
- ML-assisted ETC and EVM point forecasts are loaded from previously developed artifacts; **no ML model is retrained**, and no historical risk register is available for ML training.
- Whether the ML EAC already incorporates future risk cost is unknown. Consequently ML predictions are **not added** to risk losses. Risk-only total cost remains based on the explicitly declared risk-exclusive scenario baseline in v11.
- This simulation does **not** establish ML superiority or field effectiveness of contingency policies.

## Decision model
For a reserve R and remaining incremental risk loss L, define the illustrative cost

    J(R) = c R + p E[max(0, L - R)]

where c is an assumed opportunity cost per unit of reserve committed over the evaluation horizon, and p is an assumed economic penalty rate for uncovered risk loss. If p > c, a continuous optimum with a nonnegative loss distribution corresponds to risk-loss quantile q = 1 - c/p (under standard regularity); if p <= c, R=0. This loss function is a demonstration, not an observed company utility function.

## Scenario design vs evaluation
- Simulated draws are split into first 10,000 **design** draws to select reserves and last 10,000 **evaluation** draws to compare policies.
- Both partitions come from the **same assumed risk-generating process**, so this is reproducibility / Monte Carlo robustness, not empirical external validity.
- Decision preferences are explicitly assumed, not estimated: low penalty c=0.08, p=0.16 (target P50), balanced c=0.08, p=0.40 (target P80), high penalty c=0.08, p=0.80 (target P90).
- For balanced cost preference, benchmark fixed 5% of BAC, EMV, P80, and P90 reserve policies.
- No claimed monetary optimization for a real company until actual funding, liquidity, and risk-loss parameters are established.

## Files
- `v12_decision_policies.csv`: preference sensitivity and out-of-design-draw evaluation.
- `v12_baseline_policy_comparison.csv`: rule comparison under balanced cost preference.
- `v12_policy_summary.csv`: portfolio aggregation.
- `v12_forecast_accounting_audit.csv`: explicit no-addition guardrail for ML EAC and risk losses.
- `pages/10_Integrated_Decision_v12.py`: Streamlit readout.

## Next validations for dissertation
1. Confirm actual completion cost for DSLIB projects; improve training-label reliability.
2. Acquire genuine risk-contingency registers dated before each prediction checkpoint.
3. Validate any additive/integrated risk forecasting rule against actual costs and compare with EVM without test leakage.
4. Elicit and validate real decision costs and constraints from practitioners.
