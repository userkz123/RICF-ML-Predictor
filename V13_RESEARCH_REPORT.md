# RICF-ML v13 — Risk Integration Readiness and Accounting Guardrails

## 1. Purpose
This package audits what is **actually present** in the previously prepared DSLIB-derived table and provides a reproducible, guarded connector between historical EAC forecasts and an independent scenario-based risk register.

## 2. Observations from available artifacts
- 102 DSLIB-derived projects and 1258 historical checkpoint rows are present in the included historical_checkpoints.csv.
- That table has EVM indicators, checkpoint characteristics and proxy final-cost labels, **not a time-stamped historical risk registry**. Therefore the ML models in prior releases are **not validated risk-trained ML predictors**.
- The attached portfolio_risk_register.csv has 48 risk records from 4 projects, all labeled synthetic. They must not be joined to historical DSLIB projects as though measured.
- Closure-label reliability remains an unresolved issue: previous audits distinguished 13 explicit closures from 89 proxies; retain this caveat.

## 3. Key scientific guardrail
An ML model trained on historical final cost may implicitly reflect ordinary risk outcomes. Hence **ML EAC + all simulated risk losses is blocked** by default. The optional additive integrated-cost function requires written evidence that the supplied baseline excludes precisely the simulated risks. Selecting that mode does not validate the evidence itself.

## 4. Demonstration
`V13_SAFE_RISK_OVERLAY.csv` shows both forecasts and risk-only P80/P90 amounts for the four artificial projects; combined ML+Risk EAC is intentionally blank. The risk-only Monte Carlo inputs are *synthetic*, and the numbers describe scenario assumptions, not observed company losses.

## 5. How to make genuine ML + risks possible
1. Collect dated risk snapshots and real loss realization for the same historical projects as EVM checkpoints.
2. Confirm final costs from primary project records, record currency and whether BAC includes contingency.
3. Define risk-inclusion accounting at the checkpoint: which costs are embedded in EAC, which are genuinely incremental.
4. Train a **risk-aware** model in nested project-level cross-validation; compare EVM-only vs EVM+risk feature sets on the same projects and time points.
5. Freeze model/hyperparameters, then conduct independent external evaluation on new completed projects. Report errors, uncertainty coverage, and reserve decision costs.

## 6. Run
`python test_v13.py` executes the accounting guardrail tests (NumPy required). `risk_accounting_v13.py` contains no training code and deliberately refuses unverified addition.

## Conclusion
v13 improves methodological integrity and a concrete integration interface. It does **not** establish improvement over EVM or produce a newly risk-trained model.
