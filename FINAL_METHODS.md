# Research Methods and Scope — final synthesis

## Proposed dissertation contribution
A cost management decision-support architecture integrating (1) observed project-history EVM and experimental ML forecasting; (2) uncertainty and scenario-based risk loss estimation; and (3) economically justified contingency selection.

## What is validated
Historical EVM and ML algorithms were compared with grouped project-level splits. Their relative performance was inconsistent across versions/stages and does not establish the superiority of ML. Retrospective intervals were evaluated on development data, rather than external observations.

## What is illustrative
Synthetic four-project case; Monte Carlo distributions inferred from hypothetical per-risk triangular severities and Bernoulli occurrence probabilities; optimization function J(R)=cR+p E[max(0,L-R)]. Under p>c, optimum risk-loss quantile is q=1-c/p (subject to distribution assumptions).

## Critical accounting boundary
An ML predictor trained on final cost predicts a cost incorporating whatever ordinary project risk losses were realized. It is *not* generally valid to add simulated losses to this forecast. The program requires an explicit risk-exclusive accounting assertion and non-empty evidence text before showing additive totals; this is a user statement, not independent verification.

## Minimum future-data requirements
Project ID, checkpoint date, BAC, AC, EV, PV, verified actual final cost, source documents, dated risk probability and severities, date of risk realization, contingency balance, clarity on cost inclusion, and categorical ESG type if present.

## Evaluation design
Group split by whole projects, nested CV for any tuning, freeze final candidate once, evaluate on new projects; report stage-stratified MAE/MAPE, paired project bootstrap uncertainty, coverage/width of predictive intervals and realized reserve decision cost.
