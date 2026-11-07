# RICF-ML v8: Estimate-to-Complete (ETC) modelling

Data: 76 development projects; 902 checkpoints. The 26 previously inspected holdout projects were excluded.

## Five-fold GroupKFold, project-balanced MAPE (%)

| method                      |   project_macro_MAPE |   median_project_MAPE |   project_SD |
|:----------------------------|---------------------:|----------------------:|-------------:|
| Ridge_ETC_residual_stage20  |                6.745 |                 3.757 |        8.930 |
| EVM                         |                6.767 |                 3.544 |        9.271 |
| RF_ETC_residual_stage20     |                6.805 |                 3.321 |        9.342 |
| Ridge_ETC_residual_shrink20 |                6.912 |                 3.824 |        8.857 |
| RF_ETC_residual_shrink20    |                7.027 |                 3.436 |        9.637 |
| RandomForest_ETC            |                9.147 |                 3.021 |       16.666 |
| HistGB_ETC                  |                9.238 |                 3.913 |       15.285 |
| Ridge_ETC_residual          |                9.403 |                 5.023 |       13.046 |
| Ridge_ETC                   |                9.893 |                 5.146 |       13.297 |
| RF_ETC_residual             |                9.936 |                 3.700 |       16.377 |
| BAC_remaining               |               12.904 |                 4.909 |       22.995 |

Best ML versus EVM paired difference (EVM error minus ML error): 0.021 percentage points; 95% cluster bootstrap [-0.332, 0.311].

## Notes
- Target: ETC = independently recorded final project cost minus costs already incurred. The final cost is used only as a supervised-learning label, never as an input feature.
- EAC = AC + predicted ETC; ETC constrained nonnegative.
- EVM baseline ETC = EVM EAC minus AC.
- Group split by project prevents checkpoint mixing across train/test.
- Closure status of most projects is indirectly inferred and needs confirmation.
- No historical risk or ESG predictors are present. No independent confirmatory test is claimed.
- Project-stage analyses are exploratory.
