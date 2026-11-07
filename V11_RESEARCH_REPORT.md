# RICF-ML v11 — Synthetic Portfolio Decision Experiment

## Design
Four explicitly fictional infrastructure projects. All budgets, checkpoint histories and risk inputs are synthetic assumptions. We use the **unchanged v8 ML-assisted ETC artifact** (packaged in v10), without retraining on the fictional data, to display EVM and ML forecasts. No actual final cost exists for these fictional projects, so no accuracy comparison can be made.

The **risk-only Monte Carlo** uses 12 risk categories inherited from the v6 illustrative register, scaled by published scenario rules. Every project has 20,000 risk realizations with fixed independent random seeds. The scenario assumes independent Bernoulli events and conditional triangular severity; these parameters are not observed or estimated from DSLIB or Kazakhstan companies.

## Project parameters
| project_id        | project_type      |   BAC_million_KZT |   duration |   n_activities |   risk_exclusive_baseline_million_KZT |   available_reserve_million_KZT |   probability_multiplier |   severity_multiplier |
|:------------------|:------------------|------------------:|-----------:|---------------:|--------------------------------------:|--------------------------------:|-------------------------:|----------------------:|
| ABC-ROAD-001      | Road construction |              1000 |         18 |             62 |                                  1100 |                              75 |                     0.94 |                   1   |
| ABC-ENERGY-002    | Energy substation |              1650 |         22 |             78 |                                  1770 |                             105 |                     1.08 |                   1.2 |
| ABC-WATER-003     | Water treatment   |               780 |         15 |             48 |                                   850 |                              50 |                     0.87 |                   0.8 |
| ABC-LOGISTICS-004 | Logistics center  |              1250 |         20 |             71 |                                  1350 |                              88 |                     1    |                   1.1 |

## Portfolio reserve-policy comparison
| policy              |   total_reserve |   expected_shortfall |   mean_probability_insufficient |
|:--------------------|----------------:|---------------------:|--------------------------------:|
| Expected loss (EMV) |         494.529 |              111.995 |                           0.463 |
| Fixed 5% BAC        |         234.000 |              282.645 |                           0.773 |
| Risk P80            |         724.225 |               38.130 |                           0.200 |
| Risk P90            |         871.309 |               16.696 |                           0.100 |

## Interpretation
- Expected shortfall is average `max(0, simulated incremental risk loss − allocated reserve)`; unused reserve is not necessarily an economic loss.
- Fixed 5% BAC vs risk EMV vs P80 vs P90 are **alternative allocation policies** evaluated against an explicitly chosen risk-generating process. P80 is not optimized for cost and cannot be called optimal without a decision-loss function.
- Risk-inclusive EAC is **not** computed by adding the risk distribution to historical-data ML forecasts; that could double count risk.
- Because scenarios are generated from the same assumed distributions used for P80/P90, high nominal coverages are mathematical consequences of assumptions, not out-of-sample evidence.
- Project labels and 102-project proxy-closeout limitations persist in trained ML artifacts. The development history was already explored; independent projects remain necessary.

## Reproducibility
Run `python reproduce_portfolio.py` in this directory. It regenerates the scenario files from v6 input risk register, scales probability and severity by the parameters above, uses seeds 20261107–20261110, and applies the existing pretrained ETC estimator to each fictional project. No fabricated result is presented as observed.
