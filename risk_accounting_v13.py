"""RICF-ML v13. Validated accounting-mode guardrails. Not an ML model trainer."""
from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass(frozen=True)
class RiskAccounting:
    mode: Literal['inclusion_unknown','verified_risk_exclusive','historical_risk_inclusive']
    evidence: str = ''

def risk_quantiles(loss_draws, reserve=0.0):
    x=np.asarray(loss_draws,dtype=float)
    if x.ndim != 1 or x.size < 100 or not np.all(np.isfinite(x)) or np.any(x<0):
        raise ValueError('At least 100 finite nonnegative loss draws are required')
    if reserve < 0: raise ValueError('reserve must be nonnegative')
    return {'expected_loss':float(x.mean()),'p50_loss':float(np.quantile(x,.5)),
            'p80_loss':float(np.quantile(x,.8)),'p90_loss':float(np.quantile(x,.9)),
            'probability_of_reserve_exceedance':float(np.mean(x>reserve))}

def integrated_cost_forecast(base_eac, loss_draws, accounting: RiskAccounting):
    if not np.isfinite(base_eac) or base_eac < 0: raise ValueError('bad EAC')
    if accounting.mode != 'verified_risk_exclusive':
        raise ValueError('Addition BLOCKED: ML/EVM EAC may already include these risks')
    if not accounting.evidence.strip():
        raise ValueError('Documented evidence for risk-exclusive baseline is mandatory')
    stats=risk_quantiles(loss_draws)
    return {'expected_total_cost':base_eac+stats['expected_loss'],
            'p80_total_cost':base_eac+stats['p80_loss'],
            'p90_total_cost':base_eac+stats['p90_loss'],
            'accounting_evidence':accounting.evidence}
