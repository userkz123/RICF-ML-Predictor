import numpy as np
from risk_accounting_v13 import RiskAccounting, integrated_cost_forecast, risk_quantiles
x=np.zeros(1000); x[::2]=10
s=risk_quantiles(x,5)
assert s['expected_loss']==5
for mode in ('inclusion_unknown','historical_risk_inclusive'):
    try: integrated_cost_forecast(100,x,RiskAccounting(mode))
    except ValueError: pass
    else: raise AssertionError('Should block unknown overlap')
try: integrated_cost_forecast(100,x,RiskAccounting('verified_risk_exclusive'))
except ValueError: pass
else: raise AssertionError('Evidence should be required')
y=integrated_cost_forecast(100,x,RiskAccounting('verified_risk_exclusive','test fixture only, synthetic'))
assert y['expected_total_cost']==105
print('5 checks passed; all fixture data are synthetic.')
