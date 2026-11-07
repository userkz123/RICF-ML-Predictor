"""RICF-ML v2 numerical functions. Risk simulation independent of trained ML."""
import numpy as np
import pandas as pd

def forecast(model, features, params, halfwidth80, halfwidth90):
    b=float(params['bac']);ev=float(params['ev']);ac=float(params['ac']);pv=float(params['pv'])
    if min(b,ev,ac,pv)<=0: raise ValueError('BAC, EV, AC and PV must be positive')
    if ev>b*1.002: raise ValueError('EV > BAC: outside study task')
    x=pd.DataFrame([dict(progress=ev/b,pv_ratio=pv/b,ac_ratio=ac/b,cpi=ev/ac,spi=ev/pv,planned_duration=float(params['planned_duration']),n_activities=int(params['n_activities']))])[features]
    ratio=float(model.predict(x)[0]);
    return {'ml_eac':ratio*b,'evm_eac':b/(ev/ac),'bac':b,'ratio':ratio,'lower80':max(0,ratio-halfwidth80)*b,'upper80':(ratio+halfwidth80)*b,'lower90':max(0,ratio-halfwidth90)*b,'upper90':(ratio+halfwidth90)*b,'cpi':ev/ac,'spi':ev/pv}

def simulate_risks(baseline, risks, remaining_reserve, n=10000,seed=2026):
    """Independent Bernoulli risks with triangular conditional severity. Only *incremental* future risks.

    NOTE: In the combined display, baseline is considered exclusive of entered future risk costs.
    This is a modeling assumption, not guaranteed for ML EAC, which may already reflect generic risk.
    """
    if n<100 or n>200000: raise ValueError('n out of range')
    if baseline<0 or remaining_reserve<0: raise ValueError('Amounts must be nonnegative')
    rng=np.random.default_rng(seed)
    total=np.zeros(n)
    for r in risks:
        p=float(r['probability']);low=float(r['min_loss']);mode=float(r['most_likely_loss']);high=float(r['max_loss'])
        if not(0<=p<=1 and 0<=low<=mode<=high):raise ValueError('Risk values invalid')
        total += (rng.random(n)<p)*rng.triangular(low,mode,high,size=n)
    c=baseline+total
    p50,p80,p90=np.quantile(c,[.5,.8,.9]);reserve80=max(0,p80-baseline)
    return {'mean':float(c.mean()),'p50':float(p50),'p80':float(p80),'p90':float(p90),'risk_loss_mean':float(total.mean()),'reserve_required_p80':float(reserve80),'reserve_gap_p80':float(reserve80-remaining_reserve),'prob_exceeds_baseline_plus_reserve':float((total>remaining_reserve).mean()),'samples':c}
