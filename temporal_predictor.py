"""Temporal EVM residual adjustment: strictly a research prototype."""
import numpy as np
import pandas as pd

def engineered_history(df,sectors):
    # Input only past/current checkpoints. Do not provide final project outcome.
    d=df.copy().sort_values(['project_id','checkpoint']).reset_index(drop=True)
    for col in ['project_id','checkpoint','bac','ev','pv','ac','planned_duration','n_activities','sector']:
        if col not in d:raise ValueError(f'Missing column: {col}')
    if (d[['bac','ev','pv','ac']]<=0).any().any():raise ValueError('BAC/EV/PV/AC must be positive')
    if (d.ev>d.bac*1.002).any():raise ValueError('EV exceeds BAC')
    d['progress']=d.ev/d.bac;d['pv_ratio']=d.pv/d.bac;d['ac_ratio']=d.ac/d.bac
    d['cpi']=d.ev/d.ac;d['spi']=d.ev/d.pv
    d['remaining']=1-d.progress;d['evm_ratio']=d.ac_ratio/d.progress.clip(lower=.001)
    d['log_duration']=np.log1p(d.planned_duration);d['log_activities']=np.log1p(d.n_activities)
    d['previous_cpi']=d.groupby('project_id').cpi.shift(1)
    d['cpi_trend']=d.cpi-d.previous_cpi
    d['previous_progress']=d.groupby('project_id').progress.shift(1)
    d['progress_increment']=d.progress-d.previous_progress
    d['cost_variance_ratio']=d.progress-d.ac_ratio
    d['schedule_variance_ratio']=d.progress-d.pv_ratio
    d['prior_ac_ratio']=d.groupby('project_id').ac_ratio.shift(1)
    d['ac_delta']=d.ac_ratio-d.prior_ac_ratio
    d['cpi_delta']=d.cpi-d.groupby('project_id').cpi.shift(1)
    d['cpi_rolling_mean']=d.groupby('project_id').cpi.transform(lambda s:s.rolling(3,min_periods=1).mean())
    d['cpi_rolling_std']=d.groupby('project_id').cpi.transform(lambda s:s.rolling(3,min_periods=2).std())
    d['progress_delta']=d.progress-d.groupby('project_id').progress.shift(1)
    d['ac_per_progress_delta']=d.ac_delta/d.progress_delta.clip(lower=.001)
    d['cost_slope_ratio']=d.ac_per_progress_delta.clip(0,5)
    d['cost_acceleration']=d.cost_slope_ratio-d.groupby('project_id').cost_slope_ratio.shift(1)
    d['sector_cat']=d.sector.fillna('Other').astype(str).str.extract(r'^(Construction|IT|Engineering|Mobility)',expand=False).fillna('Other')
    for c in sectors:d[f'sector_{c}']=(d.sector_cat==c).astype(float)
    return d

def predict_history(df,payload):
    d=engineered_history(df,payload['sectors'])
    residual=payload['model'].predict(d[payload['features']])
    evm=d.evm_ratio.to_numpy();weight=.25*(1-d.progress.to_numpy())
    d['EVM EAC']=evm*d.bac.to_numpy()
    d['experimental ML-adjusted EAC']=(evm+weight*residual)*d.bac.to_numpy()
    return d
