"""Final risk integration module: synthetic demonstration and optional user risk register."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from risk_accounting_v13 import RiskAccounting, risk_quantiles, integrated_cost_forecast

HERE=Path(__file__).resolve().parents[1]
st.title('Risk integration & contingency decisions')
st.warning('Risk distributions here are user-provided or SYNTHETIC. ML was NOT trained on the risk register. Do not add risk costs to ML EAC unless the baseline explicitly excludes those same risks.')
upload=st.file_uploader('Optional CSV with risk register',type=['csv'])
if upload is None:
    df=pd.read_csv(HERE/'portfolio_risk_register.csv')
    st.caption('Demo: synthetic ABC portfolio risk assumptions (not actual company observations).')
else:
    df=pd.read_csv(upload)
required=['probability','min_loss','most_likely_loss','max_loss']
missing=[c for c in required if c not in df.columns]
if missing:
    st.error('Missing columns: '+', '.join(missing)); st.stop()
if 'project_id' in df.columns:
    options=sorted(df['project_id'].dropna().astype(str).unique().tolist())
    project=st.selectbox('Project',options)
    df=df[df['project_id'].astype(str)==project].copy()
if df.empty: st.error('No risks found'); st.stop()
for c in required: df[c]=pd.to_numeric(df[c],errors='coerce')
if df[required].isna().any().any() or (df['probability']<0).any() or (df['probability']>1).any() or (df['min_loss']<0).any() or (df['min_loss']>df['most_likely_loss']).any() or (df['most_likely_loss']>df['max_loss']).any():
    st.error('Risk values invalid: probability 0..1, 0 ≤ min ≤ likely ≤ max.'); st.stop()
if 'cost_unit' in df.columns and df['cost_unit'].nunique()>1:
    st.error('Different cost units: standardize before estimating reserves.'); st.stop()
unit=df['cost_unit'].iloc[0] if 'cost_unit' in df.columns else 'same cost units'
st.dataframe(df,use_container_width=True)
seed=int(st.number_input('Reproducible random seed',min_value=0,value=42))
n=int(st.select_slider('Monte Carlo simulations',options=[1000,5000,10000,20000],value=10000))
reserve=float(st.number_input('Available reserve (same units)',min_value=0.0,value=80.0,step=10.0))
rng=np.random.default_rng(seed)
losses=np.zeros(n)
for _,r in df.iterrows():
    event=rng.random(n)<float(r.probability)
    severity=rng.triangular(float(r.min_loss),float(r.most_likely_loss),float(r.max_loss),n)
    losses+=event*severity
stats=risk_quantiles(losses,reserve)
cols=st.columns(4)
for col,(name,val) in zip(cols,[('Expected loss',stats['expected_loss']),('P50 loss',stats['p50_loss']),('P80 loss',stats['p80_loss']),('P90 loss',stats['p90_loss'])]): col.metric(name,f'{val:,.2f}')
st.caption('Units: '+str(unit)+'. Independent Bernoulli occurrences and triangular severities are simplifying assumptions; correlations are not modeled.')
st.metric('P(loss > available reserve)',f"{100*stats['probability_of_reserve_exceedance']:.1f}%")
st.subheader('Economically informed reserve')
a,b=st.columns(2)
with a: carry=float(st.number_input('Cost of holding reserve per unit (c)',min_value=0.001,value=0.08,step=0.01,format='%.3f'))
with b: penalty=float(st.number_input('Penalty per unit uncovered loss (p)',min_value=0.001,value=0.40,step=0.05,format='%.3f'))
q=float(np.clip(1-carry/penalty,0,1))
optimal=float(np.quantile(losses,q))
st.metric('Optimal reserve under stated loss function',f'{optimal:,.2f} ({unit})')
st.caption(f'J(R)=c×R+p×E[max(0,L−R)]. Implied loss quantile: {q:.1%}. Financial coefficients are scenarios, not observed or empirically calibrated.')
st.subheader('Safe handling of ML EAC')
mode=st.radio('Accounting evidence',['Unknown or risks already included','Verified risk-exclusive baseline'],index=0)
eac=st.number_input('Optional baseline forecast (same units)',min_value=0.0,value=1100.0,step=10.0)
if mode.startswith('Verified'):
    evidence=st.text_area('Evidence / provenance explaining exactly which risks were excluded')
    if st.button('Calculate integrated totals'):
        try:
            outcome=integrated_cost_forecast(eac,losses,RiskAccounting('verified_risk_exclusive',evidence))
            st.json(outcome)
            st.warning('Research demonstration: entering evidence does not independently verify it. Do not use totals as empirically validated risk-adjusted ML predictions.')
        except ValueError as exc: st.error(str(exc))
else:
    st.info('Integrated total deliberately not calculated: adding risks may double count historical cost overruns.')
out=pd.DataFrame({'simulated_risk_loss':losses})
st.download_button('Export simulated risk losses',out.to_csv(index=False),'simulated_losses.csv','text/csv')
