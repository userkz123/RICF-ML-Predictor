from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from etc_v8_inference import predict_history, sector_options
from risk_simulation import simulate_risks
ROOT=Path(__file__).resolve().parent.parent
st.title("RICF-ML | Probabilistic Forecasting")
st.caption("Project cost forecasts, historical error intervals and a separate risk reserve assessment")
st.info("Research status: interval levels are nominal and calibrated on previously studied DSLIB projects. They have not been externally validated. Most project completion endpoints rely on proxy criteria.")
@st.cache_resource
def load_artifact():
    return joblib.load(ROOT/'etc_v8_model.joblib')
@st.cache_data
def load_margins():
    return pd.read_csv(ROOT/'research_v10'/'exploratory_calibration_margins.csv')
artifact=load_artifact()
sectors=sector_options(artifact)

st.subheader('Project performance history')
mode=st.radio('Input method',['Manual entry','Upload CSV'],horizontal=True)
example=pd.DataFrame([dict(project_id='DEMO',checkpoint=i,bac=1_000_000,ev=e,pv=p,ac=a,planned_duration=18,n_activities=50,sector='Construction (Civil)') for i,(e,p,a) in enumerate([(200000,250000,230000),(400000,430000,460000),(600000,640000,670000)],1)])
st.download_button('Download example CSV',example.to_csv(index=False),file_name='sample_history.csv',mime='text/csv')
raw=None
if mode=='Manual entry':
    with st.expander('Project details',expanded=True):
        c1,c2,c3=st.columns(3)
        with c1:
            project_id=st.text_input('Project ID',value='PROJECT_01')
            bac=st.number_input('BAC (project currency)',min_value=0.01,value=1000000.,step=10000.)
        with c2:
            sector=st.selectbox('Project sector (training categories)',sectors,index=sectors.index('Construction (Civil)') if 'Construction (Civil)' in sectors else 0)
            duration=st.number_input('Planned duration (source units)',min_value=0.01,value=18.,step=1.)
        with c3:
            activities=int(st.number_input('Number of activities',min_value=1,value=50,step=1))
    history=st.data_editor(pd.DataFrame({'checkpoint':[1,2,3],'ev':[200000.,400000.,600000.],'pv':[250000.,430000.,640000.],'ac':[230000.,460000.,670000.]}),num_rows='dynamic',use_container_width=True,hide_index=True)
    if not history.empty:
        raw=history.copy()
        for col,val in [('project_id',project_id),('bac',bac),('planned_duration',duration),('n_activities',activities),('sector',sector)]:raw[col]=val
else:
    upload=st.file_uploader('Upload checkpoint history CSV',type=['csv'])
    if upload is not None:
        raw=pd.read_csv(upload)
        st.dataframe(raw.head(10),use_container_width=True,hide_index=True)
coverage=st.selectbox('Nominal interval coverage',[0.8,0.9],format_func=lambda x:f'{x:.0%}')
if raw is not None:
    try:
        hist=predict_history(raw,artifact)
        margins=load_margins()
        m=margins[np.isclose(margins.nominal_coverage,coverage)]
        if m.empty:raise ValueError('No error calibration available for the selected interval level')
        stages=np.array([0.25,0.5,0.75])
        nearest=np.abs(hist.progress.to_numpy()[:,None]-stages[None,:]).argmin(axis=1)
        hist['interval_stage']=[['25%','50%','75%'][i] for i in nearest]
        for meth,pred,prefix in [('EVM','EVM_EAC','EVM'),('ML-assisted ETC','experimental_EAC','ML')]:
            calib=m[m.method==meth]
            lookup=dict(zip(calib.stage,calib.half_width_ratio_BAC))
            if any(s not in lookup for s in hist.interval_stage):raise ValueError('Calibration table is incomplete')
            r=hist.interval_stage.map(lookup)*hist.bac
            # Final cost cannot be below costs already incurred.
            hist[prefix+'_lower']=np.maximum(hist.ac,hist[pred]-r)
            hist[prefix+'_upper']=np.maximum(hist.ac,hist[pred]+r)
        last=hist.iloc[-1]
        a,b,c=st.columns(3)
        a.metric('Latest ML-assisted ETC EAC',f"{last.experimental_EAC:,.2f}")
        b.metric('Latest EVM EAC',f"{last.EVM_EAC:,.2f}")
        c.metric('ML − EVM',f"{last.experimental_EAC-last.EVM_EAC:+,.2f}")
        st.caption('Values use the same currency units as uploaded BAC, EV, PV and AC. Error intervals are exploratory, not guaranteed coverage.')
        view_cols=['project_id','checkpoint','sector','progress','EVM_EAC','EVM_lower','EVM_upper','experimental_EAC','ML_lower','ML_upper']
        st.dataframe(hist[view_cols],use_container_width=True,hide_index=True)
        if hist.project_id.nunique()==1:
            st.line_chart(hist.set_index('checkpoint')[['EVM_EAC','experimental_EAC']])
        else:
            st.caption('Upload one project to view a single trajectory; complete table is available above.')
        st.download_button('Export forecasts and intervals',hist[view_cols].to_csv(index=False),'ricf_forecast_intervals.csv',mime='text/csv')
    except Exception as exc:
        st.error('Unable to calculate forecasts: '+str(exc))
else:
    st.caption('Enter a project history or upload a CSV to calculate forecasts.')
st.divider()
st.subheader('Risk and contingency assessment')
st.caption('This scenario engine is separate from ML training. It models only additional risks excluded from a separately documented baseline, so its losses must not automatically be added to ML EAC.')
risks=pd.DataFrame([
    {'risk':'Materials cost change','probability':0.30,'min_loss':10000.,'most_likely_loss':40000.,'max_loss':90000.},
    {'risk':'Permit delay','probability':0.20,'min_loss':5000.,'most_likely_loss':25000.,'max_loss':75000.},
    {'risk':'Supplier interruption','probability':0.15,'min_loss':10000.,'most_likely_loss':30000.,'max_loss':120000.},
])
edit=st.data_editor(risks,num_rows='dynamic',use_container_width=True,hide_index=True)
reserve=st.number_input('Remaining contingency reserve (same units)',min_value=0.,value=80000.,step=5000.)
n=st.select_slider('Monte Carlo trials',[1000,5000,10000,20000],value=10000)
seed=int(st.number_input('Random seed',min_value=0,value=2026))
if st.button('Calculate contingency',type='primary'):
    try:
        if edit.empty:raise ValueError('Add at least one risk')
        for col in ['probability','min_loss','most_likely_loss','max_loss']:
            edit[col]=pd.to_numeric(edit[col],errors='raise')
        # Risk simulation from zero: no baseline is added to ML prediction.
        s=simulate_risks(0.,edit.to_dict('records'),reserve,n=n,seed=seed)
        x,y,z=st.columns(3)
        x.metric('Expected incremental loss',f"{s['risk_loss_mean']:,.2f}")
        y.metric('P80 incremental loss',f"{s['p80']:,.2f}")
        z.metric('P90 incremental loss',f"{s['p90']:,.2f}")
        st.metric('Additional reserve needed at P80',f"{max(0,s['reserve_gap_p80']):,.2f}")
        st.metric('Probability of reserve shortfall',f"{s['prob_exceeds_baseline_plus_reserve']:.1%}")
        st.bar_chart(pd.Series(np.histogram(s['samples'],bins=30)[0]))
        st.download_button('Export risk scenarios',pd.DataFrame({'incremental_risk_loss':s['samples']}).to_csv(index=False),'risk_scenarios.csv',mime='text/csv')
    except Exception as exc: st.error('Unable to calculate contingency: '+str(exc))
