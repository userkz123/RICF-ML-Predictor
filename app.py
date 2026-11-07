"""RICF-ML v5: auditable project history and cost forecasting prototype."""
from pathlib import Path
import sqlite3, json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from risk_simulation import simulate_risks

HERE=Path(__file__).resolve().parent
st.set_page_config(page_title='RICF-ML · Final Research Prototype',page_icon='📊',layout='wide')
@st.cache_resource
def get_model():return joblib.load(HERE/'trained_model.joblib')
@st.cache_data
def query(sql,params=()):
    with sqlite3.connect(HERE/'historical_projects.sqlite') as conn:
        return pd.read_sql_query(sql,conn,params=params)
@st.cache_data
def summary():return json.loads((HERE/'METRICS.json').read_text(encoding='utf-8'))
model_info=get_model(); clf=model_info['model'];features=model_info['features']; m=summary()
st.title('RICF-ML · Final Research Prototype')
st.caption('Research prototype | Ghent University DSLIB 3.4 historical checkpoints | ML vs EVM')
st.warning('**Scientific status:** 89 project endpoints use proxy completion; the 26-project development set was already reviewed. No ESG/risk/reserve variables were used to train ML. This is not an externally validated commercial predictor.')
t1,t2,t3,t4,t5=st.tabs(['Predict a project','Historical database','Company scenario','Model validation','ABC case study'])

def current_features(bac,ev,ac,pv,duration,activities,prior_cpi,prior_ev):
    cpi=ev/ac;progress=ev/bac
    prev_cpi = float(prior_cpi) if prior_cpi > 0 else np.nan
    prev_progress=float(prior_ev)/bac if prior_ev > 0 else np.nan
    fields={'progress':progress,'pv_ratio':pv/bac,'ac_ratio':ac/bac,'cpi':cpi,'spi':ev/pv,'planned_duration':duration,'n_activities':activities,'previous_cpi':prev_cpi,'cpi_trend':cpi-prev_cpi if np.isfinite(prev_cpi) else np.nan,'progress_increment':progress-prev_progress if np.isfinite(prev_progress) else np.nan,'cost_variance_ratio':(ev-ac)/bac,'schedule_variance_ratio':(ev-pv)/bac}
    return pd.DataFrame([fields])[features]
with t1:
    st.subheader('New project checkpoint')
    c1,c2,c3=st.columns(3)
    with c1:
        bac=st.number_input('BAC — approved baseline cost',min_value=0.01,value=1000000.0,step=10000.0)
        ev=st.number_input('EV — earned value',min_value=0.01,value=450000.0,step=10000.0)
    with c2:
        ac=st.number_input('AC — actual costs to date',min_value=0.01,value=500000.0,step=10000.0)
        pv=st.number_input('PV — planned value',min_value=0.01,value=480000.0,step=10000.0)
    with c3:
        duration=st.number_input('Planned duration (source unit)',min_value=0.0,value=180.0)
        activities=st.number_input('Number of activities',min_value=1,value=45)
    with st.expander('Optional preceding checkpoint — dynamic features'):
        prior_ev=st.number_input('Prior EV (0 if unavailable)',min_value=0.0,value=0.0)
        prior_cpi=st.number_input('Prior CPI (0 if unavailable)',min_value=0.0,value=0.0)
    if st.button('Predict final cost',type='primary'):
        if ev>bac*1.002 or (prior_ev>0 and prior_ev>ev):st.error('Check EV ≤ BAC and prior EV ≤ EV.')
        else:
            sample=current_features(bac,ev,ac,pv,duration,activities,prior_cpi,prior_ev)
            predicted_ratio=float(clf.predict(sample)[0]);ml_cost=predicted_ratio*bac; evm_cost=bac*ac/ev
            st.session_state['latest']={'BAC':bac,'EV':ev,'AC':ac,'PV':pv,'CPI':ev/ac,'SPI':ev/pv,'ML EAC':ml_cost,'EVM EAC':evm_cost,'Difference ML minus EVM':ml_cost-evm_cost}
    if 'latest' in st.session_state:
        r=st.session_state.latest
        c1,c2,c3=st.columns(3)
        c1.metric('ML estimate at completion',f"{r['ML EAC']:,.2f}")
        c2.metric('Traditional EVM estimate',f"{r['EVM EAC']:,.2f}")
        c3.metric('CPI',f"{r['CPI']:.3f}")
        st.caption('All results are in the same currency units as BAC. ML output is conditional on source population and unverified closure proxies. No ESG or risk effects learned.')
        st.download_button('Export this prediction CSV',pd.DataFrame([r]).to_csv(index=False),'ricf_prediction.csv',mime='text/csv')
with t2:
    st.subheader('Historical project library')
    p=query('SELECT * FROM projects ORDER BY project_id')
    kind=st.selectbox('Completion evidence',['All','Explicit Closure only','Proxy completion'])
    if kind=='Explicit Closure only': p=p[p.completion_evidence=='explicit_Closure_label']
    if kind=='Proxy completion': p=p[p.completion_evidence=='proxy_only_review_needed']
    st.metric('Displayed projects',len(p))
    st.dataframe(p,use_container_width=True,hide_index=True)
    st.download_button('Export catalog CSV',p.to_csv(index=False),'project_catalog.csv',mime='text/csv')
    if len(p):
        pid=st.selectbox('Inspect one project',p.project_id.tolist())
        history=query('SELECT checkpoint,progress,ev,pv,ac,cpi,spi,evm_eac FROM checkpoints WHERE project_id=? ORDER BY checkpoint',(pid,))
        st.dataframe(history,use_container_width=True,hide_index=True)
        st.line_chart(history.set_index('checkpoint')[['ev','pv','ac']])
        st.caption('Project table shows checkpoint records, not proof of an independently verified final cost.')
with t3:
    st.subheader('Company risk scenario calculator')
    st.info('Synthetic company data are only for illustrating the tool. Enter a baseline that EXCLUDES the future risk losses below. Do not automatically add these risks to ML EAC; that would potentially double-count losses.')
    base=st.number_input('Risk-exclusive predicted final cost',min_value=0.0,value=1100000.0,step=10000.0)
    reserve=st.number_input('Unspent risk contingency',min_value=0.0,value=85000.0,step=5000.0)
    risks=pd.DataFrame([{'risk':'Materials price shock','probability':0.3,'min_loss':10000.,'most_likely_loss':40000.,'max_loss':90000.},{'risk':'Environmental permit delay','probability':0.2,'min_loss':5000.,'most_likely_loss':25000.,'max_loss':75000.},{'risk':'Supplier disruption','probability':0.15,'min_loss':10000.,'most_likely_loss':30000.,'max_loss':120000.}])
    edited=st.data_editor(risks,num_rows='dynamic',use_container_width=True)
    trials=st.select_slider('Simulation runs',[1000,5000,10000,20000],value=10000)
    if st.button('Simulate cost risks'):
        try:
            s=simulate_risks(base,edited.to_dict('records'),reserve,n=trials,seed=2026)
            c1,c2,c3=st.columns(3);c1.metric('P50',f"{s['p50']:,.0f}");c2.metric('P80',f"{s['p80']:,.0f}");c3.metric('P90',f"{s['p90']:,.0f}")
            st.write(f"**Additional reserve needed at P80:** {s['reserve_gap_p80']:,.0f} (negative means surplus). **Probability of losses exceeding existing reserve:** {s['prob_exceeds_baseline_plus_reserve']:.1%}")
            vals=pd.DataFrame({'Simulated final cost':s['samples']})
            st.bar_chart(pd.Series(np.histogram(vals['Simulated final cost'],bins=30)[0]))
            st.download_button('Download scenarios',vals.to_csv(index=False),'synthetic_risk_scenarios.csv',mime='text/csv')
        except Exception as exc:st.error(str(exc))
with t4:
    st.subheader('Scientific validation and limitations')
    st.write(f"**Selected ML algorithm:** {m['selected_ml_by_train_only_cv']} (selected using 76 development-training projects)")
    cv_table=pd.DataFrame([{'method':k,'train_only_project_macro_MAPE_%':v} for k,v in m['train_group_cv_macro_mape_percent'].items()]).sort_values('train_only_project_macro_MAPE_%')
    st.dataframe(cv_table,hide_index=True)
    st.write('**Previously examined development split, NOT independent validation:**')
    st.dataframe(pd.DataFrame([{'method':k,'project_macro_MAPE_%':v} for k,v in m['previously_seen_dev_project_macro_mape_percent'].items()]),hide_index=True)
    st.write('**Completion evidence categories:**',m['completion_evidence_counts'])
    for warning in m['limitations']:st.write('• '+warning)
    st.caption('Project-balanced MAPE gives each project equal weight even when the number of checkpoints varies.')

with t5:
    st.subheader('ABC Infrastructure Ltd. — Project case study')
    st.warning('All company inputs and risks are invented for demonstration. This is not external validation of the ML predictor.')
    case_dir=HERE/'company_case'
    history=pd.read_csv(case_dir/'synthetic_project_checkpoints.csv')
    st.dataframe(history[['month','progress_pct','BAC_million_KZT','PV_million_KZT','EV_million_KZT','AC_million_KZT','ML_EAC_million_KZT','EVM_EAC_million_KZT','final_actual_cost_million_KZT']],hide_index=True,use_container_width=True)
    plot=history.set_index('month')[['ML_EAC_million_KZT','EVM_EAC_million_KZT','final_actual_cost_million_KZT']]
    st.line_chart(plot)
    st.caption('Comparison is on one synthetic trajectory, not scientific evidence of ML outperforming EVM.')
    st.write('**Synthetic risk register**')
    risks_df=pd.read_csv(case_dir/'synthetic_risk_register.csv')
    st.dataframe(risks_df,use_container_width=True,hide_index=True)
    ss=json.loads((case_dir/'synthetic_simulation_summary.json').read_text(encoding='utf-8'))
    c1,c2,c3=st.columns(3)
    c1.metric('Scenario P50, mln KZT',f"{ss['p50']:,.1f}")
    c2.metric('Scenario P80, mln KZT',f"{ss['p80']:,.1f}")
    c3.metric('Reserve gap P80, mln KZT',f"{ss['reserve_gap_p80']:,.1f}")
    st.caption('Simulation baseline of 1,100 million KZT is explicitly assumed risk-exclusive and separate from model EAC to avoid double-counting.')
    st.download_button('Download case checkpoints',history.to_csv(index=False),'abc_case_checkpoints.csv',mime='text/csv')
    st.download_button('Download risk register',risks_df.to_csv(index=False),'abc_case_risks.csv',mime='text/csv')
