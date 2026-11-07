import streamlit as st
import pandas as pd, joblib
from pathlib import Path
from temporal_predictor import predict_history
HERE=Path(__file__).resolve().parent.parent
st.title('Experimental temporal ML correction (v7)')
st.warning('Exploratory only. Training CV: conventional EVM 6.845% vs best ML-assisted candidate 7.000% project-balanced MAPE. No statistically confirmed improvement. Completion proxy is unverified for 89 projects.')
@st.cache_resource
def get_model():return joblib.load(HERE/'temporal_residual_model.joblib')
payload=get_model()
st.write('Upload a CSV with sequential checkpoints for one or more projects; the model will use only current and prior records. Required fields: project_id, checkpoint, bac, ev, pv, ac, planned_duration, n_activities, sector.')
template=pd.DataFrame([{'project_id':'DEMO','checkpoint':1,'bac':1000000,'ev':250000,'pv':280000,'ac':280000,'planned_duration':18,'n_activities':45,'sector':'Construction'}, {'project_id':'DEMO','checkpoint':2,'bac':1000000,'ev':500000,'pv':530000,'ac':550000,'planned_duration':18,'n_activities':45,'sector':'Construction'}, {'project_id':'DEMO','checkpoint':3,'bac':1000000,'ev':700000,'pv':740000,'ac':770000,'planned_duration':18,'n_activities':45,'sector':'Construction'}])
st.download_button('Download hypothetical history template',template.to_csv(index=False),'synthetic_example.csv','text/csv')
f=st.file_uploader('Project checkpoint CSV',type=['csv'])
if f is not None:
    try:
        x=pd.read_csv(f)
        d=predict_history(x,payload)
        cols=['project_id','checkpoint','progress','cpi','EVM EAC','experimental ML-adjusted EAC']
        st.dataframe(d[cols],use_container_width=True,hide_index=True)
        for pid,g in d.groupby('project_id'):
            st.subheader(str(pid));st.line_chart(g.set_index('checkpoint')[['EVM EAC','experimental ML-adjusted EAC']])
        st.download_button('Export predictions CSV',d[cols].to_csv(index=False),'temporal_forecasts.csv','text/csv')
    except Exception as e:st.error(str(e))
st.caption('The example is synthetic and does not establish true forecast accuracy. Predictions have the same currency units as BAC. ESG/risk/contingency data are NOT used for training.')
