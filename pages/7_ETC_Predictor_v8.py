from pathlib import Path
import pandas as pd, streamlit as st, joblib
from etc_v8_inference import predict_history
HERE=Path(__file__).resolve().parent.parent
st.title("ETC forecasting | RICF-ML v8")
st.warning("Research experiment, not validated for real deployment. Grouped OOF project MAPE: ETC-assisted 6.745% vs EVM 6.767%. Difference 0.021 percentage points; 95% CI includes zero. Most project closure labels not independently verified.")
st.write("**Model:** EAC = AC + predicted remaining cost (ETC). The experimental model learns a conservative correction to EVM-based ETC from historical patterns.")
@st.cache_resource
def load_model():return joblib.load(HERE/"etc_v8_model.joblib")
example=pd.DataFrame([dict(project_id="COMPANY_DEMO",checkpoint=i,bac=1_000_000,ev=ev,pv=pv,ac=ac,planned_duration=18,n_activities=50,sector="Construction") for i,(ev,pv,ac) in enumerate([(200_000,250_000,230_000),(400_000,430_000,460_000),(600_000,640_000,670_000)],1)])
st.download_button("Download illustrative input CSV",example.to_csv(index=False),file_name="etc_case_template.csv")
upload=st.file_uploader("Upload the history of a new project (CSV)",type=["csv"])
if upload is not None:
    try:
        d=predict_history(pd.read_csv(upload),load_model())
        cols=["project_id","checkpoint","progress","EVM_EAC","experimental_EAC","experimental_ETC"]
        st.dataframe(d[cols],hide_index=True,use_container_width=True)
        for pid,g in d.groupby("project_id"):
            st.subheader(str(pid));st.line_chart(g.set_index("checkpoint")[["EVM_EAC","experimental_EAC"]])
        st.download_button("Export forecasts",d[cols].to_csv(index=False),file_name="etc_predictions.csv")
    except Exception as exc:st.error(str(exc))
st.caption("All example project values are synthetic. Risk, reserve, and ESG histories are not training features. Do not interpret the predicted value as a certified final budget.")
