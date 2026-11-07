import streamlit as st
import pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
st.title("RICF-ML v12 | Decision-aware contingency")
st.warning("Research demonstration: entirely synthetic risk parameters. No external forecasting validation, and no additive ML + risk EAC is asserted.")
sc=pd.read_csv(ROOT/"v12_decision_policies.csv")
bs=pd.read_csv(ROOT/"v12_baseline_policy_comparison.csv")
au=pd.read_csv(ROOT/"v12_forecast_accounting_audit.csv")
choice=st.selectbox("Project",sorted(sc.project_id.unique()))
a=sc[sc.project_id==choice].copy()
b=bs[bs.project_id==choice].copy()
st.subheader("Decision preference sensitivity")
st.dataframe(a[["policy","holding_cost_rate","shortfall_penalty_rate","chosen_quantile","reserve_million_KZT","validation_probability_exceed","validation_decision_loss_million_KZT"]],hide_index=True,use_container_width=True)
st.subheader("Alternative policies (balanced cost setting)")
st.dataframe(b[["policy","reserve_million_KZT","validation_shortfall_million_KZT","validation_decision_loss_million_KZT"]],hide_index=True,use_container_width=True)
st.subheader("Historical ML forecast: risk accounting check")
st.info("The ML point forecast may include historical risk costs. The simulation is anchored to a separate declared risk-exclusive baseline; adding losses to ML would double count unknown components.")
st.dataframe(au[au.project_id==choice],hide_index=True,use_container_width=True)
st.download_button("Download decision results",data=a.to_csv(index=False).encode(),file_name="v12_decisions.csv")
