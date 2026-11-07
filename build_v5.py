from pathlib import Path
import sqlite3, json, shutil, zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.base import clone
from joblib import dump
root=Path('/mnt/data'); out=root/'RICF_ML_v5';out.mkdir(exist_ok=True)
raw=pd.read_csv(root/'RICF_ML_v2/training_data.csv').sort_values(['project_id','checkpoint']).reset_index(drop=True)
evidence=pd.read_csv(root/'RICF_ML_v3/completion_evidence.csv')
raw=raw.merge(evidence[['project_id','completion_evidence']],on='project_id',how='left',validate='many_to_one')
raw['ev']=raw.progress*raw.bac;raw['pv']=raw.pv_ratio*raw.bac;raw['ac']=raw.ac_ratio*raw.bac
raw['evm_eac']=raw.bac/raw.cpi
raw['previous_cpi']=raw.groupby('project_id').cpi.shift(1)
raw['cpi_trend']=raw.cpi-raw.previous_cpi
raw['previous_progress']=raw.groupby('project_id').progress.shift(1)
raw['progress_increment']=raw.progress-raw.previous_progress
raw['cost_variance_ratio']=raw.progress-raw.ac_ratio
raw['schedule_variance_ratio']=raw.progress-raw.pv_ratio
features=['progress','pv_ratio','ac_ratio','cpi','spi','planned_duration','n_activities','previous_cpi','cpi_trend','progress_increment','cost_variance_ratio','schedule_variance_ratio']
# Original holdout is already seen in earlier model development: diagnostic only.
held=pd.read_csv(root/'RICF_ML_v2/holdout_predictions_v2.csv')
held_ids=set(held.project_id.unique());train=raw[~raw.project_id.isin(held_ids)].copy();dev=raw[raw.project_id.isin(held_ids)].copy()
algorithms={
 'Ridge':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=30.)),
 'Random Forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=120,min_samples_leaf=5,max_features=.8,random_state=19,n_jobs=2)),
 'Histogram Gradient Boosting':make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=90,learning_rate=.04,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,random_state=19))
}
def pmape(yy,pp,ids):
 frame=pd.DataFrame({'id':np.asarray(ids),'ape':np.abs(np.asarray(pp)/np.asarray(yy)-1)*100})
 return float(frame.groupby('id').ape.mean().mean())
oof={name:np.zeros(len(train)) for name in algorithms}
cv=GroupKFold(n_splits=5)
foldrows=[]
for fold,(a,b) in enumerate(cv.split(train[features],train.final_ratio,train.project_id),1):
 for name,mod in algorithms.items():
  candidate=clone(mod);candidate.fit(train.iloc[a][features],train.iloc[a].final_ratio)
  p=candidate.predict(train.iloc[b][features]);oof[name][b]=p
  foldrows.append({'fold':fold,'model':name,'project_macro_mape':pmape(train.iloc[b].final_ratio,p,train.iloc[b].project_id)})
 for name,p in [('EVM',train.iloc[b].evm_eac/train.iloc[b].bac),('BAC',np.ones(len(b)))]:
  foldrows.append({'fold':fold,'model':name,'project_macro_mape':pmape(train.iloc[b].final_ratio,p,train.iloc[b].project_id)})
cv_scores={name:pmape(train.final_ratio,p,train.project_id) for name,p in oof.items()}
cv_scores['EVM']=pmape(train.final_ratio,train.evm_eac/train.bac,train.project_id)
cv_scores['BAC']=pmape(train.final_ratio,np.ones(len(train)),train.project_id)
best_ml=min(algorithms,key=lambda name:cv_scores[name]);chosen=clone(algorithms[best_ml]);chosen.fit(train[features],train.final_ratio)
dump({'model':chosen,'features':features,'algorithm':best_ml},out/'trained_model.joblib')
dev_pred=chosen.predict(dev[features]);dev_out=dev[['project_id','checkpoint','progress','bac','final_ac','closure_name','completion_evidence']].copy()
dev_out['predicted_ml_eac']=dev_pred*dev.bac;dev_out['predicted_evm_eac']=dev.evm_eac
for k in ['ml','evm']:
 dev_out[k+'_absolute_percentage_error']=np.abs(dev_out['predicted_'+k+'_eac']/dev.final_ac-1)*100
dev_out.to_csv(out/'previously_seen_development_predictions.csv',index=False)
projects=raw[['project_id','sector','bac','planned_duration','n_activities','final_ac','final_ratio','closure_name','completion_evidence']].drop_duplicates('project_id').copy()
projects['research_split']=projects.project_id.map(lambda x:'previously_seen_dev_26' if x in held_ids else 'train_76')
projects.to_csv(out/'project_catalog.csv',index=False)
raw.to_csv(out/'historical_checkpoints.csv',index=False)
evidence.to_csv(out/'completion_evidence.csv',index=False)
pd.DataFrame(foldrows).to_csv(out/'train_only_cv_folds.csv',index=False)
with sqlite3.connect(out/'historical_projects.sqlite') as con:
 projects.to_sql('projects',con,index=False,if_exists='replace')
 raw[['project_id','checkpoint','n_checkpoints','ev','pv','ac','progress','cpi','spi','evm_eac','previous_cpi','cpi_trend','progress_increment','cost_variance_ratio','schedule_variance_ratio']].to_sql('checkpoints',con,index=False,if_exists='replace')
 dev_out.to_sql('previously_seen_development_predictions',con,index=False,if_exists='replace')
 con.execute('CREATE INDEX checkpoint_project_index ON checkpoints(project_id,checkpoint)')
 con.execute('CREATE INDEX project_id_index ON projects(project_id)')
metrics={'train_projects':train.project_id.nunique(),'previously_seen_dev_projects':dev.project_id.nunique(),'train_checkpoints':len(train),'dev_checkpoints':len(dev),'all_projects':projects.project_id.nunique(),'selected_ml_by_train_only_cv':best_ml,'train_group_cv_macro_mape_percent':cv_scores,'previously_seen_dev_project_macro_mape_percent':{'ML':pmape(dev.final_ratio,dev_pred,dev.project_id),'EVM':pmape(dev.final_ratio,dev.evm_eac/dev.bac,dev.project_id),'BAC':pmape(dev.final_ratio,np.ones(len(dev)),dev.project_id)},'completion_evidence_counts':projects.completion_evidence.value_counts(dropna=False).to_dict(),'limitations':['89 project endpoints based on inferred completion proxy, not independent proof of closure','26-project development set inspected in earlier iterations: NOT a fresh holdout','No verified risk/ESG/contingency predictors in DSLIB ML training','ML model may not outperform EVM; show both','No production or external company validation']}
(out/'METRICS.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
# separate scenario calculation from ML prediction; no double counting implied
shutil.copyfile(root/'RICF_ML_v2/calculations.py',out/'risk_simulation.py')
(out/'requirements.txt').write_text('streamlit>=1.35,<2\npandas>=2.0,<3\nnumpy>=1.24,<3\nscikit-learn>=1.3,<2\njoblib>=1.3,<2\n',encoding='utf-8')
(out/'README.md').write_text('''# RICF-ML v5 — Historical Database and Project Cost Predictor\n\n## Purpose\nResearch prototype based on project checkpoint histories in DSLIB 3.4. Includes a persistent SQLite database, an ML predictor trained on prior project histories, traditional EVM benchmark, and a **separate** hypothetical company scenario module.\n\n## Launch\nRun `pip install -r requirements.txt` then `streamlit run app.py` from this folder. No coding needed for using the app.\n\n## Source and provenance\nOriginal source: `DSLIB3.4.zip` provided by user. `historical_checkpoints.csv` reflects previously extracted checkpoint values plus derived historical-only features. The repository does **not** contain the original 132 MB ZIP. `project_catalog.csv` records endpoint evidence and training/dev membership. `completion_evidence.csv` distinguishes explicit Closure from proxy-only endpoint identification.\n\n## Data and validation limits\n- 102 projects have candidate final-cost proxies. Only 13 explicitly show `Closure`; 89 require independent closure verification.\n- First 76 projects used for GroupKFold model selection and ML fitting; remaining 26 were used during earlier iterations, so they **cannot** count as an untouched test or external validation.\n- ML selected among Ridge, Random Forest and HistGradientBoosting by macro-project MAPE on out-of-fold predictions grouped by project. EVM and BAC are reported as references.\n- The Streamlit 'company simulator' is a synthetic input demonstration; it does not prove real-world accuracy.\n- ML uses EVM and history features but **does not** learn ESG / risk / contingency effects. Risk simulations must be calculated on explicitly risk-exclusive baselines to avoid double counting.\n- Do not claim that ML is better than EVM without genuinely new independent out-of-sample evidence.\n\n## Files\n- `historical_projects.sqlite`: relational base, tables `projects`, `checkpoints`, `previously_seen_development_predictions`.\n- `project_catalog.csv`, `historical_checkpoints.csv`: CSV exports.\n- `trained_model.joblib`: model and feature list, loaded locally from trusted package only.\n- `METRICS.json`, `train_only_cv_folds.csv`: validation details.\n- `app.py`, `risk_simulation.py`: user-facing app and scenario utilities.\n- `build_v5.py`: reproducibility script using earlier locally extracted DSLIB files; rebuild needs parent source files.\n''',encoding='utf-8')
shutil.copyfile(root/'build_v5.py',out/'build_v5.py')
print(json.dumps(metrics,indent=2));print('build complete')
