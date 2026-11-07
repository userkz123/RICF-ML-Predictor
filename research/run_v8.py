from pathlib import Path
import numpy as np,pandas as pd, json, joblib, shutil, zipfile
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
root=Path('/mnt/data'); out=root/'RICF_ML_v8';out.mkdir(exist_ok=True)
df=pd.read_csv(root/'RICF_ML_v5/historical_checkpoints.csv').sort_values(['project_id','checkpoint']).reset_index(drop=True)
seen=set(pd.read_csv(root/'RICF_ML_v2/holdout_predictions_v2.csv').project_id.unique())
df=df[~df.project_id.isin(seen)].copy().reset_index(drop=True)
df=df[(df.progress>0.01)&(df.progress<1)&(df.bac>0)&(df.ac>=0)&(df.final_ac>0)].copy()
# Ensure all features could exist at current checkpoint
for field in ['cpi','spi','ac_ratio','progress','pv_ratio']:
 df[f'{field}_lag1']=df.groupby('project_id')[field].shift(1)
df['cost_run_rate']=(df.ac_ratio-df.ac_ratio_lag1)/(df.progress-df.progress_lag1).clip(lower=.005)
df['cost_run_rate']=df.cost_run_rate.clip(0,10)
df['cpi_change']=df.cpi-df.cpi_lag1
df['remaining_progress']=1-df.progress
df['evm_etc_ratio']=(df.evm_eac-df.ac)/df.bac
df['etc_target_ratio']=(df.final_ac-df.ac)/df.bac
df['actual_eac_ratio']=df.final_ac/df.bac
# project type categories train only
sector=pd.get_dummies(df.sector.fillna('Other').astype(str),prefix='sector',dtype=int);df=pd.concat([df,sector],axis=1)
features=['progress','remaining_progress','ac_ratio','pv_ratio','cpi','spi','planned_duration','n_activities','cpi_lag1','spi_lag1','ac_ratio_lag1','progress_lag1','cost_run_rate','cpi_change','cost_variance_ratio','schedule_variance_ratio','evm_etc_ratio']+sector.columns.tolist()
models={
 'Ridge_ETC':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100)),
 'RandomForest_ETC':make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=140,max_depth=8,min_samples_leaf=5,max_features=.85,random_state=42,n_jobs=2)),
 'HistGB_ETC':make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=9,learning_rate=.04,l2_regularization=3,random_state=42)),
 'Ridge_ETC_residual':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=100)),
 'RF_ETC_residual':make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=140,max_depth=8,min_samples_leaf=5,max_features=.85,random_state=42,n_jobs=2))}
X=df[features];target=df.etc_target_ratio.to_numpy(); base=df.evm_etc_ratio.to_numpy(); ac=df.ac_ratio.to_numpy(); real=df.actual_eac_ratio.to_numpy();groups=df.project_id.to_numpy();
rows=[]
for fold,(a,b) in enumerate(GroupKFold(5).split(X,target,groups),1):
  preds={'EVM':base[b], 'BAC_remaining':1-ac[b]}
  for name,m in models.items():
   model=clone(m)
   if name.endswith('residual'):
    model.fit(X.iloc[a], target[a]-base[a]); residual=model.predict(X.iloc[b]);
    preds[name]=base[b]+residual
    preds[name+'_shrink20']=base[b]+.2*residual
    preds[name+'_stage20']=base[b]+.2*residual*(1-df.iloc[b].progress.to_numpy())
   else:
    model.fit(X.iloc[a],target[a]);preds[name]=model.predict(X.iloc[b])
  for method, pred in preds.items():
   eac=(ac[b]+np.maximum(0,np.asarray(pred)))
   for ii,ri in enumerate(b):
    rows.append(dict(fold=fold,project_id=groups[ri],checkpoint=int(df.iloc[ri].checkpoint),progress=float(df.iloc[ri].progress),method=method,predicted_eac_ratio=float(eac[ii]),actual_eac_ratio=float(real[ri]),APE_pct=float(abs(eac[ii]/real[ri]-1)*100),completion_evidence=df.iloc[ri].completion_evidence))
res=pd.DataFrame(rows);res.to_csv(out/'grouped_oof_predictions.csv',index=False)
pp=res.groupby(['method','project_id'],as_index=False).APE_pct.mean();summary=pp.groupby('method').APE_pct.agg(['mean','median','std']).reset_index().sort_values('mean');summary.columns=['method','project_macro_MAPE','median_project_MAPE','project_SD'];summary.to_csv(out/'method_comparison.csv',index=False)
stage=res.assign(stage=pd.cut(res.progress,[0,.25,.5,.75,1],labels=['0-25%','25-50%','50-75%','75-100%'],include_lowest=True)).groupby(['method','stage'],observed=True).APE_pct.mean().reset_index();stage.to_csv(out/'stage_comparison.csv',index=False)
best=summary[summary.method!='EVM'].iloc[0]['method'];w=pp.pivot(index='project_id',columns='method',values='APE_pct');delta=(w.EVM-w[best]).dropna().values
rng=np.random.default_rng(20261107);boots=rng.choice(delta,size=(5000,len(delta)),replace=True).mean(axis=1);CI=np.quantile(boots,[.025,.975])
# save ETC model fitted to development data, clearly EXPERIMENTAL
kind='Ridge_ETC' if best.startswith('Ridge_ETC') and 'residual' not in best else 'RandomForest_ETC'
model=clone(models[kind]);model.fit(X,target);joblib.dump(dict(model=model,features=features,method=kind,study='development-only experimental',sector_columns=sector.columns.tolist()),out/'etc_experimental.joblib')
report={'training_projects':int(df.project_id.nunique()),'training_checkpoints':len(df),'previous_holdout_projects_excluded':len(seen),'best_non_EVM':best,'EVM_minus_best_ML_macro_MAPE_pp':float(delta.mean()),'bootstrap_CI95_pp':CI.tolist(),'evidence_counts_projects':df.groupby('project_id').completion_evidence.first().value_counts().to_dict(),'winner':summary.iloc[0]['method'],'test_status':'exploratory grouped OOF on 76 projects; 26 prior holdout projects not used'}
(out/'experiment_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(out/'VALIDATION_REPORT.md').write_text('# RICF-ML v8: Estimate-to-Complete (ETC) modelling\n\nData: '+str(report['training_projects'])+' development projects; '+str(len(df))+' checkpoints. The 26 previously inspected holdout projects were excluded.\n\n## Five-fold GroupKFold, project-balanced MAPE (%)\n\n'+summary.to_markdown(index=False,floatfmt='.3f')+'\n\nBest ML versus EVM paired difference (EVM error minus ML error): '+f'{delta.mean():.3f} percentage points; 95% cluster bootstrap [{CI[0]:.3f}, {CI[1]:.3f}].'+'\n\n## Notes\n- Target: ETC = independently recorded final project cost minus costs already incurred. The final cost is used only as a supervised-learning label, never as an input feature.\n- EAC = AC + predicted ETC; ETC constrained nonnegative.\n- EVM baseline ETC = EVM EAC minus AC.\n- Group split by project prevents checkpoint mixing across train/test.\n- Closure status of most projects is indirectly inferred and needs confirmation.\n- No historical risk or ESG predictors are present. No independent confirmatory test is claimed.\n- Project-stage analyses are exploratory.\n',encoding='utf-8')
print(summary.to_string(index=False));print('best',best,'difference',delta.mean(),'CI',CI)
