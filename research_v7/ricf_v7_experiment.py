from pathlib import Path
import json, copy
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from scipy.stats import wilcoxon
P=Path('/mnt/data/RICF_ML_v7');P.mkdir(exist_ok=True)
df=pd.read_csv('/mnt/data/RICF_ML_v5/historical_checkpoints.csv').sort_values(['project_id','checkpoint']).reset_index(drop=True)
seen=set(pd.read_csv('/mnt/data/RICF_ML_v2/holdout_predictions_v2.csv').project_id.unique())
train=df[~df.project_id.isin(seen)].copy().reset_index(drop=True)
# Features are calculated only with information available at checkpoint (never n_checkpoints, final costs, completion status)
for d in [train]:
    d['remaining']=1-d.progress
    d['evm_ratio']=d.ac_ratio/d.progress.clip(lower=.001)
    d['log_duration']=np.log1p(d.planned_duration)
    d['log_activities']=np.log1p(d.n_activities)
    d['prior_ac_ratio']=d.groupby('project_id').ac_ratio.shift(1)
    d['ac_delta']=d.ac_ratio-d.prior_ac_ratio
    d['cpi_delta']=d.cpi-d.groupby('project_id').cpi.shift(1)
    d['cpi_rolling_mean']=d.groupby('project_id').cpi.transform(lambda s:s.shift(0).rolling(3,min_periods=1).mean())
    d['cpi_rolling_std']=d.groupby('project_id').cpi.transform(lambda s:s.shift(0).rolling(3,min_periods=2).std())
    d['progress_delta']=d.progress-d.groupby('project_id').progress.shift(1)
    d['ac_per_progress_delta']=d.ac_delta/d.progress_delta.clip(lower=.001)
    d['cost_slope_ratio']=d.ac_per_progress_delta.clip(0,5)
    d['evm_minus_bac']=d.evm_ratio-1
    d['cost_acceleration']=d.cost_slope_ratio-d.groupby('project_id').cost_slope_ratio.shift(1)
    d['sector_cat']=d.sector.fillna('Other').astype(str).str.extract(r'^(Construction|IT|Engineering|Mobility)',expand=False).fillna('Other')
cats=sorted(train.sector_cat.unique())
for c in cats:train[f'sector_{c}']=(train.sector_cat==c).astype(float)
features=['progress','pv_ratio','ac_ratio','cpi','spi','planned_duration','n_activities','remaining','evm_ratio','log_duration','log_activities','previous_cpi','cpi_trend','progress_increment','cost_variance_ratio','schedule_variance_ratio','ac_delta','cpi_delta','cpi_rolling_mean','cpi_rolling_std','progress_delta','cost_slope_ratio','cost_acceleration']+[f'sector_{c}' for c in cats]
# Exclude final outturn values, completion annotations, n_checkpoints (future information), and future rolling shifts
models={
'Ridge_full':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=80)),
'RF_full':make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=120,max_features=.8,min_samples_leaf=6,max_depth=9,random_state=42,n_jobs=2)),
'RF_residual':make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=120,max_features=.8,min_samples_leaf=6,max_depth=9,random_state=42,n_jobs=2)),
'Ridge_residual':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=80)),
}
Y=train.final_ratio.to_numpy();G=train.project_id.to_numpy();E=train.evm_ratio.to_numpy();X=train[features]
folds=GroupKFold(5)
records=[]
for fold,(a,b) in enumerate(folds.split(X,Y,G),1):
    p={'BAC':np.ones(len(b)),'EVM':E[b]}
    for name,model in models.items():
        m=clone(model)
        if 'residual' in name:
            m.fit(X.iloc[a],Y[a]-E[a]);raw=E[b]+m.predict(X.iloc[b])
        else:
            m.fit(X.iloc[a],Y[a]);raw=m.predict(X.iloc[b])
        p[name]=raw
        if 'residual' in name:
            p[name+'_shrink25']=E[b]+.25*(raw-E[b])
            p[name+'_shrink50']=E[b]+.5*(raw-E[b])
            p[name+'_shrink25_early']=E[b]+.25*(raw-E[b])*(1-train.iloc[b].progress.to_numpy())
    for j,row_i in enumerate(b):
        for name,v in p.items():
            records.append({'fold':fold,'project_id':G[row_i],'checkpoint':int(train.iloc[row_i].checkpoint),'stage':float(train.iloc[row_i].progress),'method':name,'predicted_ratio':float(v[j]),'true_ratio':float(Y[row_i]),'absolute_percentage_error':float(abs(v[j]/Y[row_i]-1)*100),'closure_evidence':train.iloc[row_i].completion_evidence})
res=pd.DataFrame(records);res.to_csv(P/'training_only_oof_predictions.csv',index=False)
perproj=res.groupby(['method','project_id'],as_index=False).absolute_percentage_error.mean()
rank=perproj.groupby('method').absolute_percentage_error.agg(['mean','median','std']).sort_values('mean').reset_index();rank.columns=['method','project_macro_MAPE_pct','median_project_MAPE_pct','sd_project_MAPE_pct'];rank.to_csv(P/'method_comparison.csv',index=False)
stage=res.assign(stage_bucket=pd.cut(res.stage,[0,.25,.5,.75,1.01],labels=['0-25%','25-50%','50-75%','75-100%'],include_lowest=True)).groupby(['method','stage_bucket'],observed=True).absolute_percentage_error.mean().reset_index();stage.to_csv(P/'stage_comparison.csv',index=False)
# Paired project-bootstrap difference, positive means new model improves relative to EVM
best_ml=rank[rank.method!='EVM'].iloc[0]['method']
wide=perproj.pivot(index='project_id',columns='method',values='absolute_percentage_error');delta=(wide['EVM']-wide[best_ml]).dropna().to_numpy()
rng=np.random.default_rng(2026);rep=np.mean(rng.choice(delta,size=(5000,len(delta)),replace=True),axis=1)
ci=np.quantile(rep,[.025,.975]);summary={'training_projects':int(train.project_id.nunique()),'training_checkpoints':len(train),'models_ranked_by_train_oof':rank.to_dict(orient='records'),'best_non_evm':best_ml,'difference_EVM_minus_best_model_project_MAPE_pp':float(delta.mean()),'bootstrap_difference_95ci_pp':ci.tolist(),'completion_evidence_counts_train':train.groupby('project_id').completion_evidence.first().value_counts().to_dict(),'known_previous_holdout_projects_excluded':len(seen),'status':'training-only exploratory grouped out-of-fold; not independent final test'}
(P/'experiment_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
# Train deployable residual model, on 76 training projects only, regardless of whether better, label experimental
model=clone(models['RF_residual']);model.fit(X,Y-E)
import joblib
joblib.dump({'model':model,'features':features,'sectors':cats,'method':'RF_residual','description':'Experimental residual correction (EVM + learned residual), trained on development-training data only'},P/'experimental_residual_model.joblib')
# Base v6 historical app remains functional; train-only report avoids claiming improvement
report=['# RICF-ML v7 — Temporal feature & EVM residual-correction experiment','',f'Training: **{summary["training_projects"]} projects**, **{len(train)} checkpoint observations**. All 26 previously studied development-holdout projects excluded from model selection.','', '## Model selection — GroupKFold out-of-fold error (project-balanced)', '',rank.to_markdown(index=False,floatfmt='.3f'),'',f'Best non-EVM candidate: **{best_ml}**. Paired project error advantage over EVM (positive = model better): **{delta.mean():.3f} pp**, 95% project bootstrap **[{ci[0]:.3f}, {ci[1]:.3f}] pp**.','','## Caveats','- 89 of 102 project end-points use inferred closure, not verified independent financial close-out.','- More than one checkpoint per project: group splitting prevents a project appearing in both training and validation.','- No future observations, final cost, final number of checkpoints, or closure evidence are features.','- Current and lagged cost performance only; risk and ESG historical data are not represented.','- Model comparisons are exploratory and do not constitute an untouched confirmatory test.','- An experimental residual model artifact is supplied regardless of rank; it is not automatically recommended for deployment.','- Stage analyses and comparisons across many variants are exploratory; multiple comparisons can overstate improvements.','','## Technical approach','Residual model predicts final_cost / BAC minus current EVM EAC / BAC. At inference the predicted residual is added to the conventional EVM estimate. Shrunk and stage-gated variants are assessed solely through training-fold predictions.']
(P/'VALIDATION_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
print(rank.to_string(index=False)); print('Best non-EVM:',best_ml,'difference',delta.mean(),'CI',ci)
