from pathlib import Path
import pandas as pd,numpy as np
root=Path(__file__).resolve().parent
base=pd.read_csv(root/'source_v6_synthetic_risks.csv')
cases=[('ABC-ROAD-001',1000,.94,1.00),('ABC-ENERGY-002',1650,1.08,1.20),('ABC-WATER-003',780,.87,.80),('ABC-LOGISTICS-004',1250,1.0,1.10)]
rows=[]
for j,(pid,bac,p_mult,s_mult) in enumerate(cases):
    risks=base.copy();risks['probability']=(risks.probability*p_mult).clip(upper=.95)
    for col in ['min_loss','most_likely_loss','max_loss']:risks[col]=(risks[col]*s_mult*(bac/1000)).round(3)
    rng=np.random.default_rng(20261107+j); loss=np.zeros(20000)
    for r in risks.itertuples():loss+=(rng.random(20000)<r.probability)*rng.triangular(r.min_loss,r.most_likely_loss,r.max_loss,20000)
    emv=loss.mean();q80,q90=np.quantile(loss,[.8,.9])
    for name,amount in {'Fixed 5% BAC':.05*bac,'Expected loss (EMV)':emv,'Risk P80':q80,'Risk P90':q90}.items():
        rows.append({'project_id':pid,'policy':name,'allocated_reserve_million_KZT':amount,'probability_reserve_insufficient':(loss>amount).mean(),'expected_shortfall_million_KZT':np.maximum(0,loss-amount).mean(),'mean_unused_reserve_million_KZT':np.maximum(0,amount-loss).mean()})
pd.DataFrame(rows).to_csv(root/'reproduced_policy_check.csv',index=False)
print('Reproduced scenario policies:',len(rows))
