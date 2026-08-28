"""Fold-to-fold dispersion under the two splits -- persisted so the claim is checkable."""
import json,glob,numpy as np,pandas as pd
from pathlib import Path
rows=[]
for run,lbl in [('v2_runB','random'),('v2_blocked','blocked')]:
    for f in range(5):
        j=json.load(open(f'outputs/{run}/fold{f}/transfer/summary.json'))
        h=pd.read_csv(f'outputs/{run}/fold{f}/pretrain/training_history.csv')
        v=h['val/median_kge'].to_numpy()
        rows.append({'split':lbl,'fold':f,
                     'M0':j['step1_M0_target_hourly']['median_kge'],
                     'M1':j['step2_M1_target_hourly']['median_kge'],
                     'pretrain_stopped_epoch':len(v),
                     'pretrain_best_epoch':int(np.nanargmax(v))+1,
                     'val_curve_jitter':float(np.median(np.abs(np.diff(v))))})
frame=pd.DataFrame(rows)
out=Path('outputs/split_dispersion'); out.mkdir(parents=True,exist_ok=True)
frame.to_csv(out/'per_fold.csv',index=False)
summ={}
for lbl,g in frame.groupby('split'):
    summ[lbl]={m:{'mean':float(g[m].mean()),'sd':float(g[m].std(ddof=1)),
                  'min':float(g[m].min()),'max':float(g[m].max()),
                  'range':float(g[m].max()-g[m].min())} for m in ('M0','M1')}
    summ[lbl]['pretrain_best_epoch']=sorted(int(x) for x in g['pretrain_best_epoch'])
    summ[lbl]['pretrain_stopped_epoch']=sorted(int(x) for x in g['pretrain_stopped_epoch'])
    summ[lbl]['val_curve_jitter_median']=float(g['val_curve_jitter'].median())
from scipy.stats import levene
for m in ('M0','M1'):
    a=frame[frame.split=='random'][m]; b=frame[frame.split=='blocked'][m]
    summ.setdefault('ratios',{})[m]={'sd_ratio_blocked_over_random':float(b.std(ddof=1)/a.std(ddof=1)),
                                     'levene_p':float(levene(a,b).pvalue)}
(out/'summary.json').write_text(json.dumps(summ,indent=2))
print(frame.to_string(index=False,float_format=lambda v:f'{v: .4f}'))
print()
for m in ('M0','M1'):
    r=summ['ratios'][m]
    print(f"  {m}: sd ratio blocked/random = {r['sd_ratio_blocked_over_random']:.1f}x, Levene p = {r['levene_p']:.3f}")
print(f"\nwrote {out}")
