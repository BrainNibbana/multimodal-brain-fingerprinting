# -*- coding: utf-8 -*-
"""Read-only, local-only preprint exploratory statistics. Python 3.9+, pandas/numpy."""
import argparse, csv, re, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

SUBJECTS={100307,102816,106521,108323,109123,111514,112920,113922,116726,133019,146129,149741,156334,158136,162026,162935,164636,166438,169040,172029}

def metadata(name):
    s=re.search(r'mTE_(\d+)_\d+_(.+)\.h5$',name,re.I)
    if not s: return None
    sub=int(s.group(1)); desc=s.group(2)
    if sub not in SUBJECTS: return None
    if 'rfMRI_REST' in desc:
        m=re.search(r'rfMRI_(REST[12]_[LR]{2})',desc)
        if not m:return None
        return sub,'fMRI',m.group(1)
    m=re.search(r'MEG_([345])-Restin.*?_(.+)$',desc)
    if not m:return None
    return sub,'MEG_'+m.group(2),m.group(1)

def safe_corr(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():return np.nan
    a=a-a.mean();b=b-b.mean();den=np.sqrt(np.sum(a*a)*np.sum(b*b))
    return float(np.sum(a*b)/den) if den>1e-14 else np.nan

def evaluate(group,feature,rows,perms,rng):
    # Complete, explicit subject/run matching; no train/test parameter fitting.
    by={(s,r):x for s,r,x in rows}; runs=sorted(set(r for _,r,_ in rows)); subjects=sorted(set(s for s,_,_ in rows))
    output=[]
    for i,ra in enumerate(runs):
      for rb in runs[i+1:]:
        ss=[s for s in subjects if (s,ra) in by and (s,rb) in by]
        if len(ss)<3:continue
        A=np.array([by[s,ra] for s in ss]);B=np.array([by[s,rb] for s in ss]);n=len(ss)
        sim=np.array([[safe_corr(a,b) for b in B] for a in A])
        if not np.isfinite(sim).all():
            output.append(dict(group=group,feature=feature,run_a=ra,run_b=rb,n=n,status='DEGENERATE_OR_MISSING',accuracy=np.nan,idiff=np.nan,p_perm=np.nan));continue
        predicted=np.argmax(sim,axis=1)
        observed=int(np.sum(predicted==np.arange(n)))
        within=float(np.mean(np.diag(sim))); between=float((sim.sum()-np.trace(sim))/(n*(n-1)))
        # Fixed similarity matrix; permute B subject labels, not individual time points.
        hits=np.empty(perms,dtype=int)
        for p in range(perms):
            permutation=rng.permutation(n)
            hits[p]=int(np.sum(predicted==permutation))
        pvalue=(1+int(np.sum(hits>=observed)))/(perms+1)
        output.append(dict(group=group,feature=feature,run_a=ra,run_b=rb,n=n,status='OK',accuracy=observed/n,within=within,between=between,idiff=within-between,p_perm=pvalue))
    return output

def read_manifold(folder):
    result=[]
    if not folder.exists():return result
    for path in sorted(folder.glob('Manifold_*.csv')):
      try:
        df=pd.read_csv(path)
        cols={c.lower():c for c in df.columns}
        # Recognize common exported column schemas only; never assume row order as subject.
        sc=next((cols[k] for k in ['subject','subject_id','subjectid','subj'] if k in cols),None)
        xy=[c for c in df.columns if c.lower() in ['umap1','umap2','umap_1','umap_2','dim1','dim2','x','y','component_1','component_2']]
        if sc is None or len(xy)<2:
            result.append(dict(file=path.name,status='SKIPPED_SCHEMA',rows=len(df),within=np.nan,between=np.nan,ratio=np.nan));continue
        coords=df[xy[:2]].apply(pd.to_numeric,errors='coerce').to_numpy()
        subj=df[sc].astype(str).to_numpy()
        ds=np.sqrt(np.sum((coords[:,None,:]-coords[None,:,:])**2,axis=2))
        upper=np.triu(np.ones(ds.shape,dtype=bool),1)&np.isfinite(ds)
        same=(subj[:,None]==subj[None,:])&upper;different=(subj[:,None]!=subj[None,:])&upper
        w=float(np.mean(ds[same])) if same.any() else np.nan
        b=float(np.mean(ds[different])) if different.any() else np.nan
        result.append(dict(file=path.name,status='DESCRIPTIVE_2D_ONLY',rows=len(df),within=w,between=b,ratio=w/b if b>0 else np.nan))
      except Exception as e:result.append(dict(file=path.name,status='ERROR_'+str(e)[:80]))
    return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default=r'D:\Brain_Fingerprint_scripts');p.add_argument('--permutations',type=int,default=999);args=p.parse_args()
 root=Path(args.root);src=root/'mte_saved_export';out=root/'preprint_final_statistics';out.mkdir(parents=True,exist_ok=True)
 start=time.time();log=[]
 def note(s):
    msg=f'[{(time.time()-start):.1f}s] {s}';print(msg,flush=True);log.append(msg);(out/'progress.txt').write_text('\n'.join(log),encoding='utf-8')
 note('START: D-drive only; read-only inputs; exploratory analyses')
 a=pd.read_csv(src/'omnibus_by_target.csv');v=pd.read_csv(src/'selected_variable_te.csv')
 note(f'Loaded {len(a)} omnibus target rows, {len(v)} variable rows')
 a['target']=pd.to_numeric(a['target'],errors='coerce');v['target']=pd.to_numeric(v['target'],errors='coerce')
 v['source']=pd.to_numeric(v['source'],errors='coerce');v['lag_samples']=pd.to_numeric(v['lag_samples'],errors='coerce')
 groups={};quality=[]
 for fn,g in a.groupby('file',sort=True):
    m=metadata(str(fn));
    if m is None:continue
    subject,band,run=m;arr=np.full(166,np.nan)
    for row in g.itertuples(index=False):
      t=int(row.target)
      if 0<=t<166:arr[t]=float(row.omnibus_te) if pd.notna(row.omnibus_te) else np.nan
    # NaN means no valid omnibus estimate, NOT a measured zero. Explicit sensitivity: zero-filled vector.
    quality.append(dict(file=fn,group=band,subject=subject,run=run,finite_targets=int(np.isfinite(arr).sum()),missing_targets=int(np.isnan(arr).sum())))
    groups.setdefault((band,'omnibus_zero_filled'),[]).append((subject,run,np.nan_to_num(arr,nan=0.0)))
 note(f'Prepared {len(quality)} recordings')
 # Full lag-specific sparse features, fixed (source,lag,target) index.
 valid=v.dropna(subset=['source','target','lag_samples']).copy()
 valid=valid[(valid.source>=0)&(valid.source<166)&(valid.target>=0)&(valid.target<166)&(valid.lag_samples>=0)&(valid.lag_samples<=20)]
 # Use a common 0..20 lag axis, but avoid inventing missing values as estimated zeros: absent selection=0 indicator.
 for ix,(fn,g) in enumerate(valid.groupby('file',sort=True),1):
    m=metadata(str(fn));
    if m is None:continue
    subject,band,run=m
    vec=np.zeros(166*21*166,dtype=np.float32)
    for r in g.itertuples(index=False):
       if pd.notna(r.variable_te):vec[(int(r.source)*21+int(r.lag_samples))*166+int(r.target)]=float(r.variable_te)
    groups.setdefault((band,'variable_lag_sparse'),[]).append((subject,run,vec))
    if ix%100==0:note(f'Prepared variable vectors {ix}')
 del valid
 pd.DataFrame(quality).to_csv(out/'01_omnibus_quality.csv',index=False,encoding='utf-8-sig')
 rng=np.random.default_rng(20261011);results=[]
 for i,((band,feature),rows) in enumerate(sorted(groups.items()),1):
    results.extend(evaluate(band,feature,rows,args.permutations,rng));note(f'Fingerprint {i}/{len(groups)}: {band}, {feature}')
 pd.DataFrame(results).to_csv(out/'02_mte_fingerprint_permutations.csv',index=False,encoding='utf-8-sig')
 # Historical results are referenced, never recomputed from old incorrect matrices.
 old=root/'Optimization_Ultimate_Engine_V4_1_Final'/'Ultimate_Fingerprint_Results.csv'
 if old.exists():
    pd.read_csv(old).to_csv(out/'03_historical_results_reference.csv',index=False,encoding='utf-8-sig')
 manifold=root/'Optimization_Ultimate_Engine_V4_1_Final'/'Manifold_Plots_Multiplex'
 mr=read_manifold(manifold)
 pd.DataFrame(mr).to_csv(out/'04_existing_umap_2d_descriptive.csv',index=False,encoding='utf-8-sig')
 note(f'Existing manifold CSVs examined: {len(mr)}')
 summary=pd.DataFrame(results)
 if not summary.empty:
    summary.groupby(['group','feature','status'],dropna=False).agg(pairs=('status','size'),mean_accuracy=('accuracy','mean'),mean_idiff=('idiff','mean')).reset_index().to_csv(out/'05_mte_summary.csv',index=False,encoding='utf-8-sig')
 caveats='''EXPLORATORY PREPRINT STATISTICS\n\n1. These are representations of SAVED mTE estimates, not recovered source-level joint link TE.\n2. Omnibus missing estimates are zero-filled solely for a transparent sensitivity analysis; missing is NOT a measured zero. See 01_omnibus_quality.csv.\n3. Variable features are source x lag x target; absent selected variables use zero as a selection indicator, not as a measured TE.\n4. Fingerprint is directed A-run to B-run Pearson similarity, with exact subject/run matching and permutation of B-run subject identities. Permutation p values are descriptive and uncorrected for multiple testing.\n5. No preprocessing is fit across test data; zero/constant vectors are flagged as degenerate.\n6. UMAP CSV analysis uses only existing 2D coordinates and DOES NOT establish manifold stability or shared neural mechanisms. A valid high-dimensional geometry and seed-stability test requires original feature arrays and rerunning UMAP.\n7. Old MEG source-space localization and decimation concerns persist; fMRI mTE selection saturation persists.\n8. This script does not rerun old 455 experiments, re-estimate TE, compute source-level link TE, or validate high-dimensional group templates.\n9. No E-drive access; no GPU required.\n'''
 (out/'README_LIMITATIONS.txt').write_text(caveats,encoding='utf-8');note('DONE. Reports in '+str(out))
if __name__=='__main__':main()
