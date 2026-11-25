"""Read-only final exploratory embedding audit. D: only. Python >=3.9, numpy, pandas; sklearn and umap optional."""
from pathlib import Path
import argparse, re, time, json, traceback
import numpy as np
import pandas as pd

SUBJECTS={100307,102816,106521,108323,109123,111514,112920,113922,116726,133019,146129,149741,156334,158136,162026,162935,164636,166438,169040,172029}

def group_name(fn):
 s=fn.replace('mTE_','')
 if '_rfMRI_' in s:return 'fMRI'
 m=re.search(r'_rmegpreproc_(.+)\.h5$',s)
 return 'MEG_'+m.group(1) if m else None

def run_name(fn):
 m=re.search(r'_MEG_(\d+)-',fn)
 if m:return m.group(1)
 m=re.search(r'_rfMRI_(REST\d+_[LR]{2})',fn)
 return m.group(1) if m else None

def within_between(x,sub):
 d=np.sqrt(np.maximum(((x[:,None,:]-x[None,:,:])**2).sum(axis=2),0))
 iu=np.triu_indices(len(sub),1)
 same=np.array(sub)[iu[0]]==np.array(sub)[iu[1]]
 if not same.any() or same.all():return np.nan,np.nan,np.nan
 a=float(np.mean(d[iu][same]));b=float(np.mean(d[iu][~same]));return a,b,a/b if b>0 else np.nan

def write(df,p):pd.DataFrame(df).to_csv(p,index=False,encoding='utf-8-sig')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default=r'D:\Brain_Fingerprint_scripts');ap.add_argument('--out',default=None);ap.add_argument('--seeds',type=int,default=3);args=ap.parse_args()
 root=Path(args.root);out=Path(args.out) if args.out else root/'preprint_umap_final_audit';out.mkdir(parents=True,exist_ok=True)
 t=time.time();logs=[]
 def log(s):
  print(f'[{time.time()-t:.1f}s] {s}',flush=True);logs.append(s);(out/'progress.txt').write_text('\n'.join(logs),encoding='utf-8')
 log('START; D drive only; read-only sources')
 src=root/'Optimization_Ultimate_Engine_V4_1_Final'/'Manifold_Plots_Multiplex'
 files=sorted(src.rglob('Manifold_*.csv')) if src.exists() else []
 log(f'Existing manifold CSV: {len(files)}')
 rows=[];details=[]
 for i,p in enumerate(files,1):
  rec=dict(file=p.name,status='OK',n=0,subject_count=0,within=np.nan,between=np.nan,ratio=np.nan,run_mapping='UNVERIFIED: CSV has no run column')
  try:
   df=pd.read_csv(p)
   cols={c.lower():c for c in df.columns}
   sc=next((cols[k] for k in ['subject_id','subject','subjectid'] if k in cols),None)
   c1=next((cols[k] for k in ['umap_dimension_1','umap_1','umap1','x'] if k in cols),None)
   c2=next((cols[k] for k in ['umap_dimension_2','umap_2','umap2','y'] if k in cols),None)
   if not all([sc,c1,c2]):rec['status']='SKIPPED_SCHEMA';rows.append(rec);continue
   d=df[[sc,c1,c2]].copy();d[sc]=pd.to_numeric(d[sc],errors='coerce');d[c1]=pd.to_numeric(d[c1],errors='coerce');d[c2]=pd.to_numeric(d[c2],errors='coerce');d=d.dropna()
   rec['n']=len(d);rec['subject_count']=d[sc].nunique()
   if not set(d[sc].astype(int)).issubset(SUBJECTS):rec['status']='UNEXPECTED_SUBJECTS'
   if d.groupby(sc).size().nunique()!=1:rec['status']='UNBALANCED_REPEATS'
   w,b,r=within_between(d[[c1,c2]].to_numpy(float),d[sc].to_numpy());rec.update(within=w,between=b,ratio=r)
   details.append(dict(file=p.name,dimension_1_range=float(d[c1].max()-d[c1].min()),dimension_2_range=float(d[c2].max()-d[c2].min()),duplicate_points=int(d[[c1,c2]].duplicated().sum())))
  except Exception as e:rec['status']='ERROR: '+str(e)[:200]
  rows.append(rec)
  if i%20==0:log(f'Existing UMAP {i}/{len(files)}')
 write(rows,out/'01_existing_umap_descriptive.csv');write(details,out/'02_umap_coordinate_quality.csv')
 # High-dimensional mTE Omnibus: each file is one recording, missingness kept as mask; zero-fill sensitivity analysis only.
 te=root/'mte_saved_export'/'omnibus_by_target.csv'
 hi=[];stab=[];pca_rows=[]
 if te.exists():
  log('Loading high-dimensional mTE Omnibus vectors')
  df=pd.read_csv(te,usecols=['file','subject','target','omnibus_te'])
  df['group']=df['file'].map(group_name);df['run']=df['file'].map(run_name)
  df=df.dropna(subset=['group','run']);df['target']=pd.to_numeric(df['target'],errors='coerce')
  for group,g in df.groupby('group'):
   piv=g.groupby(['file','subject','run','target'],observed=True)['omnibus_te'].first().unstack('target')
   piv=piv.reindex(columns=range(166));valid=piv.notna().to_numpy();X=piv.fillna(0).to_numpy(float)
   subjects=np.array([i[1] for i in piv.index]);runs=np.array([i[2] for i in piv.index]);n=len(X)
   if n<8:continue
   # non-informative features removed before embedding; no subject-level imputation
   variable=np.std(X,axis=0)>1e-12;X=X[:,variable]
   if X.shape[1]<2:hi.append(dict(group=group,status='DEGENERATE',n=n));continue
   X=(X-X.mean(axis=0))/(X.std(axis=0)+1e-12)
   w,b,r=within_between(X,subjects)
   hi.append(dict(group=group,status='OK',n=n,n_features=X.shape[1],missing_fraction=1-float(valid.mean()),within=w,between=b,ratio=r,run_labels=','.join(sorted(set(runs)))))
   # PCA via NumPy SVD: no external sklearn requirement
   try:
    U,S,V=np.linalg.svd(X,full_matrices=False);Y=U[:,:2]*S[:2];pw,pb,pr=within_between(Y,subjects)
    pca_rows.append(dict(group=group,method='PCA',seed='deterministic',n=n,within=pw,between=pb,ratio=pr,variance_2d=float((S[:2]**2).sum()/(S**2).sum())))
   except Exception as e:log(f'PCA failed {group}: {e}');continue
   try:
    import umap
    embeddings=[]
    for seed in range(args.seeds):
     emb=umap.UMAP(n_neighbors=min(15,n-1),min_dist=0.1,n_components=2,metric='cosine',random_state=seed).fit_transform(X)
     embeddings.append(emb)
     uw,ub,ur=within_between(emb,subjects)
     pca_rows.append(dict(group=group,method='UMAP',seed=seed,n=n,within=uw,between=ub,ratio=ur,variance_2d=np.nan))
    # Pairwise-distance correlation invariant to rotation / translation / scale
    vecs=[]
    for emb in embeddings:
     d=np.sqrt(((emb[:,None,:]-emb[None,:,:])**2).sum(axis=2));v=d[np.triu_indices(n,1)];vecs.append(v)
    corrs=[np.corrcoef(vecs[i],vecs[j])[0,1] for i in range(len(vecs)) for j in range(i+1,len(vecs))]
    stab.append(dict(group=group,seeds=len(embeddings),distance_correlation_mean=float(np.nanmean(corrs)) if corrs else np.nan,status='OK'))
   except ImportError:stab.append(dict(group=group,status='UMAP_NOT_INSTALLED',seeds=0,distance_correlation_mean=np.nan))
   except Exception as e:stab.append(dict(group=group,status='UMAP_ERROR: '+str(e)[:120],seeds=0,distance_correlation_mean=np.nan))
   log(f'High-dimensional {group}: {n} recordings')
 else:log('Omnibus CSV missing; high-dimensional tests skipped')
 write(hi,out/'03_mte_highdim_geometry.csv');write(pca_rows,out/'04_pca_umap_mte_comparison.csv');write(stab,out/'05_umap_seed_stability.csv')
 (out/'README_INTERPRETATION.txt').write_text('Existing 91 CSV: subject-level 2D descriptive geometry only; no scan/run column, therefore scan identity is UNVERIFIED. Do not infer run-specific reliability from these coordinates.\nHigh-dimensional analysis uses SAVED mTE Omnibus features ONLY, not historical Pearson/GCA original edges. NaN filled with zero as sensitivity analysis, NOT a measured zero.\nPCA 2D variance and UMAP seed stability do not establish a neural manifold or brain blueprint. Cross-condition UMAP absolute distances cannot be compared.\nNo source localization, TE estimator validity, multiple comparison correction or group template bootstrapping performed.\n',encoding='utf-8')
 log('DONE; inspect README and CSVs')
if __name__=='__main__':main()
