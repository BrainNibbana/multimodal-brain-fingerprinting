"""Read-only export of stored IDTxl mTE results; no TE re-estimation."""
import argparse, csv, re, traceback, time, json
from pathlib import Path
import numpy as np
import h5py
SUBJECTS=set('100307 102816 106521 108323 109123 111514 112920 113922 116726 133019 146129 149741 156334 158136 162026 162935 164636 166438 169040 172029'.split())
FIELDS_OM=['modality','file','subject','target','omnibus_te','omnibus_pval','omnibus_sign','selected_variable_count','status']
FIELDS_VAR=['modality','file','subject','target','source','lag_samples','variable_te','variable_pval']
FIELDS_SUM=['modality','file','subject','targets_found','targets_with_variables','variable_rows','repeated_source_target_assignments','omnibus_finite_count','error']
def scalar(v):
 if v is None: return ''
 a=np.asarray(v)
 if a.size!=1:return ''
 v=a.reshape(-1)[0]
 if isinstance(v,(np.bool_,bool)):return bool(v)
 if isinstance(v,(np.number,int,float)):return v.item() if hasattr(v,'item') else v
 return str(v)
def main():
 p=argparse.ArgumentParser();p.add_argument('--meg-results',default=r'E:\Results\meg_results');p.add_argument('--fmri-results',default=r'E:\Results\fmri_results');p.add_argument('--out',default=r'D:\Brain_Fingerprint_scripts\mte_saved_export');p.add_argument('--limit',type=int,default=0);p.add_argument('--progress-every',type=int,default=10);a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 specs=[('MEG',Path(a.meg_results)),('fMRI',Path(a.fmri_results))]
 paths=[]
 for mod,folder in specs:
  if folder.is_dir():
   for f in sorted(folder.glob('mTE_*.h5')):
    if f.name=='mTE_selection_summary.h5':continue
    m=re.match(r'mTE_(\d{6})_',f.name)
    if m and m.group(1) in SUBJECTS:paths.append((mod,f,m.group(1)))
 if a.limit:paths=paths[:a.limit]
 start=time.monotonic(); total=len(paths); print(f'Start: {total} files, read-only; output={out}',flush=True)
 with (out/'omnibus_by_target.csv').open('w',newline='',encoding='utf-8-sig') as fo,(out/'selected_variable_te.csv').open('w',newline='',encoding='utf-8-sig') as fv,(out/'file_summary.csv').open('w',newline='',encoding='utf-8-sig') as fs:
  wo=csv.DictWriter(fo,FIELDS_OM);wv=csv.DictWriter(fv,FIELDS_VAR);ws=csv.DictWriter(fs,FIELDS_SUM)
  wo.writeheader();wv.writeheader();ws.writeheader()
  for i,(mod,path,subj) in enumerate(paths,1):
   summary=dict(modality=mod,file=path.name,subject=subj,targets_found=0,targets_with_variables=0,variable_rows=0,repeated_source_target_assignments=0,omnibus_finite_count=0,error='')
   try:
    with h5py.File(path,'r') as f:
     if 'results/mte_results' not in f:raise KeyError('results/mte_results missing')
     group=f['results/mte_results']
     for name in group:
      if not re.fullmatch(r'target_\d+',name):continue
      target=int(name.split('_')[1]);g=group[name];summary['targets_found']+=1
      src=np.asarray(g['selected_sources_idx_lag']) if 'selected_sources_idx_lag' in g else np.empty((0,2),dtype=int)
      te=np.asarray(g['te_values']).reshape(-1) if 'te_values' in g else np.array([])
      pv=np.asarray(g['p_values']).reshape(-1) if 'p_values' in g else np.array([])
      status='OK' if src.ndim==2 and src.shape[1]==2 and len(src)==len(te)==len(pv) else 'ARRAY_LENGTH_MISMATCH'
      om=scalar(g.attrs.get('omnibus_te'));op=scalar(g.attrs.get('omnibus_pval'));sign=scalar(g.attrs.get('omnibus_sign'))
      if isinstance(om,(float,int)) and np.isfinite(om):summary['omnibus_finite_count']+=1
      wo.writerow(dict(modality=mod,file=path.name,subject=subj,target=target,omnibus_te=om,omnibus_pval=op,omnibus_sign=sign,selected_variable_count=len(src),status=status))
      if len(src):summary['targets_with_variables']+=1
      if status=='OK':
       counts={}
       for (s,lag),t,v in zip(src,te,pv):
        s=int(s);counts[s]=counts.get(s,0)+1
        wv.writerow(dict(modality=mod,file=path.name,subject=subj,target=target,source=s,lag_samples=int(lag),variable_te=float(t),variable_pval=float(v)))
        summary['variable_rows']+=1
       summary['repeated_source_target_assignments']+=sum(max(0,n-1) for n in counts.values())
   except Exception as e:summary['error']=f'{type(e).__name__}: {e}'
   ws.writerow(summary)
   if i%max(1,a.progress_every)==0 or i==total:
    elapsed=time.monotonic()-start; speed=i/elapsed if elapsed else 0; eta=(total-i)/speed if speed else 0
    message=f'[{i}/{total} {i/total*100:.1f}%] elapsed={elapsed/60:.1f} min | ETA={eta/60:.1f} min | last={path.name} | error={bool(summary["error"])}'
    print(message,flush=True)
    with (out/'progress.txt').open('w',encoding='utf-8') as progress: progress.write(message+'\n')
    fo.flush();fv.flush();fs.flush()
 print(f'Finished in {(time.monotonic()-start)/60:.1f} min. Saved read-only exports to: {out}',flush=True)
if __name__=='__main__':main()
