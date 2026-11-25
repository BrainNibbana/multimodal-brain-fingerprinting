import csv, json, re, zipfile, os, time
from pathlib import Path
import numpy as np

SUBJECTS='100307 102816 106521 108323 109123 111514 112920 113922 116726 133019 146129 149741 156334 158136 162026 162935 164636 166438 169040 172029'.split()
ATLAS=Path(r'E:\AAL3_Atlas'); ZIPROOT=Path(r'E:\HCP_MEG_preprocessing'); OUT=Path(r'D:\Brain_Fingerprint_scripts\MEG_final_spatial_decision')

def save_csv(path, rows, cols):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)

def parse(txt,key):
    m=re.search(r'transform\.'+re.escape(key)+r'\s*=\s*\[(.*?)\];',txt,re.S)
    if not m:return None
    a=np.array([[float(v) for v in ln.split()] for ln in m.group(1).strip().splitlines()]);return a if a.shape==(4,4) else None

def main():
    t=time.time();OUT.mkdir(parents=True,exist_ok=True)
    import nibabel as nib
    img=nib.load(str(ATLAS/'AAL3v1.nii')); a=np.asanyarray(img.dataobj); ids=sorted(set(np.unique(a).astype(int))-{0})
    txtrows=[]
    for line in (ATLAS/'AAL3v1.nii.txt').read_text(encoding='utf-8-sig').splitlines():
        p=line.split()
        if p and p[0].isdigit():txtrows.append((int(p[0]),p[1],int(p[2]) if len(p)>2 and p[2].isdigit() else None))
    labels={i:n for i,n,v in txtrows}; valid=[i for i in ids if i in labels]
    mapping=[]
    for pos,roi in enumerate(valid):
        wrong=txtrows[pos][1] if pos<len(txtrows) else ''
        mapping.append(dict(matrix_zero_based_index=pos,atlas_id=roi,correct_roi=labels[roi],naive_170_row_label=wrong,label_match=(wrong==labels[roi])))
    save_csv(OUT/'roi_index_alignment.csv',mapping,list(mapping[0]))
    zipfiles=list(ZIPROOT.rglob('*.zip')) if ZIPROOT.exists() else []
    checks=[]
    for j,sub in enumerate(SUBJECTS,1):
        matches=[p for p in zipfiles if sub in p.name and 'anatomy' in p.name.lower()]
        if not matches:matches=[p for p in zipfiles if sub in p.name]
        row={'subject':sub,'zip':'','status':'NOT_FOUND','inverse_error':'','rotation_orthogonality_error':'','translation_mm':'','warning':'SPM frame is not automatically fsaverage MRI frame'}
        for p in matches:
            try:
                with zipfile.ZipFile(p) as z:
                    names=[n for n in z.namelist() if n.lower().endswith('transform.txt')]
                    if not names:continue
                    raw=z.read(names[0]).decode('utf-8',errors='replace')
                    b=parse(raw,'bti2spm');c=parse(raw,'spm2bti')
                    if b is None:continue
                    row['zip']=str(p);row['status']='PARSED'
                    row['translation_mm']=float(np.linalg.norm(b[:3,3]));row['rotation_orthogonality_error']=float(np.linalg.norm(b[:3,:3].T@b[:3,:3]-np.eye(3)))
                    if c is not None:row['inverse_error']=float(np.max(np.abs(b@c-np.eye(4))))
                    break
            except Exception as e:row['status']='ERROR';row['warning']=str(e)[:180]
        checks.append(row);print(f'[{j}/20] {sub}: {row["status"]} | elapsed {time.time()-t:.1f}s',flush=True)
    save_csv(OUT/'subject_transform_qc.csv',checks,list(checks[0]))

    code=[]
    for name in ['HCP_fMRI_MEG_Preparing.ipynb','Timeseries_Extracting.ipynb']:
        p=Path(r'D:\Brain_Fingerprint_scripts')/name
        if not p.exists():continue
        d=json.loads(p.read_text(encoding='utf-8'));s='\n'.join(''.join(c.get('source',[])) for c in d['cells'])
        for term in ['find_parcellation_cut_coords','setup_volume_source_space','parse_subject_transform','mne_src_coords - target_coord','df_labels[\'AAL_Label\']']:
            code.append(dict(file=name,term=term,present=(term in s)))
    if code:save_csv(OUT/'code_checks.csv',code,list(code[0]))
    # Check the actual fsaverage MRI affine if the local template exists, without downloading or loading source models.
    candidates=[]
    for root in [Path.home()/'mne_data',Path(r'D:\python\mne_data'),Path(r'D:\Brain_Fingerprint_scripts')]:
        if root.exists():candidates.extend(root.glob('**/fsaverage/mri/brain.mgz'))
    fsinfo=[]
    for p in candidates[:3]:
        try:
            m=nib.load(str(p));fsinfo.append(dict(path=str(p),affine=np.array(m.affine).tolist(),shape=list(m.shape)))
        except Exception as e:fsinfo.append(dict(path=str(p),error=str(e)))
    (OUT/'template_mri_evidence.json').write_text(json.dumps(fsinfo,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else (x.tolist() if isinstance(x,np.ndarray) else str(x))),encoding='utf-8')
    bad=sum(not r['label_match'] for r in mapping)
    note=[f'20 subjects parsed: {sum(x["status"]=="PARSED" for x in checks)}/20',f'AAL3 valid ROI count: {len(valid)}; text rows: {len(txtrows)}',f'Naive position-based labels disagree with valid atlas IDs: {bad}/{len(mapping)}',f'First disagreements: {[(r["matrix_zero_based_index"],r["correct_roi"],r["naive_170_row_label"]) for r in mapping if not r["label_match"]][:6]}', 'IMPORTANT: this proves a positional label mismatch risk for the 170-row labeling method, not which exact labels are in each saved file.', 'CRITICAL: HCP bti2spm is SPM-space, not automatically MNE fsaverage MRI-space. Transform parsing does NOT establish valid coregistration.', 'CRITICAL: comparing src[0][rr] directly to AAL3 NIfTI world coordinates without a verified mapping is not certified.', 'The script does not claim successful anatomical registration; if code was executed as written, treat anatomical localization as unvalidated.', 'No source file was modified. No source reconstruction was performed.',f'Elapsed seconds: {time.time()-t:.1f}']
    (OUT/'DECISION_README.txt').write_text('\n'.join(note),encoding='utf-8')
    print('\n'.join(note));print('REPORT:',OUT)
if __name__=='__main__':main()
