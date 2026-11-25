# -*- coding: utf-8 -*-
r"""No-BLAS, low-memory, read-only audit of 20-subject resting MEG HDF5 networks.
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import subprocess
from pathlib import Path

SUBJECTS = (
    '100307', '102816', '106521', '108323', '109123',
    '111514', '112920', '113922', '116726', '133019',
    '146129', '149741', '156334', '158136', '162026',
    '162935', '164636', '166438', '169040', '172029',
)
METHODS = ('Pearson', 'GCA', 'mTE')
BANDS = (
    'Low-Frequency', 'Delta', 'Theta', 'Alpha', 'Beta',
    'Low-Gamma', 'High-Gamma', 'Wide-band', 'Aperiodic-VMD',
)
RUNS = (3, 4, 5)
SAMPLE_NODES = 166
FILE_RE = re.compile(r'^(Pearson|GCA|mTE)_(\d{6})_(.+)\.h5$', re.IGNORECASE)
EMBED_RE = re.compile(r'(\d{6})_MEG_(\d+)-Restin_')


def parse_file(p):
    match = FILE_RE.match(p.name)
    if not match:
        return None
    m, subject, body = match.groups()
    method = next((s for s in METHODS if s.lower() == m.lower()), None)
    if subject not in SUBJECTS or 'rmegpreproc' not in body.lower():
        return None
    embedded = EMBED_RE.search(body)
    if not embedded:
        return None
    actual_subject, run_txt = embedded.groups()
    run = int(run_txt)
    if run not in RUNS:
        return None
    band = next((b for b in BANDS if body.endswith('_' + b)), None)
    if band is None:
        return None
    return dict(path=str(p), method=method, band=band, subject=subject,
                run=run, embedded_subject=actual_subject,
                subject_mismatch=(subject != actual_subject))


def csv_atomic(df, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, path)


def vector_from_file(meta):
    import hashlib
    import h5py
    import numpy as np
    method = meta['method']
    row = dict(meta)
    row.update(status='OK', n_nodes='', edge_std=np.nan, edge_absmax=np.nan,
               frac_nearzero=np.nan, frac_nonfinite=np.nan, num_positive=0,
               binary_n_edges='', binary_weight_disagreement='',
               gca_fraction_fdr05='', roi_label_signature='', roi_label_hash='',
               usable_for_similarity=True)
    vec = None
    try:
        with h5py.File(meta['path'], 'r') as f:
            key = {'Pearson': 'results/results/corr_matrix',
                   'GCA': 'results/results/gca_fdr_pval',
                   'mTE': 'results/te_weighted_adjacency_matrix'}[method]
            if key not in f:
                raise KeyError(f'Missing {key}')
            mat = np.asarray(f[key][()], dtype=np.float64)
            if mat.ndim != 2 or mat.shape != (SAMPLE_NODES, SAMPLE_NODES):
                raise ValueError(f'Unexpected matrix shape: {mat.shape}')
            row['n_nodes'] = mat.shape[0]
            if 'brain_regions' in f:
                labels = [x.decode('utf-8', 'replace') if isinstance(x, bytes) else str(x)
                          for x in f['brain_regions'][()]]
                if len(labels) != SAMPLE_NODES:
                    raise ValueError('Wrong length of brain_regions')
                row['roi_label_hash'] = hashlib.sha256('\n'.join(labels).encode()).hexdigest()
                row['roi_label_signature'] = '|'.join(labels[:5]) + ' ... ' + '|'.join(labels[-3:])
            mask = ~np.eye(SAMPLE_NODES, dtype=bool)
            vals = mat[mask]
            row['frac_nonfinite'] = float((~np.isfinite(vals)).mean())
            arr = np.nan_to_num(vals, nan=0., posinf=0., neginf=0.)
            if method == 'GCA':
                valid = vals[np.isfinite(vals)]
                row['gca_fraction_fdr05'] = float(np.mean(valid <= 0.05)) if valid.size else np.nan
                arr = -np.log10(np.clip(np.nan_to_num(vals, nan=1., posinf=1., neginf=1.), 1e-15, 1.))
            row['edge_std'] = float(np.std(arr))
            row['edge_absmax'] = float(np.max(np.abs(arr)))
            row['frac_nearzero'] = float(np.mean(np.abs(arr) < 1e-8))
            row['num_positive'] = int(np.sum(arr > 1e-8))
            if method == 'mTE' and 'results/binary_adjacency_matrix' in f:
                binary = np.asarray(f['results/binary_adjacency_matrix'][()])
                if binary.shape != mat.shape:
                    raise ValueError('binary_adjacency_matrix shape differs from TE matrix')
                b = binary[mask] != 0
                row['binary_n_edges'] = int(b.sum())
                row['binary_weight_disagreement'] = int(np.sum(b != (np.abs(vals) > 1e-12)))
            vec = arr.astype(np.float32, copy=True)
            # Zero-lag Pearson edges near floating-point noise are not biologically
            # interpretable; normalizing them could create fake identifiability.
            if method == 'Pearson' and row['edge_absmax'] < 1e-8:
                row['usable_for_similarity'] = False
                vec = None
    except Exception as e:
        row['status'] = type(e).__name__ + ': ' + str(e)
    return row, vec


def norm(vec):
    import numpy as np
    if vec is None:
        return None
    x = np.asarray(vec, dtype=np.float64)
    if not np.isfinite(x).all():
        return None
    x -= x.mean()

    length = float(np.sqrt(np.sum(x * x, dtype=np.float64)))
    if not np.isfinite(length) or length < 1e-10:
        return None
    return (x / length).astype(np.float32)


def calc_pair(vectors, method, band, a, b):
    import numpy as np
    aa = [norm(vectors.get((s, a))) for s in SUBJECTS]
    bb = [norm(vectors.get((s, b))) for s in SUBJECTS]
    va = np.array([x is not None for x in aa]); vb = np.array([x is not None for x in bb])
    out = dict(method=method, band=band, query_run=a, database_run=b,
               queries_usable=int(va.sum()), database_usable=int(vb.sum()),
               n_subjects_expected=len(SUBJECTS), n_scored=0, accuracy_pct=np.nan,
               mean_within_similarity=np.nan, mean_between_similarity=np.nan,
               idiff=np.nan, status='')
    if va.sum() < 1 or vb.sum() < 2:
        out['status'] = 'NOT_ENOUGH_NONDEGENERATE_NETWORKS'
        return out
    nvec = len(next(x for x in aa if x is not None))
    A = np.stack([x if x is not None else np.zeros(nvec, dtype=np.float32) for x in aa]);
    B = np.stack([x if x is not None else np.zeros(nvec, dtype=np.float32) for x in bb]);

    sim = np.einsum('ik,jk->ij', A, B, optimize=False).astype(np.float64)
    sim[~va, :] = np.nan
    sim[:, ~vb] = np.nan
    diag = np.diag(sim)
    same = diag[np.isfinite(diag)]
    other = sim[~np.eye(len(SUBJECTS), dtype=bool)]
    other = other[np.isfinite(other)]
    if len(same):
        out['mean_within_similarity'] = float(same.mean())
    if len(other):
        out['mean_between_similarity'] = float(other.mean())
    if len(same) and len(other):
        out['idiff'] = float(same.mean()-other.mean())
    hit=[]
    for i in range(len(SUBJECTS)):
        if not va[i] or not np.isfinite(sim[i]).any():
            continue
        best = np.flatnonzero(np.isclose(sim[i], np.nanmax(sim[i]), rtol=0., atol=1e-7))
        hit.append(len(best)==1 and int(best[0])==i)
    out['n_scored'] = len(hit)
    if hit:
        out['accuracy_pct'] = 100.*sum(hit)/len(hit)
    out['status'] = 'ALL_20_VALID' if va.sum()==20 and vb.sum()==20 else 'MISSING_OR_DEGENERATE_RUNS'
    return out


def worker(args):
    import hashlib
    import pandas as pd
    import numpy as np
    root = Path(args.meg_results)
    output = Path(args.out) / '_checkpoints'
    output.mkdir(parents=True, exist_ok=True)
    metas = []
    for p in root.glob('*.h5'):
        meta = parse_file(p)
        if meta and meta['band'] == args.band and meta['method'] == args.method:
            metas.append(meta)
    rows=[]; vectors={}; duplicates=[]
    stem = f'{args.band}__{args.method}'
    current_file_log = output / (stem + '_last_attempted_file.txt')
    phase_file = output / (stem + '_CURRENT_STAGE.txt')
    phase_file.write_text(f'START\nFound files: {len(metas)}\n', encoding='utf-8')
    for file_idx, meta in enumerate(sorted(metas, key=lambda a:a['path']), 1):
        # A phase log identifies whether a native crash occurred in file I/O or after reading.
        current_file_log.write_text(meta['path'], encoding='utf-8')
        phase_file.write_text(f'READING {file_idx}/{len(metas)}\n{meta["path"]}\n', encoding='utf-8')
        row, vec = vector_from_file(meta)
        phase_file.write_text(f'READ_COMPLETE {file_idx}/{len(metas)}; status={row["status"]}\n{meta["path"]}\n', encoding='utf-8')
        if file_idx % 10 == 0:
            print(f'{stem}: read {file_idx}/{len(metas)} files', flush=True)
        rows.append(row)
        k=(meta['subject'],meta['run'])
        if k in vectors:
            duplicates.append(str(k))
        else:
            vectors[k]=vec
    # Check consistency of ROI order before comparing any matrices.
    hashes = {row['roi_label_hash'] for row in rows if row['roi_label_hash']}
    mismatch = len(hashes)>1
    if mismatch:
        vectors = {k:None for k in vectors}
    pairs = []
    for a, b in [(3, 4), (3, 5), (4, 5)]:
        phase_file.write_text(f'SIMILARITY_STARTED run={a} vs run={b}\n', encoding='utf-8')
        pairs.append(calc_pair(vectors, args.method, args.band, a, b))
        phase_file.write_text(f'SIMILARITY_FINISHED run={a} vs run={b}\n', encoding='utf-8')
    phase_file.write_text('WRITING_CSV\n', encoding='utf-8')
    df=pd.DataFrame(rows, columns=list(rows[0]) if rows else ['path','method','band','subject','run','status','roi_label_hash','edge_std','frac_nearzero'])
    pf=pd.DataFrame(pairs)
    csv_atomic(df, output / (stem+'_files.csv'))
    csv_atomic(pf, output / (stem+'_pairs.csv'))
    # Done marker is written last so resuming skips ONLY a completely finished group.
    marker=output / (stem+'.done')
    marker.write_text(f'files={len(rows)}; ROI_order_conflict={mismatch}; duplicate_keys={duplicates}\n',encoding='utf-8')
    if current_file_log.exists():
        current_file_log.unlink()
    phase_file.write_text('DONE\n', encoding='utf-8')
    print(f'{stem}: {len(rows)} HDF5 read; errors={sum(r["status"]!="OK" for r in rows)}; order_conflict={mismatch}',flush=True)


def driver(args):
    import pandas as pd
    root=Path(args.meg_results)
    output=Path(args.out)
    if not root.is_dir():
        raise FileNotFoundError(f'Input directory not found: {root}')
    output.mkdir(parents=True, exist_ok=True)
    check_dir=output/'_checkpoints';check_dir.mkdir(exist_ok=True)
    pairs=[(band,method) for band in BANDS for method in METHODS]
    if args.max_groups is not None:
        pairs=pairs[:args.max_groups]
    failures=[]
    print(f'NO-BLAS in-process audit: {len(pairs)} groups; input is READ ONLY.', flush=True)
    for seq, (band,method) in enumerate(pairs,1):
        stem=f'{band}__{method}'
        complete=(check_dir / (stem+'.done')).exists() and (check_dir/(stem+'_files.csv')).exists() and (check_dir/(stem+'_pairs.csv')).exists()
        if complete and not args.force:
            print(f'[{seq}/{len(pairs)}] Skip completed {stem}', flush=True)
            continue

        # Only one 60-file group is held at a time. BLAS and LAPACK are not used for similarity.
        try:
            args.band, args.method = band, method
            worker(args)
            print(f'[{seq}/{len(pairs)}] OK {stem}', flush=True)
        except Exception as exc:
            failures.append(dict(group=stem, exit_code='PYTHON_EXCEPTION', message=repr(exc)))
            print(f'[{seq}/{len(pairs)}] PYTHON ERROR {stem}: {type(exc).__name__}: {exc}', flush=True)
        finally:
            import gc
            gc.collect()
    csv_atomic(pd.DataFrame(failures, columns=['group','exit_code','message']), output/'08_worker_failures.csv')
    files=[]; comparisons=[]
    for band in BANDS:
        for method in METHODS:
            stem=f'{band}__{method}'
            if (check_dir/(stem+'.done')).exists():
                f=check_dir/(stem+'_files.csv');p=check_dir/(stem+'_pairs.csv')
                if f.exists() and p.exists():
                    files.append(pd.read_csv(f, dtype={'subject':str,'embedded_subject':str}))
                    comparisons.append(pd.read_csv(p))
    if not files:
        print('No completed groups yet; please upload 08_worker_failures.csv or the error output.',flush=True)
        return
    file_df=pd.concat(files,ignore_index=True)
    pair_df=pd.concat(comparisons,ignore_index=True)
    csv_atomic(file_df,output/'01_MEG_file_quality.csv')
    csv_atomic(pair_df,output/'02_MEG_within_between_by_run.csv')
    summary = file_df.groupby(['band','method'],as_index=False).agg(
        n_files=('path','size'),median_edge_std=('edge_std','median'),
        mean_frac_nearzero=('frac_nearzero','mean'))
    csv_atomic(summary,output/'03_MEG_band_method_summary.csv')
    def nvalid(x):
        return int((x=='ALL_20_VALID').sum())
    summary2=pair_df.groupby(['band','method'],as_index=False).agg(
        valid_pair_count=('status',nvalid), average_accuracy_pct=('accuracy_pct','mean'),
        average_idiff=('idiff','mean'))
    csv_atomic(summary2,output/'04_MEG_identifiability_summary.csv')
    found={ (str(r.method),str(r.band),str(r.subject),int(r.run)) for r in file_df.itertuples() }
    expected={ (m,b,s,r) for m in METHODS for b in BANDS for s in SUBJECTS for r in RUNS }
    missing=sorted(expected-found)
    (output/'05_MEG_missing_expected_files.txt').write_text('\n'.join(map(str,missing)),encoding='utf-8')
    duplicates=file_df[file_df.duplicated(['method','band','subject','run'],keep=False)]
    csv_atomic(duplicates,output/'06_MEG_duplicate_keys.csv')
    conflicts=[]
    for (b,m),group in file_df.groupby(['band','method']):
        labels=set(group['roi_label_hash'].dropna())-{''}
        if len(labels)>1:
            conflicts.append(dict(band=b,method=m,label_sets=len(labels)))
    csv_atomic(pd.DataFrame(conflicts,columns=['band','method','label_sets']),output/'07_MEG_ROI_order_conflicts.csv')
    readme=(
        'LOW-MEMORY MEG AUDIT (READ-ONLY)\n'
        f'Results dir: {root}\n'
        f'Completed groups: {len(files)}/27\n'
        f'HDF5 files examined: {len(file_df)}/1620\n'
        f'File-level errors: {int((file_df.status!="OK").sum())}\n'
        f'Missing expected files relative to ALL 27 groups: {len(missing)}\n'
        f'Worker failures this run: {len(failures)}\n'
        'NOTE: after a 2-group trial, missing files are EXPECTED; run again without --max-groups.\n'
        'Pearson: raw correlations. GCA: -log10 FDR p-values, NOT effect sizes.\n'
        'Pearson matrices with all |r| < 1e-8 are excluded from similarity (not interpreted as real fingerprints).\n'
        'mTE: original weighted matrices, still susceptible to overwriting of lag-specific values.\n'
        'Networks with zero variance are excluded from cross-run similarity.\n'
        'This procedure does not validate brain anatomy, inference significance or MEG source signals.\n'
        'All source HDF5 files were opened READ ONLY and never changed.\n')
    (output/'README_MEG_audit.txt').write_text(readme,encoding='utf-8')
    print('\n'+readme,flush=True)
    print(summary2[['band','method','valid_pair_count','average_accuracy_pct','average_idiff']].to_string(index=False),flush=True)
    print(f'\nReport folder: {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description='Low-memory resting MEG connectivity audit (read only)')
    parser.add_argument('--meg-results',default=r'E:\Results\meg_results')
    parser.add_argument('--out',default=r'D:\Brain_Fingerprint_scripts\MEG_noBLAS_audit_reports')
    parser.add_argument('--max-groups',type=int,default=None,help='For initial safety test e.g. 2 groups')
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--band',choices=BANDS,help=argparse.SUPPRESS)
    parser.add_argument('--method',choices=METHODS,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        if not args.band or not args.method:
            raise ValueError('Worker requires --band and --method')
        worker(args)
    else:
        driver(args)

if __name__=='__main__':
    main()
