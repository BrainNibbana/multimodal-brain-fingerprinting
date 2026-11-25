#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot and summarize recovered mTE analyses without raw fMRI/MEG time series.

Dependencies: numpy, pandas, matplotlib
Inputs:
    --input  Brain_Fingerprint_scripts.zip
          OR extracted Brain_Fingerprint_scripts directory
    --mask   optional mte_selection_mask_control.csv from a separate sensitivity check
    --output local folder for PNG + CSV outputs

This program does NOT re-estimate TE, update original HDF5s, or rerun
permutation tests. BH and Holm adjustments are POST-HOC adjustments
of previously saved FINGERPRINT permutation p values (NOT of TE link p values).
"""
from __future__ import annotations

import argparse
import io
import textwrap
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_ORDER = [
    "fMRI", "MEG_rmegpreproc_Low-Frequency", "MEG_rmegpreproc_Delta",
    "MEG_rmegpreproc_Theta", "MEG_rmegpreproc_Alpha",
    "MEG_rmegpreproc_Beta", "MEG_rmegpreproc_Low-Gamma",
    "MEG_rmegpreproc_High-Gamma", "MEG_rmegpreproc_Wide-band",
    "MEG_rmegpreproc_Aperiodic-VMD",
]
PRETTY = {
    "fMRI": "fMRI",
    **{x: "MEG " + x.split("MEG_rmegpreproc_")[1] for x in GROUP_ORDER if x != "fMRI"},
}
FEATURE_NAMES = {
    "omnibus_zero_filled": "Omnibus TE (zero-fill sensitivity)",
    "variable_lag_sparse": "Selected-variable TE",
}
QUA = "preprint_final_statistics/01_omnibus_quality.csv"
PAIR = "preprint_final_statistics/02_mte_fingerprint_permutations.csv"
SUMM = "preprint_final_statistics/05_mte_summary.csv"
OMNI = "mte_saved_export/omnibus_by_target.csv"
MCNEMAR = "Optimization_Ultimate_Engine_V4_1_Final/McNemar_Statistical_Comparisons.csv"


class Source:
    def __init__(self, location: Path):
        self.location = location
        self.archive = zipfile.ZipFile(location) if location.is_file() and location.suffix.lower() == ".zip" else None
        if not self.archive and not location.is_dir():
            raise ValueError("Input must be a ZIP or an extracted directory: " + str(location))

    def read_csv(self, suffix: str) -> pd.DataFrame:
        suffix = suffix.replace("\\", "/").strip("/")
        if self.archive:
            names = [n for n in self.archive.namelist()
                     if n.replace("\\", "/").endswith("/" + suffix) or n == suffix]
            if len(names) != 1:
                raise FileNotFoundError(f"Expected one '{suffix}' in ZIP; found {len(names)}")
            return pd.read_csv(io.BytesIO(self.archive.read(names[0])), encoding="utf-8-sig", low_memory=False)
        possibilities = [self.location / suffix, self.location / "Brain_Fingerprint_scripts" / suffix]
        found = [p for p in possibilities if p.exists()]
        if not found:
            found = [p for p in self.location.rglob(Path(suffix).name)
                     if str(p).replace("\\", "/").endswith(suffix)]
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one '{suffix}' in folder; found {len(found)}")
        return pd.read_csv(found[0], encoding="utf-8-sig", low_memory=False)

    def close(self):
        if self.archive:
            self.archive.close()


def corrections(p_values):
    """BH-FDR and Holm step-down adjustments on saved fingerprint p values."""
    p = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p) & (p >= 0) & (p <= 1)
    out_bh = np.full(p.shape, np.nan)
    out_holm = np.full(p.shape, np.nan)
    if not valid.any():
        return out_bh, out_holm
    pv = p[valid]
    m = len(pv)
    ordering = np.argsort(pv, kind="stable")
    ordered = pv[ordering]
    bh = np.minimum.accumulate((ordered * m / np.arange(1, m+1))[::-1])[::-1]
    holm = np.maximum.accumulate(ordered * (m - np.arange(m)))
    bh, holm = np.minimum(bh, 1), np.minimum(holm, 1)
    bh_original = np.empty(m)
    holm_original = np.empty(m)
    bh_original[ordering] = bh
    holm_original[ordering] = holm
    out_bh[valid] = bh_original
    out_holm[valid] = holm_original
    return out_bh, out_holm


def wilson_bounds(correct, n, z=1.959963984540054):
    """Marginal 95% Wilson interval for one 20-person run-pair accuracy."""
    p = correct / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    rad = z*np.sqrt((p*(1-p) + z*z/(4*n))/n)/den
    return center - rad, center + rad


def fig_setup():
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "figure.dpi": 120,
        "savefig.dpi": 220,
    })


def save(fig, out: Path, filename: str):
    fig.savefig(out / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out / filename)


def run(input_path: Path, output: Path, mask: Path | None):
    output.mkdir(parents=True, exist_ok=True)
    src = Source(input_path)
    try:
        quality = src.read_csv(QUA)
        pairs = src.read_csv(PAIR)
        summary = src.read_csv(SUMM)
        try:
            omnibus = src.read_csv(OMNI)
        except FileNotFoundError:
            omnibus = None
        try:
            historical_mcnemar = src.read_csv(MCNEMAR)
        except FileNotFoundError:
            historical_mcnemar = None
    finally:
        src.close()

    if set(pairs["status"]) - {"OK"}:
        print("WARNING: some saved fingerprint rows are non-OK; they will be omitted.")
    pairs = pairs.loc[(pairs["status"] == "OK") &
                      np.isfinite(pairs["accuracy"]) &
                      np.isfinite(pairs["idiff"])].copy()
    pairs["fingerprint_p_bh_posthoc"], pairs["fingerprint_p_holm_posthoc"] = corrections(pairs["p_perm"].to_numpy())
    if "n" in pairs:
        low, hi = [], []
        for rec in pairs.itertuples():
            n = int(rec.n)
            # count is an integer for these 20-person classifications
            n_correct = int(round(rec.accuracy * n))
            a, b = wilson_bounds(n_correct, n)
            low.append(a); hi.append(b)
        pairs["wilson95_lower_nominal"] = low
        pairs["wilson95_upper_nominal"] = hi
    pairs.to_csv(output / "fingerprint_pairwise_checked.csv", index=False, encoding="utf-8-sig")

    q = quality.groupby("group", as_index=False).agg(
        recordings=("file", "size"),
        finite_targets=("finite_targets", "sum"),
        missing_targets=("missing_targets", "sum"),
    )
    q["finite_percent"] = 100 * q["finite_targets"] / (q["finite_targets"] + q["missing_targets"])
    q["group_label"] = q["group"].map(PRETTY)
    q = q.set_index("group").reindex(GROUP_ORDER).reset_index()
    q.to_csv(output / "omnibus_quality_summary.csv", index=False, encoding="utf-8-sig")

    agg = pairs.groupby(["group", "feature"], as_index=False).agg(
        n_run_pairs=("accuracy", "size"),
        mean_accuracy=("accuracy", "mean"),
        min_accuracy=("accuracy", "min"),
        max_accuracy=("accuracy", "max"),
        mean_idiff=("idiff", "mean"),
        min_idiff=("idiff", "min"),
        max_idiff=("idiff", "max"),
        nominal_perm_p_below_05=("p_perm", lambda v: int((v < .05).sum())),
        bh_perm_q_below_05=("fingerprint_p_bh_posthoc", lambda v: int((v < .05).sum())),
    )
    agg.to_csv(output / "recovered_mte_group_summary.csv", index=False, encoding="utf-8-sig")

    fig_setup()
    groups = [PRETTY[k] for k in GROUP_ORDER]
    y = np.arange(len(GROUP_ORDER))

    # 01: quality and missingness, based on saved omnibus estimates
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.barh(y, q["finite_percent"].fillna(0).to_numpy())
    ax.set_yticks(y, groups)
    ax.invert_yaxis()
    ax.set_xlabel("Targets with finite saved omnibus TE (%)")
    ax.set_xlim(0, 108)
    ax.set_title("Recovered omnibus TE: estimate availability")
    for i, rec in enumerate(q.itertuples()):
        if pd.notna(rec.finite_percent):
            ax.text(rec.finite_percent + 1, i, f"{rec.finite_percent:.1f}%", va="center", fontsize=9)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.12, .0, "Missing estimates are not measured zero-valued TE.", fontsize=9)
    save(fig, output, "01_omnibus_availability.png")

    def wide_metric(metric):
        wide = agg.pivot(index="group", columns="feature", values=metric).reindex(GROUP_ORDER)
        return (wide["omnibus_zero_filled"].to_numpy(dtype=float),
                wide["variable_lag_sparse"].to_numpy(dtype=float))

    # 02: pair-averaged accuracy, first two scientific feature representations
    a, b = wide_metric("mean_accuracy")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    h = .37
    ax.barh(y - h/2, a*100, height=h, label=FEATURE_NAMES["omnibus_zero_filled"])
    ax.barh(y + h/2, b*100, height=h, label=FEATURE_NAMES["variable_lag_sparse"])
    ax.axvline(5, linestyle="--", linewidth=1, label="20-person chance (5%)")
    ax.set_yticks(y, groups)
    ax.invert_yaxis()
    ax.set_xlabel("Mean identification accuracy across acquisition pairs (%)")
    ax.set_xlim(0, max(70, np.nanmax([a,b])*115))
    ax.set_title("Recovered mTE representations: fingerprinting accuracy")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.12, -.015, "fMRI: 6 run pairs; each MEG group: 3 pairs. Omnibus results use zero-fill sensitivity only.", fontsize=8)
    save(fig, output, "02_fingerprint_accuracy.png")

    # 03: differential identifiability, do not imply group superiority
    a, b = wide_metric("mean_idiff")
    fig, ax = plt.subplots(figsize=(10, 5.7))
    ax.barh(y - h/2, a, height=h, label=FEATURE_NAMES["omnibus_zero_filled"])
    ax.barh(y + h/2, b, height=h, label=FEATURE_NAMES["variable_lag_sparse"])
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y, groups)
    ax.invert_yaxis()
    ax.set_xlabel("Mean differential identifiability (within - between Pearson similarity)")
    ax.set_title("Recovered mTE representations: differential identifiability")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    save(fig, output, "03_fingerprint_idiff.png")

    # 04: per-acquisition fMRI comparisons (do not obscure variability with averages)
    fmri = pairs.loc[pairs["group"] == "fMRI"].copy()
    fmri["pair"] = fmri["run_a"].astype(str) + " vs " + fmri["run_b"].astype(str)
    w = fmri.pivot(index="pair", columns="feature", values="accuracy")
    labels = list(w.index)
    pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.bar(pos - h/2, w["omnibus_zero_filled"].to_numpy()*100, width=h, label="Omnibus TE")
    ax.bar(pos + h/2, w["variable_lag_sparse"].to_numpy()*100, width=h, label="Selected-variable TE")
    ax.axhline(5, linewidth=1, linestyle="--")
    ax.set_xticks(pos, labels, rotation=25, ha="right")
    ax.set_ylim(0, max(85, fmri["accuracy"].max() * 115))
    ax.set_ylabel("Identification accuracy (%)")
    ax.set_title("fMRI: six explicit acquisition-pair comparisons")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    save(fig, output, "04_fmri_run_pair_accuracy.png")

    # 05: saved fMRI variable-count saturation (NOT a new TE inference)
    if omnibus is not None:
        is_fmri = omnibus["modality"].astype(str).str.lower().eq("fmri")
        eligible = omnibus.loc[is_fmri &
                               pd.to_numeric(omnibus["omnibus_te"], errors="coerce").notna()].copy()
        counts = pd.to_numeric(eligible["selected_variable_count"], errors="coerce").dropna().astype(int)
        if len(counts):
            fig, ax = plt.subplots(figsize=(8.8, 4.6))
            vc = counts.value_counts().sort_index()
            ax.bar(vc.index.to_numpy(), vc.values)
            ax.set_xlabel("Number of selected source-lag variables per target")
            ax.set_ylabel("Number of fMRI target records")
            ax.set_title("Historical fMRI mTE: selected-variable count among finite targets")
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            sat = int((counts == 30).sum())
            fig.text(.11, -.005, f"30 selected variables: {sat:,}/{len(counts):,} ({sat/len(counts)*100:.2f}%).", fontsize=9)
            save(fig, output, "05_fmri_selection_saturation.png")
            pd.DataFrame([{"finite_fmri_targets":len(counts),
                           "targets_with_30_selected":sat,
                           "percent_with_30_selected":100*sat/len(counts)}]).to_csv(
                               output/"fmri_variable_selection_saturation.csv",
                               index=False, encoding="utf-8-sig")

    # 06 (optional): separately created binary-selection-mask control
    if mask is not None:
        m = pd.read_csv(mask, encoding="utf-8-sig")
        needed = {"group", "feature", "accuracy"}
        if not needed.issubset(m):
            raise ValueError("Mask control CSV missing columns: " + ", ".join(sorted(needed-set(m.columns))))
        compare = m.groupby(["group","feature"])["accuracy"].mean().unstack().reindex(GROUP_ORDER)
        if not {"TE-magnitude", "Selection-only"}.issubset(compare.columns):
            raise ValueError("Mask CSV needs TE-magnitude and Selection-only feature rows")
        fig, ax = plt.subplots(figsize=(10, 5.7))
        ax.barh(y - h/2, compare["TE-magnitude"]*100, height=h, label="Selected-variable TE values")
        ax.barh(y + h/2, compare["Selection-only"]*100, height=h, label="Binary variable-selection mask")
        ax.axvline(5, linewidth=1, linestyle="--")
        ax.set_yticks(y, groups)
        ax.invert_yaxis()
        ax.set_xlabel("Mean identification accuracy across acquisition pairs (%)")
        ax.set_title("Sensitivity check: TE magnitudes versus selection pattern")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        fig.text(.12, -.012, "Post-hoc control; differences are descriptive, not paired-significance tested.", fontsize=9)
        save(fig, output, "06_selection_pattern_control.png")
        compare.reset_index().to_csv(output/"selection_pattern_group_summary.csv",
                                     index=False, encoding="utf-8-sig")
    else:
        print("Optional selection-mask chart skipped (pass --mask path/to/mte_selection_mask_control.csv).")

    summary_lines = [
        "RECOMPUTATION: NONE. All figures derive from saved CSVs.",
        "Saved fingerprint comparisons (OK): " + str(len(pairs)),
        "Nominal fingerprint permutation p<.05: " + str(int((pairs["p_perm"]<.05).sum())),
        "Post-hoc BH-FDR q<.05 across saved fingerprint comparisons: "
            + str(int((pairs["fingerprint_p_bh_posthoc"]<.05).sum())),
        "Post-hoc Holm adjusted p<.05 across saved fingerprint comparisons: "
            + str(int((pairs["fingerprint_p_holm_posthoc"]<.05).sum())),
        "Historical McNemar comparisons (NOT recovered mTE comparisons): "
            + (str(len(historical_mcnemar)) if historical_mcnemar is not None else "not found"),
        "",
        "WARNING: BH/Holm above apply to FINGERPRINT permutation p values,",
        "NOT to IDTxl mTE link-level p values or network-level FDR.",
        "The 999-permutation resolution is approximately 0.001.",
        "Across-run results share subjects and are correlated; summary statistics are descriptive.",
        "Omnibus missing values were zero-filled as a sensitivity analysis only.",
        "Historical MEG gamma-band mTE was calculated after 250-to-50 Hz decimation.",
        "MEG source-localization/coordinate issues remain unresolved.",
    ]
    (output/"README_interpretation.txt").write_text("\n".join(summary_lines)+"\n",encoding="utf-8")
    print("\n".join(summary_lines))
    print("\nCompleted; output:", output.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot recovered mTE preprint analyses from ZIP or extracted folder.")
    ap.add_argument("--input", type=Path, required=True, help="Brain_Fingerprint_scripts.zip or its extracted directory")
    ap.add_argument("--output", type=Path, default=Path("mte_recovery_figures"), help="Output folder")
    ap.add_argument("--mask", type=Path, default=None, help="Optional mte_selection_mask_control.csv")
    args = ap.parse_args()
    run(args.input.expanduser(), args.output.expanduser(), args.mask.expanduser() if args.mask else None)
