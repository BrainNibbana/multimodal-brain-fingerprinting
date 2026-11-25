EXPLORATORY PREPRINT STATISTICS

1. These are representations of SAVED mTE estimates, not recovered source-level joint link TE.
2. Omnibus missing estimates are zero-filled solely for a transparent sensitivity analysis; missing is NOT a measured zero. See 01_omnibus_quality.csv.
3. Variable features are source x lag x target; absent selected variables use zero as a selection indicator, not as a measured TE.
4. Fingerprint is directed A-run to B-run Pearson similarity, with exact subject/run matching and permutation of B-run subject identities. Permutation p values are descriptive and uncorrected for multiple testing.
5. No preprocessing is fit across test data; zero/constant vectors are flagged as degenerate.
6. UMAP CSV analysis uses only existing 2D coordinates and DOES NOT establish manifold stability or shared neural mechanisms. A valid high-dimensional geometry and seed-stability test requires original feature arrays and rerunning UMAP.
7. Old MEG source-space localization and decimation concerns persist; fMRI mTE selection saturation persists.
8. This script does not rerun old 455 experiments, re-estimate TE, compute source-level link TE, or validate high-dimensional group templates.

