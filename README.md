\# Multi-Modal and Multi-Estimator Brain Fingerprinting



\*\*An Exploratory Analysis of Connectivity Features and Geometry in Resting-State fMRI and MEG\*\*



This repository accompanies the study \*Multi-Modal and Multi-Estimator Brain Fingerprinting: An Exploratory Analysis of Connectivity Features and Geometry in Resting-State fMRI and MEG\*.



\## Overview



This exploratory project investigates individual-specific patterns in resting-state brain connectivity using two neuroimaging modalities:



\- \*\*fMRI:\*\* four resting-state runs per participant (`REST1\_LR`, `REST1\_RL`, `REST2\_LR`, and `REST2\_RL`).

\- \*\*MEG:\*\* three resting-state recordings per participant (`MEG\_3`, `MEG\_4`, and `MEG\_5`).



The study used data from \*\*20 participants\*\* in the Human Connectome Project (HCP) Young Adult dataset. The AAL3v1 atlas was used as the intended common anatomical reference and contained 166 non-empty ROI positions in the analysis.



Three connectivity estimators were examined:



1\. Pearson correlation

2\. Pairwise Granger causality analysis (GCA)

3\. Multivariate transfer entropy (mTE), estimated with IDTxl



Five feature-engineering strategies were evaluated: raw edges, mean-absolute-weight edge selection, high-variance edge selection, PCA-reconstructed features, and maximum-flow features. The original fingerprinting analysis covered 455 combinations of feature-engineering strategies, modality/frequency representations, and estimator combinations.



Additional retrospective analyses examined the numerical quality of the saved connectivity matrices, recovered previously saved mTE statistics, and characterized the geometry of connectivity-feature representations.



\## Repository contents



```text

.

├── code/

│   ├── original\_pipeline/

│   │   ├── Timeseries\_Extracting.ipynb

│   │   ├── Matrix\_Calculating\_fMRI.ipynb

│   │   ├── Matrix\_Calculating\_MEG.ipynb

│   │   ├── Brain\_Fingerprint\_Calculating.ipynb

│   │   └── Neural\_Manifold.ipynb

│   └── retrospective\_audit/

│       ├── m05\_03\_09\_03\_extract\_saved\_idtxl\_statistics.py

│       ├── m06\_04\_08\_02\_recovered\_mte\_fingerprinting.py

│       ├── m07\_02\_03\_09\_04\_umap\_geometry\_stability\_audit.py

│       ├── m08\_04\_recovered\_mte\_posthoc\_statistics\_sensitivity.py

│       ├── m09\_01\_meg\_connectivity\_matrix\_qc.py

│       └── m09\_02\_meg\_anatomical\_correspondence\_audit.py

├── figures/

│   ├── main\_figures/

│   └── all\_91\_UMAP\_plots/

└── results/

&#x20;   ├── original\_fingerprinting/

&#x20;   ├── retrospective\_analysis/

&#x20;   │   ├── meg\_anatomical\_correspondence\_audit/

&#x20;   │   ├── meg\_connectivity\_quality\_control/

&#x20;   │   ├── mte\_statistics\_recovery/

&#x20;   │   └── recovered\_mte\_fingerprinting/

&#x20;   └── UMAP\_results/

&#x20;       ├── Manifold\_Plots\_Multiplex/

&#x20;       └── recovered\_omnibus\_te/

```



\### Main folders



\- `code/original\_pipeline/` contains the original analysis notebooks for time-series extraction, connectivity estimation, fingerprinting, and UMAP visualization.

\- `code/retrospective\_audit/` contains scripts for read-only recovery of saved IDTxl statistics, retrospective fingerprinting, post-hoc statistical sensitivity analysis, MEG connectivity quality control, and anatomical correspondence checks.

\- `figures/main\_figures/` contains the seven principal figures associated with the study.

\- `figures/all\_91\_UMAP\_plots/` contains overview and supplementary UMAP visualizations.

\- `results/original\_fingerprinting/` contains the original fingerprinting summary tables and paired McNemar-test results.

\- `results/retrospective\_analysis/` contains the retrospective mTE recovery, fingerprinting, MEG quality-control, and anatomical-audit outputs.

\- `results/UMAP\_results/` contains the original embedding coordinates and plots, together with the retrospective geometry and stability summaries.



\## Analysis outline



\### fMRI



Regional BOLD time series were extracted from HCP minimally preprocessed volumetric resting-state data using AAL3v1 labels. Signals were averaged within each ROI, band-pass filtered from 0.01 to 0.15 Hz, detrended, and standardized. Each run was processed separately using the original TR of 0.72 seconds.



\### MEG



HCP-preprocessed resting-state MEG recordings were processed using a custom MNE-Python source-reconstruction workflow. The original pipeline examined eight frequency representations and a wide-band representation. Regional signals were derived using atlas-based representative coordinates and an 8-mm neighborhood in a template-based source space.



The original MEG pipeline also applied covariance whitening to the regional waveforms. The retrospective audit found that this processing step was associated with near-zero waveform-based Pearson correlations in most MEG representations. In addition, the anatomical correspondence between the template-based MEG source space and the AAL3v1 atlas was not independently validated.



\### Connectivity and fingerprinting



Pearson correlation, pairwise GCA, and mTE were estimated separately for each available recording. Fingerprinting compared query feature vectors with reference feature vectors using Pearson similarity. The original analysis evaluated five feature-engineering strategies and combinations of connectivity estimators and modalities.



For MEG mTE, the time series were decimated from 250 Hz to 50 Hz before estimation. The resulting 25-Hz Nyquist frequency limits the frequency content that can be represented in those mTE estimates.



\### Retrospective mTE recovery and geometry



The retrospective recovery scripts extracted statistics that had already been saved in the original IDTxl HDF5 outputs. They did \*\*not\*\* repeat IDTxl inference or reconstruct a complete joint source-level TE matrix. The recovered selected source-lag conditional TE features and target-level omnibus TE features were analyzed as distinct representations.



The geometry analyses included the original saved UMAP embeddings and a separate assessment of recovered target-level omnibus TE features using high-dimensional distances, PCA, and UMAP embeddings generated with multiple random seeds.



\## Selected findings



The original fingerprinting results varied substantially across the 455 analysis conditions, with identification accuracy ranging from 0% to 100%. These values describe the original saved results before retrospective quality-control exclusions.



The retrospective audit identified several important methodological constraints:



\- \*\*MEG Pearson matrices:\*\* 480 of 540 original MEG Pearson matrices (88.9%) had maximum absolute off-diagonal correlations below `1e-8` and were classified as numerically degenerate under the retrospective quality-control criterion.

\- \*\*mTE storage:\*\* the original TE-weighted adjacency-matrix construction could overwrite earlier lag-specific estimates when multiple selected source-lag variables mapped to the same source-target position. The separately saved per-target IDTxl records retained information that could be extracted for retrospective analyses.

\- \*\*Anatomical correspondence:\*\* the audit identified a potential positional mismatch risk in the original 166-ROI label assignment. The audit did not validate the anatomical localization of the MEG sources.

\- \*\*Recovered mTE fingerprinting:\*\* the retrospective analysis evaluated 66 acquisition-pair comparisons using recovered mTE representations. The selected-variable and target-level omnibus representations were treated separately.

\- \*\*UMAP:\*\* the original 91 embeddings were analyzed descriptively. Their saved coordinate files did not preserve explicit acquisition identifiers, so they cannot independently establish run-specific reproducibility.



These findings should be interpreted together with the limitations below. High identification accuracy in an original analysis condition does not, by itself, establish that the underlying connectivity representation is numerically or anatomically reliable.



\## Important limitations



This is an \*\*exploratory analysis\*\*, not a confirmatory or clinical validation study.



1\. The study included 20 participants. The original permutation tests across the 455 fingerprinting conditions and the 585 original McNemar comparisons were not adjusted across their complete respective sets of tests.

2\. Most original waveform-based MEG Pearson matrices were numerically near-degenerate after covariance whitening and should not be interpreted as usable zero-lag connectivity fingerprints.

3\. The MEG source-space procedure used a template-based reconstruction. Its anatomical correspondence with the HCP transformations and AAL3v1 ROI labels was not independently validated. Direct anatomical equivalence between the fMRI and MEG regional signals was therefore not established.

4\. The original mTE-weighted adjacency matrices did not preserve all lag-specific conditional TE estimates. The retrospective recovery extracted saved per-target statistics but did not reconstruct a complete joint source-level TE matrix.

5\. MEG mTE estimation followed decimation to 50 Hz (Nyquist frequency: 25 Hz). The resulting estimates cannot be interpreted as covering the full nominal frequency ranges of the original higher-frequency MEG inputs.

6\. Missing target-level omnibus TE estimates were replaced with zero only in explicitly designated sensitivity analyses. A zero-filled value is not a measured zero-valued TE estimate.

7\. The original UMAP coordinate files did not contain explicit run/acquisition identifiers. UMAP coordinates are descriptive feature-space embeddings, not anatomical coordinates or direct measurements of neural-manifold geometry.

8\. GCA and mTE results characterize statistical predictability or information-theoretic dependence under the specified estimators; they should not be interpreted as direct proof of neural causation.



\## Data availability and HCP terms



The repository archive contains code, figures, and analysis outputs. \*\*Raw HCP MRI/MEG acquisitions and the original intermediate time-series and connectivity files are not included.\*\*



The analysis uses data from the HCP Young Adult dataset. Users who wish to reproduce the analyses must obtain the required data through the official HCP access process and comply with the applicable \*\*WU-Minn HCP Consortium Open Access Data Use Terms\*\*:



\- \[HCP Young Adult — Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)

\- \[HCP Young Adult — Documentation](https://www.humanconnectome.org/study/hcp-young-adult/documentation)



The HCP Open Access Data Use Terms permit redistribution of original Open Access data and derived data only when redistributed under the same terms. Do not include Restricted Access data in this repository. Check the access category and terms for every source dataset before sharing files publicly.



The HCP terms also require acknowledgement of the data source. The following wording is provided by HCP for publications and public presentations using WU-Minn HCP data:



> Data were provided \[in part] by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.



\## Reproducibility notes



This repository is currently a \*\*research-artifact archive\*\*, not a fully self-contained, one-command reproduction package.



\- The raw HCP acquisitions and several intermediate inputs required by the original notebooks are not included.

\- The original notebooks and some scripts contain machine-specific absolute file paths. These paths must be changed to match the local data and output directories before execution.

\- A pinned software environment (`requirements.txt` or `environment.yml`) is not currently provided. The code uses scientific Python packages including NumPy, pandas, SciPy, scikit-learn, statsmodels, h5py, Matplotlib, MNE-Python, Nilearn, IDTxl, UMAP, FOOOF, and VMD-related tools. Compatible versions must be established for the local environment.

\- The retrospective recovery and audit scripts operate on previously saved outputs. They do not recreate missing raw data or rerun every original analysis.



Please inspect the relevant script and its input/output path settings before running it.



\## Associated Manuscript



A manuscript associated with this repository is currently being prepared for public release as a preprint.



The manuscript is not yet publicly available. Full citation information and the preprint link will be added to this repository once the preprint is publicly released.



\## How to Cite This Repository



If you use the code or research materials provided in this repository, please refer to this GitHub repository.



Please also cite the relevant HCP data-acquisition and preprocessing publications, as appropriate for the data used.





\## License



A separate license for the original code in this repository has not yet been specified. Until a license is added, do not assume that the code is licensed for unrestricted reuse.



HCP-derived data remain subject to the HCP Open Access Data Use Terms and any other applicable source-data terms. A code license, if added later, will not replace those data-use conditions.



\## Contact



For questions about the analysis or repository, contact:



\- \*\*Maintainer:\*\* Kay Zhang

\- \*\*Email:\*\* brainnibbana@163.com

\- \*\*Institution:\*\* none

