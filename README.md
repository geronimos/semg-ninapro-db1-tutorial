# An Introduction to sEMG Signal Processing and Movement Classification with NinaPro DB1

This repository is a notebook-first tutorial series that reproduces core steps from Atzori et al. [1] using the NinaPro DB1 dataset [1, 2, 3]. The emphasis is a classic sEMG ML pipeline designed for self-study.

## Tutorial notebooks

All tutorial content lives in the notebooks; start with `notebooks/00_introduction.ipynb` and proceed in numeric order (00-06).

## Project structure

```txt
ninapro-db1-tutorial/
├── data/
│   ├── raw/                # Original NinaPro DB1
│   ├── processed/          # Models, dataset splits, artifacts
├── notebooks/
│   ├── 00_introduction.ipynb
│   ├── 01_data_loading_and_overview.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_windowing_and_feature_extraction.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_evaluation_and_error_analysis.ipynb
│   ├── 06_discussion_and_real_world_constraints.ipynb
├── src/
│   ├── datasets.py         # Built dataset from `.mat`-files
│   ├── io.py               # Data loading and parsing
│   ├── features.py         # MAV, WL, VAR
│   ├── windowing.py        # Sliding window logic
├── scripts/
│   ├── download_data.py    # Download and extract NinaPro DBs
├── pyproject.toml
└── README.md
```

## References

[1] M. Atzori, A. Gijsberts, I. Kuzborskij, S. Elsig, A.-G. Mittaz Hager, O. Deriaz, C. Castellini, H. Müller, and B. Caputo,
“Characterization of a benchmark database for myoelectric movement classification,”
IEEE Trans. Neural Syst. Rehabil. Eng., vol. 23, no. 1, pp. 73–83, Jan. 2015,
doi: 10.1109/TNSRE.2014.2328495.

[2] M. Atzori, A. Gijsberts, C. Castellini, B. Caputo, A.-G. Mittaz Hager, S. Elsig, G. Giatsidis, F. Bassetto, and H. Müller,
“Electromyography data for non-invasive naturally-controlled robotic hand prostheses,”
Sci. Data, vol. 1, Art. no. 140053, Dec. 2014,
doi: 10.1038/sdata.2014.53.

[3] M. Atzori, A. Gijsberts, S. Heynen, A.-G. Mittaz-Hager, O. Deriaz,
P. v. d. Smagt, C. Castellini, B. Caputo, and H. Müller, “Building
the NINAPRO database: A resource for the biorobotics community,”
in Proc. IEEE Int. Conf. Biomed. Robot. Biomechatron., 2012, pp. 1258–1265.
