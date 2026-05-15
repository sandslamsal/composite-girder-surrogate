Composite Girder Fibre-Section Dataset
======================================

Author : Sandesh Lamsal
License: Creative Commons Attribution 4.0 International (CC-BY-4.0)

This Zenodo record accompanies the paper

    Sandesh Lamsal, "Neural-Network Surrogate and AASHTO Stiffness
    Quantification for Composite Bridge Girders," 2026.

and supplements the code archive at

    https://doi.org/10.5281/zenodo.20195641

The code archive (above) contains the surrogate architecture,
training and inference scripts, a 100-section smoke-test subset, and
the trained headline-model weights. This DATASET record (the one you
are reading) contains the artefacts that are too large for the
GitHub repository: the full 3.9M-row training dataset and (optionally)
trained baseline models used for Table 3 in the paper.

Linked records
--------------
- Source code (GitHub):
      https://github.com/sandslamsal/composite-girder-surrogate
- Source code (Zenodo archive):
      https://doi.org/10.5281/zenodo.20195641

File-by-file contents
---------------------

dataset/full_dataset.parquet
    Full LHS-sampled training dataset. 48,917 valid composite-girder
    sections, 80 monotonic-curvature increments each, for a total of
    3,912,852 rows. Parquet format with Snappy compression, ~200 MB
    on disk. Column schema is documented in
    dataset/feature_descriptions.md. The Latin Hypercube Sampling
    parameters (random seed, validity-check pass rate) are documented
    in dataset/lhs_sampling_info.txt.

dataset/feature_descriptions.md
    Column-by-column schema of full_dataset.parquet: 14 continuous
    inputs, 1 categorical section type (string), 4 regression targets,
    and 3 bookkeeping columns. Units, physical meaning, and LHS
    sampling ranges are listed for every column.

dataset/lhs_sampling_info.txt
    LHS sampling parameters: number of samples drawn, validity-check
    pass rate, final number of valid sections, total row count, and
    the random seed used.

baselines/xgboost_model.json (optional)
    Trained XGBoost regressor used for the ML-baseline comparison in
    Table 3 (200k-row subsample). Native XGBoost JSON serialisation.
    If absent from this record, re-run
    `scripts/run_baselines_200k.py` from the code archive to
    regenerate it.

baselines/plain_mlp_weights.pt (optional)
    Trained plain (non-residual) MLP used for the second ML-baseline
    row in Table 3 (200k-row subsample). PyTorch state-dict. If
    absent, re-run `scripts/run_mlps_200k.py` from the code archive.

Note: the trained surrogate weights (`weights/headline_model.pt`) live
in the code archive and are sufficient to reproduce Tables 2, 4, 5, 6
and Figures 2-9 once the dataset above is paired with the code.

Loading the data
----------------
Python:

    import pandas as pd
    df = pd.read_parquet("dataset/full_dataset.parquet")

Loading the XGBoost baseline (if shipped):

    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model("baselines/xgboost_model.json")

Citation
--------
If you use this dataset, please cite:

    @dataset{Lamsal2026CompositeGirderDataset,
      author    = {Lamsal, Sandesh},
      title     = {Composite Girder Fibre-Section Dataset},
      year      = {2026},
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.PLACEHOLDER_DATASET_DOI}
    }
