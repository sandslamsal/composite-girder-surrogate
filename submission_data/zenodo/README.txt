Composite Girder Fibre-Section Dataset and Trained Model Weights
================================================================

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
are reading) contains the artefacts that are too large or peripheral
for the GitHub repository: the full 3.9M-row training dataset, the
5-member deep ensemble weights, and the trained baseline models used
for Table 3 in the paper.

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

weights/ensemble/ensemble_member_{1..5}.pt
    Five independently trained residual-MLP members of the deep
    ensemble described in the manuscript. The members were NOT
    pre-trained for the present study (the paper's reported
    uncertainty band uses MC-Dropout, see Fig. 9); these weights are
    provided here for follow-up work and as the recommended drop-in
    replacement for deployment-critical applications. Each file is a
    PyTorch state-dict and is loaded via the deep-ensemble training
    script `scripts/train_ensemble.py` in the linked code repository.

    NOTE: at the time of this upload, no ensemble members have yet
    been generated. The folder is empty.

baselines/xgboost_model.json
    Trained XGBoost regressor used for the ML-baseline comparison in
    Table 3 (200k-row subsample). Native XGBoost JSON serialisation.

baselines/plain_mlp_weights.pt
    Trained plain (non-residual) MLP used for the second ML-baseline
    row in Table 3 (200k-row subsample). PyTorch state-dict.

    NOTE: if the baselines/ subfolder is empty in this record, the
    baseline weights were not regenerated for archival; the trained
    surrogate (headline_model.pt in the code archive) is sufficient
    to reproduce Tables 2, 4, 5, 6 and Figures 2-9. Table 3 requires
    re-running `scripts/run_baselines_200k.py` and
    `scripts/run_mlps_200k.py` from the code archive.

Loading the data
----------------
Python:

    import pandas as pd
    df = pd.read_parquet("dataset/full_dataset.parquet")

Loading the ensemble:

    import torch
    w = torch.load("weights/ensemble/ensemble_member_1.pt",
                   map_location="cpu")

Loading the XGBoost baseline:

    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model("baselines/xgboost_model.json")

Citation
--------
If you use this dataset, please cite:

    @dataset{Lamsal2026CompositeGirderDataset,
      author    = {Lamsal, Sandesh},
      title     = {Composite Girder Fibre-Section Dataset and
                   Trained Model Weights},
      year      = {2026},
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.PLACEHOLDER_DATASET_DOI}
    }
