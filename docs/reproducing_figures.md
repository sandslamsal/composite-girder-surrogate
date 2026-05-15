# Reproducing figures and tables

Every figure and table in the paper is reproduced by an entry-point
script in `scripts/`. All commands assume the working directory is the
repository root and that the full dataset has been placed at
`data/raw/full_50k.parquet` (downloaded from the Zenodo archive at
[10.5281/zenodo.20195641](https://doi.org/10.5281/zenodo.20195641) or
regenerated locally with `scripts/generate_dataset.py`).

A 100-section smoke-test subset is shipped at
`data/sample/smoke_100.parquet`; substitute that path in any command
below to verify the pipeline runs end-to-end.

## Figures

| Figure | Description | Script (run order) |
|---|---|---|
| 2 | Training and validation loss curves with best-epoch marker | `python scripts/make_figures.py --history weights/history.json` |
| 3 | Parity plots for all four targets (Tier 1) | `python scripts/make_figures.py --checkpoint weights/headline_model.pt --data data/raw/full_50k.parquet` |
| 4 | Per-target relative-error violin distributions | `python scripts/make_figures.py --checkpoint weights/headline_model.pt --data data/raw/full_50k.parquet` (same call as Fig. 3) |
| 5 | AASHTO curvature over-prediction by composite-action bin (Tier 2) | `python scripts/validate_aashto.py --data data/raw/full_50k.parquet --out reports/aashto_full/` then `python scripts/make_figures.py --aashto reports/aashto_full/aashto_comparison.parquet` |
| 6 | Neutral-axis migration vs. composite action | `python scripts/make_figures.py --checkpoint weights/headline_model.pt --aashto reports/aashto_full/aashto_comparison.parquet` |
| 7 | Nie–Cai analytical cross-validation | `python scripts/validate_nie_cai.py --data data/raw/full_50k.parquet --aashto reports/aashto_full/aashto_comparison.parquet --out reports/niecai/` then `python scripts/make_figures.py --niecai reports/niecai/niecai_comparison.parquet` |
| 8 | Moment–curvature reproduction for four representative sections | `python scripts/make_figures.py --checkpoint weights/headline_model.pt --data data/raw/full_50k.parquet` |
| 9 | MC-Dropout uncertainty band (T = 50 forward passes) | `python scripts/make_figures.py --checkpoint weights/headline_model.pt --mc-samples 50` |

## Tables

| Table | Description | Source |
|---|---|---|
| 2 | Tier 1 surrogate metrics by target (R², RMSE, MAPE) | Console output of the Fig. 3 / Fig. 4 call to `scripts/make_figures.py` |
| 3 | ML-baseline comparison at 200k subsample (XGBoost vs. plain MLP vs. residual MLP) | `python scripts/run_baselines_200k.py` and `python scripts/run_mlps_200k.py`; metrics written to `reports/baselines/` |
| 4 | AASHTO over-prediction by composite-action bin (Tier 2) | `reports/aashto_full/aashto_summary.csv` produced by `scripts/validate_aashto.py` |
| 5 | Nie–Cai cross-validation against surrogate and OpenSeesPy | `reports/niecai/niecai_summary.csv` produced by `scripts/validate_nie_cai.py` |
| 6 | Stiffness-reduction factor R_EI(η_c) for service-load and extended-elastic regimes | Bin-mean values in `reports/aashto_full/aashto_summary.csv` (service-load and extended-elastic columns) |

## Pre-generated reports

Running the AASHTO and Nie–Cai validators writes intermediate Parquet
and CSV summaries under `reports/`. These are the inputs to the
figure-generation script and can be regenerated end-to-end from the
full dataset.

## Smoke test

To verify the pipeline runs end-to-end on a laptop (no full dataset
required):

```bash
python scripts/train_surrogate.py --config configs/training.yaml \
    --data data/sample/smoke_100.parquet --out checkpoints/smoke/
python scripts/make_figures.py --history checkpoints/smoke/history.json
```

The smoke run will not reproduce paper-quality metrics; it only
verifies that the data pipeline, training loop, and figure-generation
code execute without error.
