# Neural-Network Surrogate and AASHTO Stiffness Quantification for Composite Bridge Girders

**Author:** Sandesh Lamsal

This repository contains the code, trained model weights, and reproduction
scripts for the manuscript *"Neural-Network Surrogate and AASHTO Stiffness
Quantification for Composite Bridge Girders."* It includes the OpenSeesPy
fibre-section data-generation pipeline, the surrogate architecture and
training scripts, MC-Dropout uncertainty inference (plus a deep-ensemble training script for follow-up work), and the
AASHTO and Nie–Cai validation comparators that produce every figure and
table in the paper.

## Installation

```bash
pip install -r requirements.txt
```

OpenSeesPy on Apple Silicon requires an x86 (Rosetta) Python. The
environment used for the published results is recorded in
`environment.yml`.

## Dataset

The full dataset of 48,917 OpenSeesPy fibre-section analyses
(3.9 × 10⁶ rows) is archived separately at
**Zenodo DOI: *(to be added upon publication)***. A 100-section
smoke-test subset is shipped in `data/sample/` for verifying the
pipeline runs end-to-end. See `data/README.md` for details.

## Reproducing the paper

| Output | Script | Notes |
|---|---|---|
| Trained surrogate | `scripts/train_pinn.py --config configs/training.yaml --data data/raw/full_50k.parquet --out checkpoints/run/` | Reproduces the headline model (`weights/best.pt`) |
| Deep-ensemble training script | `scripts/train_ensemble.py --members 5 --data data/raw/full_50k.parquet --out checkpoints/ensemble/` | Provided for follow-up work; members were not pre-trained for the paper (paper uses MC-Dropout for uncertainty) |
| ML baselines (Table 3) | `scripts/run_baselines_200k.py` | XGBoost + plain MLP + residual MLP at 200k subsample |
| Fig. 2 — Training curves | `scripts/make_figures.py --history weights/history.json` | Loss curves with best-epoch marker |
| Fig. 3 — Parity plots (Tier 1, Table 2) | `scripts/make_figures.py --checkpoint weights/best.pt --data data/raw/full_50k.parquet` | All four targets, hexbin density |
| Fig. 4 — Error-distribution violins | `scripts/make_figures.py` (same call) | Per-target relative-error distribution |
| Fig. 5 — AASHTO over-prediction (Tier 2, Table 4) | `scripts/validate_aashto.py --data data/raw/full_50k.parquet --out reports/aashto_full/` then `scripts/make_figures.py --aashto reports/aashto_full/aashto_comparison.parquet` | 426,025-row comparison |
| Fig. 6 — Neutral-axis migration | `scripts/make_figures.py --checkpoint weights/best.pt --aashto reports/aashto_full/aashto_comparison.parquet` | |
| Fig. 7 — Nie–Cai cross-validation (Table 5) | `scripts/validate_nie_cai.py --data data/raw/full_50k.parquet --aashto reports/aashto_full/aashto_comparison.parquet --out reports/niecai/` then `scripts/make_figures.py --niecai reports/niecai/niecai_comparison.parquet` | Lab-test-calibrated analytical reference |
| Fig. 8 — Moment–curvature reproduction | `scripts/make_figures.py --checkpoint weights/best.pt --data data/raw/full_50k.parquet` | Four representative sections |
| Fig. 9 — MC-Dropout uncertainty band | `scripts/make_figures.py --checkpoint weights/best.pt --mc-samples 50` | Deterministic + uncertainty modes both supported by `src/models/inference.py` |
| Table 6 — Design-recommendation $R_{EI}(\eta_c)$ | Bin-mean values in `reports/aashto_full/aashto_summary.csv` | Service-load and extended-elastic columns |

All figures land in `figures/` (created if missing). Intermediate
parquet/JSON outputs land under `reports/`.

## Repository layout

```text
.
├── src/
│   ├── data_generation/      LHS sampler, section builder, M-phi driver
│   ├── models/               Residual-MLP architecture, training, inference,
│   │                         deep-ensemble aggregator
│   ├── validation/           AASHTO comparator, Nie-Cai comparator,
│   │                         experimental-comparison loader
│   └── utils/                Feature/target normaliser, plotting style
├── scripts/                  Entry-point scripts that reproduce every figure
│                             and every table number
├── configs/                  YAML configs (data_gen.yaml, training.yaml)
├── data/
│   ├── sample/               100-section subset for smoke testing
│   ├── experimental/         Literature beam-test rows (Chapman/Balakrishnan,
│   │                         Nie/Cai, Ansourian)
│   └── README.md             Dataset description; full data at Zenodo
├── weights/                  Trained model state-dict + training history
├── requirements.txt          Pinned Python dependencies
├── LICENSE                   MIT
└── README.md                 (this file)
```

## License

MIT License. See `LICENSE`.

## Citation

A BibTeX entry will be added here upon journal acceptance.
