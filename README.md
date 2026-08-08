# Composite Girder Surrogate

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20195640.svg)](https://doi.org/10.5281/zenodo.20195640)

Code, model weights, validation artifacts, and reproduction scripts for
*"Stiffness Deviation of the AASHTO Transformed-Section Method Under
Partial Composite Action and a Neural-Network Surrogate for Composite
Steel Bridge Girders"* by Sandesh Lamsal.

The study quantifies how far the AASHTO transformed-section method
departs from a fiber-section reference when composite action is partial,
and releases a surrogate that reproduces the reference at a fraction of
the cost.

## Contents

### Section-level model and datasets

- Latin Hypercube sampler, fiber-section builder, moment–curvature driver
- AASHTO transformed-section comparator
- Nie–Cai slip-corrected analytical comparator

### Beam-level model

- Two-chord model with discrete `zeroLength` connector springs, in which
  the degree of composite action emerges from connector equilibrium
  rather than being imposed
- Force-based and displacement-based connector chains, reported side by
  side

### Surrogate

- Residual MLP, 662,530 parameters, two outputs (neutral-axis depth and
  curvature)
- Trained weights (`weights/best.pt`, epoch 296)
- MC-Dropout uncertainty wrapper

### Comparison and validation

- Baselines: plain MLP, tuned XGBoost, sparse variational GP, deep
  ensemble, and a physics-loss ablation
- Validation against published composite-beam tests, including the
  Sheehan et al. (2018) beam at a degree of shear connection η = 0.33
- Per-specimen outputs under `reports/model_validation/`

### Figures

- Every manuscript figure regenerated from one shared style module
  (`src/utils/figstyle.py`), drawn at final printed width, with a
  minimum type size audit and greyscale-safe encoding

## Datasets

Two full databases back the results, and neither is shipped here:

| database | sections | reinforcement |
| --- | --- | --- |
| `data/raw/full_50k.parquet` | 48,917 | none |
| `data/raw/full_50k_rebar007.parquet` | 48,902 | ρℓ = 0.7 % |

Both are regenerated deterministically from the released code in roughly
twelve minutes each on a single CPU core:

```bash
python scripts/generate_dataset.py                          # unreinforced
python scripts/generate_dataset.py --deck-rho-long 0.007    # reinforced
```

A 100-section subset (`data/sample/smoke_100.parquet`) ships for
smoke-testing.

## Installation

```bash
pip install -r requirements.txt
```

OpenSeesPy on Apple Silicon requires an x86 (Rosetta) Python.

## Reproducing the paper

See [`docs/reproducing_figures.md`](docs/reproducing_figures.md) for the
full mapping of scripts to figures and tables.

Quick smoke test:

```bash
python scripts/train_surrogate.py --config configs/training.yaml \
    --data data/sample/smoke_100.parquet \
    --out checkpoints/smoke/
```

## Repository layout

```text
.
├── src/
│   ├── data_generation/   LHS sampler, section builder, M–φ driver
│   ├── models/            Residual-MLP architecture and inference wrapper
│   ├── physics/           Soft-physics loss terms
│   ├── validation/        AASHTO, Nie–Cai, and Sheehan comparators
│   ├── beam_model.py      Two-chord beam model with discrete connectors
│   └── utils/             Normaliser, plotting, and figure style module
├── scripts/
│   ├── figs/              One script per manuscript figure
│   ├── run_*.py           Training and baseline comparisons
│   └── validate_*.py      AASHTO, Nie–Cai, experimental, connector chains
├── reports/
│   └── model_validation/  Per-specimen validation outputs (CSV, JSON)
├── configs/               YAML configs (data_gen.yaml, training.yaml)
├── data/
│   ├── experimental/      Digitised published test data
│   └── sample/            100-section smoke-test subset
├── weights/
│   ├── best.pt            Deployed surrogate (662,530 parameters)
│   └── history.json       Training-loss history
├── docs/                  Figure-by-figure reproduction guide
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

Paper:

```bibtex
@article{Lamsal2026CompositeGirderStiffness,
  author  = {Lamsal, Sandesh},
  title   = {Stiffness Deviation of the {AASHTO} Transformed-Section
             Method Under Partial Composite Action and a
             Neural-Network Surrogate for Composite Steel Bridge
             Girders},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Code archive (Zenodo). This is the *concept* DOI: it always resolves to
the latest version.

```bibtex
@software{Lamsal2026CompositeGirder,
  author    = {Lamsal, Sandesh},
  title     = {Composite Girder Surrogate: code, weights, and
               reproduction scripts},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20195640},
  url       = {https://doi.org/10.5281/zenodo.20195640}
}
```

## License

MIT License. See [`LICENSE`](LICENSE).
