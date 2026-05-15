# Composite Girder Surrogate

[![DOI](https://img.shields.io/badge/Mendeley%20Data-10.17632%2Fzjzyz6nrh5.1-blue)](https://doi.org/10.17632/zjzyz6nrh5.1)

Code, model weights, and reproduction scripts for the paper
*"Neural-Network Surrogate and AASHTO Stiffness Quantification for
Composite Bridge Girders"* by Sandesh Lamsal.

## Contents

- Surrogate architecture (residual MLP, 663,556 parameters)
- Training and inference scripts
- Trained model weights (headline model, `weights/headline_model.pt`)
- MC-Dropout uncertainty wrapper
- AASHTO transformed-section comparator
- Nie–Cai analytical comparator
- Sample dataset (100 sections) for smoke-testing

The full dataset (48,917 sections, 3.9 × 10⁶ rows, ~200 MB) is
archived together with the code and trained weights on Mendeley Data
at [10.17632/zjzyz6nrh5.1](https://doi.org/10.17632/zjzyz6nrh5.1).

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
│   ├── validation/        AASHTO, Nie–Cai, and beam-level comparators
│   └── utils/             Normaliser and plotting style
├── scripts/               Entry-point scripts (see docs/reproducing_figures.md)
├── configs/               YAML configs (data_gen.yaml, training.yaml)
├── data/
│   └── sample/            100-section smoke-test subset
├── weights/
│   ├── headline_model.pt  Trained surrogate state-dict
│   └── history.json       Training-loss history
├── docs/                  Figure-by-figure reproduction guide
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

Paper (preprint):

```bibtex
@article{Lamsal2026CompositeGirderSurrogate,
  author  = {Lamsal, Sandesh},
  title   = {Neural-Network Surrogate and AASHTO Stiffness
             Quantification for Composite Bridge Girders},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Archive (Mendeley Data) — code, weights, and full dataset under one DOI:

```bibtex
@dataset{Lamsal2026CompositeGirder,
  author    = {Lamsal, Sandesh},
  title     = {Composite Girder Surrogate: Code, Dataset, and
               Trained Model Weights for AASHTO Stiffness
               Quantification},
  year      = {2026},
  publisher = {Mendeley Data},
  version   = {V1},
  doi       = {10.17632/zjzyz6nrh5.1},
  url       = {https://doi.org/10.17632/zjzyz6nrh5.1}
}
```

## License

MIT License. See [`LICENSE`](LICENSE).
