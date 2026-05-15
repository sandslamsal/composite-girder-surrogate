# Dataset

The full dataset of **48,917 OpenSeesPy fibre-section analyses**
(3.9 × 10⁶ rows) used to train the surrogate and to drive the AASHTO
and Nie–Cai comparisons is archived alongside this repository's code
and trained weights at:

> **Mendeley Data DOI**: [10.17632/zjzyz6nrh5.1](https://doi.org/10.17632/zjzyz6nrh5.1)

The Parquet file is approximately 200 MB compressed and lives in the
Mendeley Data record rather than in this Git repository.

## Files in this folder

- `sample/smoke_100.parquet` — 100-section subset (≈ 8,000 rows) for
  smoke-testing the pipeline end-to-end.
- `experimental/literature_tests.csv` — digitised values from the
  three composite-beam test programmes referenced in the Limitations
  section of the paper (Chapman & Balakrishnan 1964; Nie & Cai 2003;
  Ansourian 1982). These lab tests are below the bridge-girder
  training distribution and are not used for surrogate evaluation.
- `feature_descriptions.md` — full column-by-column schema of the
  full dataset (14 continuous inputs, 1 categorical section type,
  4 regression targets, 3 bookkeeping columns) with units,
  physical meaning, and LHS sampling ranges.
- `lhs_sampling_info.txt` — Latin Hypercube Sampling parameters
  (random seed, validity-check pass rate, total row count).

## Regenerating the full dataset

If you have OpenSeesPy installed (see `requirements.txt`), the full
dataset can be regenerated from scratch on a single CPU core in
roughly 12 wall-clock minutes:

```bash
python scripts/generate_dataset.py \
    --config configs/data_gen.yaml \
    --n 50000 \
    --out data/raw/full_50k.parquet
```

Out of 50,000 LHS samples, approximately 48,917 (97.8 %) pass the
validity check (Newton convergence at every curvature increment and
a neutral axis within the section under elastic loading). The
resulting file has approximately 3.9 × 10⁶ rows.

## Loading

```python
import pandas as pd
df = pd.read_parquet("data/raw/full_50k.parquet")
```

## Citation

See the top-level [`README.md`](../README.md). The dataset, code,
and trained weights all share the single Mendeley Data DOI
[10.17632/zjzyz6nrh5.1](https://doi.org/10.17632/zjzyz6nrh5.1).
