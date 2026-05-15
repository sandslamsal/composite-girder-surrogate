# Dataset

The full dataset of **48,917 OpenSeesPy fibre-section analyses** (3.9 × 10⁶ rows)
used to train the surrogate and to drive the AASHTO and Nie–Cai comparisons
is archived separately at:

> **Zenodo DOI**: *(to be added upon publication)*

The full dataset is approximately 200 MB (Parquet) and is too large to
distribute in this repository.

## What is included here

A small subset is shipped in `data/sample/` for smoke-testing the pipeline:

- `smoke_100.parquet` — 100 sections × 80 curvature increments
  (≈ 8,000 rows). Sufficient to verify the training pipeline runs
  end-to-end and produces sensible loss curves; not large enough to
  train a useful surrogate.

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

Out of the 50,000 LHS samples, approximately 48,917 (97.8 %) pass the
validity check (Newton convergence at every curvature increment and a
neutral axis within the section under elastic loading). The resulting
file has approximately 3.9 × 10⁶ rows.

## Schema

Each row corresponds to one composite section at one curvature
increment. Columns:

- `sample_id` — integer identifier shared by the 80 rows of a section
- `section_type` — `W` or `plate`
- `span_in`, `deck_thickness_in`, `deck_width_in`, `girder_spacing_in` — geometry
- `fc_deck_ksi`, `fy_ksi` — material strengths
- `composite_action` ($\eta_c$), `shear_stud_stiffness_ratio` ($K_s$)
- `steel_depth_in`, `flange_width_in`, `flange_thickness_in`, `web_thickness_in`
- `total_depth_in`, `mp_estimate_kip_in`, `step_index`, `moment_ratio` ($M/M_p$)
- `curvature_1_per_in` ($\varphi$, target)
- `moment_kip_in` ($M$, derived)
- `axial_strain` ($\varepsilon_0$, OpenSees zero-strain reference)
- `neutral_axis_in` ($y_{na}$, target)
- `slip_in` ($\delta$, target — analytical surrogate)

## Experimental data

`data/experimental/literature_tests.csv` contains digitised values from
the three composite-beam test programmes referenced in the manuscript
(Chapman & Balakrishnan 1964; Nie & Cai 2003; Ansourian 1982). See the
manuscript Limitations section for the discussion of why these
laboratory-scale tests fall below the bridge-girder training
distribution.
