# Submission Data

This folder organizes all artefacts accompanying the paper
*"Neural-Network Surrogate and AASHTO Stiffness Quantification for
Composite Bridge Girders"* by Sandesh Lamsal.

## Subfolder layout

- `github/` — files that go into the public GitHub repository at
  https://github.com/sandslamsal/composite-girder-surrogate.
  The Zenodo code archive (DOI: 10.5281/zenodo.20195641) was generated
  from a release of this repository.

- `zenodo/` — files for the Zenodo DATASET record (uploaded manually,
  separate from the code archive). Contains the full 3.9M-row dataset
  (and any baseline or follow-up model files too large or too peripheral
  for inclusion in the GitHub repository).

## Workflow

1. Files in `github/` are pushed to the GitHub repo via `git push`.
2. Files in `zenodo/` are uploaded manually at
   https://zenodo.org/uploads/new as a Dataset resource type with
   license CC-BY-4.0.
3. The resulting dataset DOI is added back into both the README on
   GitHub and the manuscript's Data and Code Availability section.
