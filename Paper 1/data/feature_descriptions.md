# Feature descriptions — full_dataset.parquet

Each row corresponds to one composite section at one monotonic-curvature
increment. The 80 rows that share the same `sample_id` form a single
moment–curvature curve.

Units follow the kip–inch system unless explicitly noted.

## Identifier / bookkeeping columns

| Column | Type | Description |
|---|---|---|
| `sample_id` | int64 | Section identifier shared by the 80 rows of one moment–curvature curve. |
| `step_index` | int32 | Curvature-increment index along the curve (0 to 79). |
| `section_type` | string | `W` (hot-rolled wide-flange + concrete deck) or `plate` (welded plate girder + concrete deck). One-hot encoded inside the model. |

## Continuous input features (sampled by Latin Hypercube)

| Column | Symbol | Unit | LHS range | Description |
|---|---|---|---|---|
| `span_in` | $L$ | in | 240 – 2880 | Simple-span length. Sampled in ft (20–240) then converted. |
| `deck_thickness_in` | $t_s$ | in | 4.5 – 12.0 | Concrete deck slab thickness. |
| `deck_width_in` | $b_{\text{eff}}$ | in | derived | Effective deck width. Derived as `deck_width_ratio × girder_spacing_in`. |
| `girder_spacing_in` | $S$ | in | 60 – 144 | Centre-to-centre girder spacing. Sampled in ft (5–12) then converted. |
| `fc_deck_ksi` | $f_c'$ | ksi | 3.0 – 10.0 | Deck-concrete cylinder strength. |
| `fy_ksi` | $f_y$ | ksi | 36 – 100 | Steel yield strength. |
| `composite_action` | $\eta_c$ | – | 0.25 – 1.0 | Degree of composite action. Used to scale the AISC effective deck width. |
| `shear_stud_stiffness_ratio` | $K_s$ | – | 0.10 – 1.00 | Normalised shear-stud spring stiffness. |
| `steel_depth_in` | $d_s$ | in | 12 – 84 | Total steel-section depth. |
| `flange_width_in` | $b_f$ | in | derived | Steel-flange width. Derived as `flange_width_ratio × steel_depth_in` with LHS ratio in 0.30–0.70. |
| `flange_thickness_in` | $t_f$ | in | derived | Steel-flange thickness. Derived as `flange_thickness_ratio × flange_width_in` with LHS ratio in 0.04–0.12. |
| `web_thickness_in` | $t_w$ | in | derived | Steel-web thickness. Derived as `web_thickness_ratio × steel_depth_in` with LHS ratio in 0.015–0.040. |
| `total_depth_in` | $h$ | in | derived | `steel_depth_in + deck_thickness_in`. Convenience feature. |
| `mp_estimate_kip_in` | $M_{p,\text{est}}$ | kip·in | derived | Plastic-moment estimate from transformed-section bookkeeping; used as a non-dimensionalising scale for the moment target. |

## Curvature-sweep variable

| Column | Symbol | Unit | Description |
|---|---|---|---|
| `moment_ratio` | $M/M_p$ | – | Normalised target moment for the current step. Curvature is incremented until either the moment ratio reaches 1.2 or Newton iteration fails to converge. |

## Regression targets (predicted by the surrogate)

| Column | Symbol | Unit | Description |
|---|---|---|---|
| `curvature_1_per_in` | $\varphi$ | 1/in | Section curvature returned by OpenSeesPy. Primary target. |
| `moment_kip_in` | $M$ | kip·in | Section moment at the current curvature step. Derived from `moment_ratio × mp_estimate_kip_in`. |
| `axial_strain` | $\varepsilon_0$ | – | Zero-strain reference returned by the OpenSeesPy zero-length section. |
| `neutral_axis_in` | $y_{\text{na}}$ | in | Distance from the deck top to the elastic neutral axis (positive into the steel section). |
| `slip_in` | $\delta$ | in | Slip at the deck–steel interface. Analytical surrogate derived from `composite_action` and `shear_stud_stiffness_ratio` (not from OpenSeesPy). |

## Notes

- All four targets share the same 80-row curve structure; for the
  parity-plot evaluation in Fig. 3, predictions are made independently
  at each `(sample_id, step_index)` pair.
- The dataset is split 80/10/10 train/val/test by `sample_id` (never by
  row), so all 80 increments of a given section stay in the same fold.
  The split is reproducible from random seed `20260513`.
- The data-generation random seed (LHS) is recorded in
  `lhs_sampling_info.txt`.
