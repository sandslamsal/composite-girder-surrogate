# Table tab:sensitivity re-binned by emergent eta_c

Same 400 sections, same saved beam runs, same `summarise()` code; only the bin label per section changes.

* Pearson r(eta_c_target, eta_c_emergent) = 0.79 (Spearman 0.77)
* Pearson r(eta_c_layout, eta_c_emergent) = 0.81 (Spearman 0.79)
* Emergent values above 1.0 clipped into the 90-100% bin: 1
* Sections with emergent eta_c < 0.25 (outside all bins, dropped): 10
* Sections per emergent bin: 25-50% n=150, 50-70% n=121, 70-90% n=106, 90-100% n=13

**Bins with fewer than 20 sections (emergent assignment): 90-100%**

## Crosstab: target bin (rows) vs emergent bin (columns)

| target bin \ emergent bin | 25-50% | 50-70% | 70-90% | 90-100% | <25% (dropped) | total |
|---|---|---|---|---|---|---|
| 25-50% | 85 | 5 | 0 | 0 | 10 | 100 |
| 50-70% | 33 | 67 | 0 | 0 | 0 | 100 |
| 70-90% | 15 | 35 | 49 | 1 | 0 | 100 |
| 90-100% | 17 | 14 | 57 | 12 | 0 | 100 |
| total | 150 | 121 | 106 | 13 | 10 | 400 |

## Bin means (%) under both assignments

Mean ± section-clustered bootstrap SE (500 resamples). Shift = emergent minus target. Gap = section_matched minus beam_defl.

| bin | regime | n (target / emergent) | metric | target assignment | emergent assignment | shift |
|---|---|---|---|---|---|---|
| 25-50% | service | 100 / 150 | section_matched | +17.26 ± 0.81 | +11.48 ± 0.81 | -5.79 |
|  |  |  | beam_defl | +9.07 ± 0.99 | +8.24 ± 0.84 | -0.83 |
|  |  |  | beam_curv | +13.24 ± 0.96 | +12.58 ± 0.78 | -0.65 |
| | | | gap (sec - beam_defl) | +8.19 | +3.23 | -4.96 |
| 25-50% | extended | 100 / 150 | section_matched | +24.86 ± 4.77 | +23.31 ± 4.66 | -1.56 |
|  |  |  | beam_defl | +11.56 ± 0.98 | +11.27 ± 1.05 | -0.29 |
|  |  |  | beam_curv | +16.87 ± 1.35 | +16.63 ± 1.31 | -0.25 |
| | | | gap (sec - beam_defl) | +13.30 | +12.03 | -1.27 |
| 50-70% | service | 100 / 121 | section_matched | +7.30 ± 0.47 | +4.45 ± 0.51 | -2.85 |
|  |  |  | beam_defl | +3.22 ± 0.67 | +2.28 ± 0.65 | -0.94 |
|  |  |  | beam_curv | +7.52 ± 0.64 | +6.56 ± 0.60 | -0.96 |
| | | | gap (sec - beam_defl) | +4.08 | +2.17 | -1.91 |
| 50-70% | extended | 100 / 121 | section_matched | +13.42 ± 3.40 | +5.35 ± 0.54 | -8.07 |
|  |  |  | beam_defl | +6.04 ± 1.04 | +3.88 ± 0.68 | -2.17 |
|  |  |  | beam_curv | +10.50 ± 1.14 | +8.12 ± 0.63 | -2.38 |
| | | | gap (sec - beam_defl) | +7.37 | +1.47 | -5.90 |
| 70-90% | service | 100 / 106 | section_matched | +1.41 ± 0.47 | -0.37 ± 0.48 | -1.78 |
|  |  |  | beam_defl | +2.89 ± 0.87 | +1.53 ± 0.61 | -1.36 |
|  |  |  | beam_curv | +6.97 ± 0.81 | +5.38 ± 0.57 | -1.60 |
| | | | gap (sec - beam_defl) | -1.48 | -1.90 | -0.42 |
| 70-90% | extended | 100 / 106 | section_matched | +7.94 ± 4.75 | +18.02 ± 8.52 | +10.08 |
|  |  |  | beam_defl | +5.46 ± 1.28 | +5.87 ± 1.34 | +0.41 |
|  |  |  | beam_curv | +9.75 ± 1.47 | +10.79 ± 1.85 | +1.04 |
| | | | gap (sec - beam_defl) | +2.48 | +12.15 | +9.67 |
| 90-100% | service | 100 / 13 | section_matched | -1.91 ± 0.53 | +0.06 ± 1.28 | +1.97 |
|  |  |  | beam_defl | +3.09 ± 0.97 | +2.27 ± 1.45 | -0.82 |
|  |  |  | beam_curv | +7.14 ± 0.87 | +4.40 ± 1.06 | -2.73 |
| | | | gap (sec - beam_defl) | -5.01 | -2.21 | +2.80 |
| 90-100% | extended | 100 / 13 | section_matched | +19.10 ± 8.78 | +5.41 ± 3.23 | -13.69 |
|  |  |  | beam_defl | +7.48 ± 1.63 | +4.68 ± 1.80 | -2.80 |
|  |  |  | beam_curv | +12.73 ± 2.03 | +7.69 ± 1.83 | -5.04 |
| | | | gap (sec - beam_defl) | +11.62 | +0.73 | -10.89 |
