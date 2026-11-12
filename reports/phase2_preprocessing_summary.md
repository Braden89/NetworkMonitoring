# Phase 2 preprocessing summary

Generated UTC: 2026-11-12T04:20:28.849380+00:00
Dataset revision: `b33f3052398ffd63463ce3120fec2acbb68e2ff6`

Uniform random priority reservoir per raw label across every CSV. Class-capped development sample; not natural traffic prevalence.
These are development-sample statistics, not natural traffic prevalence or model evaluation results.

## Cleaning and split policy

Trim headers and labels; normalize observed web-attack separators. Drop the verified redundant Fwd Header Length.1 column. Coerce nonnumeric/infinite features to missing; treat -1 in the two initial TCP-window columns as missing. Other negative values remain for domain review.
Remove exact duplicate cleaned feature rows and exclude conflicting-label feature groups before splitting. Use seeded per-class splits of approximately 70/15/15. Classes with fewer than three distinct rows remain training-only.
Fit median imputation, constant-feature removal, and standard scaling on training only. Drop transformed holdout duplicates (including overlaps caused by imputation or constant removal). Label encoding is fitted on training labels. Binary labels: BENIGN=0, attack=1.

## Observed results

Source rows scanned: 2,830,743
Sampled rows: 49,193
Conflicting-label rows excluded: 2
Repeated feature rows removed: 5,290
Feature count after preprocessing: 65
Training missing cells after stateless cleaning (before imputation): 9,132
Other negative training cells retained for domain review: 22
Training-only classes: []
Transformed overlap removals: {'train': 0, 'validation': 0, 'test': 0}
Constant/all-missing training features removed: ['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'RST Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']

| Label | Train | Validation | Test |
|---|---:|---:|---:|
| BENIGN | 3473 | 744 | 744 |
| Bot | 1369 | 292 | 292 |
| DDoS | 3500 | 750 | 750 |
| DoS GoldenEye | 3500 | 749 | 749 |
| DoS Hulk | 2737 | 586 | 586 |
| DoS Slowhttptest | 3334 | 714 | 714 |
| DoS slowloris | 3259 | 697 | 697 |
| FTP-Patator | 2699 | 577 | 577 |
| Heartbleed | 9 | 1 | 1 |
| Infiltration | 26 | 5 | 5 |
| PortScan | 3403 | 729 | 729 |
| SSH-Patator | 1933 | 414 | 414 |
| Web Attack - Brute Force | 1030 | 220 | 220 |
| Web Attack - Sql Injection | 15 | 3 | 3 |
| Web Attack - XSS | 458 | 97 | 97 |

## Validation and limits

All saved feature matrices are finite. Source record IDs are disjoint across splits. No exact transformed feature rows cross splits. Artifacts are reload-checked against the saved raw test split.
Classes absent from validation/test after overlap purging: {'validation': [], 'test': []}
Rare classes with tiny holdouts do not support reliable per-class estimates. Per-class capping changes prevalence; future results must be labeled as a development benchmark. Duplicate checks apply to retained sample records, not every source record.
Random within-class splitting does not eliminate host, session, or temporal similarity. This mirror lacks the IDs and timestamps needed for strong session/time isolation. Source files are retained only as provenance. Future cross-file evaluation needs explicit handling of classes absent from training.
For future cross-validation, fit a fresh preprocessing pipeline inside each fold using the saved raw training features. Do not reuse these globally training-fitted matrices across folds.
No classifier was fitted. No validation/test metrics were computed.

See `data/processed/phase2/summary.json` for parameters, package versions, counts, and label mappings.
