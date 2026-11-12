"""Create reproducible Phase 2 artifacts without training a classifier."""
import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

from src.data_loader import ROOT
from src.preprocessing import (PROVENANCE, build_preprocessor, deduplicate_sample,
                               normalize_headers, purge_transformed_overlap, split_by_class, clean_numeric)
from src.sampling import sample_all_files


def write_summary(summary, path):
    lines = ['# Phase 2 preprocessing summary', '',
             f"Generated UTC: {summary['generated_utc']}",
             f"Dataset revision: `{summary['sampling']['metadata']['revision']}`", '',
             summary['sampling']['sampling'],
             'These are development-sample statistics, not natural traffic prevalence or model evaluation results.', '',
             '## Cleaning and split policy', '',
             'Trim headers and labels; normalize observed web-attack separators. Drop the verified redundant Fwd Header Length.1 column. Coerce nonnumeric/infinite features to missing; treat -1 in the two initial TCP-window columns as missing. Other negative values remain for domain review.',
             'Remove exact duplicate cleaned feature rows and exclude conflicting-label feature groups before splitting. Use seeded per-class splits of approximately 70/15/15. Classes with fewer than three distinct rows remain training-only.',
             'Fit median imputation, constant-feature removal, and standard scaling on training only. Drop transformed holdout duplicates (including overlaps caused by imputation or constant removal). Label encoding is fitted on training labels. Binary labels: BENIGN=0, attack=1.', '',
             '## Observed results', '',
             f"Source rows scanned: {sum(summary['sampling']['source_file_counts'].values()):,}",
             f"Sampled rows: {summary['cleaning']['input_records']:,}",
             f"Conflicting-label rows excluded: {summary['cleaning']['conflicting_label_rows_removed']:,}",
             f"Repeated feature rows removed: {summary['cleaning']['duplicate_feature_rows_removed']:,}",
             f"Feature count after preprocessing: {len(summary['feature_names'])}",
             f"Training missing cells after stateless cleaning (before imputation): {summary['training_quality']['missing_cells_after_stateless_cleaning']:,}",
             f"Other negative training cells retained for domain review: {summary['training_quality']['remaining_negative_cells']:,}",
             f"Training-only classes: {summary['train_only_classes']}",
             f"Transformed overlap removals: {summary['transformed_overlap_removed']}",
             f"Constant/all-missing training features removed: {summary['removed_constant_features']}", '',
             '| Label | Train | Validation | Test |', '|---|---:|---:|---:|']
    for label in summary['label_mapping']:
        lines.append('| ' + label + ' | ' + ' | '.join(str(summary['splits'][k]['classes'].get(label, 0)) for k in ('train', 'validation', 'test')) + ' |')
    lines += ['', '## Validation and limits', '',
              'All saved feature matrices are finite. Source record IDs are disjoint across splits. No exact transformed feature rows cross splits. Artifacts are reload-checked against the saved raw test split.',
              f"Classes absent from validation/test after overlap purging: {summary['missing_holdout_classes']}",
              'Rare classes with tiny holdouts do not support reliable per-class estimates. Per-class capping changes prevalence; future results must be labeled as a development benchmark. Duplicate checks apply to retained sample records, not every source record.',
              'Random within-class splitting does not eliminate host, session, or temporal similarity. This mirror lacks the IDs and timestamps needed for strong session/time isolation. Source files are retained only as provenance. Future cross-file evaluation needs explicit handling of classes absent from training.',
              'For future cross-validation, fit a fresh preprocessing pipeline inside each fold using the saved raw training features. Do not reuse these globally training-fitted matrices across folds.',
              'No classifier was fitted. No validation/test metrics were computed.', '',
              'See `data/processed/phase2/summary.json` for parameters, package versions, counts, and label mappings.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def prepare(per_class=5000, seed=42, revision=None, refresh_sample=False):
    if per_class < 1:
        raise ValueError('per_class must be positive')
    sample_path = ROOT / 'data/raw/phase2_sample.parquet'
    provenance_path = sample_path.with_suffix('.json')
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    reuse = (not refresh_sample and sample_path.exists() and provenance is not None
             and provenance['seed'] == seed and provenance['per_class'] == per_class
             and (revision is None or provenance['metadata']['revision'] == revision))
    if reuse:
        print(f"Using cached sample at revision {provenance['metadata']['revision']}", flush=True)
        frame = pd.read_parquet(sample_path)
    else:
        frame, provenance = sample_all_files(per_class, seed, revision)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(sample_path, index=False)
        provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    frame = normalize_headers(frame)
    original_counts = frame['Label'].value_counts().to_dict()
    clean, cleaning = deduplicate_sample(frame)
    if clean['Label'].nunique() < 2 or 'BENIGN' not in clean['Label'].values:
        raise ValueError('Need BENIGN and attack records after cleaning')
    splits, train_only = split_by_class(clean, seed)
    if any(part.empty for part in splits.values()):
        raise ValueError('Sample too small for nonempty train/validation/test splits; increase --per-class')
    feature_columns = [c for c in clean if c not in ['Label', *PROVENANCE]]
    training_features = splits['train'][feature_columns]
    numeric_training = clean_numeric(training_features)
    training_quality = {
        'missing_cells_after_stateless_cleaning': int(numeric_training.isna().sum().sum()),
        'rows_with_missing_after_stateless_cleaning': int(numeric_training.isna().any(axis=1).sum()),
        'remaining_negative_cells': int((numeric_training < 0).sum().sum()),
        'missing_by_feature': {c: int(n) for c, n in numeric_training.isna().sum().items() if n},
    }
    pipeline = build_preprocessor()
    arrays = {'train': pipeline.fit_transform(splits['train'][feature_columns])}
    arrays.update({k: pipeline.transform(splits[k][feature_columns]) for k in ('validation', 'test')})
    purged = purge_transformed_overlap(splits, arrays)
    encoder = LabelEncoder().fit(splits['train']['Label'])
    for name, values in arrays.items():
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f'{name} is empty or nonfinite after preprocessing')
    all_ids = pd.concat([part[PROVENANCE] for part in splits.values()], ignore_index=True)
    if all_ids.duplicated().any():
        raise ValueError('Source record IDs overlap across splits')
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if pd.MultiIndex.from_frame(pd.DataFrame(arrays[a])).isin(
                pd.MultiIndex.from_frame(pd.DataFrame(arrays[b]))).any():
            raise ValueError(f'Transformed features overlap: {a}/{b}')
    folder = ROOT / 'data/processed/phase2'
    folder.mkdir(parents=True, exist_ok=True)
    model_folder = ROOT / 'models'
    model_folder.mkdir(exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(folder / f'{name}_raw.parquet', index=False)
        np.savez_compressed(folder / f'{name}.npz', X=arrays[name],
                            y_multiclass=encoder.transform(part['Label']),
                            y_binary=part['Label'].ne('BENIGN').to_numpy(dtype=np.int8))
    artifact = {'preprocessor': pipeline, 'label_encoder': encoder,
                'raw_feature_columns': feature_columns, 'provenance_columns': PROVENANCE}
    artifact_path = model_folder / 'phase2_preprocessor.joblib'
    joblib.dump(artifact, artifact_path)
    # Check persistence against independently reloaded inputs and outputs.
    restored = joblib.load(artifact_path)
    saved_test = pd.read_parquet(folder / 'test_raw.parquet')
    with np.load(folder / 'test.npz') as saved:
        np.testing.assert_allclose(restored['preprocessor'].transform(saved_test[feature_columns]), saved['X'])
        np.testing.assert_array_equal(restored['label_encoder'].transform(saved_test['Label']), saved['y_multiclass'])
    names = pipeline.named_steps['clean'].get_feature_names_out()
    support = pipeline.named_steps['constant'].get_support()
    summary = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'sampling': provenance,
               'sample_raw_class_counts': original_counts, 'cleaning': cleaning,
               'training_quality': training_quality,
               'seed': seed, 'split_policy': 'per-class ~70/15/15; <3 rows train-only',
               'train_only_classes': train_only, 'transformed_overlap_removed': purged,
               'feature_names': names[support].tolist(), 'removed_constant_features': names[~support].tolist(),
               'label_mapping': {str(label): int(i) for i, label in enumerate(encoder.classes_)},
               'binary_mapping': {'BENIGN': 0, 'attack': 1},
               'splits': {k: {'records': len(v), 'classes': v['Label'].value_counts().to_dict(),
                              'source_files': v['_source_file'].value_counts().to_dict()} for k, v in splits.items()},
               'missing_holdout_classes': {k: sorted(set(encoder.classes_) - set(splits[k]['Label'])) for k in ('validation', 'test')},
               'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                            'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                            **{name: version(name) for name in ('datasets', 'huggingface-hub', 'pyarrow', 'joblib', 'scipy')}}}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = ROOT / 'reports/phase2_preprocessing_summary.md'
    report.parent.mkdir(exist_ok=True)
    write_summary(summary, report)
    print(f"Saved splits: { {k: len(v) for k, v in splits.items()} }; features: {int(support.sum())}", flush=True)
    print(f'Report: {report}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-class', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--revision')
    parser.add_argument('--refresh-sample', action='store_true')
    args = parser.parse_args()
    try:
        prepare(args.per_class, args.seed, args.revision, args.refresh_sample)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f'Phase 2 failed: {exc}\nCheck dependencies/network and retry: python -m src.prepare_data\n')


if __name__ == '__main__':
    main()
