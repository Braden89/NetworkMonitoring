"""Binary baselines: select on validation macro F1, then test only the winner."""
import argparse
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from src.data_loader import ROOT


def binary_metrics(y, prediction, probability):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {'accuracy': float(accuracy_score(y, prediction)),
            'precision_attack': float(precision_score(y, prediction, zero_division=0)),
            'recall_attack': float(recall_score(y, prediction, zero_division=0)),
            'f1_attack': float(f1_score(y, prediction, zero_division=0)),
            'macro_f1': float(f1_score(y, prediction, average='macro', zero_division=0)),
            'false_positive_rate': float(fp / (tn + fp)) if tn + fp else None,
            'roc_auc': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            'average_precision': float(average_precision_score(y, probability)) if np.any(y == 1) else None,
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
            'classification_report': classification_report(y, prediction, labels=[0, 1],
                target_names=['BENIGN', 'ATTACK'], output_dict=True, zero_division=0)}


def choose_model(results):
    eligible = [name for name, item in results.items() if item['converged']]
    if not eligible:
        raise ValueError('No converged candidate; inspect validation results')
    return max(eligible, key=lambda name: (results[name]['metrics']['macro_f1'],
                -results[name]['metrics']['false_positive_rate'], name))


def load_split(folder, name):
    with np.load(folder / f'{name}.npz', allow_pickle=False) as data:
        X, y = data['X'].copy(), data['y_binary'].copy()
    if X.ndim != 2 or len(X) != len(y) or not len(y) or not np.isfinite(X).all():
        raise ValueError(f'Invalid {name} feature matrix')
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f'{name} must contain BENIGN=0 and ATTACK=1')
    return X, y


def write_report(result, path):
    lines = ['# Phase 3: binary baseline results', '',
             f"Generated UTC: {result['generated_utc']}",
             f"Source revision: `{result['dataset_revision']}`; seed: {result['seed']}.", '',
             'BENIGN=0, ATTACK=1. Fixed default probability decision threshold (0.5); no threshold search or hyperparameter tuning. Both candidates use balanced class weights to account for the attack-heavy development sample.',
             'Models fit training data only, with the Phase 2 training-fitted preprocessing. Selection uses validation macro F1, then lower false-positive rate as a tie-breaker. Nonconverged candidates are ineligible. Only the winner is scored on test; it is not refitted on validation.', '',
             '## Validation comparison', '',
             '| Model | Accuracy | Attack precision | Attack recall | Attack F1 | Macro F1 | Benign FPR |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, item in {**result['candidates'], 'majority_dummy': result['dummy']}.items():
        m = item['metrics']
        lines.append('| ' + name + ' | ' + ' | '.join(f'{m[k]:.4f}' for k in
            ('accuracy', 'precision_attack', 'recall_attack', 'f1_attack', 'macro_f1', 'false_positive_rate')) + ' |')
    lines += ['', f"Selected: **{result['selected_model']}**.", '', '## Selected model: test results', '']
    for k in ('accuracy', 'precision_attack', 'recall_attack', 'f1_attack', 'macro_f1', 'false_positive_rate', 'roc_auc', 'average_precision'):
        lines.append(f"- {k}: {result['test'][k]:.6f}")
    lines += ['', 'Confusion matrix: rows=actual, columns=predicted; order BENIGN, ATTACK.',
              '```json', json.dumps(result['test']['confusion_matrix']), '```', '',
              '![Test confusion matrix](binary_test_confusion_matrix.png)', '',
              '## Limits', '',
              'This class-capped sample is attack-heavy and is not natural traffic prevalence. Precision, accuracy, and probability estimates must not be treated as deployment performance. Random record splitting cannot rule out session/host/time similarity. Rare source attacks have tiny support. This model predicts only BENIGN or ATTACK, not specific attack types.',
              'The test result is a final baseline measurement, not a tuning target. Future tuning should use validation or training-only cross-validation with fold-local preprocessing; repeated test inspection compromises an untouched evaluation claim.',
              'Saved model includes preprocessing and supports compatible raw flow-feature tables. It does not capture packets or establish live detection performance.', '',
              'Full metrics, parameters, convergence warnings, fit times, versions, and input hashes: `binary_baseline_metrics.json`.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def train(seed=42):
    folder = ROOT / 'data/processed/phase2'
    report_folder = ROOT / 'reports'
    result_path = report_folder / 'binary_baseline_metrics.json'
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if seed != existing['seed']:
            raise ValueError('Existing baseline uses a different seed; keep it as the recorded baseline')
        for relative, digest in existing['input_sha256'].items():
            path = ROOT / relative
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Phase 2 artifacts changed since baseline evaluation: {relative}')
        if not (ROOT / 'models/binary_detector.joblib').exists():
            raise ValueError('Recorded results exist but saved model is missing; restore the matching artifact')
        print(f'Existing results: {result_path}. Preserving the recorded test evaluation.')
        return existing
    phase2 = json.loads((folder / 'summary.json').read_text())
    preprocessing_path = ROOT / 'models/phase2_preprocessor.joblib'
    preprocessing = joblib.load(preprocessing_path)
    X_train, y_train = load_split(folder, 'train')
    X_val, y_val = load_split(folder, 'validation')
    for name, X, y in [('train', X_train, y_train), ('validation', X_val, y_val)]:
        raw = pd.read_parquet(folder / f'{name}_raw.parquet')
        np.testing.assert_allclose(preprocessing['preprocessor'].transform(raw[preprocessing['raw_feature_columns']]), X)
        np.testing.assert_array_equal(raw['Label'].ne('BENIGN').astype(int), y)
    candidates = {
        'logistic_regression': LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000,
                                                   class_weight='balanced', random_state=seed),
        'random_forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                            class_weight='balanced', random_state=seed, n_jobs=2),
    }
    results = {}
    for name, model in candidates.items():
        print(f'Fitting {name} on {len(y_train):,} training records', flush=True)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(X_train, y_train)
        results[name] = {'fit_seconds': time.perf_counter() - started,
                         'converged': not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                         'warnings': [str(w.message) for w in caught], 'parameters': model.get_params(),
                         'metrics': binary_metrics(y_val, model.predict(X_val), model.predict_proba(X_val)[:, 1])}
        print(f"{name}: validation macro F1={results[name]['metrics']['macro_f1']:.4f}", flush=True)
    dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    dummy_result = {'metrics': binary_metrics(y_val, dummy.predict(X_val), dummy.predict_proba(X_val)[:, 1])}
    winner = choose_model(results)
    model = candidates[winner]
    # Do not load or score the test set until validation selection is complete.
    X_test, y_test = load_split(folder, 'test')
    predictions = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    test_metrics = binary_metrics(y_test, predictions, probability)
    combined = Pipeline([('preprocessing', preprocessing['preprocessor']), ('classifier', model)])
    bundle = {'pipeline': combined, 'raw_feature_columns': preprocessing['raw_feature_columns'],
              'label_mapping': {0: 'BENIGN', 1: 'ATTACK'}, 'selected_model': winner,
              'dataset_revision': phase2['sampling']['metadata']['revision']}
    model_path = ROOT / 'models/binary_detector.joblib'
    joblib.dump(bundle, model_path)
    restored = joblib.load(model_path)
    raw_test = pd.read_parquet(folder / 'test_raw.parquet')
    np.testing.assert_array_equal(raw_test['Label'].ne('BENIGN').astype(int), y_test)
    np.testing.assert_allclose(restored['pipeline'].named_steps['preprocessing'].transform(
        raw_test[restored['raw_feature_columns']]), X_test)
    np.testing.assert_array_equal(restored['pipeline'].predict(raw_test[restored['raw_feature_columns']]), predictions)
    paths = [folder / f'{name}.npz' for name in ('train', 'validation', 'test')] + [preprocessing_path, folder / 'summary.json']
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed,
              'dataset_revision': bundle['dataset_revision'], 'sklearn_version': sklearn.__version__,
              'selection_rule': 'validation macro F1; lower FPR breaks ties; converged candidates only',
              'threshold': 0.5, 'selected_model': winner, 'candidates': results, 'dummy': dummy_result,
              'split_records': {'train': len(y_train), 'validation': len(y_val), 'test': len(y_test)},
              'test': test_metrics, 'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    report_folder.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(np.array(test_metrics['confusion_matrix']), display_labels=['BENIGN', 'ATTACK']).plot(ax=ax, colorbar=False, values_format='d')
    ax.set_title(f'{winner}: held-out test')
    fig.tight_layout()
    fig.savefig(report_folder / 'binary_test_confusion_matrix.png', dpi=150)
    plt.close(fig)
    write_report(result, report_folder / 'phase3_binary_baseline.md')
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Selected {winner}; test results saved to {result_path}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        train(args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Training failed: {exc}\nEnsure Phase 2 artifacts exist: python -m src.prepare_data\n')


if __name__ == '__main__':
    main()
