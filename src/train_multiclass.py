"""Multiclass baselines on the existing Phase 2 benchmark splits."""
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.data_loader import ROOT
from src.train import choose_model


def multiclass_metrics(y, prediction, classes):
    labels = np.arange(len(classes))
    if 'BENIGN' not in classes:
        raise ValueError('Class mapping must include BENIGN')
    if not set(np.unique(np.concatenate([y, prediction]))).issubset(set(labels)):
        raise ValueError('Target or prediction outside saved class mapping')
    benign = list(classes).index('BENIGN')
    benign_rows = y == benign
    attack_rows = ~benign_rows
    matrix = confusion_matrix(y, prediction, labels=labels)
    confusions = [{'actual': str(classes[a]), 'predicted': str(classes[b]), 'count': int(matrix[a, b])}
                  for a in labels for b in labels if a != b and matrix[a, b]]
    confusions.sort(key=lambda row: (-row['count'], row['actual'], row['predicted']))
    return {'accuracy': float(accuracy_score(y, prediction)),
            'macro_f1': float(f1_score(y, prediction, labels=labels, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y, prediction, labels=labels, average='weighted', zero_division=0)),
            'false_positive_rate': float(np.mean(prediction[benign_rows] != benign)) if benign_rows.any() else None,
            'attack_detection_recall': float(np.mean(prediction[attack_rows] != benign)) if attack_rows.any() else None,
            'classification_report': classification_report(y, prediction, labels=labels, target_names=list(classes),
                                                            output_dict=True, zero_division=0),
            'confusion_matrix': matrix.tolist(), 'top_confusions': confusions[:15]}


def load_split(folder, name, encoder, preprocessing, columns):
    with np.load(folder / f'{name}.npz', allow_pickle=False) as data:
        X, y = data['X'].copy(), data['y_multiclass'].copy()
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y) or not np.isfinite(X).all():
        raise ValueError(f'Invalid {name} matrix')
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= len(encoder.classes_):
        raise ValueError(f'Invalid {name} class IDs')
    raw = pd.read_parquet(folder / f'{name}_raw.parquet')
    np.testing.assert_array_equal(encoder.transform(raw['Label']), y)
    np.testing.assert_allclose(preprocessing.transform(raw[columns]), X)
    return X, y, raw


def write_report(result, path):
    lines = ['# Phase 3: multiclass baseline', '', f"Generated UTC: {result['generated_utc']}",
             f"Source revision: `{result['dataset_revision']}`; seed: {result['seed']}.", '',
             'Both models fit training only with the frozen Phase 2 training-fitted preprocessing. Selection uses validation macro F1 over the full saved label set, then lower benign false-positive rate. Nonconverged candidates are excluded. Predictions use the highest-scoring class; there is no threshold search or hyperparameter tuning. Only the selected model is scored on test, without refitting on validation.',
             '**This is the same benchmark split already evaluated for the binary baseline, not a new independent or untouched test set.**', '',
             '## Validation comparison', '',
             '| Model | Accuracy | Macro F1 | Weighted F1 | Benign FPR | Converged |',
             '|---|---:|---:|---:|---:|---|']
    for name, item in {**result['candidates'], 'majority_dummy': result['dummy']}.items():
        m = item['metrics']
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | {m['false_positive_rate']:.4f} | {item.get('converged', 'N/A')} |")
    lines += ['', f"Selected: **{result['selected_model']}**.", '', '## Selected model: test results', '']
    for key in ('accuracy', 'macro_f1', 'weighted_f1', 'false_positive_rate', 'attack_detection_recall'):
        lines.append(f"- {key}: {result['test'][key]:.6f}")
    lines += ['', 'Attack detection recall collapses all attacks together; it does not measure correct attack-type identification.', '',
              '| Class | Precision | Recall | F1 | Test support |', '|---|---:|---:|---:|---:|']
    for label in result['classes']:
        row = result['test']['classification_report'][label]
        lines.append(f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |")
    lines += ['', '![Test confusion matrices](multiclass_test_confusion_matrix.png)', '',
              '## Largest attack-type confusions', '', '| Actual | Predicted | Records |', '|---|---|---:|']
    for item in result['test']['top_confusions']:
        lines.append(f"| {item['actual']} | {item['predicted']} | {item['count']} |")
    lines += ['', '## Limits and validation', '',
              f"Test classes with fewer than 20 records (descriptive flag, not a statistical guarantee): {result['rare_test_classes']}",
              'A perfect score with one or a handful of examples is weak evidence. Zero-support classes, if any, receive zero in the fixed-label macro F1; inspect support alongside scores. Macro F1 weights tiny and common classes equally, while weighted F1 can hide rare-class failures.',
              'Class-capped sampling changes prevalence; this is a development benchmark. Random record splits retain possible session, host, and time similarity. These results do not establish performance on unseen attacks or current live traffic. Scores from predict_proba are uncalibrated.',
              'The saved bundle includes preprocessing, the selected classifier, and the original label mapping. Reloaded raw-input predictions and probability arrays are checked against the evaluated model. Input and model SHA-256 hashes are recorded. The notebook reads saved results and uses validation rows for inference; it does not rescore test.',
              'Do not tune against these test results. Any follow-up tuning should use validation or training-only cross-validation with fold-local preprocessing. Live packet capture and PCAP feature extraction remain unimplemented.', '',
              'Full parameters, warnings, fit times, package versions, per-class metrics, and provenance: `multiclass_baseline_metrics.json`.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def plot_confusions(result, path):
    matrix = np.asarray(result['test']['confusion_matrix'])
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                           out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0)
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    for ax, values, title in zip(axes, [matrix, normalized], ['Record counts', 'Recall by actual class']):
        im = ax.imshow(values, cmap='Blues', aspect='auto')
        for i in range(len(values)):
            for j in range(len(values)):
                if matrix[i, j]:
                    ax.text(j, i, str(int(values[i, j])) if ax is axes[0] else f'{values[i,j]:.2f}',
                            ha='center', va='center', fontsize=7,
                            color='white' if values[i,j] > values.max()/2 else 'black')
        ax.set_xticks(range(len(values)), result['classes'], rotation=90, fontsize=8)
        ax.set_yticks(range(len(values)), result['classes'], fontsize=8)
        ax.set(xlabel='Predicted class', ylabel='Actual class', title=title)
        fig.colorbar(im, ax=ax, shrink=.65)
    fig.suptitle(f"{result['selected_model']}: existing benchmark test split")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(seed=42):
    folder = ROOT / 'data/processed/phase2'
    reports = ROOT / 'reports'
    result_path = reports / 'multiclass_baseline_metrics.json'
    model_path = ROOT / 'models/multiclass_detector.joblib'
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing['seed'] != seed:
            raise ValueError('Seed differs from recorded baseline')
        for relative, digest in existing['input_sha256'].items():
            path = ROOT / relative
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Artifact changed since evaluation: {relative}')
        if not model_path.exists() or hashlib.sha256(model_path.read_bytes()).hexdigest() != existing['model_sha256']:
            raise ValueError('Saved model missing or changed since recorded evaluation')
        print(f'Preserving existing multiclass results: {result_path}', flush=True)
        return existing
    phase2 = json.loads((folder / 'summary.json').read_text())
    pre_path = ROOT / 'models/phase2_preprocessor.joblib'
    artifact = joblib.load(pre_path)
    encoder, preprocessor, columns = artifact['label_encoder'], artifact['preprocessor'], artifact['raw_feature_columns']
    classes = encoder.classes_.tolist()
    X_train, y_train, _ = load_split(folder, 'train', encoder, preprocessor, columns)
    X_val, y_val, _ = load_split(folder, 'validation', encoder, preprocessor, columns)
    if set(np.unique(y_train)) != set(range(len(classes))) or not np.any(y_val == classes.index('BENIGN')):
        raise ValueError('Training must cover saved labels; validation must include BENIGN')
    candidates = {
        'logistic_regression': LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, class_weight='balanced', random_state=seed),
        'random_forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight='balanced', random_state=seed, n_jobs=2),
    }
    results = {}
    for name, model in candidates.items():
        print(f'Fitting multiclass {name} on {len(y_train):,} rows ({len(classes)} labels)', flush=True)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(X_train, y_train)
        results[name] = {'fit_seconds': time.perf_counter() - started, 'parameters': model.get_params(),
                         'converged': not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                         'warnings': [str(w.message) for w in caught],
                         'metrics': multiclass_metrics(y_val, model.predict(X_val), classes)}
        print(f"{name}: validation macro F1={results[name]['metrics']['macro_f1']:.4f}; converged={results[name]['converged']}", flush=True)
    dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    dummy_result = {'metrics': multiclass_metrics(y_val, dummy.predict(X_val), classes)}
    winner = choose_model(results)
    model = candidates[winner]
    print(f'Selected on validation: {winner}. Loading test split now.', flush=True)
    X_test, y_test, raw_test = load_split(folder, 'test', encoder, preprocessor, columns)
    prediction = model.predict(X_test)
    metrics = multiclass_metrics(y_test, prediction, classes)
    bundle = {'pipeline': Pipeline([('preprocessing', preprocessor), ('classifier', model)]),
              'raw_feature_columns': columns, 'label_mapping': dict(enumerate(classes)),
              'task': 'multiclass', 'selected_model': winner, 'dataset_revision': phase2['sampling']['metadata']['revision']}
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, model_path)
    restored = joblib.load(model_path)
    np.testing.assert_array_equal(restored['pipeline'].predict(raw_test[columns]), prediction)
    np.testing.assert_allclose(restored['pipeline'].predict_proba(raw_test[columns]), model.predict_proba(X_test))
    paths = [folder / f'{name}{suffix}' for name in ('train', 'validation', 'test') for suffix in ('.npz', '_raw.parquet')]
    paths += [pre_path, folder / 'summary.json']
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed, 'sklearn_version': sklearn.__version__,
              'dataset_revision': bundle['dataset_revision'], 'classes': classes,
              'selection_rule': 'validation macro F1 over all saved classes; lower benign FPR breaks ties; converged candidates only',
              'test_scope': 'Same Phase 2 benchmark split previously evaluated for binary classification; not a fresh independent test.',
              'selected_model': winner, 'candidates': results, 'dummy': dummy_result, 'test': metrics,
              'split_records': {'train': len(y_train), 'validation': len(y_val), 'test': len(y_test)},
              'rare_test_classes': {c: int(metrics['classification_report'][c]['support']) for c in classes if metrics['classification_report'][c]['support'] < 20},
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest()}
    reports.mkdir(exist_ok=True)
    plot_confusions(result, reports / 'multiclass_test_confusion_matrix.png')
    write_report(result, reports / 'phase3_multiclass_baseline.md')
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved multiclass metrics and model: {result_path}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        train(args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Multiclass training failed: {exc}\nCheck Phase 2 artifacts and requirements.txt.\n')


if __name__ == '__main__':
    main()
