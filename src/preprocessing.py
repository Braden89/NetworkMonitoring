"""Deterministic cleaning and train-only fitted preprocessing; no classifiers."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

PROVENANCE = ['_source_file', '_source_row']
REDUNDANT = 'Fwd Header Length.1'


def normalize_headers(frame):
    result = frame.copy()
    result.columns = [str(c).strip() for c in result.columns]
    if result.columns.duplicated().any():
        raise ValueError('Header normalization produced duplicate column names')
    return result


def normalize_labels(labels):
    """Repair only observed web-attack separator variants; reject unknown corruption."""
    values = labels.astype('string').str.strip()
    for suffix in ('Brute Force', 'XSS', 'Sql Injection'):
        for separator in ('\ufffd', '\u2013', '\u2014', '-'):
            values = values.replace(f'Web Attack {separator} {suffix}', f'Web Attack - {suffix}')
    if values.isna().any() or values.eq('').any() or values.str.contains('\ufffd', regex=False).any():
        raise ValueError('Missing, empty, or unrecognized corrupted labels need manual inspection')
    return values.astype(str)


def clean_numeric(frame):
    """Stateless rules also used before deduplication; leave other negatives intact."""
    result = normalize_headers(frame).drop(columns=[REDUNDANT], errors='ignore')
    if 'Label' in result or any(c.startswith('_') for c in result.columns):
        raise ValueError('Labels and provenance must not enter feature preprocessing')
    result = result.apply(pd.to_numeric, errors='coerce').astype('float64')
    result = result.replace([np.inf, -np.inf], np.nan)
    for column in ('Init_Win_bytes_forward', 'Init_Win_bytes_backward'):
        if column in result:
            result[column] = result[column].mask(result[column].eq(-1))
    return result


class NumericCleaner(TransformerMixin, BaseEstimator):
    """Enforce the fitted feature schema and apply deterministic cleaning rules."""
    def fit(self, X, y=None):
        cleaned = clean_numeric(X)
        self.feature_names_in_ = np.asarray(cleaned.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        check_is_fitted(self, 'feature_names_in_')
        cleaned = clean_numeric(X)
        if set(cleaned.columns) != set(self.feature_names_in_):
            raise ValueError('Input feature schema differs from fitted training schema')
        return cleaned.loc[:, self.feature_names_in_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_in_')
        return self.feature_names_in_.copy()


def build_preprocessor():
    return Pipeline([
        ('clean', NumericCleaner()),
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('constant', VarianceThreshold()),
        ('scale', StandardScaler()),
    ])


def deduplicate_sample(frame):
    """Remove conflicting feature groups, then exact repeated cleaned feature rows."""
    frame = normalize_headers(frame).reset_index(drop=True)
    required = {'Label', *PROVENANCE}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing required columns: {sorted(required - set(frame.columns))}')
    frame['Label'] = normalize_labels(frame['Label'])
    features = clean_numeric(frame.drop(columns=['Label', *PROVENANCE]))
    keyed = features.copy()
    keyed['Label'] = frame['Label']
    distinct_pairs = keyed.drop_duplicates()
    conflicts = distinct_pairs.duplicated(subset=list(features.columns), keep=False)
    # MultiIndex membership uses exact values (including missing values), not just fingerprints.
    bad_keys = pd.MultiIndex.from_frame(distinct_pairs.loc[conflicts, features.columns])
    bad = pd.MultiIndex.from_frame(features).isin(bad_keys)
    duplicate = features.loc[~bad].duplicated()
    keep = duplicate.index[~duplicate]
    stats = {'input_records': len(frame), 'conflicting_label_rows_removed': int(bad.sum()),
             'duplicate_feature_rows_removed': int(duplicate.sum()), 'retained_records': len(keep)}
    return frame.loc[keep].reset_index(drop=True), stats


def split_by_class(frame, seed=42):
    """Approximate 70/15/15; ensure each class with >=3 rows appears in each split."""
    rng = np.random.default_rng(seed)
    indices = {'train': [], 'validation': [], 'test': []}
    train_only = []
    for label, group in frame.groupby('Label', sort=True):
        rows = rng.permutation(group.index.to_numpy())
        if len(rows) < 3:
            indices['train'].extend(rows)
            train_only.append(label)
            continue
        n_holdout = max(1, int(len(rows) * 0.15))
        indices['test'].extend(rows[:n_holdout])
        indices['validation'].extend(rows[n_holdout:2*n_holdout])
        indices['train'].extend(rows[2*n_holdout:])
    return {name: frame.loc[rng.permutation(rows)].reset_index(drop=True)
            for name, rows in indices.items()}, train_only


def purge_transformed_overlap(splits, arrays):
    """Imputation/constant removal can collapse distinct rows; protect holdouts too."""
    order = ['train', 'validation', 'test']
    combined = pd.DataFrame(np.concatenate([arrays[k] for k in order]))
    repeated = combined.duplicated().to_numpy()
    offset, removed = 0, {}
    for name in order:
        count = len(splits[name])
        keep = np.ones(count, dtype=bool) if name == 'train' else ~repeated[offset:offset+count]
        removed[name] = int((~keep).sum())
        splits[name] = splits[name].loc[keep].reset_index(drop=True)
        arrays[name] = arrays[name][keep]
        offset += count
    return removed
