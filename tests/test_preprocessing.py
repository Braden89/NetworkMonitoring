"""Leakage, rare-class, schema, sampling, and artifact regression checks."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.prepare_data import prepare
from src.preprocessing import (build_preprocessor, clean_numeric, deduplicate_sample,
                               normalize_headers, normalize_labels, purge_transformed_overlap,
                               split_by_class)
from src.sampling import retain_by_priority


class PreprocessingTests(unittest.TestCase):
    def test_imputer_scaler_fit_training_only(self):
        train = pd.DataFrame({'rate': [1., 3., np.inf], 'constant': [0., 0., 0.],
                              'empty': [np.nan] * 3})
        pipeline = build_preprocessor()
        values = pipeline.fit_transform(train)
        np.testing.assert_allclose(values[:, 0], [-np.sqrt(1.5), np.sqrt(1.5), 0])
        original_medians = pipeline.named_steps['impute'].statistics_.copy()
        original_means = pipeline.named_steps['scale'].mean_.copy()
        pipeline.transform(pd.DataFrame({'rate': [1e9], 'constant': [9.], 'empty': [7.]}))
        np.testing.assert_array_equal(original_medians, pipeline.named_steps['impute'].statistics_)
        np.testing.assert_array_equal(original_means, pipeline.named_steps['scale'].mean_)
        self.assertEqual(values.shape, (3, 1))

    def test_cleaning_schema_and_labels(self):
        frame = pd.DataFrame({' rate ': ['bad', np.inf, -7],
                              'Init_Win_bytes_forward': [-1, 5, 6],
                              'Fwd Header Length.1': [1, 2, 3]})
        cleaned = clean_numeric(frame)
        self.assertTrue(cleaned['rate'].iloc[:2].isna().all())
        self.assertEqual(cleaned['rate'].iloc[2], -7)
        self.assertTrue(np.isnan(cleaned['Init_Win_bytes_forward'].iloc[0]))
        self.assertNotIn('Fwd Header Length.1', cleaned)
        with self.assertRaises(ValueError):
            normalize_headers(pd.DataFrame(columns=['rate', ' rate ']))
        with self.assertRaises(ValueError):
            clean_numeric(pd.DataFrame({'Label': ['BENIGN']}))
        self.assertEqual(normalize_labels(pd.Series([' Web Attack \ufffd XSS '])).iloc[0], 'Web Attack - XSS')
        with self.assertRaises(ValueError):
            normalize_labels(pd.Series([None]))
        p = build_preprocessor().fit(pd.DataFrame({'rate': [1, 2]}))
        with self.assertRaises(ValueError):
            p.transform(pd.DataFrame({'other': [1]}))

    def test_exact_conflicts_duplicates_and_missing_keys(self):
        frame = pd.DataFrame({'x': [1, 1, 2, 2, np.nan, np.inf, 7],
                              'Label': ['A', 'A', 'A', 'B', 'A', 'B', 'B'],
                              '_source_file': ['file'] * 7, '_source_row': range(7)})
        clean, stats = deduplicate_sample(frame)
        self.assertEqual(stats['conflicting_label_rows_removed'], 4)
        self.assertEqual(stats['duplicate_feature_rows_removed'], 1)
        self.assertEqual(clean['x'].tolist(), [1, 7])

    def test_reproducible_splits_and_rare_class_policy(self):
        frame = pd.DataFrame({'Label': ['A'] * 20 + ['B'] * 3 + ['rare'] * 2,
                              'id': range(25)})
        a, rare = split_by_class(frame, 42)
        b, _ = split_by_class(frame, 42)
        self.assertEqual(rare, ['rare'])
        for name in a:
            pd.testing.assert_frame_equal(a[name], b[name])
            self.assertIn('B', a[name]['Label'].values)
        self.assertNotIn('rare', a['test']['Label'].values)
        self.assertEqual(len(set(a['train']['id']) & set(a['test']['id'])), 0)
        self.assertEqual(sum(len(v) for v in a.values()), len(frame))

    def test_transformation_collisions_purged_from_holdouts(self):
        arrays = {'train': np.array([[1.], [2.]]), 'validation': np.array([[2.], [3.]]),
                  'test': np.array([[1.], [3.], [4.]])}
        splits = {k: pd.DataFrame({'id': range(len(v))}) for k, v in arrays.items()}
        removed = purge_transformed_overlap(splits, arrays)
        self.assertEqual(removed, {'train': 0, 'validation': 1, 'test': 2})
        self.assertEqual(arrays['test'].tolist(), [[4.]])

    def test_reservoir_keeps_late_low_priorities(self):
        first = pd.DataFrame({'_priority': [.9, .8, .2], 'id': [0, 1, 2]})
        second = pd.DataFrame({'_priority': [.1, .7], 'id': [3, 4]})
        retained = retain_by_priority(retain_by_priority(None, first, 2), second, 2)
        self.assertEqual(retained['id'].tolist(), [3, 2])

    def test_artifact_round_trip_and_cached_reproducibility(self):
        size = 60
        frame = pd.DataFrame({'x': np.arange(size, dtype=float), 'constant': np.zeros(size),
                              'Label': ['BENIGN'] * 30 + ['Attack'] * 30,
                              '_source_file': ['synthetic'] * size, '_source_row': range(size)})
        provenance = {'metadata': {'revision': 'synthetic'}, 'seed': 42, 'per_class': 30,
                      'sampling': 'Synthetic test fixture, not dataset observations',
                      'source_file_counts': {'synthetic': size}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch('src.prepare_data.ROOT', root), patch('src.prepare_data.sample_all_files', return_value=(frame, provenance)) as sampler:
                first = prepare(per_class=30)
                with np.load(root / 'data/processed/phase2/test.npz') as values:
                    initial = values['X'].copy()
                second = prepare(per_class=30)
                sampler.assert_called_once()
                self.assertEqual(first['splits'], second['splits'])
                with np.load(root / 'data/processed/phase2/test.npz') as values:
                    np.testing.assert_array_equal(initial, values['X'])
                summary = json.loads((root / 'data/processed/phase2/summary.json').read_text())
                self.assertEqual(summary['binary_mapping']['BENIGN'], 0)
                self.assertEqual(summary['feature_names'], ['x'])


if __name__ == '__main__':
    unittest.main()
