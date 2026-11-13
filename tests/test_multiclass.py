"""Multiclass metric semantics, inference, and saved-result protection."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.predict import predict_frame
from src.train_multiclass import multiclass_metrics, train


class MulticlassTests(unittest.TestCase):
    def test_benign_index_and_attack_type_confusion(self):
        names = ['Attack A', 'BENIGN', 'Attack B']
        y = np.array([1, 1, 0, 2])
        prediction = np.array([1, 0, 2, 2])
        result = multiclass_metrics(y, prediction, names)
        self.assertEqual(result['false_positive_rate'], .5)
        self.assertEqual(result['attack_detection_recall'], 1)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['classification_report']['Attack A']['recall'], 0)
        self.assertEqual(result['confusion_matrix'], [[0, 0, 1], [1, 1, 0], [0, 0, 1]])
        self.assertEqual(sum(row['count'] for row in result['top_confusions']), 2)

    def test_missing_class_is_retained_in_fixed_macro_average(self):
        result = multiclass_metrics(np.array([0, 1]), np.array([0, 1]), ['BENIGN', 'Attack', 'Rare'])
        self.assertAlmostEqual(result['macro_f1'], 2/3)
        self.assertEqual(result['classification_report']['Rare']['support'], 0)
        with self.assertRaises(ValueError):
            multiclass_metrics(np.array([0]), np.array([9]), ['BENIGN', 'Attack'])

    def test_multiclass_inference_uses_class_mapping_and_max_score(self):
        pipeline = MagicMock()
        pipeline.predict.return_value = np.array([2])
        pipeline.predict_proba.return_value = np.array([[.1, .2, .7]])
        artifact = {'pipeline': pipeline, 'raw_feature_columns': ['rate'], 'task': 'multiclass',
                    'label_mapping': {0: 'BENIGN', 1: 'DoS', 2: 'PortScan'}}
        result = predict_frame(pd.DataFrame({'rate': [1], 'Label': ['ignored']}), artifact)
        self.assertEqual(result['prediction'].iloc[0], 'PortScan')
        self.assertEqual(result['predicted_class_score'].iloc[0], .7)
        self.assertNotIn('attack_score', result)
        self.assertEqual(list(pipeline.predict.call_args.args[0]), ['rate'])

    def test_existing_result_checks_hashes_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'reports').mkdir()
            (root / 'models').mkdir()
            source = root / 'input.bin'
            source.write_bytes(b'input')
            model = root / 'models/multiclass_detector.joblib'
            model.write_bytes(b'model')
            saved = {'seed': 42, 'input_sha256': {'input.bin': hashlib.sha256(source.read_bytes()).hexdigest()},
                     'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest()}
            (root / 'reports/multiclass_baseline_metrics.json').write_text(json.dumps(saved))
            with patch('src.train_multiclass.ROOT', root), patch('src.train_multiclass.load_split') as loader:
                self.assertEqual(train(), saved)
                loader.assert_not_called()
                with self.assertRaises(ValueError):
                    train(seed=99)
                model.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'model missing or changed'):
                    train()
                source.write_bytes(b'changed input')
                with self.assertRaisesRegex(ValueError, 'Artifact changed'):
                    train()


if __name__ == '__main__':
    unittest.main()
