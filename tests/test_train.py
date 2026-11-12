"""Checks of metric definitions, validation selection, and inference schema."""
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.predict import predict_frame
from src.train import binary_metrics, choose_model


class TrainingTests(unittest.TestCase):
    def test_false_positive_rate_uses_benign_denominator(self):
        result = binary_metrics(np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]),
                                np.array([.1, .2, .7, .9]))
        self.assertEqual(result['confusion_matrix'], [[2, 1], [0, 1]])
        self.assertAlmostEqual(result['false_positive_rate'], 1/3)
        self.assertEqual(result['recall_attack'], 1)
        self.assertEqual(result['precision_attack'], .5)

    def test_selection_uses_macro_f1_then_fpr_and_rejects_nonconvergence(self):
        def item(f1, fpr, converged=True):
            return {'converged': converged, 'metrics': {'macro_f1': f1, 'false_positive_rate': fpr}}
        self.assertEqual(choose_model({'a': item(.8, .2), 'b': item(.8, .1),
                                       'c': item(.99, 0, False)}), 'b')
        with self.assertRaises(ValueError):
            choose_model({'c': item(.99, 0, False)})

    def test_inference_excludes_labels_and_provenance(self):
        pipeline = MagicMock()
        pipeline.predict.return_value = np.array([1])
        pipeline.predict_proba.return_value = np.array([[.2, .8]])
        artifact = {'pipeline': pipeline, 'raw_feature_columns': ['rate'],
                    'label_mapping': {0: 'BENIGN', 1: 'ATTACK'}}
        frame = pd.DataFrame({' rate ': [2], 'Label': ['BENIGN'], '_source_file': ['test']})
        result = predict_frame(frame, artifact)
        self.assertEqual(result['prediction'].iloc[0], 'ATTACK')
        self.assertEqual(list(pipeline.predict.call_args.args[0]), ['rate'])
        with self.assertRaises(ValueError):
            predict_frame(pd.DataFrame({'other': [1]}), artifact)


if __name__ == '__main__':
    unittest.main()
