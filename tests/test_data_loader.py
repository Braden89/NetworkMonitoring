"""Offline checks for bounded loading and metadata-driven split validation."""
import unittest
from unittest.mock import MagicMock, patch

from src.data_loader import load_sample


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {"config": "default", "revision": "test-commit",
                         "splits": {"train": {"records": None}}}

    @patch("src.data_loader.load_dataset")
    def test_stream_is_bounded_and_revision_pinned(self, load):
        stream = MagicMock()
        load.return_value = stream
        stream.shuffle.return_value.take.return_value = [{"Label": "BENIGN"}]
        frame = load_sample(1, 7, 12, metadata=self.metadata)
        self.assertEqual(len(frame), 1)
        self.assertTrue(load.call_args.kwargs["streaming"])
        self.assertEqual(load.call_args.kwargs["revision"], "test-commit")
        stream.shuffle.assert_called_once_with(seed=7, buffer_size=12)
        stream.shuffle.return_value.take.assert_called_once_with(1)

    @patch("src.data_loader.load_dataset")
    def test_invalid_inputs_do_not_load_data(self, load):
        for kwargs in ({"records": 0}, {"buffer_size": 0}, {"split": "missing"}):
            with self.assertRaises(ValueError):
                load_sample(metadata=self.metadata, **kwargs)
        load.assert_not_called()

    @patch("src.data_loader.load_dataset", side_effect=OSError("network unavailable"))
    def test_network_failure_preserves_cause(self, load):
        with self.assertRaisesRegex(RuntimeError, "Cannot stream") as caught:
            load_sample(metadata=self.metadata)
        self.assertIsInstance(caught.exception.__cause__, OSError)


if __name__ == "__main__":
    unittest.main()
