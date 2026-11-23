import unittest

import numpy as np

from cifar10 import sample_subset
from stock_forecasting.data import prepare_sequences


class PreparationTests(unittest.TestCase):
    def test_subsample_deterministic(self):
        images = np.arange(100)
        labels = np.arange(100)
        a, b = sample_subset(images, labels, 10, seed=2)
        c, d = sample_subset(images, labels, 10, seed=2)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)

    def test_chronological_target_and_train_only_scaler(self):
        prices = np.arange(100, dtype=float)
        x_train, y_train, x_test, y_test, scaler = prepare_sequences(
            prices, lookback=10, train_fraction=0.8
        )
        self.assertEqual(x_train.shape, (70, 10, 1))
        self.assertEqual(x_test.shape, (20, 10, 1))
        self.assertEqual(float(scaler.data_max_[0]), 79.0)
        first_test_target = scaler.inverse_transform(y_test[:1, None])[0, 0]
        first_test_previous = scaler.inverse_transform(x_test[:1, -1, :])[0, 0]
        self.assertAlmostEqual(first_test_target, 80.0, places=3)
        self.assertAlmostEqual(first_test_previous, 79.0, places=3)

    def test_reject_invalid_window(self):
        with self.assertRaises(ValueError):
            prepare_sequences(np.arange(10.0), lookback=9)


if __name__ == "__main__":
    unittest.main()
