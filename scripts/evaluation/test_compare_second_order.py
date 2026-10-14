import unittest

import numpy as np
import pandas as pd

from scripts.evaluation.compare_second_order import build_features, make_pairs, score


class ChronologicalEvaluationTests(unittest.TestCase):
    def raw(self):
        index = pd.date_range("2026-01-01", periods=500, freq="h", tz="UTC")
        price = 60 * np.exp(.0005 * np.arange(500) + .02 * np.sin(np.arange(500) / 8))
        return pd.DataFrame({"Open": price, "High": price + 1, "Low": price - 1,
                             "Close": price, "Volume": 1000}, index=index)

    def test_features_and_age_do_not_read_future(self):
        raw = self.raw()
        whole = build_features(raw)
        for n in [180, 300, 420]:
            prefix = build_features(raw.iloc[:n])
            pd.testing.assert_frame_equal(prefix, whole.loc[prefix.index])

    def test_targets_are_next_hour_and_gaps_are_excluded(self):
        raw = self.raw().drop(self.raw().index[350])
        features = build_features(raw)
        pairs, excluded = make_pairs(features, features.index[0])
        self.assertEqual(excluded, 1)
        self.assertTrue((pairs.target_timestamp - pairs.index.to_series() == pd.Timedelta(hours=1)).all())
        np.testing.assert_array_equal(pairs.target, features.loc[pairs.target_timestamp, "state"])

    def test_accuracy_is_realized_correctness_not_confidence(self):
        p = np.array([[.8, .05, .05, .05, .05], [.05, .8, .05, .05, .05]])
        scored = score(p, np.array([1, 1]), np.array([0, 1]))
        self.assertEqual(scored["accuracy"], .5)
        self.assertEqual(scored["observed_flips"], 1)

    def test_duplicate_timestamps_are_rejected(self):
        raw = self.raw()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_features(pd.concat([raw, raw.iloc[[-1]]]))


if __name__ == "__main__":
    unittest.main()
