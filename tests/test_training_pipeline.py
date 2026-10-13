import unittest

import pandas as pd

from data.split_data import create_splits, prepare_dataset
from training.train import compute_class_weights, compute_metrics


class SplitDataTests(unittest.TestCase):
    def make_dataset(self):
        rows = []
        for label in ("group-a", "group-b", "group-c"):
            for index in range(30):
                rows.append(
                    {"text": f"{label} ticket {index}", "Assignment Group": label}
                )
        return pd.DataFrame(rows)

    def test_prepare_dataset_deduplicates_ticket_text(self):
        df = pd.concat(
            [self.make_dataset(), self.make_dataset().iloc[[0]]],
            ignore_index=True,
        )

        prepared = prepare_dataset(df)

        self.assertEqual(len(prepared), 90)
        self.assertEqual(prepared["text"].nunique(), 90)

    def test_prepare_dataset_rejects_conflicting_labels(self):
        df = pd.DataFrame(
            [
                {"text": "same ticket", "Assignment Group": "group-a"},
                {"text": "same ticket", "Assignment Group": "group-b"},
            ]
        )

        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            prepare_dataset(df)

    def test_splits_are_reproducible_stratified_and_disjoint(self):
        df = prepare_dataset(self.make_dataset())

        first = create_splits(df)
        second = create_splits(df)

        self.assertEqual(
            {name: split["text"].tolist() for name, split in first.items()},
            {name: split["text"].tolist() for name, split in second.items()},
        )
        expected_labels = set(df["Assignment Group"])
        for split in first.values():
            self.assertEqual(set(split["Assignment Group"]), expected_labels)

        text_sets = {
            name: set(split["text"]) for name, split in first.items()
        }
        self.assertFalse(text_sets["train"] & text_sets["validation"])
        self.assertFalse(text_sets["train"] & text_sets["test"])
        self.assertFalse(text_sets["validation"] & text_sets["test"])


class TrainingMetricTests(unittest.TestCase):
    def test_class_weights_increase_for_minority_groups(self):
        weights = compute_class_weights([0] * 9 + [1], num_classes=2)

        self.assertEqual(weights.shape[0], 2)
        self.assertGreater(weights[1].item(), weights[0].item())

    def test_metrics_include_macro_and_weighted_f1(self):
        metrics = compute_metrics(
            (
                [[3.0, 0.0], [0.0, 3.0], [2.0, 1.0]],
                [0, 1, 1],
            )
        )

        self.assertEqual(metrics["accuracy"], 2 / 3)
        self.assertIn("f1_macro", metrics)
        self.assertIn("f1_weighted", metrics)


if __name__ == "__main__":
    unittest.main()
