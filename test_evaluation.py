"""Evaluation split, benchmark, and metric-contract tests."""

import unittest

from evaluation import (
    BEHAVIORAL_CATEGORIES,
    load_manifest,
    load_fixed_evaluation_set,
    score_behavioral_answer,
    validate_behavioral_benchmark,
)


class EvaluationTests(unittest.TestCase):
    def test_manifest_declares_three_disjoint_partitions_and_train_only_tokenizer(self) -> None:
        manifest = load_manifest("dataset_manifest.json")
        paths = [manifest["splits"][name]["path"] for name in ("train", "validation", "test")]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(
            manifest["tokenizer"]["training_source"],
            "train_documents_only",
        )
        self.assertEqual(manifest["isolation"]["model_training_split"], "train")
        self.assertEqual(manifest["isolation"]["checkpoint_selection_split"], "validation")
        self.assertEqual(manifest["isolation"]["final_test_split"], "test")

    def test_fixed_validation_and_final_test_sets_are_distinct(self) -> None:
        validation = load_fixed_evaluation_set(
            "evaluation/validation_eval.json",
            "dataset_manifest.json",
            "validation",
            32,
        )
        final_test = load_fixed_evaluation_set(
            "evaluation/test_eval.json",
            "dataset_manifest.json",
            "test",
            32,
        )
        self.assertEqual(validation["role"], "checkpoint_validation")
        self.assertEqual(final_test["role"], "final_test")
        self.assertEqual(validation["evaluation_set_id"], "fixed-validation-v1")
        self.assertEqual(final_test["evaluation_set_id"], "fixed-final-test-v1")
        self.assertEqual(final_test["prohibited_uses"], [
            "tokenizer_training",
            "model_training",
            "hyperparameter_selection",
            "checkpoint_selection",
            "prompt_engineering",
            "instruction_fine_tuning_decisions",
        ])
        self.assertNotEqual(validation["split"], final_test["split"])
        with self.assertRaisesRegex(ValueError, "not 'validation'"):
            load_fixed_evaluation_set(
                "evaluation/test_eval.json",
                "dataset_manifest.json",
                "validation",
                32,
            )

    def test_behavioral_benchmark_covers_all_required_categories(self) -> None:
        benchmark = validate_behavioral_benchmark(
            "evaluation/a321neo_behavioral_benchmark.json"
        )
        categories = {item["category"] for item in benchmark["questions"]}
        self.assertEqual(categories, BEHAVIORAL_CATEGORIES)
        self.assertGreaterEqual(len(benchmark["questions"]), len(BEHAVIORAL_CATEGORIES))

    def test_behavioral_metrics_remain_separate(self) -> None:
        benchmark = validate_behavioral_benchmark(
            "evaluation/a321neo_behavioral_benchmark.json"
        )
        item = benchmark["questions"][0]
        result = score_behavioral_answer(
            item,
            "NEO means New Engine Option. The A321neo is an A320 Family aircraft variant.",
        )
        self.assertEqual(
            set(result),
            {
                "relevance",
                "factual_correctness",
                "completeness",
                "hallucination",
                "question_following",
            },
        )
        self.assertNotIn("overall", result)


if __name__ == "__main__":
    unittest.main()