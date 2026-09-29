"""Evaluation split, benchmark, and metric-contract tests."""

import unittest

from evaluation import (
    BEHAVIORAL_CATEGORIES,
    load_fixed_evaluation_set,
    score_behavioral_answer,
    validate_behavioral_benchmark,
)


class EvaluationTests(unittest.TestCase):
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