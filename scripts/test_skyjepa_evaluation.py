import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("evaluation", Path(__file__).with_name("evaluate-skyjepa-remediation.py"))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationProtocolTests(unittest.TestCase):
    def test_warm_start_requires_tracking_and_timing_improvement(self):
        fresh = dict(complete=True, tracking=81, timing=81, maximum_aggregate_p95_ms=9.0,
                     mean_rmse_m=0.2, worst_rmse_m=0.4)
        shifted = {**fresh, "mean_rmse_m": 0.19}
        self.assertEqual(evaluation.choose_warm_start(fresh, shifted), "shifted-residual")
        for regression in ({"complete": False}, {"tracking": 80}, {"timing": 80},
                           {"maximum_aggregate_p95_ms": 10.01}, {"mean_rmse_m": 0.199},
                           {"worst_rmse_m": 0.421}):
            self.assertEqual(evaluation.choose_warm_start(fresh, {**shifted, **regression}), "fresh-prior")

    def test_incomplete_cases_are_not_silently_dropped(self):
        report = {"results": [{"position_vector_rmse_m": 0.2, "tracking_passed": True, "timing_passed": True},
                              {"position_vector_rmse_m": None, "tracking_passed": False, "timing_passed": False}],
                  "aggregate_p95_plan_ms": 9.0}
        result = evaluation.aggregate([report])
        self.assertEqual(result["runs"], 2)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["mean_rmse_m"])
        self.assertEqual(result["tracking"], 1)

    def test_baselines_cannot_be_reused_across_different_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in evaluation.SEEDS:
                evaluation.save_new(root / f"seed-{seed}" / "prober" / "checkpoint.json",
                                    {"contract": {"normalization": [1.0, 2.0]}})
            self.assertEqual(evaluation.verify_shared_baselines(root)["reused_for_training_seeds"], [7, 17, 29])
            path = root / "seed-29/prober/checkpoint.json"
            path.write_text(json.dumps({"contract": {"normalization": [1.0, 3.0]}}))
            with self.assertRaises(AssertionError):
                evaluation.verify_shared_baselines(root)


if __name__ == "__main__":
    unittest.main()
