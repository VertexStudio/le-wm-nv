import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("evaluation", Path(__file__).with_name("evaluate-skyjepa-remediation.py"))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
summary_spec = importlib.util.spec_from_file_location("summary", Path(__file__).with_name("summarize-skyjepa-remediation.py"))
summary = importlib.util.module_from_spec(summary_spec)
summary_spec.loader.exec_module(summary)


class EvaluationProtocolTests(unittest.TestCase):
    def test_dataset_fingerprint_binds_all_canonical_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("metadata.json", "data.h5", "domains.json"):
                (root / name).write_bytes(name.encode())
            original = evaluation.dataset_sha(root)
            for name in ("metadata.json", "data.h5", "domains.json"):
                (root / name).write_bytes(b"changed")
                self.assertNotEqual(original, evaluation.dataset_sha(root))
                (root / name).write_bytes(name.encode())
            self.assertEqual(original, evaluation.dataset_sha(root))

    def test_paired_summary_checks_domains_and_reports_regressions(self):
        case = {"reference": "hover", "randomized": True, "domain_seed": 7, "domain": {"mass": 1.3},
                "position_vector_rmse_m": 0.8, "tracking_passed": False}
        baseline = {"results": [case]}
        trained = {"results": [{**case, "position_vector_rmse_m": 0.4, "tracking_passed": True}]}
        comparison = summary.compare(trained, baseline)
        self.assertEqual(comparison["mean_rmse_delta_m"], -0.4)
        self.assertEqual(comparison["tracking_gains"], 1)
        self.assertEqual(summary.compare(baseline, trained)["tracking_losses"], 1)
        trained["results"][0]["position_vector_rmse_m"] = None
        self.assertIsNone(summary.compare(trained, baseline)["mean_rmse_delta_m"])
        trained["results"][0]["domain"] = {"mass": 1.4}
        with self.assertRaises(AssertionError):
            summary.compare(trained, baseline)

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
