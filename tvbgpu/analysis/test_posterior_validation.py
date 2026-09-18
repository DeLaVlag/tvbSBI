"""CPU-only checks; never import the GPU-initializing Posterior_tests module.

Run: python -B -m unittest tvbgpu.analysis.test_posterior_validation
The orchestration functions are AST-loaded with simulator/posterior stand-ins.
"""
import argparse
import ast
import contextlib
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from tvbgpu.analysis import posterior_validation as v


def checkpoint_fixture():
    names = ["dfa_1", "dfa_2", "lya_1", "fc_pca_1", "dfa_curve_pca_1",
             "lya_curve_pca_1", "pli_pca_1", "aecc_pca_1", "alpha_power_global_mean"]
    blocks = {"dfa": (0, 2), "lya": (2, 3), "fc_pca": (3, 4),
              "dfa_curve_pca": (4, 5), "lya_curve_pca": (5, 6),
              "pli_pca": (6, 7), "aecc_pca": (7, 8), "alpha": (8, 9)}
    keep = np.array([True, False, True, True, True, True, True, True, True])
    return {"parameter_names": list(v.PARAMETER_ORDER),
            "prior_low": np.array([.2, -6, .25, 1, 1, .5, .2], dtype=np.float32),
            "prior_high": np.array([1.2, -3, .55, 20, 1.3, .9, 2.5], dtype=np.float32),
            "feature_names_full": names, "feature_names": [n for n, k in zip(names, keep) if k],
            "feature_keep": keep, "feature_block_order": list(blocks), "feature_block_slices": blocks,
            "feature_config": {}, "x_mean": np.arange(8, dtype=np.float32),
            "x_std": np.arange(1, 9, dtype=np.float32)}


def load_orchestration():
    path = Path(__file__).parents[1] / "Posterior_tests.py"
    tree = ast.parse(path.read_text())
    wanted = {"parse_validation_args", "require_single_rank", "_draw_posteriors", "_resimulate_features",
              "posterior_predictive_check", "feature_sensitivity_test", "posterior_width_test",
              "parameter_recovery_test", "coverage_test", "validate_training_pairs", "resimulation_consistency_test", "main"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    import os
    ns = {"np": np, "torch": torch, "validation": v, "argparse": argparse, "os": os, "sys": sys,
          "__file__": str(path),
          "comm": SimpleNamespace(Get_rank=lambda: 0, Get_size=lambda: 1),
          "logger": logging.getLogger("test"),
          "TEST_TITLES": {n: n.upper() for n in ("consistency", "predictive", "width", "recovery", "coverage", "sensitivity")}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


class NumericValidation(unittest.TestCase):
    def setUp(self):
        self.ckpt = checkpoint_fixture()

    def test_blocks_use_saved_full_slices_and_mask(self):
        blocks = v.feature_block_indices(self.ckpt)
        np.testing.assert_array_equal(blocks["dfa"], [0])
        np.testing.assert_array_equal(blocks["dfa_curve_pca"], [3])
        np.testing.assert_array_equal(blocks["lya"], [1])
        np.testing.assert_array_equal(blocks["lya_curve_pca"], [4])
        self.ckpt["feature_block_slices"]["lya"] = (0, 3)
        with self.assertRaisesRegex(ValueError, "partition"):
            v.feature_block_indices(self.ckpt)

    def test_width_physical_noise_and_prior_units(self):
        names, low, high = v.parameter_metadata(self.ckpt)
        samples = np.stack([low, (low + high) / 2, high])[None, ...]
        result = v.width_summary(samples, low, high, names)
        np.testing.assert_allclose(result["widths"], samples.std(axis=1, ddof=1))
        self.assertAlmostEqual(result["physical_widths"][0, 1], np.std(10. ** samples[0, :, 1], ddof=1), places=9)
        self.assertAlmostEqual(result["physical_widths_normalized"][0, 1],
                               result["physical_widths"][0, 1] / (10. ** high[1] - 10. ** low[1]), places=6)
        self.assertEqual(result["physical_parameter_names"][1], "weight_noise")

    def test_recovery_preserves_log_coordinates_and_undefined_correlations(self):
        truth = np.array([[1., -6.], [2., -5.], [3., -4.]])
        samples = np.repeat((truth + [.5, .25])[:, None, :], 4, axis=1)
        result = v.recovery_summary(samples, truth, np.array([0., -6.]), np.array([4., -3.]))
        np.testing.assert_allclose(result["mean_abs_error_per_param"], [.5, .25])
        np.testing.assert_allclose(result["mean_normalized_abs_error_per_param"], [.125, .25 / 3])
        np.testing.assert_allclose(result["rmse_per_param"], [.5, .25])
        np.testing.assert_allclose(result["pearson_r_per_param"], [1., 1.])
        small = v.recovery_summary(samples[:2], truth[:2], np.array([0., -6.]), np.array([4., -3.]))
        self.assertTrue(np.isnan(small["pearson_r_per_param"]).all())
        constant = v.recovery_summary(np.ones((3, 4, 2)), truth, np.array([0., -6.]), np.array([4., -3.]))
        self.assertTrue(np.isnan(constant["pearson_r_per_param"]).all())

    def test_coverage_is_marginal_equal_tailed(self):
        samples = np.broadcast_to(np.linspace(0, 1, 101)[None, :, None], (2, 101, 2))
        truth = np.array([[.5, .5], [.05, .95]])
        result = v.coverage_summary(samples, truth, [.5, .8, .9, .95])
        np.testing.assert_allclose(result["lower"][0, :, 0], [.25, .1, .05, .025])
        np.testing.assert_allclose(result["upper"][0, :, 0], [.75, .9, .95, .975])
        np.testing.assert_allclose(result["coverage_per_param"][[0, 1, 3]], [[.5, .5], [.5, .5], [1, 1]])
        self.assertEqual(result["inside_matrix"].shape, (2, 4, 2))

    def test_predictive_distances_baseline_exclusion_and_seed(self):
        training = np.arange(30, dtype=float).reshape(10, 3)
        indices = np.array([1, 7])
        targets = training[indices]
        predictions = targets[:, None, :] + np.array([[[1, 0, 0], [0, 2, 0], [0, 0, 3]]])
        blocks = {"dfa": np.array([0]), "lya": np.array([1, 2])}
        result = v.predictive_summary(predictions, targets, training, indices, blocks, np.random.default_rng(42))
        np.testing.assert_allclose(result["distances"], [[1, 2, 3], [1, 2, 3]])
        for i, target in enumerate(indices):
            self.assertNotIn(target, result["baseline_indices"][i])
            self.assertEqual(len(set(result["baseline_indices"][i])), 3)
        np.testing.assert_allclose(result["baseline_distances"], np.linalg.norm(result["baseline_x"] - targets[:, None, :], axis=-1))
        again = v.predictive_summary(predictions, targets, training, indices, blocks, np.random.default_rng(42))
        np.testing.assert_array_equal(result["baseline_indices"], again["baseline_indices"])
        self.assertAlmostEqual(result["median_distance_ratio"], 2 / np.median(result["baseline_distances"]))
        with self.assertRaisesRegex(ValueError, "K\\+1"):
            v.predictive_summary(predictions, targets, training[:3], indices, blocks, np.random.default_rng(42))
        self.assertTrue(np.isnan(v.ratio(0., 0.)))

    def test_sensitivity_attribution_bounds_and_noise(self):
        _, low, high = v.parameter_metadata(self.ckpt)
        design, steps = v.sensitivity_design(np.stack([low, high]), low, high, .05)
        self.assertTrue(np.all((design >= low) & (design <= high)))
        for j in range(7):
            np.testing.assert_array_equal(np.flatnonzero(design[0, j + 1] != design[0, 0]), [j])
        self.assertTrue((steps[0] > 0).all())
        self.assertTrue((steps[1] < 0).all())
        # A known diagonal linear simulator in prior-normalized coordinates.
        features = ((design - low) / (high - low))[:, :, None, :] + np.array([-.001, .001])[None, None, :, None]
        result = v.sensitivity_summary(features, steps, low, high, {"all": np.arange(7)})
        np.testing.assert_allclose(result["feature_derivative"], np.broadcast_to(np.eye(7), (2, 7, 7)), atol=2e-6)
        self.assertTrue((result["delta_standard_error"] > 0).all())

    def test_output_has_plot_arrays_and_no_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            v.save_result(directory, "result", {"distances": np.ones((2, 3)), "r": np.nan,
                                               "samples": [torch.ones(2, 7), torch.zeros(2, 7)]})
            meta = json.loads((Path(directory) / "result.json").read_text())
            self.assertIsNone(meta["r"])
            with np.load(Path(directory) / "result.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays[meta["distances"]["npz_key"]].shape, (2, 3))
                self.assertEqual(arrays["samples"].shape, (2, 2, 7))

    def test_nonfinite_inputs_are_reported_without_sanitizing(self):
        for value in (np.array(np.nan), np.array([[1., np.inf]])):
            with self.assertRaisesRegex(ValueError, "NaN/Inf"):
                v.require_finite("fixture", value)


class OrchestrationValidation(unittest.TestCase):
    def setUp(self):
        self.ns = load_orchestration()
        self.ckpt = checkpoint_fixture()

    def test_cli_selection_legacy_aliases_and_required_simulator_settings(self):
        parse = self.ns["parse_validation_args"]
        args, sim = parse(["--tests", "all", "--n-samples", "5", "--max-items", "2", "-n", "4000", "-dt", ".1", "-cs", "3"])
        self.assertEqual(len(args.tests), 6)
        self.assertEqual((args.consistency_samples, args.consistency_items), (5, 2))
        self.assertEqual(sim, ["-cs", "3", "-n", "4000", "-dt", "0.1"])
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in ([], ["-n", "4000"], ["-n", "4000", "-dt", ".1", "--tests", "unknown"]):
                with self.assertRaises(SystemExit):
                    parse(argv)
        with self.assertRaisesRegex(RuntimeError, "one MPI rank"):
            self.ns["require_single_rank"](SimpleNamespace(Get_size=lambda: 4))

    def test_batched_resimulation_noise_conversion_and_saved_scaling(self):
        _, low, high = v.parameter_metadata(self.ckpt)
        theta = np.broadcast_to((low + high) / 2, (5, 7)).copy()
        seen = []
        def simulator(physical, comm, logger):
            seen.append(physical.copy())
            return physical
        def features(eeg, *args, **kwargs):
            return np.broadcast_to(np.arange(9, dtype=np.float32), (len(eeg), 9)).copy()
        self.ns.update(run_tvb_gpu=simulator, compute_features=features)
        for key in ("fcpca", "dfa_pca", "lya_pca", "pli_pca", "aecc_pca"):
            self.ckpt[key] = object()
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.ns["_resimulate_features"](theta, self.ckpt, 3)
        self.assertEqual([len(batch) for batch in seen], [3, 2])
        np.testing.assert_allclose(np.concatenate(seen)[:, 1], 10 ** theta[:, 1])
        np.testing.assert_allclose(theta[:, 1], (low[1] + high[1]) / 2)
        expected = (np.arange(9)[self.ckpt["feature_keep"]] - self.ckpt["x_mean"]) / self.ckpt["x_std"]
        np.testing.assert_allclose(result["x_resim"], np.broadcast_to(expected, (5, 8)))

    def test_predictive_resimulates_individual_draws(self):
        draws = torch.arange(2 * 3 * 7, dtype=torch.float32).reshape(2, 3, 7)
        seen = []
        self.ns["_draw_posteriors"] = lambda *args: draws
        def simulate(theta, ckpt, batch):
            seen.append(theta.clone())
            x = np.ones((len(theta), 8), dtype=np.float32)
            return {"x_resim": x, "x_raw": x, "x_raw_full": np.ones((len(theta), 9))}
        self.ns["_resimulate_features"] = simulate
        self.ckpt["xs"] = np.arange(80, dtype=np.float32).reshape(10, 8)
        result = self.ns["posterior_predictive_check"](None, torch.zeros(2, 8), self.ckpt, [0, 1], 3, 2)
        np.testing.assert_array_equal(seen[0], draws.reshape(6, 7))
        self.assertEqual(result["distances"].shape, (2, 3))
        self.assertEqual(result["theta_samples"].shape, (2, 3, 7))

    def test_main_all_tests_saves_results_and_reuses_samples(self):
        from tvbgpu.analysis.sbi_features import FeatureConfig
        checkpoint = self.ckpt
        checkpoint["feature_config"] = FeatureConfig().metadata()
        checkpoint["feature_pipeline_version"] = "test-fixture"
        low, high = checkpoint["prior_low"], checkpoint["prior_high"]
        checkpoint["thetas"] = torch.tensor(np.linspace(low, high, 12))
        checkpoint["xs"] = torch.arange(12 * 8, dtype=torch.float32).reshape(12, 8) / 100
        checkpoint["xs_raw"] = checkpoint["xs"].numpy() * checkpoint["x_std"] + checkpoint["x_mean"]
        prior = SimpleNamespace(low=torch.tensor(low), high=torch.tensor(high))
        posterior = Mock()
        posterior.sample.side_effect = lambda shape, **kw: torch.tensor(np.linspace(low, high, shape[0]))
        self.ns["load_sbi_full"] = lambda *args: (posterior, checkpoint["thetas"], checkpoint["xs"],
                                                  torch.tensor(checkpoint["x_mean"]), torch.tensor(checkpoint["x_std"]),
                                                  Mock(), prior, checkpoint)
        driver_tree = ast.parse((Path(__file__).parents[1] / "model_driver_LarterBreakspear.py").read_text())
        driver = next(n for n in driver_tree.body if isinstance(n, ast.ClassDef) and n.name == "Driver_Setup")
        parser = next(n for n in driver.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
        exec(compile(ast.Module(body=[parser], type_ignores=[]), "<simulator argument parser>", "exec"), self.ns)
        self.ns["Driver_Setup"] = SimpleNamespace(parse_args=self.ns["parse_args"])
        def resim(theta, checkpoint, batch):
            return {"x_resim": np.ones((len(theta), 8)), "x_raw": np.ones((len(theta), 8)),
                    "x_raw_full": np.ones((len(theta), 9))}
        self.ns["_resimulate_features"] = resim
        self.ns["resimulation_consistency_test"] = lambda *args, **kwargs: {
            "x_resim": np.zeros((2, 8)), "mean_diff": torch.tensor(1.), "median_diff": torch.tensor(1.)}
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "checkpoint-marker"
            marker.write_text("CPU test fixture; no checkpoint is loaded")
            output = Path(directory) / "results"
            argv = ["Posterior_tests.py", "--tests", "all", "--checkpoint", str(marker), "--output-dir", str(output),
                    "-n", "4000", "-dt", ".1", "--sensitivity-items", "2"]
            for test in ("consistency", "predictive", "width", "recovery", "coverage"):
                argv += [f"--{test}-items", "2", f"--{test}-samples", "4"]
            with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(torch.cuda, "current_device", return_value=0), \
                    patch.object(torch.cuda, "get_device_name", return_value="CPU_TEST_STAND_IN"), \
                    patch.object(torch, "manual_seed"), contextlib.redirect_stdout(io.StringIO()):
                self.ns["main"]()
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["results"]), 6)
            self.assertTrue(all(r["status"] == "completed" for r in manifest["results"].values()))
            # Predictive draws twice; width/recovery/coverage share two other draws.
            self.assertEqual(posterior.sample.call_count, 4)
            with np.load(output / "predictive.npz", allow_pickle=False) as result:
                self.assertEqual(result["distances"].shape, (2, 4))
                self.assertEqual(result["baseline_distances"].shape, (2, 4))
            with np.load(output / "coverage.npz", allow_pickle=False) as result:
                self.assertEqual(result["inside_matrix"].shape, (2, 4, 7))


if __name__ == "__main__":
    unittest.main()
