"""CPU checks for the existing consistency/predictive comparison workflow."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from tvbgpu.analysis import posterior_validation as v
from tvbgpu.analysis.test_posterior_validation import checkpoint_fixture, load_orchestration
from tvbgpu.compare_validation import plot_comparison, verify_comparison


class ComparisonTests(unittest.TestCase):
    def test_shared_observations_created_once_and_reused_with_model_transforms(self):
        reference = checkpoint_fixture()
        reference.update(thetas=np.linspace(reference['prior_low'], reference['prior_high'], 10),
                         signal_block_order=['dfa'], feature_pipeline_version='fixture')
        model = copy.deepcopy(reference)
        model['x_mean'] = reference['x_mean'] + 100
        simulate = Mock(return_value={'dfa': np.arange(6).reshape(2, 3), 'alpha_valid': np.ones(2, bool)})
        seen = []
        def transform(blocks, ckpt):
            seen.append(blocks['dfa'].copy())
            return blocks['dfa'] - ckpt['x_mean'][0]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / 'small.pt'
            checkpoint_path.write_text('fixture')
            inputs = Path(tmp) / 'inputs.npz'
            with patch.object(v, 'transform_signal_blocks', side_effect=transform), \
                 patch('tvbgpu.analysis.sbi_checkpoint.load_checkpoint', return_value=reference):
                a, _ = v.shared_inputs(inputs, reference, checkpoint_path, {'n_time': 4000}, 2, 42, simulate)
                b, scoring = v.shared_inputs(inputs, model, 'unused', {'n_time': 4000}, 2, 42, simulate)
                self.assertIs(scoring, reference)
                self.assertEqual(simulate.call_count, 1)
                np.testing.assert_array_equal(a['theta_true'], b['theta_true'])
                np.testing.assert_array_equal(a['reference_x'], b['reference_x'])
                np.testing.assert_array_equal(b['model_x'], a['model_x'] - 100)
                np.testing.assert_array_equal(seen[0], seen[2])
                self.assertEqual(a['sha256'], b['sha256'])
                with self.assertRaisesRegex(ValueError, 'seed'):
                    v.shared_inputs(inputs, model, 'unused', {'n_time': 4000}, 2, 43, simulate)
                with self.assertRaisesRegex(ValueError, 'simulator_args'):
                    v.shared_inputs(inputs, model, 'unused', {'n_time': 4}, 2, 42, simulate)
                bad = dict(model, parameter_names=list(reversed(v.PARAMETER_ORDER)))
                with self.assertRaisesRegex(ValueError, 'parameter order'):
                    v.shared_inputs(inputs, bad, 'unused', {'n_time': 4000}, 2, 42, simulate)
                checkpoint_path.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'reference checkpoint changed'):
                    v.shared_inputs(inputs, model, 'unused', {'n_time': 4000}, 2, 42, simulate)

    def test_predictive_conditions_on_model_features_but_scores_reference_features(self):
        ns, reference = load_orchestration(), checkpoint_fixture()
        reference['xs'] = np.arange(80).reshape(10, 8)
        own_x, target = torch.full((2, 8), 90.), torch.zeros(2, 8)
        draw = Mock(return_value=torch.ones(2, 5, 7))
        ns['_draw_posteriors'] = draw
        simulate = Mock(return_value={'x_resim': np.ones((10, 8)), 'x_raw': np.ones((10, 8)),
                                      'x_raw_full': np.ones((10, 9))})
        ns['_resimulate_features'] = simulate
        result = ns['posterior_predictive_check'](None, target, reference, [0, 1], 5, 2, conditioning_xs=own_x)
        np.testing.assert_array_equal(draw.call_args.args[1], own_x)
        self.assertIs(simulate.call_args.args[1], reference)
        np.testing.assert_allclose(result['distances'], np.sqrt(8))
        np.testing.assert_array_equal(result['x_target'], target)

    def test_consistency_keeps_mean_then_noise_conversion_and_reference_metric(self):
        ns, reference = load_orchestration(), checkpoint_fixture()
        reference.update(xs_raw=np.zeros((10, 8)), xs=np.zeros((10, 8)))
        for key in ('fcpca', 'dfa_pca', 'lya_pca', 'pli_pca', 'aecc_pca'):
            reference[key] = None
        low, high = reference['prior_low'], reference['prior_high']
        posterior = Mock()
        posterior.sample.return_value = torch.tensor(np.stack([low, high]))
        simulator = Mock(return_value=np.zeros((2, 61, 600)))
        ns['run_tvb_gpu'] = simulator
        ns['compute_features'] = lambda **kwargs: np.ones((2, 9))
        own_x = torch.full((2, 8), 90.)
        with contextlib.redirect_stdout(io.StringIO()):
            result = ns['resimulation_consistency_test'](
                posterior, torch.zeros(2, 8), np.zeros(8), np.ones(8), reference,
                [10, 11], [5, 6], [5, 6, 10, 11, 20, 21], n_samples=2, max_items=2,
                device='cpu', conditioning_xs=own_x)
        np.testing.assert_array_equal(posterior.sample.call_args.kwargs['x'], own_x[1])
        physical = simulator.call_args.args[0]
        np.testing.assert_allclose(physical[:, 1], 10 ** ((low[1] + high[1]) / 2))
        np.testing.assert_allclose(result['diffs'], np.sqrt(8))

    def test_compact_output_plots_and_mismatch_rejection(self):
        base = dict(seed=42, simulator_args={'n_time': 4000}, shared_inputs_sha256='shared',
                    reference_sha256='reference', checkpoint='small.pt', training_simulations=8, tests={})
        common = dict(theta_true=np.ones((2, 7)), x_target=np.zeros((2, 8)), observation_indices=np.array([1, 2]))
        consistency = dict(common, diffs=np.array([1., 2.]), mean_diff=1.5, median_diff=1.)
        predictive = v.predictive_summary(np.ones((2, 5, 8)), common['x_target'],
                                         np.arange(80).reshape(10, 8), [1, 2], {}, np.random.default_rng(42))
        predictive.update(common)
        base['tests']['consistency'] = v.compact_test_result('consistency', consistency, 50)
        base['tests']['predictive'] = v.compact_test_result('predictive', predictive, 5)
        full = copy.deepcopy(base)
        full.update(checkpoint='full.pt', training_simulations=131072)
        full['tests']['consistency']['diffs'] = [.5, 1.]
        verify_comparison(base, full)
        for key in ('shared_inputs_sha256', 'reference_sha256', 'seed'):
            with self.assertRaisesRegex(ValueError, key):
                verify_comparison(base, dict(full, **{key: None}))
        bad = copy.deepcopy(full)
        bad['tests']['predictive']['theta_true'][0][0] = 99
        with self.assertRaisesRegex(ValueError, 'theta_true'):
            verify_comparison(base, bad)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, run in [('small', base), ('full', full)]:
                (root / name).mkdir()
                v.write_comparison(root / name, run)
            import matplotlib
            matplotlib.use('Agg')
            figures = plot_comparison(root / 'small', root / 'full', root / 'plots')
            self.assertEqual(set(figures), {'consistency', 'predictive'})
            self.assertEqual(len(list((root / 'plots').glob('*.png'))), 2)
            import matplotlib.pyplot as plt
            for fig in figures.values():
                plt.close(fig)

    def test_shared_cli_is_optional_and_restricts_tests(self):
        parse = load_orchestration()['parse_validation_args']
        args, _ = parse(['--tests', 'consistency,predictive', '--shared-inputs', 'inputs.npz',
                         '--seed', '42', '-n', '4000', '-dt', '.1'])
        self.assertEqual(args.shared_inputs, 'inputs.npz')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(['--tests', 'coverage', '--shared-inputs', 'inputs.npz', '-n', '4000', '-dt', '.1'])


if __name__ == '__main__':
    unittest.main()
