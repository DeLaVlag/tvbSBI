"""Comparison contracts using known draws; no training or inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tvbgpu import compare_posteriors as cp


def metadata(checkpoint, n=101):
    return dict(checkpoint=checkpoint, eeg_input='/input/sub-001.vhdr', input_format='ebrains-synthetic',
                parameter_names=list(cp.PARAMETER_ORDER), prior_low=[0.] * 7, prior_high=[1.] * 7,
                feature_pipeline_version='test-version', feature_config={'fs': 500, 'n_channels': 61},
                preprocessing_config={'n_times': 4001}, feature_names_full=['dfa_1', 'lya_1'],
                feature_names=['dfa_1', 'lya_1'], feature_keep=[True, True],
                raw_feature_dimension=2, final_feature_dimension=2,
                raw_feature_representation='post-PCA', num_samples=n,
                eeg=dict(channel_names=['E1'], sfreq=500, n_times=4001, epoch_index=0,
                         condition='rest', start_sample=0, stop_sample=4001))


class ComparisonTests(unittest.TestCase):
    def test_statistics(self):
        small = np.tile(np.linspace(.3, .9, 101)[:, None], (1, 7))
        full = np.tile(np.linspace(.4, .6, 101)[:, None], (1, 7))
        before = small.copy()
        rows = cp.compare_samples(small, full, np.zeros(7), np.ones(7))
        np.testing.assert_array_equal(small, before)
        for row in rows:
            self.assertAlmostEqual(row['small_median'], .6)
            self.assertAlmostEqual(row['small_ci50_low'], .45)
            self.assertAlmostEqual(row['small_ci50_high'], .75)
            self.assertAlmostEqual(row['small_ci95_low'], .315)
            self.assertAlmostEqual(row['small_ci95_high'], .885)
            self.assertAlmostEqual(row['small_full_width95_ratio'], 3.)
            self.assertAlmostEqual(row['normalized_median_displacement'], .1)
        full[:] = .5
        self.assertIsNone(cp.compare_samples(small, full, np.zeros(7), np.ones(7))[0]['small_full_width95_ratio'])

    def test_metadata_checks(self):
        small, full = metadata('/models/workshop.pt'), metadata('/models/sbi_full.pt')
        cp.verify_runs(small, full, 'workshop.pt', 'sbi_full.pt')
        for key, value in [('eeg_input', '/input/different.vhdr'), ('parameter_names', list(reversed(cp.PARAMETER_ORDER))),
                           ('feature_config', {'fs': 200}), ('checkpoint', '/models/workshop.pt')]:
            changed = copy.deepcopy(full)
            changed[key] = value
            with self.assertRaises(ValueError):
                cp.verify_runs(small, changed)
        changed = copy.deepcopy(full)
        changed['eeg']['epoch_index'] = 1
        with self.assertRaisesRegex(ValueError, 'epoch_index'):
            cp.verify_runs(small, changed)
        with self.assertRaisesRegex(ValueError, 'full checkpoint'):
            cp.verify_runs(small, full, expected_full='wrong.pt')
        full['feature_keep'] = [True, False]
        full['feature_names'] = ['dfa_1']
        full['final_feature_dimension'] = 1
        self.assertFalse(cp.verify_runs(small, full)['feature_masks_identical'])

    def test_files_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            small, full = root / 'small', root / 'full'
            for directory, checkpoint in ((small, 'workshop.pt'), (full, 'sbi_full.pt')):
                directory.mkdir()
                np.save(directory / 'posterior_samples.npy', np.tile(np.linspace(.1, .9, 101)[:, None], (1, 7)))
                (directory / 'inference_metadata.json').write_text(json.dumps(metadata(checkpoint)))
            args = cp.parse_args([str(small), str(full), '--output-dir', str(root),
                                  '--expected-small-checkpoint', 'workshop.pt', '--expected-full-checkpoint', 'sbi_full.pt'])
            self.assertEqual(len(cp.run(args)), 7)
            self.assertGreater((root / 'posterior_comparison.png').stat().st_size, 1000)
            self.assertEqual(len((root / 'posterior_comparison.csv').read_text().splitlines()), 8)
            report = json.loads((root / 'posterior_comparison_metadata.json').read_text())
            self.assertEqual(report['small_samples_shape'], [101, 7])
            bad = metadata('workshop.pt', n=100)
            (small / 'inference_metadata.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'num_samples'):
                cp.load_run(small)
            bad['num_samples'] = 101
            (small / 'inference_metadata.json').write_text(json.dumps(bad))
            np.save(small / 'posterior_samples.npy', np.full((101, 7), np.nan))
            with self.assertRaisesRegex(ValueError, 'finite posterior'):
                cp.load_run(small)


if __name__ == '__main__':
    unittest.main()
