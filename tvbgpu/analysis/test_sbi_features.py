"""Small orchestration parity test; DFA/LYA GPU kernels are stubbed, not retested."""
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.decomposition import PCA
from tvbgpu.analysis import sbi_features as sf
from tvbgpu.Posterior_tests import compute_features


def fake_dfa(x, logger, rank):
    n, c, _ = x.shape
    base = np.mean(np.abs(np.diff(x, axis=-1)), axis=-1)
    windows = sf.dfa_window_sizes().size
    log_s = np.broadcast_to(np.log(sf.dfa_window_sizes())[None, None, :], (n, c, windows)).copy()
    log_f = log_s * (0.5 + base[:, :, None]) + 0.01 * np.arange(windows)[None, None, :] ** 2
    return base.astype(np.float32), log_s.astype(np.float32), log_f.astype(np.float32)


def fake_lya(x, logger, rank):
    slope = np.mean(np.abs(np.diff(x, axis=-1)), axis=-1)
    curve = slope[:, :, None] * np.linspace(0, 1, 80)[None, None, :]
    return slope.astype(np.float32), curve.astype(np.float32)


class FeatureParity(unittest.TestCase):
    def test_training_vs_resimulation(self):
        rng = np.random.default_rng(17)
        n, c, t = 8, 61, 600
        time = np.arange(t) / 500.0
        eeg = rng.normal(size=(n, c, t)).astype(np.float32)
        eeg += (0.5 * np.sin(2 * np.pi * 10 * time))[None, None, :]
        config = sf.FeatureConfig()
        with patch.object(sf, '_compute_dfa', fake_dfa), patch.object(sf, '_compute_lya', fake_lya):
            training = sf.extract_signal_features(eeg, config)
            mapping = {'fc_pca': 'fc_flat', 'dfa_curve_pca': 'dfa_curve_flat',
                       'lya_curve_pca': 'lya_curve_flat', 'pli_pca': 'pli_flat',
                       'aecc_pca': 'aecc_flat'}
            models = {}
            fitted = {'dfa': training['dfa'], 'lya': training['lya'], 'alpha': training['alpha']}
            for target, source in mapping.items():
                models[target] = PCA(n_components=2, svd_solver='full')
                fitted[target] = models[target].fit_transform(training[source])
            train_full, names, boundaries = sf.assemble_post_pca(fitted)
            observed = []
            real_extract = sf.extract_signal_features
            def capture_extract(*args, **kwargs):
                blocks = real_extract(*args, **kwargs)
                observed.append(blocks)
                return blocks
            with patch.object(sf, 'extract_signal_features', side_effect=capture_extract):
                resim_full = compute_features(
                    eeg, models['fc_pca'], models['dfa_curve_pca'], models['lya_curve_pca'],
                    models['pli_pca'], models['aecc_pca'],
                    config.precuneus_idx, config.acc_idx, config.dmn_idx, config=config,
                )
            self.assertEqual(len(observed), 1)
            for name in sf.SIGNAL_BLOCK_ORDER:
                np.testing.assert_allclose(training[name], observed[0][name], atol=1e-5, rtol=1e-5,
                                           err_msg=f'pre-PCA mismatch in {name}')
            np.testing.assert_allclose(train_full, resim_full, atol=1e-4, rtol=1e-4)
            self.assertEqual(len(names), train_full.shape[1])
            self.assertEqual(boundaries['dfa_curve_pca'].stop - boundaries['dfa_curve_pca'].start, 2)
            self.assertTrue(np.any(training['dfa_curve_flat'] != fake_dfa(sf.preprocess_eeg(eeg), None, 0)[1].reshape(n, -1)))

    def test_axes_and_temporal_standardization(self):
        rng = np.random.default_rng(23)
        eeg = rng.normal(size=(2, 61, 600)).astype(np.float32)
        shifted = eeg.copy()
        shifted[1] += 100.0
        norm = sf.preprocess_eeg(shifted)
        np.testing.assert_allclose(norm[0], sf.preprocess_eeg(eeg)[0], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(norm.mean(axis=-1), 0, atol=2e-5)
        with self.assertRaisesRegex(ValueError, 'batch, 61 channels'):
            sf.preprocess_eeg(np.transpose(eeg, (0, 2, 1)))


if __name__ == '__main__':
    unittest.main()
