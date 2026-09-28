"""CPU contract tests. Synthetic signals do not validate scientific EEG features."""
import csv
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import mne
import numpy as np
import torch
from sklearn.decomposition import PCA

from tvbgpu import infer_eeg
from tvbgpu.analysis import sbi_features as sf
from tvbgpu.analysis.posterior_validation import PARAMETER_ORDER
from tvbgpu.analysis.sbi_checkpoint import load_checkpoint, build_posterior, draw_posteriors


def fixture():
    rng = np.random.default_rng(42)
    config = sf.FeatureConfig()
    blocks = dict(dfa=rng.normal(size=(40, 61)).astype('f'),
                  lya=rng.normal(size=(40, 61)).astype('f'),
                  alpha=rng.normal(size=(40, 5)).astype('f'))
    models = {}
    for name in ('fc_pca', 'dfa_curve_pca', 'lya_curve_pca', 'pli_pca', 'aecc_pca'):
        models[name] = PCA(n_components=2).fit(rng.normal(size=(40, 4)))
        blocks[name] = models[name].transform(rng.normal(size=(40, 4)))
    full, names, slices = sf.assemble_post_pca(blocks)
    keep = np.ones(full.shape[1], dtype=bool)
    keep[[2, 6]] = False
    retained = torch.tensor(full[:, keep])
    mean, std = retained.mean(0), retained.std(0) + 1e-8
    return dict(feature_pipeline_version=sf.FEATURE_PIPELINE_VERSION,
                feature_config=config.metadata(), preprocessing_config=dict(input_axes='batch_channels_time',
                    projection_before_standardization=True, standardization_axis='time', standardization_eps=1e-8),
                feature_names_full=names, feature_names=[n for n,k in zip(names,keep) if k],
                feature_keep=keep, x_full_dim=len(names), x_dim=int(keep.sum()),
                feature_block_order=list(sf.POST_PCA_BLOCK_ORDER),
                signal_block_order=list(sf.SIGNAL_BLOCK_ORDER), alpha_feature_names=list(sf.ALPHA_FEATURE_KEYS),
                feature_block_slices={n:(s.start,s.stop) for n,s in slices.items()},
                x_mean=mean, x_std=std, xs=(retained-mean)/std, xs_raw=retained.numpy(),
                parameter_names=list(PARAMETER_ORDER), theta_dim=7,
                prior_low=torch.tensor([.2,-6,.25,1,1,.5,.2]),
                prior_high=torch.tensor([1.2,-3,.55,20,1.3,.9,2.5]),
                channel_names=[f'C{i}' for i in range(61)], n_channels=61,
                fcpca=models['fc_pca'], dfa_pca=models['dfa_curve_pca'], lya_pca=models['lya_curve_pca'],
                pli_pca=models['pli_pca'], aecc_pca=models['aecc_pca'])


class InferenceContracts(unittest.TestCase):
    def test_ebrains_256_channel_fallback_and_saved_rate(self):
        c = fixture()
        del c['channel_names']
        c['feature_config']['fs'] = 200.0
        c['preprocessing_config']['n_times'] = 600
        names = [f'E{i + 1}' for i in range(256)]
        raw = mne.io.RawArray(np.random.default_rng(11).normal(size=(256, 2000)) * 1e-6,
                              mne.create_info(names, 256.016, 'eeg'), verbose=False)
        indices = np.rint(np.linspace(0, 255, 61)).astype(int)
        expected = raw.copy().pick(indices.tolist()).resample(200).get_data(start=600, stop=1200)
        with patch('mne.io.read_raw_brainvision', return_value=raw), self.assertLogs(infer_eeg.LOG) as logs:
            data, meta = infer_eeg.load_ebrains_epoch('test.vhdr', c, epoch_index=1)
        np.testing.assert_allclose(data[0], expected)
        self.assertEqual(len(set(meta['selected_original_indices'])), 61)
        self.assertEqual(meta['selected_original_indices'], indices.tolist())
        self.assertEqual(meta['channel_names'], [names[i] for i in indices])
        self.assertEqual(meta['sfreq'], 200.0)
        self.assertTrue(meta['approximate_channel_selection'])
        self.assertIn('WORKSHOP/DEMO', meta['observation_label'])
        self.assertTrue(any('Selected channels:' in line for line in logs.output))

    def test_spatial_selection_is_deterministic_and_preserves_order(self):
        raw = mne.io.RawArray(np.zeros((256, 601)),
                              mne.create_info([f'E{i}' for i in range(256)], 500, 'eeg'), verbose=False)
        positions = np.random.default_rng(12).normal(size=(256, 3))
        positions /= np.linalg.norm(positions, axis=1, keepdims=True)
        montage = mne.channels.make_dig_montage(ch_pos=dict(zip(raw.ch_names, positions * .09)),
                                               coord_frame='head')
        raw.set_montage(montage)
        indices, method = infer_eeg.approximate_channel_indices(raw, 61)
        self.assertEqual(method, 'approximate_spatial_farthest_point')
        self.assertEqual(len(set(indices)), 61)
        self.assertTrue(np.all(np.diff(indices) > 0))
        np.testing.assert_array_equal(indices, infer_eeg.approximate_channel_indices(raw, 61)[0])
        # One missing position makes the complete montage unusable for spatial coverage.
        raw.info['chs'][0]['loc'][:3] = np.nan
        fallback, method = infer_eeg.approximate_channel_indices(raw, 61)
        self.assertEqual(method, 'approximate_evenly_spaced_indices')
        np.testing.assert_array_equal(fallback, np.rint(np.linspace(0, 255, 61)).astype(int))

    def test_pre_inference_dimensions_rate_and_finiteness(self):
        c = fixture()
        data = np.zeros((1, 61, 600))
        full, processed = np.zeros((1, c['x_full_dim'])), np.zeros((1, c['x_dim']))
        estimator = Mock(condition_shape=(c['x_dim'],))
        infer_eeg.validate_inference_observation(data, {'sfreq': 500}, full, processed, c, estimator)
        for bad_data, meta, bad_full, bad_processed, model, error in [
            (data, {'sfreq': 200}, full, processed, estimator, 'rate'),
            (data[:, :60], {'sfreq': 500}, full, processed, estimator, '61 channels'),
            (data * np.nan, {'sfreq': 500}, full, processed, estimator, 'NaN or Inf'),
            (data, {'sfreq': 500}, full[:, :-1], processed, estimator, 'Full post-PCA'),
            (data, {'sfreq': 500}, full, processed[:, :-1], estimator, 'Transformed'),
            (data, {'sfreq': 500}, full, processed * np.nan, estimator, 'Transformed'),
            (data, {'sfreq': 500}, full, processed, Mock(condition_shape=(999,)), 'condition shape'),
        ]:
            with self.assertRaisesRegex(ValueError, error):
                infer_eeg.validate_inference_observation(bad_data, meta, bad_full, bad_processed, c, model)

    def test_ebrains_real_brainvision_and_window_selection(self):
        c = fixture()
        del c['channel_names']
        values = np.random.default_rng(8).normal(size=(8400, 61)).astype('<f4')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = tmp / 'test.vhdr'
            values.tofile(tmp / 'test.eeg')
            (tmp / 'test.vmrk').write_text(
                'Brain Vision Data Exchange Marker File, Version 1.0\n'
                '[Common Infos]\nDataFile=test.eeg\n[Marker Infos]\n')
            path.write_text('Brain Vision Data Exchange Header File Version 1.0\n'
                '[Common Infos]\nDataFile=test.eeg\nMarkerFile=test.vmrk\n'
                'DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=61\n'
                'SamplingInterval=2000\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n'
                '[Channel Infos]\n' + ''.join(f'Ch{i+1}=C{i},,1,µV\n' for i in range(61)))
            with self.assertLogs(infer_eeg.LOG, level='WARNING') as logs:
                data, meta = infer_eeg.load_ebrains_epoch(path, c, epoch_index=1)
            np.testing.assert_allclose(data[0], values[4001:8002].T * 1e-6, rtol=1e-6)
            self.assertEqual(data.shape, (1, 61, 4001))
            self.assertFalse(meta['observed_eeg_projected'])
            self.assertFalse(meta['epoch_length_verified'])
            self.assertEqual(meta['channel_order_source'], 'ebrains_brainvision_order_assumed')
            self.assertTrue(any('assumes' in line for line in logs.output))
            self.assertTrue(any('4001' in line for line in logs.output))
            # Strict legacy mode must still require verified channel provenance.
            with self.assertRaisesRegex(ValueError, 'verified channel order'):
                infer_eeg.load_epoch(path, c, None, 0)

    def test_ebrains_resampling_and_rejections(self):
        c = fixture()
        c['preprocessing_config']['n_times'] = 600
        values = np.random.default_rng(9).normal(size=(61, 2400)) * 1e-6
        raw = mne.io.RawArray(values, mne.create_info(c['channel_names'], 1000, 'eeg'), verbose=False)
        expected = raw.copy().resample(500).get_data()[:, 600:1200]
        with patch('mne.io.read_raw_brainvision', return_value=raw.copy()):
            data, meta = infer_eeg.load_ebrains_epoch('test.vhdr', c, epoch_index=1)
        np.testing.assert_allclose(data[0], expected)
        self.assertTrue(meta['resampled'])
        self.assertTrue(meta['epoch_length_verified'])
        for kwargs, error in [({'window_samples': 601}, 'disagrees'),
                              ({'epoch_index': 2}, 'exceeds')]:
            with patch('mne.io.read_raw_brainvision', return_value=raw.copy()):
                with self.assertRaisesRegex(ValueError, error):
                    infer_eeg.load_ebrains_epoch('test.vhdr', c, **kwargs)
        cases = [(raw.copy().drop_channels(['C0']), 'requires 61'),
                 (raw.copy().reorder_channels(list(reversed(raw.ch_names))), 'names/order')]
        nonfinite = raw.copy()
        nonfinite._data[0, -1] = np.nan  # Even outside the selected window must fail.
        cases.append((nonfinite, 'NaN or Inf'))
        bad = raw.copy()
        bad.info['bads'] = ['C0']
        cases.append((bad, 'bad channels'))
        for input_raw, error in cases:
            with patch('mne.io.read_raw_brainvision', return_value=input_raw):
                with self.assertRaisesRegex(ValueError, error):
                    infer_eeg.load_ebrains_epoch('test.vhdr', c)

    def test_ebrains_connectivity_order_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            atlas, weights, distances = [Path(tmp) / name for name in ('atlas.tsv', 'w.tsv', 'd.tsv')]
            dk = infer_eeg.dk68_reference_labels()
            sub = ['cerebellumcortex', 'thalamus', 'caudate', 'putamen', 'pallidum',
                   'hippocampus', 'amygdala', 'accumbensarea']
            names = [n + '_L' for n in sub] + dk[:34] + [n + '_R' for n in sub] + dk[34:]
            # Actual EBRAINS layout: both headings matched the old allowlist.
            atlas.write_text('label\tname\n' + ''.join(f'{n}\tDescription {i}\n' for i, n in enumerate(names)))
            matrix = np.arange(84 * 84).reshape(84, 84)
            np.savetxt(weights, matrix, delimiter='\t')
            np.savetxt(distances, matrix + 1, delimiter='\t')
            w, d, labels, metadata = infer_eeg.load_ebrains_dk68_connectivity(
                atlas, weights, distances, return_metadata=True)
            self.assertEqual(metadata['subcortical_regions'], 14)
            self.assertEqual(metadata['cerebellar_regions'], 2)
            self.assertEqual(metadata['atlas_label_column'], 0)
            self.assertEqual(w.shape, (68, 68))
            self.assertEqual(labels, dk)
            self.assertEqual(w[0, 0], matrix[8, 8])
            self.assertEqual(w[33, 34], matrix[41, 50])
            self.assertEqual(w[-1, -1], matrix[83, 83])
            np.testing.assert_array_equal(d, w + 1)
            for invalid in (np.zeros((68, 68)), np.full((84, 84), np.nan)):
                np.savetxt(weights, invalid, delimiter='\t')
                with self.assertRaises(ValueError):
                    infer_eeg.load_ebrains_dk68_connectivity(atlas, weights, distances)
            np.savetxt(weights, matrix, delimiter='\t')
            atlas.write_text('name\n' + 'region\n' * 83)
            with self.assertRaisesRegex(ValueError, 'not verified DK68'):
                infer_eeg.load_ebrains_dk68_connectivity(atlas, weights, distances)

    def test_dk68_atlas_formats_and_no_permutation(self):
        dk = infer_eeg.dk68_reference_labels()
        with tempfile.TemporaryDirectory() as tmp:
            atlas, weights = Path(tmp) / 'atlas.tsv', Path(tmp) / 'weights.tsv'
            matrix = np.arange(68 * 68).reshape(68, 68)
            np.savetxt(weights, matrix, delimiter='\t')
            formats = [('\n'.join(dk), False),
                       ('unfamiliar\n' + '\n'.join(dk), True),
                       (' ROI_NAME \tindex\n' + '\n'.join(f'{name}\t{i}' for i, name in enumerate(dk)), True),
                       ('title\themisphere\n' + '\n'.join(
                           f'{name[:-2]}\t{"left" if name.endswith("_L") else "right"}' for name in dk), True)]
            for content, header in formats:
                atlas.write_text(content)
                w, d, labels, metadata = infer_eeg.load_ebrains_dk68_connectivity(
                    atlas, weights, return_metadata=True)
                np.testing.assert_array_equal(w, matrix)
                self.assertIsNone(d)
                self.assertEqual(len(labels), 68)
                self.assertEqual(metadata['atlas_has_header'], header)
                self.assertTrue(metadata['matches_simulator_order'])
            # Permuted DK84: selection is anatomical, not fixed blocks or first 68.
            sub = [f'{hemi}-{name}' for hemi in ('Left', 'Right') for name in (
                'Thalamus-Proper', 'Caudate', 'Putamen', 'Pallidum', 'Hippocampus',
                'Amygdala', 'Accumbens-area', 'VentralDC')]
            cortex = [('ctx-lh-' if name.endswith('_L') else 'ctx-rh-') + name[:-2] for name in dk]
            order = np.random.default_rng(14).permutation(84)
            names = [ (cortex + sub)[i] for i in order ]
            atlas.write_text('strange_label\tnumeric_id\n' + '\n'.join(f'{n}\t{i}' for i, n in enumerate(names)))
            matrix = np.arange(84 * 84).reshape(84, 84)
            np.savetxt(weights, matrix, delimiter='\t')
            w, _, labels, metadata = infer_eeg.load_ebrains_dk68_connectivity(atlas, weights, return_metadata=True)
            selected = np.flatnonzero(order < 68)
            np.testing.assert_array_equal(w, matrix[np.ix_(selected, selected)])
            self.assertEqual(metadata['cortical_indices'], selected.tolist())
            self.assertEqual(labels, [names[i] for i in selected])
            self.assertFalse(metadata['matches_simulator_order'])
            atlas.write_text('strange_label\n' + '\n'.join(names[:-1] + ['unknown_anatomy']))
            with self.assertRaisesRegex(ValueError, 'not verified DK68'):
                infer_eeg.load_ebrains_dk68_connectivity(atlas, weights)

    def test_ebrains_cli(self):
        base = ['--checkpoint', 'model.pt', '--output-dir', 'results']
        ebrains = ['--input-format', 'ebrains-synthetic', '--eeg-vhdr', 'test.vhdr']
        args = infer_eeg.parse_args(base + ebrains)
        self.assertEqual(args.eeg, Path('test.vhdr'))
        self.assertEqual(args.epoch_index, 0)
        self.assertEqual(infer_eeg.parse_args(['--checkpoint', 'model.pt'] + ebrains).output_dir, Path('.'))
        self.assertEqual(infer_eeg.parse_args(base + ['--eeg', 'test-epo.fif']).input_format, 'auto')
        for extra in (['--condition', 'EO'], ['--dk-atlas', 'atlas.tsv'],
                      ['--eeg', 'other.vhdr'], ['--window-samples', '500']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                infer_eeg.parse_args(base + ebrains + extra)

    def test_ebrains_output_metadata_and_plot(self):
        # Exercise EBRAINS routing and artifacts, without training or GPU inference.
        c = fixture()
        del c['channel_names']
        raw_eeg = mne.io.RawArray(np.random.default_rng(10).normal(size=(256, 4001)) * 1e-6,
                                 mne.create_info([f'E{i+1}' for i in range(256)], 500, 'eeg'), verbose=False)
        samples = (c['prior_low'] + c['prior_high']).repeat(4, 1) / 2
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            args = infer_eeg.parse_args(['--checkpoint', str(tmp / 'model.pt'),
                '--input-format', 'ebrains-synthetic', '--eeg-vhdr', str(tmp / 'test.vhdr'),
                '--output-dir', str(tmp), '--num-samples', '4'])
            with patch('tvbgpu.analysis.sbi_checkpoint.load_checkpoint', return_value=c), \
                 patch('tvbgpu.analysis.sbi_checkpoint.build_posterior', return_value=(Mock(), Mock(condition_shape=(c['x_dim'],)), Mock())), \
                 patch('tvbgpu.analysis.sbi_checkpoint.draw_posteriors', return_value=samples[None]), \
                 patch('mne.io.read_raw_brainvision', return_value=raw_eeg), \
                 patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.manual_seed_all'), \
                 patch.object(infer_eeg, 'extract_observation', return_value=(
                     np.zeros((1, c['x_full_dim'])), np.zeros((1, c['x_dim'])))) as extract, \
                 patch.object(infer_eeg, 'plot_posterior', wraps=infer_eeg.plot_posterior) as plot:
                out = infer_eeg.run(args)
            self.assertEqual(extract.call_args.args[0].shape, (1, 61, 4001))
            self.assertIs(extract.call_args.args[1], c)
            self.assertIn('WORKSHOP/DEMO', plot.call_args.args[3])
            self.assertGreater((out / 'posterior_distributions.png').stat().st_size, 1000)
            metadata = json.loads((out / 'inference_metadata.json').read_text())
            self.assertIn('WORKSHOP/DEMO', metadata['observation_label'])
            self.assertEqual(metadata['input_format'], 'ebrains-synthetic')
            self.assertEqual(metadata['eeg']['channel_order_source'], 'approximate_evenly_spaced_indices')
            np.testing.assert_array_equal(np.load(out / 'posterior_samples.npy'), samples.numpy())

    def test_brainvision_crop_and_epoch_dimension(self):
        c = fixture()
        values = np.random.default_rng(7).normal(size=(63, 4100)) * 1e-6
        raw = mne.io.RawArray(values, mne.create_info(
            c['channel_names'] + ['extra1', 'extra2'], 500, 'eeg'), verbose=False)
        with patch('mne.io.read_raw_brainvision', return_value=raw) as reader:
            data, meta = infer_eeg.load_epoch('input.vhdr', c, None, 0)
            reader.assert_called_once_with('input.vhdr', preload=True)
        np.testing.assert_array_equal(data, values[np.newaxis, :61, :4001])
        self.assertEqual(meta['channel_names'], c['channel_names'])
        self.assertEqual(meta['n_times'], 4001)
        for condition, index in [('EO', 0), (None, 1)]:
            with self.assertRaisesRegex(ValueError, 'provides one epoch'):
                infer_eeg.load_epoch('input.vhdr', c, condition, index)
        c['channel_names'].reverse()
        with patch('mne.io.read_raw_brainvision', return_value=raw):
            with self.assertRaisesRegex(ValueError, 'names/order'):
                infer_eeg.load_epoch('input.vhdr', c, None, 0)

    def test_saved_transform_and_order(self):
        c = fixture()
        rng = np.random.default_rng(1)
        blocks = {name:rng.normal(size=(1, width)).astype('f') for name,width in
                  [('dfa',61),('lya',61),('alpha',5),('fc_flat',4),('dfa_curve_flat',4),
                   ('lya_curve_flat',4),('pli_flat',4),('aecc_flat',4)]}
        blocks['alpha_valid'] = np.ones(1, dtype=bool)
        with patch.object(sf,'extract_signal_features',return_value=blocks):
            raw, processed = infer_eeg.extract_observation(None, c)
            np.testing.assert_allclose(processed,(raw[:,c['feature_keep']]-c['x_mean'].numpy())/c['x_std'].numpy())
            c['feature_names_full'][0]='wrong'
            c['feature_names'][0]='wrong'
            with self.assertRaisesRegex(ValueError,'names/order'):
                infer_eeg.extract_observation(None, c)

    def test_summary_support_and_physical_noise(self):
        c = fixture()
        low, high = c['prior_low'].numpy(), c['prior_high'].numpy()
        samples = np.linspace(low,high,101)
        rows = infer_eeg.summarize(samples, list(PARAMETER_ORDER), low, high)
        self.assertAlmostEqual(rows[0]['ci90_low'],np.quantile(samples[:,0],.05))
        self.assertAlmostEqual(rows[-1]['mean'],np.mean(10.**samples[:,1].astype(float)))
        samples[0,0]=-1
        with self.assertRaisesRegex(ValueError,'support'):
            infer_eeg.summarize(samples, list(PARAMETER_ORDER), low, high)

    def test_prepared_epochs_and_rejections(self):
        c=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test-epo.fif'
            data=np.random.default_rng(1).normal(size=(2,61,600))
            epochs=mne.EpochsArray(data,mne.create_info(c['channel_names'],500,'eeg'),
                                  events=np.array([[0,0,1],[600,0,2]]),event_id={'EO':1,'EC':2},verbose=False)
            epochs.save(p,overwrite=True,verbose=False)
            eeg, meta= infer_eeg.load_epoch(p, c, 'EC', 0)
            self.assertEqual(eeg.shape,(1,61,600))
            self.assertEqual(meta['original_epoch_index'],1)
            with self.assertRaisesRegex(ValueError,'Multiple'):
                infer_eeg.load_epoch(p, c, None, 0)
            c['channel_names'].reverse()
            with self.assertRaisesRegex(ValueError,'names/order'):
                infer_eeg.load_epoch(p, c, 'EO', 0)
            c['channel_names'].reverse()
            c['feature_config']['fs']=200
            with self.assertRaisesRegex(ValueError,'sampling rate'):
                infer_eeg.load_epoch(p, c, 'EO', 0)

    def test_checkpoint_rejections(self):
        c=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.pt'
            torch.save(c,p)
            self.assertEqual(load_checkpoint(p)['x_dim'],c['x_dim'])
            c['feature_pipeline_version']='old'
            torch.save(c,p)
            with self.assertRaisesRegex(ValueError,'incompatible'):
                load_checkpoint(p)

    def test_small_trained_checkpoint_cpu_sampling_and_outputs(self):
        # Genuine SBI training/sampling; synthetic feature observations, not TVB training.
        from sbi.inference import SNPE
        from sbi.utils import BoxUniform
        torch.manual_seed(4)
        c=fixture()
        prior=BoxUniform(c['prior_low'],c['prior_high'])
        c['thetas']=prior.sample((40,))
        inference=SNPE(prior=prior,show_progress_bars=False)
        c['density_estimator']=inference.append_simulations(c['thetas'],c['xs']).train(
            max_num_epochs=1,training_batch_size=10,show_train_summary=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            model=tmp/'model.pt'
            torch.save(c,model)
            loaded=load_checkpoint(model)
            posterior,estimator,_=build_posterior(loaded,'cpu')
            estimator.eval()
            samples=draw_posteriors(posterior,c['xs'][:1],20)[0].numpy()
            rows= infer_eeg.summarize(samples, list(PARAMETER_ORDER), c['prior_low'].numpy(), c['prior_high'].numpy())
            infer_eeg.plot_posterior(samples, rows, tmp / 'plot.png')
            self.assertGreater((tmp/'plot.png').stat().st_size,1000)
            # Exercise orchestration/output only; explicitly stub GPU extraction, not sampling.
            args= infer_eeg.parse_args(['--checkpoint', str(model), '--eeg', str(tmp / 'external-epo.fif'),
                                      '--output-dir', str(tmp),'--epoch-index','0','--num-samples','20'])
            with patch.object(infer_eeg, 'load_epoch', return_value=(np.zeros((1, 61, 600)), {'condition': 'EO', 'sfreq': 500.0})), \
                 patch.object(infer_eeg, 'extract_observation', return_value=(np.zeros((1, c['x_full_dim'])), c['xs'][:1].numpy())), \
                 patch('torch.cuda.is_available',return_value=True), patch('torch.cuda.manual_seed_all'):
                out= infer_eeg.run(args)
            for filename in ['posterior_samples.npy','posterior_summary.csv','inference_metadata.json',
                             'posterior_distributions.png','eeg_features_raw.npy','eeg_features_processed.npy']:
                self.assertTrue((out/filename).is_file())
            self.assertEqual(np.load(out/'posterior_samples.npy').shape,(20,7))
            metadata=json.loads((out/'inference_metadata.json').read_text())
            self.assertEqual(metadata['parameter_names'],list(PARAMETER_ORDER))
            with (out/'posterior_summary.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),8)


if __name__=='__main__':
    unittest.main()
