"""CPU contract tests. Synthetic signals do not validate scientific EEG features."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
                                      '--output-dir', str(tmp/'results'),'--epoch-index','0','--num-samples','20'])
            with patch.object(infer_eeg, 'load_epoch', return_value=(None, {'condition': 'EO'})), \
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
