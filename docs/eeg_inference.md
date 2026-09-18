# Workshop EEG inference endpoint

`../tvbgpu/infer_eeg.py` runs one EEG epoch through the frozen feature pipeline and the
validator's posterior construction. Workshop and full checkpoints follow exactly
the same path. Training budget is not an inference argument. There is no training,
EBRAINS retrieval, simulator import, posterior predictive simulation, or calibration
inside this endpoint.

## Architecture inspection

- **Active pipeline / file responsibilities:** `tvbgpu/SBI_metrics_driver.py`
  simulates, projects, extracts features, fits PCA, selects features, calls
  `analysis/sbi.py:do_sbi`, and saves the checkpoint. `Posterior_tests.py` remains
  the research validation entry point.
- **Feature flow:** `analysis/sbi_features.py:extract_signal_features` applies
  temporal per-channel z-scoring and extracts DFA, LYA, FC, DFA curves, LYA curves,
  PLI, AECC and alpha. `post_pca_blocks` applies five saved PCA objects;
  `assemble_post_pca` supplies canonical names and slices. The full feature vector
  is **post-PCA**, before retention/normalization. Its size is derived, not fixed.
- **Normalization flow:** training selects columns using `feature_keep`, then
  `do_sbi` saves the mean and sample standard deviation plus epsilon.
  `apply_saved_transform` applies the mask and divides by that saved standard
  deviation directly. Nothing is fitted on empirical EEG.
- **SBI flow:** shared `analysis/sbi_checkpoint.py` loads CPU checkpoint storage,
  validates metadata, moves only the density estimator/prior to the requested
  device, and reuses validation's SNPE MCMC construction: 2 chains, 20 warmup
  steps, thin=1, resample initialization. `draw_posteriors` retains the existing
  `posterior.sample` call for each observation. These settings are preserved,
  not a claim of convergence or calibration.
- **Resimulation flow:** remains in `Posterior_tests.py`; never invoked here.
- **Potential bugs/compatibility gaps:** the legacy empirical batch loader selects
  18 channels and resamples to 200 Hz, incompatible with the default current
  61-channel/500-Hz feature contract. The checkpoint lacks channel identities and
  epoch duration provenance. LYA uses its existing fixed kernel arguments,
  including tau=.001; this implementation does not change them. The canonical
  extractor's existing interpolation and nonfinite substitution policies remain;
  output finite checks cannot recover invalid values already replaced internally.
- **Legacy code:** `SBI_test_posterior.py` and the older model loaders in
  `analysis/sbi.py` use alternate formats. They are preserved, not classified as
  safe to delete. No dead-code removal was performed.
- **Minimal refactor:** validator checkpoint/posterior construction and sampling
  moved into a simulator-independent module. Validator retains its return tuple,
  device supplied by its caller, and MCMC settings. CPU checkpoint storage avoids
  copying training arrays to the GPU during EEG inference. Validation explicitly
  moves its training pairs as before. Shared compatibility checks are stricter.

## Input contract

Input is a **prepared MNE Epochs FIF** (`*-epo.fif` or supported compressed FIF),
not raw continuous EEG. One epoch is selected explicitly with `--epoch-index`
(zero-based **after** condition selection). Multiple conditions require
`--condition`; its value must be an exact `event_id` key. A filename containing
EO/EC is not evidence of an event label. Epochs are not averaged or pooled into a
subject posterior: each invocation conditions on one observation.

All channels must be EEG, in exactly the checkpoint channel order, with the exact
checkpoint sampling frequency, no unresolved marked bad channels, and finite
values. The canonical shape is `(1, channels, time)`; existing DFA constraints
require more than 500 samples. If checkpoint preprocessing supplies `n_times`, it
is enforced; otherwise duration is recorded as unverified. No channel dropping,
reordering, interpolation, resampling, cropping, filtering or montage guessing is
performed. These preparation steps need a verified upstream dataset protocol.
Canonical temporal standardization is reused, not reimplemented.

If `channel_names` is absent from the checkpoint, supply `--channel-order` with a
JSON array containing the verified simulator projection row names. This manifest
must come from scientific provenance, not simply copying arbitrary input channel
names. A manifest cannot override saved checkpoint names. No sensitive EEG data
or verified real channel manifest is added to the repository.

## Commands

Run inside the project environment (for Apptainer, bind the repository/data and
use `--nv`). CPU posterior sampling is supported, but **canonical DFA/LYA
extraction still needs NVIDIA CUDA/PyCUDA**; this is not a CPU-only EEG pipeline.

```bash
python -m tvbgpu.infer_eeg \
  --checkpoint output/workshop_model.pt \
  --eeg data/prepared_subject-epo.fif --condition EO --epoch-index 0 \
  --channel-order config/projection_channel_names.json \
  --output-dir output/workshop_inference \
  --num-samples 1000 --device cuda --seed 42

python -m tvbgpu.infer_eeg \
  --checkpoint checkpoints/sbi_full.pt \
  --eeg data/prepared_subject-epo.fif --condition EO --epoch-index 0 \
  --channel-order config/projection_channel_names.json \
  --output-dir output/full_model_inference \
  --num-samples 1000 --device cuda --seed 42
```

These are input-path templates: compatible checkpoints, prepared EEG and the
verified manifest must be supplied. Use `--device cpu` for CPU posterior sampling.
The output directory must be new. Fixed seeds initialize Python, NumPy and Torch;
bitwise reproducibility across GPU/software versions is not guaranteed. Exit codes:
0 success, 1 runtime/compatibility failure with traceback, 2 CLI usage error.

## Outputs

- `posterior_samples.npy`: `(draws, 7)`, canonical log-noise parameter ordering.
- `posterior_summary.csv`: seven parameter rows plus a derived `weight_noise` row.
  Mean, median, population SD, requested percentiles, equal-tailed 50/90/95%
  credible intervals, prior bounds, lower/upper boundary fractions. Physical noise
  statistics use `10 ** log10_weight_noise` **on every draw**. Boundary fractions
  refer to 5% of each row's reported prior range (linear for physical noise).
- `inference_metadata.json`: source paths, epoch/condition/channel provenance,
  checkpoint/pipeline versions where available, names/mask/configuration, prior,
  software/git state, seed/device, MCMC settings and measured phase timings.
- `posterior_distributions.png`: seven marginals, median, 90% credible interval,
  axes spanning the prior. Narrow intervals alone do not establish accuracy.
- `eeg_features_raw.npy`: `(1, full_dimension)`, full **post-PCA** observation.
- `eeg_features_processed.npy`: `(1, retained_dimension)`, normalized observation.

No original EEG is copied. Identical output schemas support notebook comparison
of workshop and full models for the same epoch. PCA bases and masks may differ
between training budgets; each checkpoint's own transforms are authoritative.

## Testing and integration prerequisites

CPU contracts (synthetic data, with explicitly stubbed GPU extraction in the
output test):

```bash
python -m unittest discover -s tests -p test_infer_eeg.py -v
```

Real GPU smoke test, accepting external EEG rather than committing it:

```bash
python tests/smoke_infer_eeg.py \
  --checkpoint output/workshop_model.pt \
  --eeg /path/to/prepared_subject-epo.fif --condition EO --epoch-index 0 \
  --channel-order /path/to/projection_channel_names.json \
  --output-dir output/workshop_smoke --num-samples 20 --device cpu --seed 42
```

Repeat with the compatible full checkpoint and a different output directory.
This invokes actual loading, extraction, transforms, sampling and all outputs;
20 samples exercise execution only, not scientifically meaningful uncertainty.

On inspection, the repository-root `sbi_full.pt` lacked feature pipeline version,
feature/preprocessing config, full dimension and block-slice metadata. It is
rejected, just as the previous validator rejected missing pipeline versions.
Do not patch in assumed provenance. The inspected example
`1-379-1-Z_1-EO_eeg_1_epo.fif` contained 129 generic EEG channel names and a
`stimulus` event, not a verified 61-channel EO input. The local NVIDIA driver
was unavailable. Therefore real GPU end-to-end timings and successful full-model
EEG inference require testing on a GPU host with compatible scientific artifacts.
The current trainer emits the shared checkpoint schema irrespective of budget;
a fresh TVB workshop training run was not performed as part of inference work.

Before EBRAINS/PyUNICORE integration, supply the authoritative projection channel
mapping, verified EEG preparation/reference/epoch-duration protocol, compatible
full checkpoint and GPU environment. Retrieval/authentication and job submission
remain outside this endpoint.

Measured local CPU checks: 5 inference tests passed in 70.61 seconds; 12 existing
posterior-validation tests passed in 0.46 seconds. The synthetic output test used
real SBI MCMC sampling with 20 draws: checkpoint/posterior loading 0.025 seconds,
sampling plus summaries 28.31 seconds, plotting 1.48 seconds, total 29.84 seconds.
EEG loading/extraction was stubbed in that particular orchestration test; its
near-zero feature timing is not a performance measurement. Separate tests read
real temporary MNE FIF epochs and validate saved PCA/mask/scaling. These figures
must not be used to promise workshop runtime; actual GPU EEG timings remain
unmeasured.
