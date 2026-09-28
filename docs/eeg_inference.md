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

The default input mode accepts **prepared MNE Epochs FIF** (`*-epo.fif` or
supported compressed FIF). One epoch is selected with `--epoch-index` (default 0)
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

The default mode also accepts BrainVision `.vhdr`, taking the first 61 channels
and up to 4001 samples, with the same strict channel manifest and sampling-rate
checks. It requires index 0 and no condition. For the workshop dataset, use the
explicit EBRAINS mode below instead.

## EBRAINS Days: synthetic TVB EEG, sub-001

`--input-format ebrains-synthetic --eeg-vhdr PATH` reads BrainVision sensor-space
EEG with MNE, which resolves the accompanying `.eeg` and `.vmrk` files. Finite data,
unique channel names, no unresolved bad channels, and at least the checkpoint's
required channel count are required. No projection is applied to observed EEG. The JSON and electrodes
sidecars are not needed by this endpoint; no montage or reference is inferred.

BrainVision channel order is preserved and printed for approximate selection. When checkpoint channel names
are absent, no external manifest is required in this mode. A prominent warning
records the assumption that these sensors correspond to the checkpoint's
observation space. Saved checkpoint names or an explicitly supplied
`--channel-order` still must match exactly for equal-size recordings. Larger inputs
are selected in saved/manifest order when all those names are available. An explicit
manifest with missing names is rejected. All other checkpoint compatibility checks
remain unchanged.

For the 256-channel `E1`…`E256` recording without verified correspondence, selection
is explicitly **approximate workshop/demo inference**, never anatomical equivalence.
If MNE provides finite, nonzero, unique, non-collinear coordinates for every channel,
the loader uses deterministic Euclidean farthest-point sampling: start farthest from
the coordinate centroid, then repeatedly choose the point farthest from its nearest
selected point. Ties use the lowest original index. The result is sorted into
original acquisition order. This spreads sensors spatially but does not align them
to the training montage. No standard montage is guessed from generic `E` names.

Without usable coordinates, selection is
`np.rint(np.linspace(0, 255, 61)).astype(int)` for this input/checkpoint. It includes
both endpoints and produces 61 unique indices. This covers the channel index range;
it cannot guarantee scalp coverage. Original indices, selected channel names, method,
and the approximation warning are logged and recorded in metadata. The plot title
also marks approximate runs as WORKSHOP/DEMO. Inputs smaller than the required
channel count are rejected without padding.

The current trainer uses **61 sensors at 500 Hz**. Its full feature order is:
DFA (61), LYA (61), FC PCA (20), DFA log-F curve PCA (10), LYA curve PCA (10),
PLI PCA (20), AECC PCA (20), alpha summaries (5): **207 total**. Saved PCA input
widths and channel-specific DFA/LYA features make channel count/order matter even
though the neural posterior only receives the final vector. For example, the
connectivity PCA inputs contain `61*60/2 = 1830` upper-triangle entries.
Training computes the near-constant mask using
`std > 1e-6 * max(median(abs(feature)), 1)`; inference reuses the saved mask rather
than recomputing this reference scale from the observation. Saved PCA, feature
names/slices, `feature_keep`, `x_mean`, and `x_std` remain authoritative.

MNE resamples the selected continuous recording to saved `feature_config.fs` when necessary,
before window selection (e.g. 256.016 → 500 Hz, or → 200 Hz if the checkpoint
explicitly specifies 200). The window length uses saved `preprocessing_config.n_times`
when present. Otherwise `--window-samples` can supply the verified training length;
the workshop fallback is **4001 samples**, with a warning that the checkpoint
does not verify this assumption. An override cannot contradict saved `n_times`.
`--epoch-index k` selects nonoverlapping window `[k*N:(k+1)*N]` after resampling;
short/incomplete windows are rejected. No filtering or rereferencing is added.
The existing shared extractor performs temporal standardization, feature
extraction, saved PCA, ordering, feature retention and saved normalization.

From the repository root, with the dataset in `/tvbgpu/input` and a compatible
checkpoint at `tvbgpu/output/sbi_heidel26.pt`, run in the CUDA/PyCUDA environment:

```bash
python -m tvbgpu.infer_eeg \
  --input-format ebrains-synthetic \
  --eeg-vhdr /tvbgpu/input/sub-001_task-rest_desc-sim_eeg.vhdr \
  --checkpoint tvbgpu/output/sbi_heidel26.pt \
  --dk-atlas /tvbgpu/input/dk_atlas.tsv \
  --connectivity-weights /tvbgpu/input/sub-001_atlas-dk_desc-weight_conndata-network_connectivity.tsv \
  --connectivity-distances /tvbgpu/input/sub-001_atlas-dk_desc-distance_conndata-network_connectivity.tsv \
  --output-dir output/ebrains_sub-001_inference \
  --epoch-index 0 --num-samples 1000 --device cuda --seed 42
```

Adjust those paths to your mounted files. The three connectivity arguments are
optional but must be supplied together. Matrices must be headerless numeric TSVs
of shape `(84, 84)`. The atlas must have 84 rows and one label column named `name`,
`label`, `region`, `region_name` or `region_label`, or be 84 headerless labels.
The loader selects indices `[8:42] + [50:84]` on both matrix axes and on labels:
34 left cortical regions, then 34 right cortical regions, with **no permutation**.
It validates finite values and reports the first/last labels. Atlas row order is
assumed to be the supplied EBRAINS DK84 order already checked against `centres.txt`.

Connectivity is validated and recorded as provenance; it does not alter the
trained posterior. This endpoint has no posterior resimulation support, so no
`--use-ebrains-connectivity-for-resimulation` option is added. The simulator's QL
vertex projection and region mapping remain untouched; the 76-region mapping is
not used.

The console, plot title and metadata label the observation
**“EBRAINS synthetic TVB EEG — sub-001”**. Existing seven-parameter samples,
means, medians, credible intervals and plots are retained. These describe the
trained model's posterior for a held-out TVB observation under possible model
mismatch; they do not establish recovery of the original generating parameters.

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
`--output-dir` is optional and defaults to the current directory. Any supplied
directory must already exist; the script creates no output directories. Repeated
runs overwrite the six result files listed below. Fixed seeds initialize Python, NumPy and Torch;
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

Repeat with the compatible full checkpoint and a different existing output directory.
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

EBRAINS/PyUNICORE deployment still requires a compatible checkpoint and GPU
environment. The explicit EBRAINS input mode records the workshop sensor-order
and epoch-duration assumptions above; independently verifying those assumptions
is needed for scientific interpretation. Retrieval/authentication and job
submission remain outside this endpoint.

Measured local CPU checks: 5 inference tests passed in 70.61 seconds; 12 existing
posterior-validation tests passed in 0.46 seconds. The synthetic output test used
real SBI MCMC sampling with 20 draws: checkpoint/posterior loading 0.025 seconds,
sampling plus summaries 28.31 seconds, plotting 1.48 seconds, total 29.84 seconds.
EEG loading/extraction was stubbed in that particular orchestration test; its
near-zero feature timing is not a performance measurement. Separate tests read
real temporary MNE FIF epochs and validate saved PCA/mask/scaling. These figures
must not be used to promise workshop runtime; actual GPU EEG timings remain
unmeasured.
