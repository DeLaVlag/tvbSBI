# Workshop: compare the existing consistency and predictive tests

Use the existing `tvbgpu.Posterior_tests` command twice, adding only
`--shared-inputs` with the same file path. Run the small model first and wait for
success before running the full model. Each output directory must be new.

```bash
python3 -u -m tvbgpu.Posterior_tests \
  --checkpoint tvbgpu/output/SMALL_CHECKPOINT.pt \
  --tests consistency,predictive \
  --output-dir tvbgpu/output/posterior_test_small \
  --consistency-items 2 --consistency-samples 50 \
  --predictive-items 2 --predictive-samples 5 \
  --shared-inputs tvbgpu/output/posterior_test_inputs.npz \
  --seed 42 -n 4000 -dt 0.1

python3 -u -m tvbgpu.Posterior_tests \
  --checkpoint tvbgpu/output/sbi_full.pt \
  --tests consistency,predictive \
  --output-dir tvbgpu/output/posterior_test_full \
  --consistency-items 2 --consistency-samples 50 \
  --predictive-items 2 --predictive-samples 5 \
  --shared-inputs tvbgpu/output/posterior_test_inputs.npz \
  --seed 42 -n 4000 -dt 0.1
```

Replace `SMALL_CHECKPOINT.pt` with the workshop model filename. Use the actual
training simulator settings; the example above uses the requested 4000 / 0.1.
Run inside the usual SIF with the repository on PYTHONPATH, using the existing
one-MPI-rank, one-GPU `srun` configuration. Both tests still require that GPU
environment. No retraining or separate dataset-generation command is involved.

## Why one extra flag is needed

`--seed` already exists. Without shared inputs, consistency uses the first
training rows and predictive uses a seeded subset of each checkpoint's own rows.
Different checkpoints can contain different simulations and different dataset
sizes; an identical seed does not give identical observations.

With `--shared-inputs`, the first run selects two training theta rows using the
seed, simulates them once, and saves the pre-PCA feature blocks and true theta.
Both tests use these same two cases. The second run loads them without generating
new target observations. This adds two target simulations to the first run.
Checkpoints do not save their original pre-PCA training features, so those fresh
simulations are necessary to allow independently fitted PCA transforms.

Each posterior is conditioned using its own PCA/mask/scaling. Resimulated outputs
from both models are scored using the **first checkpoint's** PCA/mask/scaling and
training-row baseline. This keeps the existing normalized-feature L2 metric
comparable. The first checkpoint must remain available at its recorded absolute
path during the second evaluation; its SHA256 is checked. It is not needed for
plotting. Priors, parameter order, feature configuration, simulator arguments,
seed and case count must agree. Differing fitted PCA/masks/scaling are supported.

Do not regenerate or edit the shared input file between runs. The simulator uses
clock-based CUDA randomness; `--seed 42` controls selection and posterior/baseline
RNG, but does not make GPU simulations bitwise reproducible. The saved inputs
ensure identical target observations. Predictive simulations remain stochastic.

These are fresh noisy observations at training theta from the first checkpoint,
not a held-out parameter-recovery/calibration study. The small-model-derived cases
and reference feature space can affect the comparison. No additional scientific
tests are added. Omitting `--shared-inputs` retains the original test behavior;
the comparison utility rejects runs without verified shared inputs.

## What the two tests measure

**Consistency:** for each target, draw 50 posterior theta samples and take their
arithmetic mean in inference coordinates. Convert the mean `log10_weight_noise`
to physical noise with `10**mean_log_noise`, then run one simulator realization.
Extract and normalize features and calculate their Euclidean (L2) distance from
the target. Save per-case distances, their mean, and their median. The existing
code uses `torch.median`, which takes the lower middle value for an even number
of cases. This is forward consistency of the posterior-mean parameters, not a
parameter recovery error. It is also not the mean of 50 simulated predictions.

**Predictive:** draw five posterior theta samples for each target and simulate
each draw individually. Calculate L2 distance from each simulated feature vector
to its target, globally and by feature block. For each target, also select five
unrelated reference training rows without replacement, excluding the row with
that target's source theta. Calculate their distances as the existing random-row
baseline. Save all distances, per-case means/medians, pooled summaries and ratios.
The plot shows individual posterior-draw errors and the baseline median per case.

For the example: each evaluation performs two posterior-mean simulations plus ten
posterior predictive simulations. The first evaluation additionally makes the two
shared target observations. Both tests assess agreement in summary-feature space;
neither establishes ground-truth parameter recovery, interval coverage, or SBC.

## Plotting and files to download

```bash
python3 -m tvbgpu.compare_validation \
  tvbgpu/output/posterior_test_small \
  tvbgpu/output/posterior_test_full \
  --output-dir tvbgpu/output/posterior_test_comparison
```

This creates `consistency_comparison.png` and `predictive_comparison.png`.
Plots use the recorded training-row counts; a full checkpoint containing 131072
training rows is labeled accordingly.

Only **two files** need downloading, preserving their parent directories:

```text
posterior_test_small/comparison.json
posterior_test_full/comparison.json
```

These contain case theta, common target features, sampling counts, case/draw
errors, baseline errors, summary statistics and comparison provenance. Plotting
requires only NumPy, Matplotlib and the repository code; no checkpoint, SBI, torch,
GPU or shared-input NPZ is required.

```python
from tvbgpu.compare_validation import plot_comparison
figures = plot_comparison('posterior_test_small', 'posterior_test_full')
for fig in figures.values():
    display(fig)
```

The original `consistency.json/.npz`, `predictive.json/.npz` and `manifest.json/.npz`
are still saved for detailed inspection, including true theta and raw posterior
samples. They are optional downloads. Keep the shared-input NPZ on JUSUF for
further comparable evaluations. Use a new shared-input filename and new output
directories if you intentionally change settings.

## Lightweight tests

```bash
python3 -m unittest tvbgpu.analysis.test_posterior_validation -v
python3 -m unittest discover -s tests -p test_compare_validation.py -v
```
