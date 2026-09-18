# tvbSBI

EBRAINS Days workshop source for the frozen seven-parameter TVB-GPU EEG SBI
pipeline: prior → GPU simulation → shared features → posterior training → quick
validation → prepared real EEG → posterior inference and uncertainty plots.

## Scope and provenance

This is a selected snapshot of the current `tvb_apptainer` working tree, including
its uncommitted scientific changes, not an export of a historical Git commit.
Scientific implementations are preserved; packaging changes correct inference
imports/commands and make launch scripts configurable. Only the active
Larter–Breakspear model is packaged. Dormant references to alternate models and
standalone legacy demonstrations do not make those workflows supported here.

No license has yet been selected; retain existing author notices and resolve
code/third-party licensing before public redistribution. Data redistribution
provenance is also pending. No container images, EEG recordings, model checkpoints
or generated results are committed.

## Setup

Use the external Apptainer environment described in [containers/README.md](containers/README.md).
Supply the three verified scientific assets listed in
[tvbgpu/data/README.md](tvbgpu/data/README.md) before training.

Run commands from this repository root in the configured environment.

## Training

The unchanged training entry point is:

```bash
python -m tvbgpu.SBI_metrics_driver [simulator arguments]
```

On HPC use the site's MPI/GPU launcher and its approved GPU-binding configuration.
The driver exposes `-s0` through `-s7` for parameter-grid sizes, `-n` for simulation
length and `-dt` for the integration timestep. Keep the established scientific
settings. The workshop simulation budget and two-GPU launch settings still need
benchmarking; no new budget is imposed here. Set `SBI_OUTPUT_DIR` to control
training output. Participant and full models use the same checkpoint schema.

## Quick validation

```bash
bash scripts/posterior_validation.sh CHECKPOINT TRAIN_N_TIME TRAIN_DT
```

The wrapper selects width/recovery for two observations and 100 posterior draws
per observation. It is an in-sample teaching demonstration, not held-out
calibration. The complete research validator remains available through
`python -m tvbgpu.Posterior_tests`, with its existing test options.

For Slurm, submit from the repository root and supply site settings:

```bash
export IMAGE=/absolute/path/to/sbi.sif
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  scripts/posterior_validation_width_recovery.sbatch CHECKPOINT TRAIN_N_TIME TRAIN_DT
```

Supply the exact training simulation length/timestep. The wrapper does not select
or change them. Cluster-specific settings must be checked on JUSUF.

## Real EEG inference

```bash
python -m tvbgpu.infer_eeg \
  --checkpoint output/workshop_model.pt \
  --eeg /path/to/prepared_subject-epo.fif --condition EO --epoch-index 0 \
  --channel-order /path/to/verified_channel_names.json \
  --output-dir output/workshop_inference --num-samples 1000 --device cuda --seed 42
```

Use the same command with a compatible full checkpoint and a different output
directory. Read [the EEG input contract and output documentation](docs/eeg_inference.md).
The channel mapping is deferred; inference must not guess it. EBRAINS retrieval,
authentication, notebooks and PyUNICORE submission are not implemented here.

## Tests

```bash
python -m unittest discover -s tests -p test_infer_eeg.py -v
python -m unittest discover -s tvbgpu/analysis -p test_posterior_validation.py -v
```

`test_sbi_features.py` imports GPU-initializing modules and needs the GPU
environment. The real inference smoke test accepts external artifacts:

```bash
python tests/smoke_infer_eeg.py \
  --checkpoint /path/to/compatible.pt --eeg /path/to/prepared-epo.fif \
  --condition EO --epoch-index 0 --channel-order /path/to/verified_channels.json \
  --output-dir output/smoke --num-samples 20 --device cpu
```

CPU contract tests do not substitute for this real EEG/GPU smoke test. The old
local full checkpoint lacked required pipeline metadata and cannot be silently
upgraded. Obtain the verified compatible checkpoint for the workshop.
