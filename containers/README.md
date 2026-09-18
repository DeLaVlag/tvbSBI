# SBI software environment

The repository contains the Apptainer recipe and pinned dependencies, not a SIF.
Use an externally supplied compatible image, or build from the repository root:

```bash
apptainer build --fakeroot /path/to/sbi.sif containers/sbi.def
```

The image contains software only. Bind the repository and external input files
at runtime. GPU simulation and canonical EEG feature extraction require a working
NVIDIA driver and `--nv`. CPU posterior sampling does not remove that feature
extraction requirement.

```bash
export PROJECT="$PWD"
export IMAGE=/absolute/path/to/sbi.sif
apptainer exec --cleanenv --no-home --nv \
  --bind "$PROJECT:$PROJECT" --pwd "$PROJECT" \
  --env "PYTHONPATH=$PROJECT" "$IMAGE" \
  /opt/sbi/bin/python containers/smoke_test.py --gpu --project "$PROJECT"
```

Use `--cpu` instead of `--gpu`, without `--nv`, for the limited CPU environment
check. Run project modules using `/opt/sbi/bin/python -m tvbgpu.MODULE`.
Additional input locations must be bound explicitly if not exposed by the site.
Set writable cache locations if required by local cluster policy.

The recipe is a candidate environment; source inspection and CPU tests do not
establish GPU or checkpoint compatibility on JUSUF. Benchmark and validate there
before the workshop. `capture_environment.py` records environment provenance;
the definition also records dependency versions during the image build.
