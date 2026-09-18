"""Small environment/project checks. Never run TVB or train/load an SBI model.

--cpu hides GPUs; --gpu requires Torch/PyCUDA on one GPU and compiles a tiny
kernel. Project code is imported from --project, never an image-baked copy.
"""
import argparse
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def project_import(name, root):
    module = importlib.import_module(name)
    location = Path(module.__file__).resolve()
    require(root in location.parents, f"{name} came from {location}, outside {root}")
    print(f"[IMPORT] {name}: {location}", flush=True)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu", action="store_true")
    mode.add_argument("--gpu", action="store_true")
    parser.add_argument("--project", type=Path, help="Bind-mounted repository root")
    parser.add_argument("--environment-only", action="store_true",
                        help="Skip project checks, e.g. while building the image")
    args = parser.parse_args()
    if not args.environment_only and args.project is None:
        parser.error("supply --project REPOSITORY or --environment-only")
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Avoid importing GPU-initializing entry points under an accidental MPI run.
    from mpi4py import MPI
    require(MPI.COMM_WORLD.Get_size() == 1, "Smoke test requires exactly one MPI rank")
    require(int(os.environ.get("SLURM_NTASKS", "1")) == 1,
            "Smoke test requires srun --ntasks=1")

    import numpy as np
    import torch
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    from sklearn.decomposition import PCA

    print(f"Python: {sys.version}")
    for name in ("torch", "sbi", "numpy", "scipy", "scikit-learn", "mpi4py",
                 "pycuda", "mne", "matplotlib", "numba", "tqdm"):
        print(f"{name}: {metadata.version(name)}")
    # pycuda.driver needs libcuda.so, which intentionally is NOT installed in
    # the image. The top-level package can be imported on a CPU-only host.
    for name in ("scipy.signal", "mne", "matplotlib.pyplot", "numba", "tqdm", "pycuda"):
        importlib.import_module(name)
    print(f"MPI: {MPI.Get_library_version().strip()}")
    print(f"Torch: {torch.__version__}; runtime CUDA: {torch.version.cuda}; "
          f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if args.gpu:
        require(torch.cuda.is_available(), "--gpu requires a working NVIDIA GPU/driver and --nv")
        # Import the real entry points BEFORE CUDA computation. The existing
        # training module sets CUDA_VISIBLE_DEVICES from SLURM_LOCALID itself.
        require(int(os.environ.get("SLURM_LOCALID", "0")) == 0,
                "Use one rank with SLURM_LOCALID=0, as expected by the current driver")

    root = None
    if not args.environment_only:
        root = args.project.resolve()
        require((root / "tvbgpu" / "SBI_metrics_driver.py").is_file(),
                f"No SBI source tree at {root}")
        sys.path.insert(0, str(root))
        for name in ("sbi_features", "EEGprojection", "fc_analysis", "alpha_feat_analysis",
                     "best_complexity_results", "sbi", "posterior_validation"):
            project_import("tvbgpu.analysis." + name, root)
        if args.gpu:
            for name in ("tvbgpu.SBI_metrics_driver", "tvbgpu.model_driver_LarterBreakspear",
                         "tvbgpu.Posterior_tests", "tvbgpu.analysis.gpu_dfa2",
                         "tvbgpu.analysis.gpu_lya_3"):
                project_import(name, root)
        else:
            print("[DEFERRED TO --gpu] Training/posterior entry points, simulator, DFA/LYA "
                  "modules initialize PyCUDA during import.")

    device = torch.device("cuda:0" if args.gpu else "cpu")
    values = torch.arange(8, dtype=torch.float32, device=device)
    require(float((values * values).sum().cpu()) == 140., "Torch arithmetic failed")
    # Exercise SBI's actual prior API; do not construct/train a density estimator.
    prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))
    require(tuple(prior.sample((3,)).shape) == (3, 2), "SBI BoxUniform failed")
    require(callable(SNPE), "SBI SNPE import failed")
    toy = np.random.default_rng(123).normal(size=(8, 4))
    require(PCA(n_components=2).fit_transform(toy).shape == (8, 2), "Toy PCA failed")

    if args.gpu:
        import pycuda.autoinit
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        from pycuda import gpuarray
        print(f"Torch GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyCUDA GPU: {pycuda.autoinit.device.name()}, "
              f"PCI {pycuda.autoinit.device.pci_bus_id()}, driver API {cuda.get_driver_version()}")
        # Match the model's C++ standard and require its cuRAND headers too.
        # This is a 32-element arithmetic kernel, NOT a TVB simulation.
        module = SourceModule('''
            #include <curand_kernel.h>
            extern "C" __global__ void twice(float *x) {
                int i = threadIdx.x; x[i] *= 2.0f;
            }
        ''', no_extern_c=True, options=["--std=c++14"])
        vector = gpuarray.to_gpu(np.arange(32, dtype=np.float32))
        module.get_function("twice")(vector, block=(32, 1, 1), grid=(1, 1))
        np.testing.assert_array_equal(vector.get(), 2 * np.arange(32, dtype=np.float32))
        print("[PASS] NVCC + cuRAND headers + PyCUDA kernel compilation/execution")

    if root is not None:
        from tvbgpu.analysis.EEGprojection import parallel_EEGproj_batched
        from tvbgpu.analysis.sbi_features import (
            FeatureConfig, FEATURE_PIPELINE_VERSION, preprocess_eeg,
        )
        from tvbgpu.analysis.fc_analysis import compute_fc
        config = FeatureConfig()
        rng = np.random.default_rng(123)
        # Synthetic regional input/projection only; no files, fitted PCA or model.
        regional = rng.normal(size=(2, 68, 600)).astype(np.float32)
        gain = rng.normal(size=(config.n_channels, 68)).astype(np.float32)
        eeg = parallel_EEGproj_batched(regional, np.arange(68), gain, batch_size=2)
        np.testing.assert_allclose(eeg, np.einsum("cr,nrt->nct", gain, regional),
                                   atol=2e-4, rtol=2e-4)
        standardized = preprocess_eeg(eeg, config)
        np.testing.assert_allclose(standardized.mean(axis=-1), 0, atol=2e-6)
        np.testing.assert_allclose(standardized.std(axis=-1), 1, atol=2e-6)
        fc = compute_fc(standardized, method="corr").detach().cpu().numpy()
        require(fc.shape == (2, config.n_channels, config.n_channels), "FC shape mismatch")
        require(np.isfinite(fc).all(), "Nonfinite synthetic FC")
        print(f"[PASS] Host-source projection, canonical temporal preprocessing and FC; "
              f"feature pipeline {FEATURE_PIPELINE_VERSION}")
    print("[PASS] Smoke test finished; no checkpoints/data opened, no TVB simulation or SBI training.")


if __name__ == "__main__":
    main()
