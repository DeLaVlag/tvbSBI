"""Record an environment without importing the project or opening checkpoints.

Run with the Python used by the successful native Booster job, or in an image.
Only standard-library imports are required; optional probes report failures.
"""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command_output(command):
    if not shutil.which(command[0]):
        return {"unavailable": command[0]}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def capture():
    packages = sorted(
        ({"name": dist.metadata["Name"], "version": dist.version,
          "location": str(dist.locate_file(""))}
         for dist in metadata.distributions() if dist.metadata["Name"]),
        key=lambda item: (item["name"].lower(), item["location"]),
    )
    result = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "sys_path": sys.path, "packages": packages,
        "environment": {key: os.environ.get(key) for key in (
            "LOADEDMODULES", "MODULEPATH", "PYTHONPATH", "CUDA_HOME", "CUDA_PATH",
            "CUDA_VISIBLE_DEVICES", "SLURM_LOCALID", "SLURM_NTASKS",
            "APPTAINER_CONTAINER", "PYCUDA_CACHE_DIR", "XDG_CACHE_HOME")},
        "nvcc": command_output(["nvcc", "--version"]),
        "mpicc": command_output(["mpicc", "-show"]),
        "nvidia_smi": command_output([
            "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
    }
    try:
        import torch
        result["torch"] = {"version": torch.__version__, "cuda_runtime": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available()}
    except Exception as exc:
        result["torch"] = {"error": str(exc)}
    # Do not import mpi4py here. Importing it calls MPI_Init immediately; on a
    # login node with no usable interface, Open MPI may abort the whole process
    # before Python can catch an exception. Package and launcher metadata are
    # sufficient for an environment capture; the smoke test probes MPI inside
    # an allocated rank.
    try:
        result["mpi4py"] = {"version": metadata.version("mpi4py")}
    except metadata.PackageNotFoundError:
        result["mpi4py"] = {"unavailable": True}
    result["mpi"] = {"launcher": command_output(["mpirun", "--version"]),
                     "compiler": result["mpicc"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="JSON file; otherwise print to stdout")
    args = parser.parse_args()
    report = json.dumps(capture(), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
