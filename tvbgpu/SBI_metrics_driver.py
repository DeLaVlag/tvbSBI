# ============================================================
# Project: TVB EEG Pipeline / HPC Simulation Framework
# Author:  Michiel van der Vlag
# Institution: Forschungszentrum Jülich (FZJ)
# Created: 2024 for Ebrain-Health
#
# Description:
# This project contains simulation pipelines, containerized
# environments (Apptainer), and analysis workflows for
# large-scale brain network modeling and data-driven
# computational neuroscience fitting eeg to TVB simulations
# for augmenting EEG ML pipelines

import logging, sys
from mpi4py import MPI

import os, socket

from tvbgpu.analysis.sbi_features import (
    ALPHA_FEATURE_KEYS, FEATURE_PIPELINE_VERSION, FeatureConfig,
    POST_PCA_BLOCK_ORDER, SIGNAL_BLOCK_ORDER, extract_signal_features,
    assemble_post_pca,
)

precuneus_idx = [10, 11]  # example channels
acc_idx = [5, 6]
dmn_idx = [5, 6, 10, 11, 20, 21]

local_rank = int(os.environ.get("SLURM_LOCALID", 0))
slurm_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram

import mne
import itertools
import argparse
import tqdm
import zipfile
import io
import time
import gc

try:
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
except ImportError:
    logging.warning('pycuda not available, rateML driver not usable.')

# from analysis.analysis import analysis
# from analysis.insert_database import *

# c_metrics
# from tvbgpu.analysis.pci_driver import run_pci_analy
from tvbgpu.analysis.gpu_dfa2 import computeDFA_gpu
from tvbgpu.analysis.gpu_lya_3 import computeLYA_gpu

from tvbgpu.analysis.best_complexity_results import *
from tvbgpu.analysis.alpha_feat_analysis import *
from tvbgpu.analysis.fc_analysis import *
# from tvbgpu.plotting.plotting_model import *
# from tvbgpu.plotting.plot_dashbord import *
from tvbgpu.analysis.EEGprojection import *
from tvbgpu.analysis.sbi import *

# rank = MPI.COMM_WORLD.Get_rank()
# size = MPI.COMM_WORLD.Get_size()

# Initialize MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
world_size = comm.Get_size()

# print(f"[RANK {rank}] START on {socket.gethostname()}", flush=True)
# print(f"[RANK {rank}] after imports", flush=True)
# print(f"[RANK {rank}] before GPU init", flush=True)

data_root = os.path.dirname(os.path.realpath(__file__)) + '/data/'

def get_logger_o(loggername):
    logger = logging.getLogger(loggername)
    # logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Only add handlers if none are present
    if not logger.hasHandlers():
        # Stream handler to stdout (will go to SLURM .out file)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


logger = get_logger_o('tvb.SBI')
# logger.setLevel(level='INFO' if True else 'WARNING')
logger.setLevel(level='DEBUG' if True else 'WARNING')
if my_rank == 0:
    logger.info('Starting logger for rank 0')
    logger.info('MPI World Size %d', world_size)

try:
    selected_device = pycuda.autoinit.device
    try:
        bus_id = selected_device.pci_bus_id()
    except Exception:
        bus_id = "<unavailable>"
    pycuda_device = f"{selected_device.name()} pci_bus_id={bus_id}"
except Exception as exc:
    pycuda_device = f"unavailable ({type(exc).__name__}: {exc})"
logger.info(
    "[GPU STARTUP] rank=%d world_size=%d hostname=%s SLURM_LOCALID=%s "
    "CUDA_VISIBLE_DEVICES=%s slurm_cuda_visible_devices_before=%s pycuda_device=%s",
    my_rank, world_size, socket.gethostname(),
    os.environ.get("SLURM_LOCALID", "<unset>"),
    os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    slurm_cuda_visible_devices, pycuda_device,
)

################################## SET TVB MODEL DYNAMICS ################################

# set the model to be fitted. LB for LarterBreakspear and JR for JansenRit
model = 'LarterBreakspear'
# model = 'JansenRit'

if my_rank == 0:
    logger.info('TVB Model %s', model)

if model == 'LarterBreakspear':
    from tvbgpu.model_driver_LarterBreakspear import *

    # LB
    g_col = 0  # column index for parameter g
    s_col = 3  # column index for parameter s
    dV_col = 6  # column index for parameter dV

    # oaram selection for printing, corresponds to the plotting parameters of the model_driver*
    param_sel = np.array([1, 0, 0, 1, 0, 0, 1], dtype=bool)  # for LB

    # runtime for plotting, should correspond to runtime of the model_driver*
    mdl_runtime = 4000

elif model == 'JansenRit':
    from tvbgpu.model_driver_JansenRit import *

    # JR
    g_col = 0  # column index for parameter g
    s_col = 1  # column index for parameter s
    dV_col = 4  # column index for parameter mu

    # oaram selection for printing, corresponds to the plotting parameters of the model_driver*
    param_sel = np.array([1, 1, 0, 0, 1, 0, 0], dtype=bool)  # for JR

    # runtime for plotting, should correspond to runtime of the model_driver*
    mdl_runtime = 4000


def do_dfas(data):
    def safe_window_sizes(N, min_w=10, max_w=None, n_windows=20, min_segments=4):
        if max_w is None:
            max_w = N // 4

        floats = np.logspace(np.log10(min_w), np.log10(max_w), n_windows)
        cand = np.unique(np.round(floats).astype(np.int32))
        cand = cand[(cand >= min_w) & (cand <= max_w)]
        cand = cand[np.array([N // w >= min_segments for w in cand])]

        # enforce sorted and contiguous int32:
        cand = np.sort(cand).astype(np.int32)
        cand = np.ascontiguousarray(cand)

        return cand

    N = int(0.004 * 500 * 1000 / 4)  # not needed — just set below
    N = 2000  # 4 s at 500 Hz
    window_sizes = safe_window_sizes(N, min_w=10, max_w=N // 4, n_windows=20, min_segments=4)

    # min_w = 10
    # max_w = n_timesteps // 4
    # n_windows = 20
    # window_sizes = np.logspace(np.log10(min_w), np.log10(max_w), n_windows).astype(int)

    # print('window_sizes', window_sizes)

    # window_sizes = np.array([4, 8, 16, 32, 64, 128, 256], dtype=np.int32)
    # window_sizes = np.array([ 10, 12, 15, 19, 23, 28, 34, 42, 52, 64, 78, 96, 118, 145, 179, 219, 270, 331, 407, 500 ], dtype=np.int32)
    alphas, dfa_log_s, dfa_log_F = computeDFA_gpu(np.ascontiguousarray(data), logger, my_rank, window_sizes)
    # logger.info('DFAs %s', dfas_pr.shape)

    return alphas, dfa_log_s, dfa_log_F


def do_lyas(data) -> np.ndarray:
    # emb_dim = 4
    # lag = 5
    # min_tsep = 10
    # trajectory_len = 50
    # tau = 2

    emb_dim = 8
    lag = 8
    min_tsep = 16
    trajectory_len = 80
    tau = .001

    # to test
    # print('dataaaamax', data.max())
    # print('dataaaamin', data.min())
    # data = np.random.rand(1, 61, 2000).astype(np.float32)
    # print('dataaaamaxa', data.max())
    # print('dataaaamina', data.min())
    # data = (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)

    # computeLYA_gpu(timeseries, loggerobj, mpirank, emb_dim=3, lag=1, min_tsep=2, trajectory_len=20, tau=1):
    lyas_pr, divergence_curves = computeLYA_gpu(np.ascontiguousarray(data), logger, my_rank,
                                                emb_dim, lag, min_tsep, trajectory_len, tau)
    # logger.info('LYAs %s', lyas_pr.shape)

    return lyas_pr, divergence_curves


def run_TVB(sim_params, logger, extract):
    n_runs = 1
    driver_setup = Driver_Setup(comm, logger, sim_params, extract)
    params_here = driver_setup.params
    params_allranks = driver_setup.all_params

    tvbobj = Driver_Execute(driver_setup)

    # shape is 10 x (1 row params + 1 fitness)
    bestten = np.zeros((10, params_here.shape[1] + 1))
    for run in range(n_runs):
        if run == 0:
            tavg = tvbobj.run_all()
        # print(analyres)
        else:
            b10_analyres = Driver_Execute(driver_setup).run_all()
            bestten = (bestten + b10_analyres[0]) / 2
        # print(b10_analyres)

    # comm.Barrier()
    # bestten_world = np.array(comm.gather(bestten, root=0))
    # all_analyresul = np.array(comm.gather(analyres, root=0))
    # all_tsdictlis = np.array(comm.gather(tsdictlist, root=0))

    return params_here, tavg, tvbobj, params_allranks


def projection(tavg):
    # tavgforproj = tavg.transpose(2, 1, 0)
    # print('tavgforproj', tavgforproj.shape)

    region_mapping = np.loadtxt(data_root + "/QL_20120814_RegionMapping.txt", dtype=int)
    gain_matrix_np = scipy.io.loadmat(data_root + "/QL_20120814_ProjectionMatrix.mat")["ProjectionMatrix"]

    # region_mapping = np.loadtxt("data/QL_20120814_RegionMapping.txt", dtype=int)
    # gain_matrix_np = scipy.io.loadmat("data/QL_20120814_ProjectionMatrix.mat")["ProjectionMatrix"]

    tavg_proj = parallel_EEGproj_batched(tavg, region_mapping, gain_matrix_np, batch_size=2)
    # tavg_proj = tavg_proj.transpose(2, 1, 0)

    return tavg_proj


def normalize(eeg_dfa, eeg_lya, tvb_dfa, tvb_lya):
    # Stack into (n_sim, feature_dim)
    tvb_flat = np.concatenate([tvb_dfa.reshape(tvb_dfa.shape[0], -1), tvb_lya], axis=1)
    eeg_flat = np.concatenate([eeg_dfa.flatten(), eeg_lya.flatten()])[None, :]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    tvb_flat_scaled = scaler.fit_transform(tvb_flat)
    eeg_flat_scaled = scaler.transform(eeg_flat)

    # Reshape back if needed
    tvb_dfa_scaled = tvb_flat_scaled[:, :-tvb_lya.shape[1]].reshape(tvb_dfa.shape)
    tvb_lya_scaled = tvb_flat_scaled[:, -tvb_lya.shape[1]:]
    eeg_dfa_scaled = eeg_flat_scaled[0, :-tvb_lya.shape[1]].reshape(eeg_dfa.shape)
    eeg_lya_scaled = eeg_flat_scaled[0, -tvb_lya.shape[1]:].reshape(eeg_lya.shape)

    return eeg_dfa_scaled, eeg_lya_scaled, tvb_dfa_scaled, tvb_lya_scaled

def check_nans(label, arr):
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    if nan_count > 0:
        print(f"⚠️ {label}: {nan_count} NaNs detected")
    if inf_count > 0:
        print(f"⚠️ {label}: {inf_count} Infs detected")
    if nan_count == 0 and inf_count == 0:
        print(f"✅ {label}: No NaNs or Infs")


def free_gpu_memory():
    """
    Free as much GPU memory as possible.
    Call this after finishing all kernel computations.
    """
    # 1️⃣ Delete any lingering Python references (optional if you know them)
    # del var1, var2, ...

    # 2️⃣ Run Python garbage collector
    gc.collect()

    # 3️⃣ Free PyTorch cached memory
    torch.cuda.empty_cache()

    # 4️⃣ Optionally reset peak memory stats (for monitoring/debugging)
    torch.cuda.reset_peak_memory_stats()

    # 5️⃣ Optional: force a CUDA device sync to flush everything
    torch.cuda.synchronize()


def interpolate_nans_dfa(dfa_array):
    """
    Interpolates NaNs along the time-window axis for DFA data.

    Input:
        dfa_array : np.ndarray of shape (n_sims, n_regions, n_windows)
                    may contain NaNs.

    Output:
        cleaned : same shape, with NaNs filled
    """
    dfa = np.asarray(dfa_array, dtype=float)
    n_sims, n_regions, n_windows = dfa.shape

    # Create output array
    cleaned = np.empty_like(dfa)

    # Time index reused for interpolation
    t = np.arange(n_windows)

    for i in range(n_sims):
        for r in range(n_regions):

            row = dfa[i, r, :]

            # mask of valid values
            mask = np.isfinite(row)

            if mask.sum() == 0:
                # All NaNs — choose your fallback
                cleaned[i, r, :] = 0.0
                continue

            # If only ONE valid value, fill all with that value
            if mask.sum() == 1:
                cleaned[i, r, :] = row[mask][0]
                continue

            # Interpolate only over valid segments
            interp_vals = np.interp(
                t,
                t[mask],
                row[mask]
            )

            cleaned[i, r, :] = interp_vals

    return cleaned


def zscore_fc(fc: torch.Tensor, eps: float = 1e-8):
    """
    Z-score FC tensors along the last two dimensions (NxN),
    ignoring the diagonal.

    fc: (B, N, N)
    """
    B, N, _ = fc.shape
    fc = fc.clone()

    # Diagonal mask: (1, N, N)
    diag_mask = torch.eye(N, device=fc.device, dtype=torch.bool).unsqueeze(0)

    # Mask diagonal with NaN
    fc_masked = fc.masked_fill(diag_mask, float('nan'))

    # Mean over off-diagonal
    mu = torch.nanmean(fc_masked, dim=(1, 2), keepdim=True)

    # Variance: E[(x - mu)^2]
    diff = fc_masked - mu
    var = torch.nanmean(diff * diff, dim=(1, 2), keepdim=True)
    sigma = torch.sqrt(var + eps)

    fc_z = diff / sigma

    # Set diagonal to zero (or keep NaN if you prefer)
    fc_z = fc_z.masked_fill(diag_mask, 0.0)

    return fc_z


def verify_feature_part(name, value, n_sims, n_channels):
    """Check a pre-PCA feature part; optionally log stats for same-input parity runs."""
    expected = {
        "dfa_slopes": (n_sims, n_channels),
        "dfa_log_F": (n_sims, n_channels, None),
        "lya": (n_sims, n_channels),
        "broadband_fc": (n_sims, n_channels, n_channels),
        "alpha_pli": (n_sims, n_channels, n_channels),
        "alpha_aecc": (n_sims, n_channels, n_channels),
        "alpha_summaries": (n_sims, len(ALPHA_FEATURE_KEYS)),
    }[name]
    shape = tuple(value.shape)
    if len(shape) != len(expected) or any(e is not None and got != e for got, e in zip(shape, expected)):
        raise ValueError(f"{name} shape {shape} does not match expected {expected}")
    if os.environ.get("TVB_FEATURE_PARITY_DIAG") != "1":
        return
    values = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    finite = np.isfinite(values)
    if finite.any():
        good = values[finite]
        logger.info("[FEATURE PARITY] %s shape=%s finite=%d/%d min=%.9g max=%.9g mean=%.9g std=%.9g",
                    name, shape, int(finite.sum()), values.size,
                    float(good.min()), float(good.max()), float(good.mean()), float(good.std()))
    else:
        logger.info("[FEATURE PARITY] %s shape=%s finite=0/%d", name, shape, values.size)


def training_output_paths():
    """Use legacy locations by default; a new SBI_OUTPUT_DIR isolates both artifacts.

    An override must be absolute, identical on every MPI rank, and nonexistent
    at startup. This prevents a smoke run from replacing saved checkpoint/parts.
    """
    requested = os.environ.get("SBI_OUTPUT_DIR", "").strip()
    values = comm.allgather(requested)
    if len(set(values)) != 1:
        raise RuntimeError(f"SBI_OUTPUT_DIR differs across MPI ranks: {values}")
    if not requested:
        return (
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "output"),
            "/p/project1/vbt/vandervlag1/TSDchecks/tvbgpu/output/sbi_feature_parts",
        )
    if not os.path.isabs(requested):
        raise ValueError("SBI_OUTPUT_DIR must be an absolute path on shared storage")
    output_root = os.path.normpath(requested)
    error = None
    if my_rank == 0:
        try:
            os.makedirs(output_root, exist_ok=False)
        except OSError as exc:
            error = f"Cannot create new SBI_OUTPUT_DIR {output_root!r}: {exc}"
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(error)
    feature_parts = os.path.join(output_root, "sbi_feature_parts")
    logger.info("[SBI OUTPUT] rank=%d checkpoint_dir=%s feature_parts_dir=%s",
                my_rank, output_root, feature_parts)
    return output_root, feature_parts


def runmain():
    ticall = time.time()
    output_rootpath, outdir = training_output_paths()

    import os

    params_to_simulate = []

    # ########################## EEG file reading ###########################################
    # #######################################################################################
    #

    ########################## TVB Simulation #############################################
    #######################################################################################
    tic = time.time()
    # run tvb todo make mpi here, remove logger
    tvb_params, tavg, tvbobj, params_allranks = run_TVB(params_to_simulate, logger, extract=0)
    toc = time.time()
    if my_rank == 0:
        logger.info('TVB timeseries shape %s', tavg.shape)
        logger.info('Time for TVB simulation %.2f', (toc - tic))

    free_gpu_memory()

    # set tavg to a single state variable
    state = 0
    tavg = tavg[:, state]

    if my_rank == 0:
        logger.debug('tavgdtype %s', tavg.dtype)

    # Source activity remains raw until linear projection. EEG temporal
    # standardization is performed per channel by extract_signal_features.
    check_nans("tavg", tavg)

    ########################## TVB- Interpolate NaN #######################################
    #######################################################################################

    def interpolate_nans_tavg(tavg):
        """
        Interpolate NaNs along the time axis for tavg with shape (timepoints, 1, regions, sims).
        Returns a copy with NaNs replaced using linear interpolation.
        """
        tavg_interp = np.copy(tavg)
        timepoints = tavg.shape[0]
        sims = tavg.shape[2]
        regions = tavg.shape[1]

        for s in range(sims):
            for r in range(regions):
                y = tavg[:, r, s]
                nans = np.isnan(y)
                if np.any(nans):
                    x = np.arange(timepoints)
                    # interpolate only over valid values
                    y[nans] = np.interp(x[nans], x[~nans], y[~nans])
                    tavg_interp[:, r, s] = y

        return tavg_interp


    # tavg = interpolate_nans_tavg(tavg)

    # if my_rank == 0:
    #     logger.debug('tavg_interpol.shape %s', tavg.shape)

    # transpose only once
    tavg = tavg.T
    if my_rank == 0:
        logger.debug('transposed1 %s', tavg.shape)


    ########################## TVB- Remove bad nans ########################################
    #######################################################################################

    def filter_bad_sims_with_logging(
            tavg,
            logger,
            my_rank=0,
            nan_thr=0.05,
            var_thr=0.30,
            sat_thr=0.30,
            var_eps=1e-6,
            sat_eps=0.99,
    ):
        """
        Filter out bad simulations based on NaNs, variance collapse, and saturation,
        with MPI-safe logging.

        Parameters
        ----------
        tavg : ndarray, shape (nsims, nregions, ntimesteps)
        logger : logging.Logger
        my_rank : int
            MPI rank (log only if rank == 0)
        nan_thr : float
            Maximum allowed NaN fraction.
        var_thr : float
            Maximum allowed fraction of regions with near-zero variance.
        sat_thr : float
            Maximum allowed saturation fraction.
        var_eps : float
            Variance threshold for collapse detection.
        sat_eps : float
            Absolute value threshold for saturation detection.

        Returns
        -------
        tavg_clean : ndarray
        good_mask : ndarray, shape (nsims,)
        stats : dict
        """

        nsims = tavg.shape[0]

        # ---- metrics ----
        nan_frac = np.isnan(tavg).mean(axis=(1, 2))

        region_std = np.std(tavg, axis=-1)
        var_frac = np.mean(region_std < var_eps, axis=1)

        sat_frac = np.mean(np.abs(tavg) > sat_eps, axis=(1, 2))

        # ---- mask ----
        good_mask = (
                (nan_frac <= nan_thr) &
                (var_frac <= var_thr) &
                (sat_frac <= sat_thr)
        )

        bad = np.where(~good_mask)[0]

        # ---- logging ----
        if my_rank == 0:
            frac = 100 * len(bad) / nsims if nsims else 0.0
            logger.info(
                f"Simulation quality filtering:"
                f" removed {len(bad)} / {nsims} sims ({frac:.2f}%) | "
                f"criteria: nan>{nan_thr:.2f}, var>{var_thr:.2f}, sat>{sat_thr:.2f}"
            )

            if len(bad) > 0:
                logger.info(
                    f"  Mean bad-sim stats: "
                    f"nan={nan_frac[bad].mean():.3f}, "
                    f"var={var_frac[bad].mean():.3f}, "
                    f"sat={sat_frac[bad].mean():.3f}"
                )

        stats = dict(
            nan_frac=nan_frac,
            var_frac=var_frac,
            sat_frac=sat_frac
        )

        return tavg[good_mask], good_mask, stats

    filterbads = False
    if filterbads == True:
        tavg, good_mask, stats = filter_bad_sims_with_logging(
            tavg,
            logger=logger,
            my_rank=my_rank
        )
    else:
        good_mask = np.ones(tavg.shape[0], dtype=bool)
        stats = {}

    ########################## TVB- EEG Projection ########################################
    #######################################################################################

    # compute projection matrix
    tavg_proj = projection(tavg)

    if my_rank == 0:
        logger.debug('tavgNaNremovalshape %s', tavg.shape)
        logger.debug('TVB after projection %s', tavg_proj.shape)

    free_gpu_memory()

    # Canonical deterministic EEG path; no training-fitted transforms here.
    feature_config = FeatureConfig(
        fs=500.0, n_channels=61,
        precuneus_idx=tuple(precuneus_idx), acc_idx=tuple(acc_idx),
        dmn_idx=tuple(dmn_idx),
    )
    signal_blocks = extract_signal_features(tavg_proj, feature_config, logger, my_rank)
    N = tavg_proj.shape[0]
    payload = {
        "dfas_tvb_pr": signal_blocks["dfa"],
        "lyas_tvb_pr": signal_blocks["lya"],
        "fc_flat": signal_blocks["fc_flat"],
        "dfa_curve_flat": signal_blocks["dfa_curve_flat"],
        "lya_curve_flat": signal_blocks["lya_curve_flat"],
        "pli_flat": signal_blocks["pli_flat"],
        "aecc_flat": signal_blocks["aecc_flat"],
        "alpha_flat": signal_blocks["alpha"],
        "tvb_params": np.asarray(tvb_params),
    }
    alpha_valid = signal_blocks["alpha_valid"]

    # Start with validity from the unsanitized alpha features and add a finite check for
    # every saved block. This keeps each theta row aligned with its feature row.
    local_valid = np.asarray(alpha_valid).astype(bool, copy=False)

    for name, values in payload.items():
        if values.shape[0] != N:
            raise ValueError(
                f"{name} has {values.shape[0]} simulations; expected {N}"
            )
        local_valid &= np.isfinite(values.reshape(N, -1)).all(axis=1)

    # weight_noise must be positive because rank 0 applies log10 below.
    local_valid &= payload["tvb_params"][:, 1] > 0.0

    logger.info(
        "rank %d: retaining %d/%d finite simulations",
        my_rank,
        int(local_valid.sum()),
        N,
    )

    payload = {
        name: np.ascontiguousarray(values[local_valid])
        for name, values in payload.items()
    }

    os.makedirs(outdir, exist_ok=True)

    np.savez_compressed(
        f"{outdir}/features_rank{my_rank:04d}.npz",
        **payload,
    )

    comm.Barrier()

    ########################## Merge ranks and fit transforms ############################
    #######################################################################################

    if my_rank == 0:
        part_keys = list(payload.keys())
        parts = {key: [] for key in part_keys}

        for r in range(comm.Get_size()):
            filename = f"{outdir}/features_rank{r:04d}.npz"
            with np.load(filename) as z:
                missing = [key for key in part_keys if key not in z]
                if missing:
                    raise KeyError(f"{filename} is missing arrays: {missing}")

                for key in part_keys:
                    parts[key].append(z[key])

        merged = {
            key: np.concatenate(values, axis=0)
            for key, values in parts.items()
        }

        dfa_all = np.asarray(merged["dfas_tvb_pr"], dtype=np.float32)
        lya_all = np.asarray(merged["lyas_tvb_pr"], dtype=np.float32)
        fc_all = np.asarray(merged["fc_flat"], dtype=np.float32)
        dfa_curve_all = np.asarray(merged["dfa_curve_flat"], dtype=np.float32)
        lya_curve_all = np.asarray(merged["lya_curve_flat"], dtype=np.float32)
        pli_all = np.asarray(merged["pli_flat"], dtype=np.float32)
        aecc_all = np.asarray(merged["aecc_flat"], dtype=np.float32)
        alpha_all = np.asarray(merged["alpha_flat"], dtype=np.float32)
        tvb_params_all = np.asarray(merged["tvb_params"], dtype=np.float32)

        if alpha_all.shape[1] != len(ALPHA_FEATURE_KEYS):
            raise ValueError(
                f"Merged alpha shape {alpha_all.shape}; expected "
                f"(N, {len(ALPHA_FEATURE_KEYS)})"
            )

        from sklearn.decomposition import PCA

        # Connectivity matrices are large, so randomized PCA extracts the requested
        # components without a complete 1830-dimensional SVD. random_state makes it
        # reproducible. The fitted objects are saved and reused for EEG/resimulation.
        fcpca = PCA(n_components=20, svd_solver="randomized", random_state=0)
        fc_pca = fcpca.fit_transform(fc_all)

        dfa_pca = PCA(n_components=10, svd_solver="full")
        dfa_curve_pca = dfa_pca.fit_transform(dfa_curve_all)

        lya_pca = PCA(n_components=10, svd_solver="full")
        lya_curve_pca = lya_pca.fit_transform(lya_curve_all)

        pli_pca_model = PCA(
            n_components=20, svd_solver="randomized", random_state=0
        )
        pli_pca = pli_pca_model.fit_transform(pli_all)

        aecc_pca_model = PCA(
            n_components=20, svd_solver="randomized", random_state=0
        )
        aecc_pca = aecc_pca_model.fit_transform(aecc_all)

        print("fc_pca std:", fc_pca.std(axis=0))
        print("dfa_curve_pca std:", dfa_curve_pca.std(axis=0))
        print("lya_curve_pca std:", lya_curve_pca.std(axis=0))
        print("pli_pca std:", pli_pca.std(axis=0))
        print("aecc_pca std:", aecc_pca.std(axis=0))

        # Manual DFA-PCA projection sanity check.
        manual = (dfa_curve_all - dfa_pca.mean_) @ dfa_pca.components_.T
        print("manual equals sklearn:", np.allclose(manual, dfa_curve_pca), flush=True)

        if dfa_all.ndim == 1:
            dfa_all = dfa_all[:, None]
        if lya_all.ndim == 1:
            lya_all = lya_all[:, None]

        feature_blocks = {
            "dfa": dfa_all, "lya": lya_all, "fc_pca": fc_pca,
            "dfa_curve_pca": dfa_curve_pca, "lya_curve_pca": lya_curve_pca,
            "pli_pca": pli_pca, "aecc_pca": aecc_pca, "alpha": alpha_all,
        }
        xstack_full, feature_names_full, feature_slices = assemble_post_pca(feature_blocks)
        xstack_full = xstack_full.astype(np.float32, copy=False)

        if len(feature_names_full) != xstack_full.shape[1]:
            raise RuntimeError(
                f"Generated {len(feature_names_full)} names for "
                f"{xstack_full.shape[1]} features"
            )

        theta_sbi = tvb_params_all.copy()
        theta_sbi[:, 1] = np.log10(theta_sbi[:, 1])

        # Final check after PCA and the parameter transformation.
        global_valid = (
                np.isfinite(xstack_full).all(axis=1)
                & np.isfinite(theta_sbi).all(axis=1)
        )
        if not global_valid.all():
            logger.warning(
                "Removing %d non-finite simulations after merged transformations",
                int((~global_valid).sum()),
            )
            xstack_full = xstack_full[global_valid]
            theta_sbi = theta_sbi[global_valid]

        # Remove constant/near-constant columns before do_sbi normalizes the features.
        # This avoids tiny x_std values producing enormous resimulation deltas.
        x_std_before_filter = xstack_full.std(axis=0, ddof=0)
        reference_scale = np.maximum(
            np.median(np.abs(xstack_full), axis=0),
            1.0,
        )
        feature_keep = (
                np.isfinite(x_std_before_filter)
                & (x_std_before_filter > 1e-6 * reference_scale)
        )

        removed_feature_names = [
            name
            for name, keep in zip(feature_names_full, feature_keep)
            if not keep
        ]
        if removed_feature_names:
            logger.warning(
                "Removing %d constant/near-constant features: %s",
                len(removed_feature_names),
                removed_feature_names,
            )

        if not feature_keep.any():
            raise RuntimeError("All feature columns were removed as near-constant")

        xstack = np.ascontiguousarray(xstack_full[:, feature_keep])
        feature_names = [
            name
            for name, keep in zip(feature_names_full, feature_keep)
            if keep
        ]

        # do_sbi retains responsibility for normalization and returns the normalization
        # values used during training.
        x_mean, x_std, xs, prior, posterior, density_estimator = do_sbi(
            theta_sbi,
            xstack,
            feature_names=feature_names,
        )

        ########################## Save ###################################################
        ###################################################################################

        parameter_names = [
            "coupling",
            "log10_weight_noise",
            "rNMDA",
            "global_speed",
            "tau_K",
            "phi",
            "dV",
        ]

        prior_ranges = {
            "coupling": [0.2, 1.2],
            "log10_weight_noise": [-6.0, -3.0],
            "rNMDA": [0.25, 0.55],
            "global_speed": [1.0, 20.0],
            "tau_K": [1.0, 1.3],
            "phi": [0.5, 0.9],
            "dV": [0.2, 2.5],
        }

        save_obj = {
            "posterior": posterior,
            "density_estimator": density_estimator,
            "density_estimator_state_dict": density_estimator.state_dict(),

            "thetas": theta_sbi,
            "xs": xs,
            "xs_raw": xstack,

            # Apply feature_keep to the complete post-PCA feature vector before using
            # x_mean/x_std during empirical inference or resimulation.
            "feature_keep": feature_keep,
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "x_mean": x_mean,
            "x_std": x_std,

            "prior_low": prior.low,
            "prior_high": prior.high,

            # Simulation-fitted transforms required for EEG and resimulation.
            "fcpca": fcpca,
            "dfa_pca": dfa_pca,
            "lya_pca": lya_pca,
            "pli_pca": pli_pca_model,
            "aecc_pca": aecc_pca_model,

            "parameter_names": parameter_names,
            "feature_names": feature_names,
            "feature_names_full": feature_names_full,
            "removed_feature_names": removed_feature_names,
            "alpha_feature_names": ALPHA_FEATURE_KEYS,
            "feature_block_order": list(POST_PCA_BLOCK_ORDER),
            "feature_block_slices": {name: (sl.start, sl.stop) for name, sl in feature_slices.items()},
            "signal_block_order": list(SIGNAL_BLOCK_ORDER),
            "x_full_dim": len(feature_names_full),
            "prior_ranges": prior_ranges,
            "n_channels": feature_config.n_channels,
            "theta_dim": len(parameter_names),
            "x_dim": len(feature_names),

            "feature_config": feature_config.metadata(),
            "preprocessing_config": {
                "input_axes": "batch_channels_time",
                "projection_before_standardization": True,
                "standardization_axis": "time",
                "standardization_eps": feature_config.standardization_eps,
            },

            "note": (
                "weight_noise is stored in log10-space; convert back with "
                "10 ** log10_weight_noise. EEG and resimulations must use the saved "
                "PCA objects, feature block order, feature_keep, x_mean and x_std."
            ),
        }

        torch.save(
            save_obj,
            os.path.join(output_rootpath, "sbi_full.pt"),
        )

        tocall = time.time()
        print('\n')
        if my_rank == 0:
            logger.info('FINISHED TVB - EEG FITTING')
            logger.info('Time for all operations %.2f', (tocall - ticall))

        '''
        some runtime info
        4x4x4x4 4 4000: 169s
        4x4x4x4 4 40000: 1:20
        8x8x8x8 4 40000: ???
        '''

    MPI.Finalize()

if __name__ == '__main__':
    runmain()


