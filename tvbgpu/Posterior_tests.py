#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch

from sbi.inference import SNPE
from sbi.utils import BoxUniform
import logging, sys
from mpi4py import MPI
from tvbgpu.analysis import posterior_validation as validation

from tvbgpu import SBI_metrics_driver
# from tvb_sandbox.workspace.tvbgpu.metrics_driver import projection
# from tvbgpu.model_driver_LarterBreakspear import *
# from tvbgpu.SBI_metrics_driver import logger, comm
from tvbgpu.model_driver_LarterBreakspear import Driver_Setup, Driver_Execute

comm = MPI.COMM_WORLD
# logger.setLevel(level='INFO' if True else 'WARNING')

logger = logging.getLogger('check.Posterior')
logger.setLevel(level='DEBUG' if True else 'WARNING')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Only add handlers if none are present
if not logger.hasHandlers():
    # Stream handler to stdout (will go to SLURM .out file)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def run_tvb_gpu(params_to_simulate, comm, logger):
    """
    Replace with your TVB-GPU simulation call.

    theta_np shape: (n_params,)
    returns: raw simulation output
    """

    print('theta_np', params_to_simulate.shape)

    driver_setup = Driver_Setup(comm, logger, params_to_simulate, extract=1)
    tvbobj = Driver_Execute(driver_setup)
    tavg = tvbobj.run_all()
    print('tavg', tavg.shape)
    print('tavgT', tavg[:,0].T.shape)

    tavg = SBI_metrics_driver.projection(tavg[:,0].T)
    print('tavg_proj', tavg.shape)

    return tavg

@torch.no_grad()
def compute_features(
    tavg, fcpca, dfa_pca, lya_pca, pli_pca, aecc_pca,
    precuneus_idx, acc_idx, dmn_idx, fs=500.0, device="cuda", config=None,
):
    """Apply the canonical EEG feature path and already-fitted PCA objects.

    `tavg` is projected EEG with shape (batch, channels, time). This function
    never fits a PCA, mask, or normalization statistic.
    """
    from tvbgpu.analysis.sbi_features import (
        FeatureConfig, extract_signal_features, post_pca_blocks,
        assemble_post_pca,
    )
    if config is None:
        config = FeatureConfig(
            fs=fs, n_channels=61,
            precuneus_idx=tuple(precuneus_idx), acc_idx=tuple(acc_idx),
            dmn_idx=tuple(dmn_idx),
        )
    signal_blocks = extract_signal_features(tavg, config, logger, comm.Get_rank())
    pca_models = {
        "fc_pca": fcpca, "dfa_curve_pca": dfa_pca,
        "lya_curve_pca": lya_pca, "pli_pca": pli_pca,
        "aecc_pca": aecc_pca,
    }
    blocks = post_pca_blocks(signal_blocks, pca_models)
    full, _, _ = assemble_post_pca(blocks)
    return full

def normalize_x(x_np, x_mean, x_std):
    x = torch.tensor(x_np, dtype=torch.float32, device=x_mean.device)
    return (x - x_mean) / (x_std + 1e-8)


# ============================================================
# Loading
# ============================================================

def load_sbi_full(path, device):
    from tvbgpu.analysis.sbi_checkpoint import load_checkpoint, build_posterior
    checkpoint = load_checkpoint(path)
    posterior, density_estimator, prior = build_posterior(checkpoint, device)
    if checkpoint.get("thetas") is None or checkpoint.get("xs") is None:
        raise ValueError("Checkpoint must contain `thetas` and `xs` for these tests.")
    thetas = torch.as_tensor(checkpoint["thetas"], dtype=torch.float32, device=device)
    xs = torch.as_tensor(checkpoint["xs"], dtype=torch.float32, device=device)
    x_mean = torch.as_tensor(checkpoint["x_mean"], dtype=torch.float32, device=device)
    x_std = torch.as_tensor(checkpoint["x_std"], dtype=torch.float32, device=device)
    return (posterior, thetas, xs, x_mean, x_std, density_estimator, prior,
            checkpoint, checkpoint["fcpca"], checkpoint["dfa_pca"], checkpoint["lya_pca"])


# ============================================================
# 1. Posterior width test
# ============================================================

@torch.no_grad()
def _draw_posteriors(posterior, xs, n_samples):
    from tvbgpu.analysis.sbi_checkpoint import draw_posteriors
    return draw_posteriors(posterior, xs, n_samples)


@torch.no_grad()
def posterior_width_test(posterior, xs, n_samples=200, max_items=None,
                         prior=None, parameter_names=None, samples=None):
    if max_items is not None:
        xs = xs[:max_items]
    if prior is None or parameter_names is None:
        raise ValueError("Width reporting requires the saved prior and parameter names")
    if samples is None:
        samples = _draw_posteriors(posterior, xs, n_samples)
    return validation.width_summary(samples, validation.as_numpy(prior.low),
                                    validation.as_numpy(prior.high), parameter_names)


@torch.no_grad()
def parameter_recovery_test(posterior, thetas, xs, prior, n_samples=200,
                            max_items=None, samples=None):
    if max_items is not None:
        thetas, xs = thetas[:max_items], xs[:max_items]
    if samples is None:
        samples = _draw_posteriors(posterior, xs, n_samples)
    return validation.recovery_summary(samples, thetas, validation.as_numpy(prior.low),
                                       validation.as_numpy(prior.high))


@torch.no_grad()
def _resimulate_features(theta, checkpoint, simulation_batch_size=256):
    """Batched simulator input is physical theta; returned features use saved scaling.

    theta: (draws, 7) in checkpoint coordinates. Only noise is exponentiated,
    on a copy. Each row is simulated separately by the existing batched GPU path.
    """
    from tvbgpu.analysis.sbi_features import FeatureConfig, apply_saved_transform
    names, low, high = validation.parameter_metadata(checkpoint)
    theta = validation.require_finite("resimulation theta", theta).astype(np.float32)
    if theta.ndim != 2 or theta.shape[1] != len(names) or len(theta) == 0:
        raise ValueError("Resimulation theta must have shape (draws, 7)")
    if np.any((theta < low) | (theta > high)):
        raise ValueError("Resimulation theta is outside the saved prior")
    if simulation_batch_size < 1:
        raise ValueError("simulation_batch_size must be positive")
    config = FeatureConfig(**checkpoint["feature_config"])
    full_parts, normalized_parts = [], []
    for start in range(0, len(theta), simulation_batch_size):
        physical = theta[start:start + simulation_batch_size].copy()
        physical[:, names.index("log10_weight_noise")] = 10.0 ** physical[:, names.index("log10_weight_noise")]
        print(f"[RESIM] batch rows={start}:{start + len(physical)} total={len(theta)}", flush=True)
        eeg = run_tvb_gpu(physical, comm, logger)
        full = compute_features(
            eeg, checkpoint["fcpca"], checkpoint["dfa_pca"], checkpoint["lya_pca"],
            checkpoint["pli_pca"], checkpoint["aecc_pca"],
            config.precuneus_idx, config.acc_idx, config.dmn_idx, config=config,
        )
        if full.shape[0] != len(physical):
            raise ValueError("Simulator/feature extraction changed the number of rows")
        validation.require_finite("x_raw_full", full)
        normalized = apply_saved_transform(full, checkpoint)
        validation.require_finite("x_resim", normalized)
        full_parts.append(full)
        normalized_parts.append(normalized)
        del eeg
    full = np.concatenate(full_parts)
    normalized = np.concatenate(normalized_parts)
    return {"x_raw_full": full, "x_raw": full[:, checkpoint["feature_keep"]],
            "x_resim": normalized}


@torch.no_grad()
def posterior_predictive_check(posterior, xs, checkpoint, target_indices,
                               n_theta_samples=20, max_items=10,
                               simulation_batch_size=256, seed=42):
    """Resimulate individual posterior draws, then compare to unrelated saved rows."""
    xs, target_indices = xs[:max_items], np.asarray(target_indices)[:max_items]
    draws = _draw_posteriors(posterior, xs, n_theta_samples)
    n, k, p = draws.shape
    resim = _resimulate_features(draws.reshape(n * k, p), checkpoint, simulation_batch_size)
    blocks = validation.feature_block_indices(checkpoint)
    result = validation.predictive_summary(
        resim["x_resim"].reshape(n, k, -1), validation.as_numpy(xs),
        validation.as_numpy(checkpoint["xs"]), target_indices, blocks,
        np.random.default_rng(seed),
    )
    result.update(theta_samples=validation.as_numpy(draws), observation_indices=target_indices,
                  x_raw_full=resim["x_raw_full"].reshape(n, k, -1),
                  x_raw=resim["x_raw"].reshape(n, k, -1),
                  baseline_method="Uniform random saved training rows without replacement per target; "
                                  "exclude the target row; K baseline rows for K predictive draws; no baseline resimulation")
    return result


@torch.no_grad()
def coverage_test(posterior, thetas, xs, alpha=0.95, n_samples=500,
                  max_items=None, levels=None, samples=None):
    if max_items is not None:
        thetas, xs = thetas[:max_items], xs[:max_items]
    if samples is None:
        samples = _draw_posteriors(posterior, xs, n_samples)
    result = validation.coverage_summary(samples, thetas, [alpha] if levels is None else levels)
    if levels is None:  # Preserve the original single-level return shapes.
        for key in ("inside_matrix", "lower", "upper"):
            result[key] = result[key][:, 0]
        result["coverage_per_param"] = result["coverage_per_param"][0]
        result["mean_coverage"] = result["mean_coverage"][0]
    return result


@torch.no_grad()
def feature_sensitivity_test(thetas, checkpoint, eps=0.05, max_items=3,
                             repeats=2, simulation_batch_size=256):
    """Repair the existing forward finite-perturbation test with parameter attribution.

    Perturb one parameter at a time by eps of its saved prior range, reversing
    direction near the upper bound. Repeated stochastic simulations estimate the
    noise in feature differences; no common-random-number assumption is made.
    This measures forward sensitivity, not unique posterior identifiability.
    """
    _, low, high = validation.parameter_metadata(checkpoint)
    design, steps = validation.sensitivity_design(validation.as_numpy(thetas[:max_items]), low, high, eps)
    n, conditions, p = design.shape
    repeated = np.repeat(design[:, :, None, :], repeats, axis=2)
    resim = _resimulate_features(repeated.reshape(-1, p), checkpoint, simulation_batch_size)
    features = resim["x_resim"].reshape(n, conditions, repeats, -1)
    result = validation.sensitivity_summary(features, steps, low, high,
                                             validation.feature_block_indices(checkpoint))
    result.update(theta_design=design, repeats=repeats, eps=eps,
                  x_raw_full=resim["x_raw_full"].reshape(n, conditions, repeats, -1),
                  x_raw=resim["x_raw"].reshape(n, conditions, repeats, -1))
    return result


# ============================================================
# Posterior-mean resimulation consistency (existing diagnostic)
# ============================================================

@torch.no_grad()
def resimulation_consistency_test(
    posterior,
    xs,
    x_mean,
    x_std,
    checkpoint,
    # fcpca, dfa_pca, lya_pca,
    precuneus_idx,
    acc_idx,
    dmn_idx,
    n_samples=200,
    max_items=5,
    device="cuda",
):
    """
    Posterior-mean resimulation consistency test with diagnostics.
    """
    # ---------------------------------------------------------
    # Load all simulation-fitted feature transforms
    # ---------------------------------------------------------
    fcpca = checkpoint["fcpca"]
    dfa_pca = checkpoint["dfa_pca"]
    lya_pca = checkpoint["lya_pca"]
    pli_pca = checkpoint["pli_pca"]
    aecc_pca = checkpoint["aecc_pca"]

    # feature_config = checkpoint["feature_config"]

    # fs = float(feature_config["fs"])
    # alpha_band = tuple(feature_config["alpha_band"])
    # broadband = tuple(feature_config["broadband"])
    # alpha_fc_method = feature_config["alpha_fc_method"]
    #
    # precuneus_idx = list(feature_config["precuneus_idx"])
    # acc_idx = list(feature_config["acc_idx"])
    # dmn_idx = list(feature_config["dmn_idx"])

    feature_keep = np.asarray(
        checkpoint["feature_keep"],
        dtype=bool,
    )

    feature_names_full = list(checkpoint["feature_names_full"])
    feature_names = list(checkpoint["feature_names"])
    feature_block_order = list(checkpoint["feature_block_order"])

    print(
        "[DIAG] checkpoint feature dimensions:",
        {
            "full": len(feature_names_full),
            "retained": len(feature_names),
            "feature_keep_length": feature_keep.size,
            "feature_keep_sum": int(feature_keep.sum()),
            "block_order": feature_block_order,
        },
        flush=True,
    )

    if feature_keep.size != len(feature_names_full):
        raise RuntimeError(
            f"feature_keep has length {feature_keep.size}, but "
            f"feature_names_full has length {len(feature_names_full)}"
        )

    if int(feature_keep.sum()) != len(feature_names):
        raise RuntimeError(
            f"feature_keep retains {int(feature_keep.sum())} features, "
            f"but feature_names contains {len(feature_names)} names"
        )

    def torch_stats(name, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        x_cpu = x.detach().cpu()

        finite = torch.isfinite(x_cpu)
        n_total = x_cpu.numel()
        n_finite = finite.sum().item()
        n_nan = torch.isnan(x_cpu).sum().item()
        n_inf = torch.isinf(x_cpu).sum().item()

        print(
            f"[DIAG] {name}: shape={tuple(x_cpu.shape)} "
            f"dtype={x_cpu.dtype} "
            f"finite={n_finite}/{n_total} "
            f"nan={n_nan} inf={n_inf}",
            flush=True,
        )

        if n_finite > 0:
            xf = x_cpu[finite]
            print(
                f"[DIAG] {name}: min={xf.min().item():.6g} "
                f"max={xf.max().item():.6g} "
                f"mean={xf.mean().item():.6g} "
                f"std={xf.std().item():.6g}",
                flush=True,
            )

    def numpy_stats(name, x):
        x = np.asarray(x)

        finite = np.isfinite(x)
        n_total = x.size
        n_finite = finite.sum()
        n_nan = np.isnan(x).sum()
        n_inf = np.isinf(x).sum()

        print(
            f"[DIAG] {name}: shape={x.shape} "
            f"dtype={x.dtype} "
            f"finite={n_finite}/{n_total} "
            f"nan={n_nan} inf={n_inf}",
            flush=True,
        )

        if n_finite > 0:
            xf = x[finite]
            print(
                f"[DIAG] {name}: min={xf.min():.6g} "
                f"max={xf.max():.6g} "
                f"mean={xf.mean():.6g} "
                f"std={xf.std():.6g}",
                flush=True,
            )

    def first_bad_rows_torch(name, x, max_print=10):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)

        x_cpu = x.detach().cpu()

        if x_cpu.ndim == 1:
            bad = ~torch.isfinite(x_cpu)
            bad_idx = torch.where(bad)[0][:max_print].tolist()
            if bad_idx:
                print(f"[DIAG] {name}: bad indices {bad_idx}", flush=True)
            return

        bad_rows = ~torch.isfinite(x_cpu).all(dim=1)
        bad_idx = torch.where(bad_rows)[0][:max_print].tolist()

        if bad_idx:
            print(f"[DIAG] {name}: bad rows {bad_idx}", flush=True)
            for r in bad_idx:
                bad_cols = torch.where(~torch.isfinite(x_cpu[r]))[0][:max_print].tolist()
                print(f"[DIAG] {name}: row {r} bad cols {bad_cols}", flush=True)

    def first_bad_rows_numpy(name, x, max_print=10):
        x = np.asarray(x)

        if x.ndim == 1:
            bad_idx = np.where(~np.isfinite(x))[0][:max_print]
            if len(bad_idx):
                print(f"[DIAG] {name}: bad indices {bad_idx.tolist()}", flush=True)
            return

        bad_rows = ~np.isfinite(x).all(axis=1)
        bad_idx = np.where(bad_rows)[0][:max_print]

        if len(bad_idx):
            print(f"[DIAG] {name}: bad rows {bad_idx.tolist()}", flush=True)
            for r in bad_idx:
                bad_cols = np.where(~np.isfinite(x[r]))[0][:max_print]
                print(f"[DIAG] {name}: row {r} bad cols {bad_cols.tolist()}", flush=True)

    # ---------------------------------------------------------
    # Prepare target xs
    # ---------------------------------------------------------
    xs_test = xs[:max_items].to(device)
    torch_stats("xs_test", xs_test)
    first_bad_rows_torch("xs_test", xs_test)

    # ---------------------------------------------------------
    # Check normalization stats
    # ---------------------------------------------------------
    if not torch.is_tensor(x_mean):
        x_mean = torch.tensor(x_mean, dtype=torch.float32)
    if not torch.is_tensor(x_std):
        x_std = torch.tensor(x_std, dtype=torch.float32)

    x_mean = x_mean.to(device)
    x_std = x_std.to(device)

    dfa_curve_indices = [i for i, name in enumerate(feature_names)
                         if name.startswith("dfa_curve_pca_")]
    print("[DEBUG] DFA-curve normalization stats")
    print("names =", [feature_names[i] for i in dfa_curve_indices])
    print("mean =", x_mean[dfa_curve_indices])
    print("std  =", x_std[dfa_curve_indices])

    print("global x_std min:", x_std.min())
    print("global x_std max:", x_std.max())

    small = torch.where(x_std < 1e-5)[0]
    print("small std indices:", small)
    print("small std values:", x_std[small])

    torch_stats("x_mean", x_mean)
    torch_stats("x_std", x_std)
    first_bad_rows_torch("x_mean", x_mean)
    first_bad_rows_torch("x_std", x_std)

    zero_std = (x_std == 0).detach().cpu()
    if zero_std.any():
        bad_std_idx = torch.where(zero_std)[0][:20].tolist()
        print(
            f"[DIAG] x_std has {zero_std.sum().item()} zero entries. "
            f"First indices: {bad_std_idx}",
            flush=True,
        )

    # Avoid division by zero during diagnostics
    x_std_safe = torch.where(x_std == 0, torch.ones_like(x_std), x_std)

    # ---------------------------------------------------------
    # Sample posterior means
    # ---------------------------------------------------------
    theta_means = []
    theta_samples_all = []

    for i, x_i in enumerate(xs_test):
        print(f"[DIAG] Sampling posterior for item {i}", flush=True)

        theta_samples = posterior.sample(
            (n_samples,),
            x=x_i,
            show_progress_bars=False,
        )

        torch_stats(f"theta_samples[{i}]", theta_samples)
        first_bad_rows_torch(f"theta_samples[{i}]", theta_samples)

        theta_mean = theta_samples.mean(dim=0)

        torch_stats(f"theta_mean[{i}]", theta_mean)
        first_bad_rows_torch(f"theta_mean[{i}]", theta_mean)

        theta_means.append(theta_mean)
        theta_samples_all.append(theta_samples.detach().cpu())

    theta_batch = torch.stack(theta_means, dim=0)
    torch_stats("theta_batch", theta_batch)
    first_bad_rows_torch("theta_batch", theta_batch)

    # If theta already contains NaN, stop early
    if not torch.isfinite(theta_batch).all():
        print("[DIAG] STOP: theta_batch contains NaN/Inf.", flush=True)
        return {
            "theta_batch": theta_batch.detach().cpu(),
            "theta_samples_all": theta_samples_all,
            "x_target": xs_test.detach().cpu(),
            "x_resim": None,
            "x_raw": None,
            "diffs": None,
            "mean_diff": torch.tensor(float("nan")),
            "median_diff": torch.tensor(float("nan")),
        }

    # ---------------------------------------------------------
    # Convert posterior theta to simulator parameters
    # ---------------------------------------------------------
    theta_log = theta_batch.detach().cpu().numpy().astype(np.float32)

    numpy_stats("theta_log_for_sbi", theta_log)
    first_bad_rows_numpy("theta_log_for_sbi", theta_log)

    param_names = [
        "coupling",
        "log10_weight_noise",
        "rNMDA",
        "global_speed",
        "tau_K",
        "phi",
        "dV",
    ]

    print("[PARAM CHECK] posterior theta as stored for SBI:", flush=True)
    for i, theta in enumerate(theta_log[:5]):
        print(f"[PARAM CHECK] sample {i}", flush=True)
        for name, val in zip(param_names, theta):
            print(f"  {name}: {val:.6g}", flush=True)

    print(
        "[PARAM CHECK] log10_weight_noise range:",
        theta_log[:, 1].min(),
        theta_log[:, 1].max(),
        flush=True,
    )

    print(
        "[PARAM CHECK] corresponding physical weight_noise range:",
        (10.0 ** theta_log[:, 1]).min(),
        (10.0 ** theta_log[:, 1]).max(),
        flush=True,
    )

    # IMPORTANT:
    # theta_log[:, 1] is log10_weight_noise for SBI.
    # run_tvb_gpu should usually receive physical weight_noise.
    params_to_simulate = theta_log.copy()
    params_to_simulate[:, 1] = 10.0 ** theta_log[:, 1]

    numpy_stats("params_to_simulate_physical", params_to_simulate)
    first_bad_rows_numpy("params_to_simulate_physical", params_to_simulate)

    print("[PARAM CHECK] simulator params after log-noise conversion:", flush=True)
    for i, theta in enumerate(params_to_simulate[:5]):
        print(f"[PARAM CHECK] sample {i}", flush=True)
        for name, val in zip(
                [
                    "coupling",
                    "weight_noise",
                    "rNMDA",
                    "global_speed",
                    "tau_K",
                    "phi",
                    "dV",
                ],
                theta,
        ):
            print(f"  {name}: {val:.6g}", flush=True)

    # Sanity checks for physical simulator input
    if np.any(params_to_simulate[:, 1] <= 0):
        raise ValueError("[PARAM CHECK] physical weight_noise must be > 0")

    if np.any(params_to_simulate[:, 1] > 1.0):
        print(
            "[PARAM CHECK] WARNING: physical weight_noise > 1.0. "
            "This may indicate double conversion or bad posterior samples.",
            flush=True,
        )

    # ---------------------------------------------------------
    # Run simulator
    # ---------------------------------------------------------
    tavg = run_tvb_gpu(params_to_simulate, comm, logger)

    if torch.is_tensor(tavg):
        torch_stats("tavg", tavg)
        first_bad_rows_torch("tavg", tavg)
    else:
        numpy_stats("tavg", tavg)
        first_bad_rows_numpy("tavg", tavg)

    # ---------------------------------------------------------
    # Compute raw features
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Compute complete post-PCA feature vector
    # ---------------------------------------------------------
    from tvbgpu.analysis.sbi_features import FeatureConfig
    saved_config = FeatureConfig(**checkpoint["feature_config"])
    x_raw_full = compute_features(
        tavg=tavg,
        fcpca=checkpoint["fcpca"],
        dfa_pca=checkpoint["dfa_pca"],
        lya_pca=checkpoint["lya_pca"],
        pli_pca=checkpoint["pli_pca"],
        aecc_pca=checkpoint["aecc_pca"],
        precuneus_idx=precuneus_idx,
        acc_idx=acc_idx,
        dmn_idx=dmn_idx,
        fs=saved_config.fs,
        device=device,
        config=saved_config,
    )

    if torch.is_tensor(x_raw_full):
        torch_stats("x_raw_full", x_raw_full)
        first_bad_rows_torch("x_raw_full", x_raw_full)

        x_raw_full_t = x_raw_full.detach().to(
            device=device,
            dtype=torch.float32,
        )
    else:
        numpy_stats("x_raw_full", x_raw_full)
        first_bad_rows_numpy("x_raw_full", x_raw_full)

        x_raw_full_t = torch.as_tensor(
            x_raw_full,
            dtype=torch.float32,
            device=device,
        )

    if x_raw_full_t.ndim == 1:
        x_raw_full_t = x_raw_full_t.unsqueeze(0)

    if x_raw_full_t.shape[1] != feature_keep.size:
        raise RuntimeError(
            "Resimulation feature construction does not match training. "
            f"compute_features returned {x_raw_full_t.shape[1]} features, "
            f"but the checkpoint expects {feature_keep.size} complete "
            "post-PCA features. Ensure compute_features includes the PLI-PCA, "
            "AECC-PCA, and alpha blocks."
        )

    feature_keep_t = torch.as_tensor(
        feature_keep,
        dtype=torch.bool,
        device=device,
    )

    if x_raw_full_t.shape[1] != feature_keep.size:
        raise RuntimeError(
            f"Complete resimulation vector has "
            f"{x_raw_full_t.shape[1]} features, but feature_keep "
            f"expects {feature_keep.size}"
        )

    # Apply exactly the mask fitted on the training simulations.
    x_raw_t = x_raw_full_t[:, feature_keep_t]

    torch_stats("x_raw retained", x_raw_t)
    first_bad_rows_torch("x_raw retained", x_raw_t)

    if x_raw_t.shape[1] != x_mean.numel():
        raise RuntimeError(
            f"After applying feature_keep, resimulation has "
            f"{x_raw_t.shape[1]} features, but x_mean has "
            f"{x_mean.numel()} entries"
        )

    if x_raw_t.shape[1] != xs_test.shape[1]:
        raise RuntimeError(
            f"Resimulation has {x_raw_t.shape[1]} retained features, "
            f"but the posterior observations contain "
            f"{xs_test.shape[1]} features"
        )

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------
    x_resim = (x_raw_t - x_mean) / x_std_safe

    # Diagnostic only: expose the feature responsible for each normalized extreme.
    kept_full = np.flatnonzero(feature_keep)
    train_raw = np.asarray(checkpoint["xs_raw"])
    train_x = checkpoint["xs"]
    train_x = train_x.detach().cpu().numpy() if torch.is_tensor(train_x) else np.asarray(train_x)
    if train_raw.shape[1] != x_resim.shape[1] or train_x.shape[1] != x_resim.shape[1]:
        raise RuntimeError("Checkpoint training features differ from retained re-simulation features")

    retained_blocks = validation.feature_block_indices(checkpoint)

    def feature_info(j):
        full = int(kept_full[j])
        name = feature_names[j]
        block = next((b for b, indices in retained_blocks.items() if j in indices), "unknown")
        component = int(name.rsplit("_", 1)[1]) if block.endswith("_pca") else None
        return full, name, block, component

    print("[EXTREME DIAG] 10 smallest retained x_std", flush=True)
    for j in torch.argsort(x_std.detach().cpu())[:10].tolist():
        full, name, block, component = feature_info(j)
        print(f"[EXTREME DIAG] retained={j} full={full} name={name} "
              f"block={block} PCA_component={component} x_std={x_std[j].item():.9g}", flush=True)

    for row in range(x_resim.shape[0]):
        values = x_resim[row].detach().cpu()
        finite = torch.isfinite(values)
        if not finite.any():
            print(f"[EXTREME DIAG] row={row} has no finite normalized features", flush=True)
            continue
        j = int(torch.where(finite, values.abs(), torch.full_like(values, -1)).argmax().item())
        full, name, block, component = feature_info(j)
        print(f"[EXTREME DIAG] row={row} retained={j} full={full} name={name} "
              f"block={block} PCA_component={component} feature_keep={bool(feature_keep[full])} "
              f"x_raw_t={x_raw_t[row,j].item():.9g} x_mean={x_mean[j].item():.9g} "
              f"x_std={x_std[j].item():.9g} x_resim={x_resim[row,j].item():.9g} "
              f"training_xs_raw_min={train_raw[:,j].min():.9g} "
              f"training_xs_raw_max={train_raw[:,j].max():.9g} "
              f"training_xs_min={train_x[:,j].min():.9g} "
              f"training_xs_max={train_x[:,j].max():.9g}", flush=True)

    torch_stats("x_resim", x_resim)
    first_bad_rows_torch("x_resim", x_resim)

    # ---------------------------------------------------------
    # Compare only finite rows
    # ---------------------------------------------------------
    delta = x_resim - xs_test

    for block_name, indices in retained_blocks.items():
        if not len(indices):
            print(
                f"[BLOCK DIFF] {block_name}: no retained features",
                flush=True,
            )
            continue

        indices_t = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=device,
        )
        d = delta.index_select(1, indices_t)

        d_safe = torch.nan_to_num(
            d,
            nan=0.0,
            posinf=1e12,
            neginf=-1e12,
        )

        print(
            "[BLOCK DIFF]",
            block_name,
            "n_features", len(indices),
            "finite", torch.isfinite(d).sum().item(), "/", d.numel(),
            "max_abs", d_safe.abs().max().item(),
            "mean_norm",
            torch.linalg.vector_norm(d_safe, dim=1).mean().item(),
            flush=True,
        )

    torch_stats("delta", delta)
    first_bad_rows_torch("delta", delta)

    for row in range(delta.shape[0]):
        finite = torch.isfinite(delta[row])
        if finite.any():
            j = int(torch.where(finite, delta[row].abs(), torch.full_like(delta[row], -1)).argmax().item())
            full, name, block, component = feature_info(j)
            print(f"[EXTREME DELTA] row={row} retained={j} full={full} name={name} "
                  f"block={block} PCA_component={component} "
                  f"x_resim={x_resim[row, j].item():.9g} x_target={xs_test[row, j].item():.9g} "
                  f"delta={delta[row, j].item():.9g}", flush=True)

    valid_rows = torch.isfinite(delta).all(dim=1)
    n_valid = valid_rows.sum().item()

    print(
        f"[DIAG] valid rows for diff: {n_valid}/{delta.shape[0]}",
        flush=True,
    )

    if n_valid == 0:
        diffs = torch.full((delta.shape[0],), float("nan"), device=device)
        mean_diff = torch.tensor(float("nan"))
        median_diff = torch.tensor(float("nan"))
    else:
        diffs_valid = torch.norm(delta[valid_rows], dim=1)

        diffs = torch.full((delta.shape[0],), float("nan"), device=device)
        diffs[valid_rows] = diffs_valid

        mean_diff = diffs_valid.mean().detach().cpu()
        median_diff = diffs_valid.median().detach().cpu()

    torch_stats("diffs", diffs)
    first_bad_rows_torch("diffs", diffs)

    return {
        "theta_batch": theta_batch.detach().cpu(),
        "theta_samples_all": theta_samples_all,
        "x_target": xs_test.detach().cpu(),
        "x_raw_full": x_raw_full_t.detach().cpu(),
        "x_raw": x_raw_t.detach().cpu(),
        "x_resim": x_resim.detach().cpu(),
        "diffs": diffs.detach().cpu(),
        "valid_rows": valid_rows.detach().cpu(),
        "mean_diff": mean_diff,
        "median_diff": median_diff,
    }

@torch.no_grad()
def posterior_basic_tests(
    posterior,
    thetas,
    xs,
    prior,
    n_samples=200,
    max_items=100,
    random_subset=True,
    seed=42,
    debug_n=5,
):
    """
    Combined diagnostic:
    1. Posterior width
    2. Parameter recovery
    3. Coverage

    Samples posterior only once per x_i.
    """

    # ---- optional random subset ----
    if max_items is not None and max_items < thetas.shape[0]:
        if random_subset:
            g = torch.Generator(device=thetas.device)
            g.manual_seed(seed)
            idx = torch.randperm(thetas.shape[0], generator=g, device=thetas.device)[:max_items]
        else:
            idx = torch.arange(max_items, device=thetas.device)

        thetas = thetas[idx]
        xs = xs[idx]

    prior_width = prior.high - prior.low

    widths = []
    raw_errors = []
    norm_errors = []
    abs_errors_per_param = []
    norm_abs_errors_per_param = []
    theta_means = []
    coverage_inside = []

    q_low, q_high = 0.025, 0.975

    for k, (theta_true, x_i) in enumerate(zip(thetas, xs)):
        samples = posterior.sample((n_samples,), x=x_i)

        theta_mean = samples.mean(dim=0)
        width = samples.std(dim=0)

        raw_diff = theta_mean - theta_true
        norm_diff = raw_diff / prior_width

        lower = torch.quantile(samples, q_low, dim=0)
        upper = torch.quantile(samples, q_high, dim=0)
        inside = ((theta_true >= lower) & (theta_true <= upper)).float()

        if debug_n is not None and k < debug_n:
            print("\n--- DEBUG SAMPLE", k, "---")
            print("theta_true:", theta_true)
            print("theta_mean:", theta_mean)
            print("lower:", lower)
            print("upper:", upper)
            print("inside:", inside)
            print("sample std:", width)

        theta_means.append(theta_mean.cpu())
        widths.append(width.cpu())

        raw_errors.append(torch.norm(raw_diff).cpu())
        norm_errors.append(torch.norm(norm_diff).cpu())

        abs_errors_per_param.append(torch.abs(raw_diff).cpu())
        norm_abs_errors_per_param.append(torch.abs(norm_diff).cpu())

        coverage_inside.append(inside.cpu())

    widths = torch.stack(widths)
    prior_width_cpu = prior_width.cpu()
    widths_normalized = widths / prior_width_cpu

    raw_errors = torch.stack(raw_errors)
    norm_errors = torch.stack(norm_errors)

    abs_errors_per_param = torch.stack(abs_errors_per_param)
    norm_abs_errors_per_param = torch.stack(norm_abs_errors_per_param)

    coverage_inside = torch.stack(coverage_inside)

    return {
        "theta_means": torch.stack(theta_means),

        # Posterior width
        "widths": widths,
        "mean_width_per_param": widths.mean(dim=0),
        "median_width_per_param": widths.median(dim=0).values,
        "widths_normalized": widths_normalized,
        "mean_width_normalized_per_param": widths_normalized.mean(dim=0),
        "median_width_normalized_per_param": widths_normalized.median(dim=0).values,

        # Recovery: global raw errors
        "raw_errors": raw_errors,
        "mean_raw_error": raw_errors.mean(),
        "median_raw_error": raw_errors.median(),

        # Recovery: global normalized errors
        "normalized_errors": norm_errors,
        "mean_normalized_error": norm_errors.mean(),
        "median_normalized_error": norm_errors.median(),

        # Recovery: per-parameter raw errors
        "mean_abs_error_per_param": abs_errors_per_param.mean(dim=0),
        "median_abs_error_per_param": abs_errors_per_param.median(dim=0).values,

        # Recovery: per-parameter normalized errors
        "mean_normalized_abs_error_per_param": norm_abs_errors_per_param.mean(dim=0),
        "median_normalized_abs_error_per_param": norm_abs_errors_per_param.median(dim=0).values,

        # Coverage
        "coverage_per_param": coverage_inside.mean(dim=0),
        "mean_coverage": coverage_inside.mean(),
    }

def check_theta_x_integrity(checkpoint, prior, device="cpu"):
    import torch
    import numpy as np

    parameter_names = checkpoint["parameter_names"]
    feature_names = checkpoint["feature_names"]

    thetas = torch.as_tensor(checkpoint["thetas"], dtype=torch.float32, device=device)
    xs = torch.as_tensor(checkpoint["xs"], dtype=torch.float32, device=device)
    xs_raw = torch.as_tensor(checkpoint["xs_raw"], dtype=torch.float32, device=device)

    x_mean = torch.as_tensor(checkpoint["x_mean"], dtype=torch.float32, device=device)
    x_std = torch.as_tensor(checkpoint["x_std"], dtype=torch.float32, device=device)

    print("\n=== SHAPES ===")
    print("thetas:", thetas.shape)
    print("xs:", xs.shape)
    print("xs_raw:", xs_raw.shape)
    print("x_mean:", x_mean.shape)
    print("x_std:", x_std.shape)

    print("\n=== NAMES ===")
    print("parameter_names:", parameter_names)
    print("feature_names:", feature_names)

    print("\n=== THETA PRIOR CHECK ===")
    inside = (thetas >= prior.low) & (thetas <= prior.high)
    for j, name in enumerate(parameter_names):
        frac_inside = inside[:, j].float().mean().item()
        print(
            f"{j:02d} {name:20s} "
            f"min={thetas[:, j].min().item(): .5g} "
            f"max={thetas[:, j].max().item(): .5g} "
            f"prior=[{prior.low[j].item(): .5g}, {prior.high[j].item(): .5g}] "
            f"inside={frac_inside:.4f}"
        )

    print("\n=== XS NORMALIZATION CHECK ===")
    xs_recomputed = (xs_raw - x_mean) / (x_std + 1e-8)
    max_diff = torch.max(torch.abs(xs - xs_recomputed)).item()
    mean_diff = torch.mean(torch.abs(xs - xs_recomputed)).item()

    print("max |xs - recomputed xs| :", max_diff)
    print("mean |xs - recomputed xs|:", mean_diff)

    print("\nxs mean over dataset:")
    print(xs.mean(dim=0))

    print("\nxs std over dataset:")
    print(xs.std(dim=0))

    print("\n=== FEATURE RAW STATS ===")
    for j, name in enumerate(feature_names):
        col = xs_raw[:, j]
        print(
            f"{j:02d} {name:15s} "
            f"raw_min={col.min().item(): .5g} "
            f"raw_max={col.max().item(): .5g} "
            f"raw_mean={col.mean().item(): .5g} "
            f"raw_std={col.std().item(): .5g}"
        )

    print("\n=== FIRST 5 PAIRS ===")
    for i in range(min(5, thetas.shape[0])):
        print(f"\nSample {i}")
        print("theta:")
        for j, name in enumerate(parameter_names):
            print(f"  {j:02d} {name:20s}: {thetas[i, j].item(): .6g}")

        print("x raw:")
        for j, name in enumerate(feature_names):
            print(f"  {j:02d} {name:20s}: {xs_raw[i, j].item(): .6g}")

        print("x normalized:")
        for j, name in enumerate(feature_names):
            print(f"  {j:02d} {name:20s}: {xs[i, j].item(): .6g}")

    print("specific features:")
    print(feature_names[102:112])
    print(x_mean[102:112])
    print(x_std[102:112])

# ============================================================
# Main
# ============================================================

TEST_TITLES = {
    "consistency": "Posterior-mean resimulation consistency",
    "predictive": "POSTERIOR PREDICTIVE CHECK",
    "width": "POSTERIOR WIDTH",
    "recovery": "PARAMETER RECOVERY",
    "coverage": "COVERAGE",
    "sensitivity": "FEATURE SENSITIVITY (FORWARD PERTURBATIONS)",
}


def parse_validation_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate a trained SBI checkpoint on one MPI rank / one GPU.",
        epilog="Supply the exact training -n/-dt; other simulator flags are forwarded unchanged.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False,
    )
    parser.add_argument("--checkpoint", default="tvbgpu/output/sbi_full.pt", metavar="PATH")
    parser.add_argument("--tests", default="consistency", help="Comma-separated test names or all: " + ",".join(TEST_TITLES))
    parser.add_argument("--output-dir", help="New result directory; default: validation_JOBID_TIMESTAMP in the working directory")
    parser.add_argument("--seed", type=int, default=42, help="Training-row/baseline selection and posterior RNG seed; simulator RNG unchanged")
    parser.add_argument("-n", "--n_time", type=int, required=True, help="Simulator n_time used for training (required)")
    parser.add_argument("-dt", "--delta_time", type=float, required=True, help="Simulator dt used for training (required)")
    parser.add_argument("--simulation-batch-size", type=int, default=256, help="Maximum simultaneous GPU simulations for predictive/sensitivity")
    defaults = {"consistency": (5, 3), "predictive": (10, 20), "width": (10, 200),
                "recovery": (10, 200), "coverage": (10, 500)}
    for test, (items, samples) in defaults.items():
        item_flags = [f"--{test}-items"]
        sample_flags = [f"--{test}-samples"]
        if test == "consistency":
            item_flags.append("--max-items")
            sample_flags.append("--n-samples")
        parser.add_argument(*item_flags, type=int, default=items, help=f"Conditioning observations for {test}")
        parser.add_argument(*sample_flags, type=int, default=samples,
                            help="Posterior draws per observation" + (" averaged before simulation" if test == "consistency" else ""))
    parser.add_argument("--sensitivity-items", type=int, default=3)
    parser.add_argument("--sensitivity-repeats", type=int, default=2, help="Stochastic replicates per base/perturbed theta")
    parser.add_argument("--sensitivity-eps", type=float, default=0.05, help="Perturbation as a fraction of each saved prior range")
    parser.add_argument("--sensitivity-top-features", type=int, default=10, help="Features printed per parameter; all features are saved")
    args, simulator_argv = parser.parse_known_args(argv)
    selected = [x.strip() for x in args.tests.split(",")]
    if selected == ["all"]:
        selected = list(TEST_TITLES)
    if not selected or any(x not in TEST_TITLES for x in selected) or len(set(selected)) != len(selected):
        parser.error("--tests must be all or unique comma-separated names: " + ",".join(TEST_TITLES))
    args.tests = selected
    for key, value in vars(args).items():
        if key.endswith(("_items", "_samples", "_repeats")) or key in ("n_time", "simulation_batch_size", "sensitivity_top_features"):
            if value < 1:
                parser.error(f"{key} must be positive")
    for test in ("width", "recovery", "coverage"):
        if test in selected and getattr(args, test + "_samples") < 2:
            parser.error(f"--{test}-samples must be at least 2")
    if not np.isfinite(args.delta_time) or args.delta_time <= 0:
        parser.error("-dt must be finite and positive")
    if not 0 < args.sensitivity_eps <= 0.5:
        parser.error("--sensitivity-eps must be in (0, 0.5]")
    if args.seed < 0 or args.seed >= 2 ** 32:
        parser.error("--seed must be between 0 and 2**32-1")
    # The existing Driver_Setup parser remains responsible for simulator settings.
    simulator_argv += ["-n", str(args.n_time), "-dt", str(args.delta_time)]
    return args, simulator_argv


def require_single_rank(communicator):
    if communicator.Get_size() != 1:
        raise RuntimeError("Posterior validation supports exactly one MPI rank; use --ntasks=1")


def validate_training_pairs(checkpoint):
    """Validate saved pairs and report exact saved-statistic reconstruction."""
    names, low, high = validation.parameter_metadata(checkpoint)
    blocks = validation.feature_block_indices(checkpoint)
    theta = validation.require_finite("training theta", checkpoint["thetas"])
    xs = validation.require_finite("training xs", checkpoint["xs"])
    raw = validation.require_finite("training xs_raw", checkpoint["xs_raw"])
    if (theta.ndim != 2 or xs.ndim != 2 or theta.shape != (len(xs), len(names))
            or raw.shape != xs.shape or xs.shape[1] != len(checkpoint["feature_names"]) or len(xs) == 0):
        raise ValueError("Training theta/x rows or feature dimensions disagree with checkpoint metadata")
    if np.any((theta < low) | (theta > high)):
        raise ValueError("Saved training theta contains values outside the saved prior")
    mean, std = validation.as_numpy(checkpoint["x_mean"]), validation.as_numpy(checkpoint["x_std"])
    max_error, sum_error = 0.0, 0.0
    for start in range(0, len(xs), 4096):
        delta = np.abs((raw[start:start + 4096] - mean) / std - xs[start:start + 4096])
        max_error = max(max_error, float(delta.max()))
        sum_error += float(delta.sum(dtype=np.float64))
    return names, low, high, blocks, {"max_abs_error": max_error, "mean_abs_error": sum_error / xs.size}


def main():
    from datetime import datetime, timezone
    from pathlib import Path
    import importlib.metadata
    import time

    args, simulator_argv = parse_validation_args()
    require_single_rank(comm)
    # Strip validation flags before the unchanged simulator parser is used.
    sys.argv[1:] = simulator_argv
    simulator_args = Driver_Setup.parse_args(None)
    if not torch.cuda.is_available():
        raise RuntimeError("This validation entry point requires the existing CUDA/PyCUDA environment")
    device = "cuda"
    cuda_index = torch.cuda.current_device()
    model = os.path.abspath(args.checkpoint)
    print(f"[STARTUP] checkpoint={model}", flush=True)
    print(f"[STARTUP] selected tests={','.join(args.tests)} MPI ranks={comm.Get_size()}", flush=True)
    print(f"[STARTUP] CUDA device=cuda:{cuda_index} ({torch.cuda.get_device_name(cuda_index)}) "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"[STARTUP] simulator n_time={simulator_args.n_time} dt={simulator_args.delta_time}", flush=True)

    posterior, thetas, xs, mean, std, estimator, prior, checkpoint, *_ = load_sbi_full(model, device)
    estimator.eval()
    names, low, high, blocks, normalization_check = validate_training_pairs(checkpoint)
    print(f"[STARTUP] feature pipeline={checkpoint['feature_pipeline_version']} "
          f"retained/full={len(checkpoint['feature_names'])}/{len(checkpoint['feature_names_full'])}", flush=True)
    print("[STARTUP] saved normalization reconstruction:", normalization_check, flush=True)
    print("[STARTUP] Training-pair diagnostics are in-sample, not held-out calibration or independent SBC.", flush=True)
    print("[STARTUP] Consistency uses the first saved rows; other tests use a seeded random subset.", flush=True)
    print("[STARTUP] Existing MCMC settings retained: 2 chains, 20 warmup steps, thin=1, resample initialization.", flush=True)

    counts = {}
    for test in args.tests:
        requested = getattr(args, test + "_items")
        counts[test] = {"requested_items": requested, "items": min(requested, len(xs))}
        if test == "sensitivity":
            counts[test].update(repeats=args.sensitivity_repeats, eps=args.sensitivity_eps)
        else:
            counts[test]["samples"] = getattr(args, test + "_samples")
        print(f"[STARTUP] {test}: {counts[test]}", flush=True)
    if "predictive" in args.tests and args.predictive_samples > len(xs) - 1:
        raise ValueError("Predictive baseline requires at least predictive_samples+1 saved rows")
    print(f"[STARTUP] GPU simulation batch size={args.simulation_batch_size}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    default_dir = f"validation_{os.environ.get('SLURM_JOB_ID', 'local')}_{stamp}"
    output_dir = Path(args.output_dir or default_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"[OUTPUT] {output_dir}", flush=True)
    versions = {}
    for package in ("numpy", "torch", "sbi", "scikit-learn", "mpi4py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    manifest = {
        "checkpoint": model, "checkpoint_size": os.path.getsize(model),
        "checkpoint_mtime_ns": os.stat(model).st_mtime_ns, "started_utc": stamp,
        "checkpoint_feature_pipeline_version": checkpoint["feature_pipeline_version"],
        "parameter_names": names, "prior_low": low, "prior_high": high,
        "feature_names": checkpoint["feature_names"], "feature_names_full": checkpoint["feature_names_full"],
        "feature_keep": checkpoint["feature_keep"], "feature_block_indices": blocks,
        "feature_block_slices": checkpoint["feature_block_slices"],
        "feature_config": checkpoint["feature_config"], "normalization_check": normalization_check,
        "simulator_args": vars(simulator_args), "validation_args": vars(args),
        "sample_counts": counts, "versions": versions, "source_file": os.path.realpath(__file__),
        "mpi_world_size": comm.Get_size(), "cuda_device": torch.cuda.get_device_name(cuda_index),
        "sampling": "Existing MCMC: 2 chains, warmup_steps=20, thin=1, init_strategy=resample",
        "observation_source": "Saved training pairs, not held-out data; not independent SBC",
        "seed_scope": "Row/baseline selection and posterior RNG; simulator RNG unchanged",
        "results": {},
    }
    validation.save_result(output_dir, "manifest", manifest)
    order = np.random.default_rng(args.seed).permutation(len(xs))
    # Match the existing combined basic test's reuse of posterior samples when
    # observation/sample counts coincide, while retaining independent CLI counts.
    sample_cache = {}
    cfg = checkpoint["feature_config"]
    for test in args.tests:
        print("\n" + "=" * 70 + "\n" + TEST_TITLES[test] + "\n" + "=" * 70, flush=True)
        started = time.monotonic()
        torch.manual_seed(args.seed)
        n = counts[test]["items"]
        indices = np.arange(n) if test == "consistency" else order[:n]
        ix = torch.as_tensor(indices, dtype=torch.long, device=xs.device)
        target_x, target_theta = xs[ix], thetas[ix]
        try:
            if test == "consistency":
                result = resimulation_consistency_test(
                    posterior, target_x, mean, std, checkpoint,
                    cfg["precuneus_idx"], cfg["acc_idx"], cfg["dmn_idx"],
                    n_samples=args.consistency_samples, max_items=n, device=device,
                )
            elif test == "predictive":
                result = posterior_predictive_check(
                    posterior, target_x, checkpoint, indices, args.predictive_samples, n,
                    args.simulation_batch_size, args.seed,
                )
            elif test == "sensitivity":
                result = feature_sensitivity_test(
                    target_theta, checkpoint, args.sensitivity_eps, n,
                    args.sensitivity_repeats, args.simulation_batch_size,
                )
                result["top_features"] = args.sensitivity_top_features
            else:
                count = counts[test]["samples"]
                key = (n, count)
                if key not in sample_cache:
                    sample_cache[key] = _draw_posteriors(posterior, target_x, count)
                samples = sample_cache[key]
                if test == "width":
                    result = posterior_width_test(posterior, target_x, count, n, prior, names, samples)
                elif test == "recovery":
                    result = parameter_recovery_test(posterior, target_theta, target_x, prior, count, n, samples)
                else:
                    result = coverage_test(posterior, target_theta, target_x, n_samples=count,
                                           levels=[0.50, 0.80, 0.90, 0.95], samples=samples)
            result.update(observation_indices=indices, x_target=validation.as_numpy(target_x),
                          elapsed_seconds=time.monotonic() - started)
            # Save before printing: preserve expensive results even if reporting fails.
            validation.save_result(output_dir, test, result)
            manifest["results"][test] = {"status": "completed", "json": f"{test}.json", "npz": f"{test}.npz",
                                          "elapsed_seconds": result["elapsed_seconds"]}
            validation.save_result(output_dir, "manifest", manifest)
            validation.report_result(test, result, names, checkpoint["feature_names"], blocks)
            print(f"[SAVED] {test}.json and {test}.npz", flush=True)
            if test == "consistency":
                if result["x_resim"] is None:
                    raise ValueError("Posterior-mean consistency did not produce resimulation features")
                validation.require_finite("consistency x_resim", result["x_resim"])
            del result
        except Exception as exc:
            manifest["results"][test] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            validation.save_result(output_dir, "manifest", manifest)
            raise
    print(f"[FINISHED] Validation results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
