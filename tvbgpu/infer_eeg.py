#!/usr/bin/env python3
"""Infer a seven-parameter posterior for one explicitly selected prepared EEG epoch."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import random
import subprocess
import time

LOG = logging.getLogger("infer_eeg")


def load_epoch(path, checkpoint, condition, epoch_index, channel_order=None):
    """Read prepared MNE epochs; never invent a montage or change preprocessing."""
    import mne
    import numpy as np
    from tvbgpu.analysis.sbi_features import validate_checkpoint_features, validate_eeg
    config = validate_checkpoint_features(checkpoint)
    epochs = mne.read_epochs(str(path), preload=False, verbose="ERROR")
    if condition:
        if condition not in epochs.event_id:
            raise ValueError(f"Condition {condition!r} is absent from event_id {epochs.event_id}. "
                             "Use prepared FIF epochs with explicit condition labels.")
        epochs = epochs[condition]
    elif len(epochs.event_id) > 1:
        raise ValueError("Multiple event conditions: specify --condition explicitly")
    if not 0 <= epoch_index < len(epochs):
        raise ValueError(f"--epoch-index must be between 0 and {len(epochs)-1}")
    if epochs.info["bads"]:
        raise ValueError(f"Unresolved bad channels: {epochs.info['bads']}; prepare EEG upstream")
    expected = checkpoint.get("channel_names")
    if channel_order is not None:
        supplied = json.loads(Path(channel_order).read_text())
        if expected is not None and supplied != list(expected):
            raise ValueError("Channel manifest disagrees with checkpoint channel_names")
        expected = supplied
    if not isinstance(expected, (list, tuple)) or len(expected) != config.n_channels:
        raise ValueError("Checkpoint lacks a verified channel order. Supply --channel-order "
                         "with a JSON list of simulator projection channel names in order.")
    if len(set(expected)) != len(expected) or list(expected) != epochs.ch_names:
        raise ValueError("EEG channel names/order do not exactly match the checkpoint/manifest; "
                         "no automatic selection or reordering is permitted")
    if any(kind != "eeg" for kind in epochs.get_channel_types()):
        raise ValueError("Prepared epochs must contain EEG channels only")
    if not np.isclose(epochs.info["sfreq"], config.fs, rtol=0, atol=1e-8):
        raise ValueError(f"EEG sampling rate {epochs.info['sfreq']} does not match checkpoint "
                         f"{config.fs}; prepare EEG upstream using the verified protocol")
    selected = epochs[epoch_index:epoch_index + 1]
    data = selected.get_data()
    validate_eeg(data, config)
    expected_times = checkpoint.get("preprocessing_config", {}).get("n_times")
    if expected_times is not None and data.shape[-1] != expected_times:
        raise ValueError("Epoch length does not match checkpoint preprocessing n_times")
    return data, dict(condition=condition or next(iter(epochs.event_id)),
                      epoch_index=epoch_index, original_epoch_index=int(selected.selection[0]),
                      event=selected.events[0].tolist(), channel_names=epochs.ch_names,
                      channel_order_source="checkpoint" if checkpoint.get("channel_names") is not None
                      else str(Path(channel_order).resolve()),
                      sfreq=float(epochs.info["sfreq"]), n_times=data.shape[-1],
                      tmin=float(selected.tmin), tmax=float(selected.tmax),
                      epoch_length_verified=expected_times is not None,
                      bad_channels=list(epochs.info["bads"]))


def extract_observation(data, checkpoint):
    import numpy as np
    from tvbgpu.analysis import sbi_features as sf
    from tvbgpu.analysis.posterior_validation import require_finite
    config = sf.validate_checkpoint_features(checkpoint)
    blocks = sf.extract_signal_features(data, config, LOG)
    if not np.asarray(blocks["alpha_valid"]).all():
        raise ValueError("Canonical extraction reported invalid alpha features")
    for name in sf.SIGNAL_BLOCK_ORDER:
        require_finite(name, blocks[name])
    models = {"fc_pca": checkpoint["fcpca"], "dfa_curve_pca": checkpoint["dfa_pca"],
              "lya_curve_pca": checkpoint["lya_pca"], "pli_pca": checkpoint["pli_pca"],
              "aecc_pca": checkpoint["aecc_pca"]}
    full, names, slices = sf.assemble_post_pca(sf.post_pca_blocks(blocks, models))
    if names != list(checkpoint["feature_names_full"]):
        raise ValueError("Extracted feature names/order disagree with checkpoint")
    for name, sl in slices.items():
        if (sl.start, sl.stop) != tuple(checkpoint["feature_block_slices"][name]):
            raise ValueError(f"Feature block slice mismatch: {name}")
    require_finite("full post-PCA features", full)
    processed = sf.apply_saved_transform(full, checkpoint)
    require_finite("normalized observation", processed)
    for name, sl in slices.items():
        LOG.info("Feature block %s: columns %d:%d, max abs %.6g", name, sl.start,
                 sl.stop, np.abs(full[:, sl]).max())
    return full, processed


def summarize(samples, names, low, high):
    import numpy as np
    from tvbgpu.analysis.posterior_validation import require_finite
    samples = require_finite("posterior samples", samples)
    if samples.ndim != 2 or samples.shape[1] != len(names) or not len(samples):
        raise ValueError("Posterior samples must have shape (draws, parameters)")
    if np.any(samples < low) or np.any(samples > high):
        raise ValueError("Posterior samples fall outside saved prior support")
    rows = []
    # Transform individual draws before computing physical noise statistics.
    j = names.index("log10_weight_noise")
    columns = [(name, samples[:, i], low[i], high[i]) for i, name in enumerate(names)]
    columns.append(("weight_noise", 10.0 ** samples[:, j].astype(np.float64),
                    10.0 ** float(low[j]), 10.0 ** float(high[j])))
    for name, values, lo, hi in columns:
        q = np.quantile(values, [.025, .05, .25, .5, .75, .95, .975])
        rows.append(dict(parameter=name, mean=float(np.mean(values)), median=float(q[3]),
                         std=float(np.std(values, ddof=0)), p05=float(q[1]), p25=float(q[2]),
                         p75=float(q[4]), p95=float(q[5]), ci50_low=float(q[2]), ci50_high=float(q[4]),
                         ci90_low=float(q[1]), ci90_high=float(q[5]), ci95_low=float(q[0]), ci95_high=float(q[6]),
                         prior_low=float(lo), prior_high=float(hi),
                         boundary_low_fraction=float(np.mean(values <= lo + .05 * (hi-lo))),
                         boundary_high_fraction=float(np.mean(values >= hi - .05 * (hi-lo)))))
    return rows


def plot_posterior(samples, rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for i, row in enumerate(rows[:samples.shape[1]]):
        ax = axes.flat[i]
        ax.hist(samples[:, i], bins=40, density=True, color="#377eb8", alpha=.75)
        ax.axvspan(row["ci90_low"], row["ci90_high"], color="#ffb74d", alpha=.25, label="90% credible interval")
        ax.axvline(row["median"], color="#d95f02", label="Median")
        ax.set_xlim(row["prior_low"], row["prior_high"])
        ax.set_title(row["parameter"])
        ax.set_ylabel("Posterior density")
        ax.set_xlabel("Parameter value (axis spans prior)")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center")
    fig.suptitle("Plausible parameter values given this EEG epoch and trained model")
    fig.text(.5, .01, "Narrow credible intervals alone do not establish accurate recovery or calibration.", ha="center")
    fig.tight_layout(rect=(0, .03, 1, .95))
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eeg", required=True, type=Path, help="Prepared MNE Epochs FIF file")
    parser.add_argument("--output-dir", required=True, type=Path, help="New result directory")
    parser.add_argument("--epoch-index", required=True, type=int, help="Zero-based index after condition selection")
    parser.add_argument("--condition", help="Exact MNE event_id label, e.g. EO or EC")
    parser.add_argument("--channel-order", type=Path, help="Verified JSON channel list when absent from checkpoint")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Posterior device; feature kernels still require CUDA")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.num_samples < 2 or args.epoch_index < 0 or not 0 <= args.seed < 2**32:
        parser.error("Require num-samples >= 2, epoch-index >= 0, and 0 <= seed < 2**32")
    return args


def run(args):
    started = time.perf_counter()
    import numpy as np
    import torch
    from tvbgpu.analysis.sbi_checkpoint import load_checkpoint, build_posterior, draw_posteriors, MCMC_PARAMETERS
    from tvbgpu.analysis.posterior_validation import parameter_metadata
    if args.output_dir.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    elif args.device == "cuda":
        raise RuntimeError("CUDA posterior requested but no working CUDA device is available")
    timings = {}
    tick = time.perf_counter()
    checkpoint = load_checkpoint(args.checkpoint)
    names, low, high = parameter_metadata(checkpoint)
    posterior, estimator, prior = build_posterior(checkpoint, args.device)
    estimator.eval()
    timings["posterior_loading_seconds"] = time.perf_counter() - tick
    LOG.info("Checkpoint loaded: %s; full/final features %s/%s", args.checkpoint,
             checkpoint["x_full_dim"], checkpoint["x_dim"])
    tick = time.perf_counter()
    data, eeg_metadata = load_epoch(args.eeg, checkpoint, args.condition, args.epoch_index, args.channel_order)
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical DFA/LYA feature extraction requires a working CUDA/PyCUDA "
                           "environment (Apptainer --nv). --device cpu selects posterior sampling only.")
    with torch.no_grad():
        raw, processed = extract_observation(data, checkpoint)
    timings["eeg_loading_preprocessing_features_seconds"] = time.perf_counter() - tick
    tick = time.perf_counter()
    observation = torch.as_tensor(processed, dtype=torch.float32, device=args.device)
    samples = draw_posteriors(posterior, observation, args.num_samples)[0].numpy()
    rows = summarize(samples, names, low, high)
    timings["posterior_sampling_seconds"] = time.perf_counter() - tick
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "posterior_samples.npy", samples)
    np.save(out / "eeg_features_raw.npy", raw)
    np.save(out / "eeg_features_processed.npy", processed)
    with (out / "posterior_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tick = time.perf_counter()
    plot_posterior(samples, rows, out / "posterior_distributions.png")
    timings["plotting_seconds"] = time.perf_counter() - tick
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=Path(__file__).parent,
                                             stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    versions = {}
    for package in ("torch", "sbi", "numpy", "scipy", "scikit-learn", "mne", "matplotlib", "pycuda"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    timings["total_seconds"] = time.perf_counter() - started
    metadata = dict(output_format_version=1, checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_type="trainer_density_estimator", checkpoint_format_version=checkpoint.get("checkpoint_version"),
                    eeg_input=str(args.eeg.resolve()), eeg=eeg_metadata, num_samples=args.num_samples,
                    device=args.device, feature_device="cuda", seed=args.seed, parameter_names=names,
                    prior_low=low.tolist(), prior_high=high.tolist(), feature_pipeline_version=checkpoint["feature_pipeline_version"],
                    raw_feature_representation="full post-PCA, before feature_keep and normalization",
                    raw_feature_dimension=raw.shape[1], final_feature_dimension=processed.shape[1],
                    feature_names_full=checkpoint["feature_names_full"], feature_names=checkpoint["feature_names"],
                    feature_keep=np.asarray(checkpoint["feature_keep"]).tolist(),
                    feature_config=checkpoint["feature_config"], preprocessing_config=checkpoint["preprocessing_config"],
                    mcmc_parameters=MCMC_PARAMETERS, noise_conversion="weight_noise = 10 ** log10_weight_noise",
                    intervals="equal-tailed credible intervals from posterior draws",
                    boundary_fraction_definition="lowest/highest 5% of each reported prior range",
                    software=versions, git_commit=commit, git_dirty=dirty,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(), timings=timings)
    (out / "inference_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"{'Parameter':22s} {'Median':>11s} {'90% credible interval':>27s} {'Mean':>11s} {'SD':>11s}")
    for row in rows:
        print(f"{row['parameter']:22s} {row['median']:11.5g} "
              f"[{row['ci90_low']:11.5g}, {row['ci90_high']:11.5g}] {row['mean']:11.5g} {row['std']:11.5g}")
    print("Runtime (seconds):", json.dumps(timings))
    print("Output directory:", out)
    return out


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(args)
    except Exception:
        LOG.exception("EEG inference failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
