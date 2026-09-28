#!/usr/bin/env python3
"""Infer a seven-parameter posterior from BrainVision EEG or prepared MNE epochs."""
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
EBRAINS_LABEL = "EBRAINS synthetic TVB EEG — sub-001"


def dk68_reference_labels():
    """Use the simulator's bundled centres as the DK identity reference."""
    import zipfile
    archive = Path(__file__).parent / "data" / "connectivity_zerlaut_68_newcentres.zip"
    with zipfile.ZipFile(archive) as stream:
        text = stream.read("QL_20120814_Connectivity/centres.txt").decode("utf-8")
    return [line.split()[0] for line in text.splitlines() if line.strip()]


def normalize_region_label(label):
    """Normalize spelling/separators and hemisphere placement, not anatomical identity."""
    import re
    value = re.sub(r"[^a-z0-9]", "", label.strip().lower())
    if value.startswith("ctx"):
        value = value[3:]
    for marker, hemisphere in (("left", "l"), ("right", "r"), ("lh", "l"), ("rh", "r")):
        if value.startswith(marker):
            return value[len(marker):] + hemisphere
        if value.endswith(marker):
            return value[:-len(marker)] + hemisphere
    # Short hemisphere prefixes such as L_bankssts; canonical suffixes stay put.
    if re.match(r"^[lr][_.\s-]", label.strip().lower()):
        return value[1:] + value[0]
    return value


def load_ebrains_dk68_connectivity(atlas_path, weights_path, distance_path=None, *, return_metadata=False):
    """Identify all bilateral DK cortical labels; preserve their input ordering."""
    import numpy as np
    print(f"[CONNECTIVITY]\nAtlas file: {Path(atlas_path).resolve()}", flush=True)
    with Path(atlas_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [[cell.strip() for cell in row] for row in csv.reader(stream, delimiter="\t")
                if any(cell.strip() for cell in row)]
    print(f"TSV raw shape (including any header): ({len(rows)}, {len(rows[0]) if rows else 0})\n"
          f"First 10 raw rows: {json.dumps(rows[:10], ensure_ascii=False)}", flush=True)
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Atlas TSV is empty or has inconsistent column counts; see raw rows above")
    matrices = []
    for kind, path in (("Connectivity", weights_path), ("Tract lengths", distance_path)):
        if path is None:
            matrices.append(None)
            continue
        print(f"{kind} file: {Path(path).resolve()}", flush=True)
        matrix = np.loadtxt(path, delimiter="\t", ndmin=2)
        print(f"{kind} shape: {matrix.shape}", flush=True)
        if matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError(f"{kind}: expected a finite square matrix; got {matrix.shape}")
        matrices.append(matrix)
    n_regions = matrices[0].shape[0]
    if matrices[1] is not None and matrices[1].shape != matrices[0].shape:
        raise ValueError(f"Weights shape {matrices[0].shape} != tract lengths shape {matrices[1].shape}")
    # Matrix size disambiguates the first row without guessing from unfamiliar headings.
    if len(rows) == n_regions + 1:
        header, data = rows[0], rows[1:]
    elif len(rows) == n_regions:
        header, data = None, rows
    else:
        raise ValueError(f"Atlas has {len(rows)} nonblank rows; matrix has {n_regions} regions. "
                         "Expected one atlas row per region, optionally preceded by one header row")
    print(f"Header present: {header is not None} (inferred from matrix row count)\n"
          f"Columns: {header if header is not None else list(range(len(rows[0])))}\n"
          f"TSV data shape: ({len(data)}, {len(data[0])})\nFirst 10 data rows: "
          f"{json.dumps(data[:10], ensure_ascii=False)}", flush=True)
    reference = [normalize_region_label(label) for label in dk68_reference_labels()]
    reference_set = set(reference)
    subcortical_bases = {"thalamus", "thalamusproper", "caudate", "putamen", "pallidum",
                        "hippocampus", "amygdala", "accumbens", "accumbensarea", "ventraldc"}
    cerebellar_bases = {"cerebellumcortex"}
    unqualified_regions = {label[:-1] for label in reference} | subcortical_bases | cerebellar_bases
    normalized_header = [name.strip().lower() for name in header] if header else []
    hemisphere_cols = [i for i, name in enumerate(normalized_header) if name in ("hemi", "hemisphere")]

    def column_labels(column):
        labels = [row[column] for row in data]
        if len(hemisphere_cols) == 1 and column != hemisphere_cols[0]:
            hemi = hemisphere_cols[0]
            labels = [row[hemi] + "_" + label if normalize_region_label(label) in unqualified_regions
                      else label for label, row in zip(labels, data)]
        return labels

    scores = [sum(normalize_region_label(label) in reference_set for label in column_labels(i))
              for i in range(len(data[0]))]
    candidates = [i for i, score in enumerate(scores) if score == max(scores) and score > 0]
    if not candidates:
        aliases = {"name", "label", "region", "region_name", "region_label", "regionname",
                   "regionlabel", "roi", "roi_name", "anatomical_name"}
        candidates = [i for i, name in enumerate(normalized_header) if name in aliases]
    if not candidates:
        def is_number(value):
            try:
                float(value)
                return True
            except ValueError:
                return False
        candidates = [i for i in range(len(data[0]))
                      if all(row[i] and not is_number(row[i]) for row in data)]
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous/unidentified atlas label column: candidates={candidates}, "
                         f"columns={header}, DK label match counts={scores}. No column was guessed")
    column = candidates[0]
    labels = column_labels(column)
    canonical = [normalize_region_label(label) for label in labels]
    selected = [i for i, label in enumerate(canonical) if label in reference_set]
    excluded = [i for i in range(n_regions) if i not in selected]
    # Recognize bilateral subcortical structures, rather than calling arbitrary extras subcortical.
    subcortical = {name + hemi for name in subcortical_bases for hemi in ("l", "r")}
    cerebellar = {name + hemi for name in cerebellar_bases for hemi in ("l", "r")}
    unknown = [labels[i] for i in excluded if canonical[i] not in subcortical | cerebellar]
    n_subcortical = sum(canonical[i] in subcortical for i in excluded)
    n_cerebellar = sum(canonical[i] in cerebellar for i in excluded)
    print(f"Label column: {header[column] if header else column}\nAtlas input: {len(labels)} regions\n"
          f"Detected cortical labels: {len(selected)}; subcortical labels: "
          f"{n_subcortical}; cerebellar labels: {n_cerebellar}\nUnknown labels: {unknown}", flush=True)
    if (n_regions not in (68, 84) or len(selected) != 68
            or {canonical[i] for i in selected} != reference_set or unknown
            or len(set(canonical)) != n_regions):
        missing = sorted(reference_set - set(canonical))
        raise ValueError(f"Atlas is not verified DK68 or DK68 + 16 recognized non-DK regions: "
                         f"total={n_regions}, cortical={len(selected)}, missing DK labels={missing}, "
                         f"unknown labels={unknown}. Duplicate identities are not permitted")
    labels68 = [labels[i] for i in selected]
    result = [matrix[np.ix_(selected, selected)] if matrix is not None else None for matrix in matrices]
    if any(matrix is not None and (matrix.shape != (68, 68) or not np.isfinite(matrix).all())
           for matrix in result) or len(labels68) != 68:
        raise ValueError("Final connectivity must contain 68 labels and finite (68, 68) matrices")
    same_order = [canonical[i] for i in selected] == reference
    print(("Atlas already corresponds to DK68" if n_regions == 68 else
           f"Detected: 68 DK cortical + {n_subcortical} subcortical + {n_cerebellar} cerebellar\n"
           "Selected DK68 cortical subset") +
          f"\nSelected original indices: {selected}\nFinal connectivity: 68 x 68\n"
          f"Region names: {labels68}\nMatches simulator region order: {same_order}", flush=True)
    if not same_order:
        LOG.warning("Input cortical order differs from simulator centres; preserved without permutation. "
                    "Do not use for resimulation without aligning simulator region-dependent data.")
    metadata = dict(atlas_input_regions=n_regions, atlas_columns=header, atlas_has_header=header is not None,
                    subcortical_regions=n_subcortical, cerebellar_regions=n_cerebellar,
                    atlas_shape=[len(data), len(data[0])], atlas_label_column=column,
                    cortical_indices=selected, excluded_region_labels=[labels[i] for i in excluded],
                    matches_simulator_order=same_order)
    answer = (result[0], result[1], labels68)
    return answer + (metadata,) if return_metadata else answer


def approximate_channel_indices(raw, count):
    """Cover available positions, otherwise channel indices; never infer a montage."""
    import numpy as np
    positions = np.asarray([ch["loc"][:3] for ch in raw.info["chs"]])
    spatial = (np.isfinite(positions).all()
               and np.all(np.linalg.norm(positions, axis=1) > 0)
               and len(np.unique(positions, axis=0)) == len(positions)
               and np.linalg.matrix_rank(positions - positions.mean(axis=0)) >= 2)
    if spatial:
        # Farthest-point sampling in the existing coordinate frame. Ties go to
        # the lowest original index; sorted output preserves acquisition order.
        selected = [int(np.argmax(np.sum((positions - positions.mean(axis=0)) ** 2, axis=1)))]
        distances = np.full(len(positions), np.inf)
        while len(selected) < count:
            distances = np.minimum(distances, np.sum((positions - positions[selected[-1]]) ** 2, axis=1))
            distances[selected] = -np.inf
            selected.append(int(np.argmax(distances)))
        return np.sort(selected), "approximate_spatial_farthest_point"
    return np.rint(np.linspace(0, len(raw.ch_names) - 1, count)).astype(int), "approximate_evenly_spaced_indices"


def load_ebrains_epoch(path, checkpoint, epoch_index=0, channel_order=None, window_samples=None):
    """Load sensor-space EEG; resample, then select one nonoverlapping window."""
    import mne
    import numpy as np
    from tvbgpu.analysis.sbi_features import validate_checkpoint_features, validate_eeg
    config = validate_checkpoint_features(checkpoint)
    raw = mne.io.read_raw_brainvision(str(path), preload=True)
    original_sfreq, original_samples = float(raw.info["sfreq"]), int(raw.n_times)
    finite = bool(np.isfinite(raw.get_data()).all())
    print(f"[EBRAINS]\n{EBRAINS_LABEL}\nEEG file: {path}\n"
          f"Channels: {len(raw.ch_names)}\nSampling rate: {original_sfreq} Hz\n"
          f"Samples: {original_samples}\nDuration: {original_samples / original_sfreq:.6g} s\n"
          f"Finite: {finite}\nCHANNEL ORDER: {raw.ch_names}", flush=True)
    original_names = list(raw.ch_names)
    required = config.n_channels
    LOG.info("Original EEG: %d channels, %.3f Hz. Checkpoint/feature requirements: "
             "%d channels, %.3f Hz; full/final features %d/%d",
             len(original_names), original_sfreq, required, config.fs,
             checkpoint['x_full_dim'], checkpoint['x_dim'])
    if len(original_names) < required:
        raise ValueError(f"Input EEG has {len(original_names)} channels; feature extractor requires "
                         f"{required}. Channels cannot be padded or invented.")
    if not finite:
        raise ValueError("EBRAINS EEG contains NaN or Inf")
    if raw.info["bads"] or any(kind != "eeg" for kind in raw.get_channel_types()):
        raise ValueError("EBRAINS input must contain EEG channels only, with no unresolved bad channels")
    if len(set(raw.ch_names)) != len(raw.ch_names) or not all(raw.ch_names):
        raise ValueError("EBRAINS input must have unique nonempty channel names")
    expected = checkpoint.get("channel_names")
    source = "checkpoint"
    if channel_order is not None:
        supplied = json.loads(Path(channel_order).read_text())
        if expected is not None and supplied != list(expected):
            raise ValueError("Channel manifest disagrees with checkpoint channel_names")
        expected, source = supplied, str(Path(channel_order).resolve())
    if expected is not None and (not isinstance(expected, (list, tuple))
                                or len(expected) != required or len(set(expected)) != required):
        raise ValueError(f"Checkpoint/manifest must contain {required} unique channel names")
    indices = np.arange(len(original_names))
    method = "original_order"
    approximate = False
    if len(original_names) > required:
        if expected is not None and all(name in original_names for name in expected):
            indices = np.asarray([original_names.index(name) for name in expected])
            method = "checkpoint_or_manifest_names"
        elif channel_order is not None:
            raise ValueError("Explicit channel manifest names are absent from the input EEG")
        else:
            indices, method = approximate_channel_indices(raw, required)
            approximate = True
            LOG.warning("Input EEG has %d channels. Feature extractor requires %d channels. "
                        "No verified montage mapping was found. Selecting %d/%d channels using %s. "
                        "WORKSHOP/DEMO inference only: no anatomical equivalence to the training montage.",
                        len(original_names), required, required, len(original_names), method)
        raw.pick(indices.tolist())
        LOG.info("Selected original indices (zero-based): %s", indices.tolist())
        LOG.info("Selected channels: %s", raw.ch_names)
    if approximate:
        source = method
    elif expected is None:
        source = "ebrains_brainvision_order_assumed"
    elif list(expected) != raw.ch_names:
        raise ValueError("EEG channel names/order do not exactly match the checkpoint/manifest")
    assumption = ("Approximate channel selection for workshop/demo inference; anatomical correspondence "
                  "to the training montage is unknown." if approximate else
                  f"This mode assumes the {required}-channel EBRAINS sensor ordering corresponds "
                  "to the observation space used by the checkpoint.")
    LOG.warning(assumption)
    if not np.isfinite(config.fs) or config.fs <= 0:
        raise ValueError("Checkpoint sampling frequency must be finite and positive")
    resampled = not np.isclose(original_sfreq, config.fs, rtol=0, atol=1e-8)
    if resampled:
        LOG.info("Resampling: %.3f -> %.3f Hz", original_sfreq, config.fs)
        raw.resample(config.fs)
    if not np.isclose(raw.info["sfreq"], config.fs, rtol=0, atol=1e-8):
        raise ValueError(f"EEG sampling rate {raw.info['sfreq']} does not match required {config.fs} Hz")
    expected_times = checkpoint.get("preprocessing_config", {}).get("n_times")
    if expected_times is not None and window_samples is not None and window_samples != expected_times:
        raise ValueError("--window-samples disagrees with checkpoint preprocessing n_times")
    n_times = expected_times if expected_times is not None else window_samples
    if n_times is None:
        n_times = 4001
        LOG.warning("Checkpoint lacks preprocessing n_times; assuming a 4001-sample workshop window")
    if not isinstance(n_times, (int, np.integer)) or n_times <= 500 or epoch_index < 0:
        raise ValueError("Window length must be an integer >500 and epoch index must be nonnegative")
    start, stop = epoch_index * n_times, (epoch_index + 1) * n_times
    if stop > raw.n_times:
        raise ValueError(f"Requested window [{start}:{stop}] exceeds {raw.n_times} samples after resampling")
    data = raw.get_data(start=start, stop=stop)[np.newaxis, :, :]
    validate_eeg(data, config)
    LOG.info("Selected sensor-space EEG window: shape=%s, sfreq=%g Hz, samples=%d:%d",
             data.shape, raw.info["sfreq"], start, stop)
    label = EBRAINS_LABEL + (" — WORKSHOP/DEMO: approximate channel selection" if approximate else "")
    return data, dict(observation_label=label, condition="rest", epoch_index=epoch_index,
                      approximate_channel_selection=approximate, channel_selection_method=method,
                      original_channel_names=original_names, selected_original_indices=indices.tolist(),
                      interpretation="Trained-model posterior under possible model mismatch; "
                      "not recovery of the original hidden generating parameters",
                      channel_names=raw.ch_names, channel_order_source=source,
                      channel_order_assumption=assumption, sfreq=float(raw.info["sfreq"]),
                      original_sfreq=original_sfreq, original_n_times=original_samples,
                      resampled=resampled, n_times=int(n_times), start_sample=int(start),
                      stop_sample=int(stop), tmin=start / config.fs, tmax=(stop - 1) / config.fs,
                      epoch_length_verified=expected_times is not None,
                      window_length_source="checkpoint" if expected_times is not None else
                      "cli" if window_samples is not None else "workshop_default",
                      bad_channels=[], observed_eeg_projected=False)


def load_epoch(path, checkpoint, condition, epoch_index, channel_order=None):
    """Read FIF epochs or the first 61 channels/4001 samples of BrainVision EEG."""
    import mne
    import numpy as np
    from tvbgpu.analysis.sbi_features import validate_checkpoint_features, validate_eeg
    config = validate_checkpoint_features(checkpoint)
    brainvision = Path(path).suffix.lower() == ".vhdr"
    if brainvision:
        if condition is not None or epoch_index != 0:
            raise ValueError("BrainVision input provides one epoch: use --epoch-index 0 "
                             "and omit --condition")
        raw = mne.io.read_raw_brainvision(str(path), preload=True)
        LOG.debug("EEG input info %s", raw.info)
        data = raw.get_data()[:61, :4001]
        all_epochs = data[np.newaxis, :, :]
        info = mne.pick_info(raw.info, list(range(data.shape[0])))
        epochs = mne.EpochsArray(all_epochs, info, event_id={"BrainVision": 1},
                                 verbose="ERROR")
    else:
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


def validate_inference_observation(data, eeg_metadata, full, processed, checkpoint, estimator):
    """Fail before MCMC if the observation violates the saved feature contract."""
    import numpy as np
    from tvbgpu.analysis.sbi_features import validate_checkpoint_features, validate_eeg
    config = validate_checkpoint_features(checkpoint)
    validate_eeg(data, config)
    if not np.isclose(eeg_metadata['sfreq'], config.fs, rtol=0, atol=1e-8):
        raise ValueError(f"EEG rate {eeg_metadata['sfreq']} Hz; feature extractor requires {config.fs} Hz")
    for name, values, width in (("Full post-PCA", full, checkpoint['x_full_dim']),
                                ("Transformed", processed, checkpoint['x_dim'])):
        if values.shape != (1, width) or not np.isfinite(values).all():
            raise ValueError(f"{name} features: got shape {values.shape}; expected finite (1, {width})")
    condition_shape = getattr(estimator, "condition_shape", None)
    if condition_shape is not None and tuple(condition_shape) != (checkpoint['x_dim'],):
        raise ValueError(f"Posterior estimator condition shape {tuple(condition_shape)} disagrees "
                         f"with checkpoint feature dimension {checkpoint['x_dim']}")


def plot_posterior(samples, rows, destination, observation_label=None):
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
    fig.suptitle(observation_label or "Plausible parameter values given this EEG epoch and trained model")
    fig.text(.5, .01, "Narrow credible intervals alone do not establish accurate recovery or calibration.", ha="center")
    fig.tight_layout(rect=(0, .03, 1, .95))
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-format", choices=("auto", "ebrains-synthetic"), default="auto")
    parser.add_argument("--eeg", type=Path,
                        help="BrainVision .vhdr (first 61 channels/4001 samples) or prepared MNE Epochs FIF")
    parser.add_argument("--eeg-vhdr", type=Path, help="BrainVision .vhdr for ebrains-synthetic mode")
    parser.add_argument("--dk-atlas", type=Path)
    parser.add_argument("--connectivity-weights", type=Path)
    parser.add_argument("--connectivity-distances", type=Path)
    parser.add_argument("--window-samples", type=int,
                        help="EBRAINS window length after resampling; saved n_times takes precedence, otherwise 4001")
    parser.add_argument("--output-dir", default=Path("."), type=Path,
                        help="Existing result directory (default: current directory); result files are overwritten")
    parser.add_argument("--epoch-index", default=0, type=int,
                        help="Zero-based epoch/window index (default: 0; non-EBRAINS BrainVision requires 0)")
    parser.add_argument("--condition", help="Exact MNE event_id label, e.g. EO or EC")
    parser.add_argument("--channel-order", type=Path, help="Verified JSON channel list when absent from checkpoint")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Posterior device; feature kernels still require CUDA")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.num_samples < 2 or args.epoch_index < 0 or not 0 <= args.seed < 2**32:
        parser.error("Require num-samples >= 2, epoch-index >= 0, and 0 <= seed < 2**32")
    connectivity_paths = (args.dk_atlas, args.connectivity_weights, args.connectivity_distances)
    if args.input_format == "ebrains-synthetic":
        if (args.eeg is None) == (args.eeg_vhdr is None):
            parser.error("EBRAINS mode requires exactly one of --eeg-vhdr or --eeg")
        args.eeg = args.eeg_vhdr if args.eeg_vhdr is not None else args.eeg
        if args.eeg.suffix.lower() != ".vhdr" or args.condition is not None:
            parser.error("EBRAINS mode requires a .vhdr file and no --condition")
        if any(connectivity_paths) and not all(connectivity_paths):
            parser.error("Supply --dk-atlas, --connectivity-weights and --connectivity-distances together")
        if args.window_samples is not None and args.window_samples <= 500:
            parser.error("--window-samples must be >500")
    elif (args.eeg is None or args.eeg_vhdr is not None or any(connectivity_paths)
          or args.window_samples is not None):
        parser.error("Default mode requires --eeg; EBRAINS options require --input-format ebrains-synthetic")
    return args


def run(args):
    started = time.perf_counter()
    import numpy as np
    import torch
    from tvbgpu.analysis.sbi_checkpoint import load_checkpoint, build_posterior, draw_posteriors, MCMC_PARAMETERS
    from tvbgpu.analysis.posterior_validation import parameter_metadata
    out = args.output_dir.resolve()
    if not out.is_dir():
        raise ValueError(f"Output directory does not exist: {out}; choose an existing directory")
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
    connectivity_metadata = None
    observation_label = None
    if args.input_format == "ebrains-synthetic":
        observation_label = EBRAINS_LABEL
        data, eeg_metadata = load_ebrains_epoch(args.eeg, checkpoint, args.epoch_index,
                                               args.channel_order, args.window_samples)
        observation_label = eeg_metadata["observation_label"]
        LOG.warning("Posterior describes this trained model under possible model mismatch; "
                    "it does not establish recovery of the original hidden generating parameters.")
        if args.dk_atlas is not None:
            _, _, labels, atlas_metadata = load_ebrains_dk68_connectivity(
                args.dk_atlas, args.connectivity_weights, args.connectivity_distances, return_metadata=True)
            connectivity_metadata = dict(atlas=str(args.dk_atlas.resolve()),
                weights=str(args.connectivity_weights.resolve()),
                distances=str(args.connectivity_distances.resolve()), region_labels=labels,
                **atlas_metadata,
                used_for_resimulation=False)
            LOG.info("Connectivity validated only; this endpoint does not perform resimulation")
    else:
        data, eeg_metadata = load_epoch(args.eeg, checkpoint, args.condition, args.epoch_index, args.channel_order)
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical DFA/LYA feature extraction requires a working CUDA/PyCUDA "
                           "environment (Apptainer --nv). --device cpu selects posterior sampling only.")
    with torch.no_grad():
        raw, processed = extract_observation(data, checkpoint)
    print(f"[FEATURES]\nRaw feature shape: {raw.shape} (post-PCA, before feature_keep)\n"
          f"Retained feature shape: {processed.shape}\n"
          f"Finite: {bool(np.isfinite(raw).all() and np.isfinite(processed).all())}\n"
          f"Checkpoint expected dimension: {checkpoint['x_dim']}", flush=True)
    timings["eeg_loading_preprocessing_features_seconds"] = time.perf_counter() - tick
    tick = time.perf_counter()
    validate_inference_observation(data, eeg_metadata, raw, processed, checkpoint, estimator)
    observation = torch.as_tensor(processed, dtype=torch.float32, device=args.device)
    if not torch.isfinite(observation).all():
        raise ValueError("Observation contains NaN/Inf after conversion to posterior float32 input")
    samples = draw_posteriors(posterior, observation, args.num_samples)[0].numpy()
    rows = summarize(samples, names, low, high)
    timings["posterior_sampling_seconds"] = time.perf_counter() - tick
    np.save(out / "posterior_samples.npy", samples)
    np.save(out / "eeg_features_raw.npy", raw)
    np.save(out / "eeg_features_processed.npy", processed)
    with (out / "posterior_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tick = time.perf_counter()
    plot_posterior(samples, rows, out / "posterior_distributions.png", observation_label)
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
                    input_format=args.input_format, observation_label=observation_label,
                    connectivity=connectivity_metadata,
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
