"""CPU-only summaries and output for posterior validation; no simulator imports."""
import json
from pathlib import Path

import numpy as np


PARAMETER_ORDER = (
    "coupling", "log10_weight_noise", "rNMDA", "global_speed", "tau_K", "phi", "dV",
)


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def require_finite(name, value):
    value = as_numpy(value)
    invalid = ~np.isfinite(value)
    if invalid.any():
        bad = np.argwhere(invalid)
        raise ValueError(f"{name} contains NaN/Inf at indices {bad[:10].tolist()}")
    return value


def parameter_metadata(checkpoint):
    names = list(checkpoint["parameter_names"])
    if tuple(names) != PARAMETER_ORDER:
        raise ValueError("Checkpoint parameter order does not match the seven-parameter simulator")
    low = require_finite("prior_low", checkpoint["prior_low"])
    high = require_finite("prior_high", checkpoint["prior_high"])
    if low.shape != (len(names),) or high.shape != low.shape or np.any(high <= low):
        raise ValueError("Invalid saved prior bounds")
    return names, low, high


def feature_block_indices(checkpoint):
    """Map saved full-vector slices through feature_keep, without prefix matching."""
    keep = np.asarray(checkpoint["feature_keep"], dtype=bool)
    full_names = list(checkpoint["feature_names_full"])
    names = list(checkpoint["feature_names"])
    if keep.ndim != 1 or len(full_names) != keep.size:
        raise ValueError("Full feature names and feature_keep disagree")
    if [n for n, k in zip(full_names, keep) if k] != names:
        raise ValueError("Retained feature names and feature_keep disagree")
    retained_full = np.flatnonzero(keep)
    covered = np.zeros(keep.size, dtype=int)
    blocks = {}
    for name in checkpoint["feature_block_order"]:
        start, stop = checkpoint["feature_block_slices"][name]
        if not (0 <= start <= stop <= keep.size):
            raise ValueError(f"Invalid saved feature slice for {name}: {(start, stop)}")
        covered[start:stop] += 1
        blocks[name] = np.flatnonzero((retained_full >= start) & (retained_full < stop))
    if not np.all(covered == 1):
        raise ValueError("Saved feature block slices must partition the full feature vector")
    return blocks


def distribution(values):
    values = require_finite("distribution", values).astype(np.float64)
    return {"count": values.size, "mean": values.mean(), "median": np.median(values),
            "std": values.std(ddof=0), "min": values.min(), "max": values.max()}


def ratio(numerator, denominator):
    """Undefined ratios remain NaN; do not introduce a denominator floor."""
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan),
                     where=denominator != 0)


def width_summary(samples, low, high, names):
    """Width means sample standard deviation, as in the original width test.

    Inference-coordinate widths are retained. For physical widths, only the
    log10_weight_noise samples and bounds are exponentiated, before taking std.
    """
    samples = require_finite("posterior samples", samples)
    if samples.ndim != 3 or samples.shape[1] < 2:
        raise ValueError("Widths require (observations, >=2 draws, parameters)")
    physical = samples.astype(np.float64, copy=True)
    physical_low, physical_high = np.array(low, dtype=float), np.array(high, dtype=float)
    physical_names = list(names)
    j = names.index("log10_weight_noise")
    physical[..., j] = 10.0 ** physical[..., j]
    physical_low[j], physical_high[j] = 10.0 ** low[j], 10.0 ** high[j]
    physical_names[j] = "weight_noise"
    result = {"samples": samples, "physical_parameter_names": physical_names,
              "physical_prior_low": physical_low, "physical_prior_high": physical_high}
    for prefix, draws, span in (
        ("", samples, np.asarray(high) - low),
        ("physical_", physical, physical_high - physical_low),
    ):
        widths = draws.std(axis=1, ddof=1)
        normalized = widths / span
        result[prefix + "widths"] = widths
        result[prefix + "widths_normalized"] = normalized
        for label, values in (("width", widths), ("width_normalized", normalized)):
            result[prefix + "mean_" + label + "_per_param"] = values.mean(axis=0)
            result[prefix + "median_" + label + "_per_param"] = np.median(values, axis=0)
            result[prefix + "std_" + label + "_per_param"] = values.std(axis=0, ddof=0)
    return result


def recovery_summary(samples, theta_true, low, high):
    """Recovery stays entirely in checkpoint/inference parameter coordinates."""
    samples = require_finite("posterior samples", samples)
    truth = require_finite("theta_true", theta_true)
    means = samples.mean(axis=1)
    medians = np.median(samples, axis=1)
    if means.shape != truth.shape:
        raise ValueError("Posterior samples and known theta rows disagree")
    diff = means - truth
    normalized = diff / (np.asarray(high) - low)
    correlations = np.full(truth.shape[1], np.nan)
    # Two points yield a trivial +/-1. Constant columns have undefined Pearson r.
    if len(truth) >= 3:
        for j in range(truth.shape[1]):
            if np.ptp(truth[:, j]) > 0 and np.ptp(means[:, j]) > 0:
                correlations[j] = np.corrcoef(truth[:, j], means[:, j])[0, 1]
    raw_errors = np.linalg.norm(diff, axis=1)
    norm_errors = np.linalg.norm(normalized, axis=1)
    return {
        "samples": samples, "theta_true": truth, "theta_means": means,
        "theta_medians": medians, "abs_errors_per_param": np.abs(diff),
        "normalized_abs_errors_per_param": np.abs(normalized),
        "median_abs_errors_per_param": np.abs(medians - truth),
        "median_normalized_abs_errors_per_param": np.abs(medians - truth) / (np.asarray(high) - low),
        "mean_abs_error_per_param": np.abs(diff).mean(axis=0),
        "median_abs_error_per_param": np.median(np.abs(diff), axis=0),
        "mean_normalized_abs_error_per_param": np.abs(normalized).mean(axis=0),
        "median_normalized_abs_error_per_param": np.median(np.abs(normalized), axis=0),
        "rmse_per_param": np.sqrt(np.mean(diff ** 2, axis=0)),
        "normalized_rmse_per_param": np.sqrt(np.mean(normalized ** 2, axis=0)),
        "pearson_r_per_param": correlations,
        "raw_errors": raw_errors, "mean_raw_error": raw_errors.mean(),
        "median_raw_error": np.median(raw_errors), "normalized_errors": norm_errors,
        "mean_normalized_error": norm_errors.mean(), "median_normalized_error": np.median(norm_errors),
    }


def coverage_summary(samples, theta_true, levels):
    """Marginal equal-tailed intervals, preserving the existing coverage definition."""
    samples = require_finite("posterior samples", samples)
    truth = require_finite("theta_true", theta_true)
    levels = np.asarray(levels, dtype=float)
    if np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("Coverage levels must lie strictly between 0 and 1")
    lower = np.quantile(samples, (1 - levels) / 2, axis=1).transpose(1, 0, 2)
    upper = np.quantile(samples, (1 + levels) / 2, axis=1).transpose(1, 0, 2)
    inside = (truth[:, None, :] >= lower) & (truth[:, None, :] <= upper)
    return {"samples": samples, "theta_true": truth, "levels": levels,
            "lower": lower, "upper": upper, "inside_matrix": inside,
            "coverage_per_param": inside.mean(axis=0),
            "mean_coverage": inside.mean(axis=(0, 2)), "n_observations": len(truth)}


def predictive_summary(x_resim, targets, training_xs, target_indices, blocks, rng):
    """Compare each predictive draw to its target and to K unrelated saved rows.

    Baseline rows are sampled uniformly without replacement per target, excluding
    that target's row. No extra baseline simulations or parameter averaging.
    """
    x_resim = require_finite("predictive x_resim", x_resim)
    targets = require_finite("predictive targets", targets)
    training_xs = require_finite("baseline training xs", training_xs)
    n, k, f = x_resim.shape
    if targets.shape != (n, f) or len(target_indices) != n:
        raise ValueError("Predictive target rows do not match resimulation rows")
    if k > len(training_xs) - 1:
        raise ValueError("Random baseline needs at least K+1 saved training rows")
    baseline_indices = []
    for target in target_indices:
        if not 0 <= target < len(training_xs):
            raise ValueError("Baseline target index is outside saved training rows")
        indices = rng.choice(len(training_xs) - 1, size=k, replace=False)
        baseline_indices.append(indices + (indices >= target))
    baseline_indices = np.stack(baseline_indices)
    baseline_x = training_xs[baseline_indices]
    delta, baseline_delta = x_resim - targets[:, None, :], baseline_x - targets[:, None, :]
    distances = np.linalg.norm(delta, axis=-1)
    baseline = np.linalg.norm(baseline_delta, axis=-1)
    result = {
        "x_target": targets, "x_resim": x_resim,
        "baseline_indices": baseline_indices, "baseline_x": baseline_x,
        "distances": distances, "baseline_distances": baseline,
        "posterior_summary": distribution(distances), "baseline_summary": distribution(baseline),
        "median_distance_ratio": ratio(np.median(distances), np.median(baseline)),
        "per_observation_mean": distances.mean(axis=1),
        "per_observation_median": np.median(distances, axis=1),
        "baseline_per_observation_mean": baseline.mean(axis=1),
        "baseline_per_observation_median": np.median(baseline, axis=1),
        "per_observation_median_ratio": ratio(np.median(distances, axis=1), np.median(baseline, axis=1)),
        "blocks": {},
    }
    for name, indices in blocks.items():
        if not len(indices):
            continue
        d = np.linalg.norm(delta[..., indices], axis=-1)
        b = np.linalg.norm(baseline_delta[..., indices], axis=-1)
        result["blocks"][name] = {
            "n_features": len(indices), "distances": d, "baseline_distances": b,
            "posterior_summary": distribution(d), "baseline_summary": distribution(b),
            "median_distance_ratio": ratio(np.median(d), np.median(b)),
        }
    return result


def sensitivity_design(theta, low, high, eps):
    """One-at-a-time forward perturbations; reverse direction near the upper bound."""
    theta = require_finite("sensitivity theta", theta).astype(np.float32)
    if not 0 < eps <= 0.5:
        raise ValueError("Sensitivity eps must be in (0, 0.5] of each prior range")
    if np.any((theta < low) | (theta > high)):
        raise ValueError("Sensitivity theta is outside the saved prior")
    n, p = theta.shape
    design = np.repeat(theta[:, None, :], p + 1, axis=1)
    for j in range(p):
        step = eps * (high[j] - low[j])
        direction = np.where(theta[:, j] + step <= high[j], 1.0, -1.0)
        design[:, j + 1, j] += direction * step
    steps = (design[:, 1:, :] - theta[:, None, :])[:, np.arange(p), np.arange(p)]
    return design, steps


def sensitivity_summary(features, steps, low, high, blocks):
    """Finite feature differences per unit prior-range fraction, plus repeat noise.

    features: (observations, base+parameters, repeats, retained_features).
    These are forward responses, not posterior information/identifiability scores.
    """
    features = require_finite("sensitivity features", features)
    mean_features = features.mean(axis=2)
    delta = mean_features[:, 1:] - mean_features[:, :1]
    scaled_steps = steps / (np.asarray(high) - low)
    derivative = delta / scaled_steps[..., None]
    repeats = features.shape[2]
    if repeats >= 2:
        variance = features.var(axis=2, ddof=1)
        delta_se = np.sqrt((variance[:, 1:] + variance[:, :1]) / repeats)
    else:
        delta_se = np.full_like(delta, np.nan)
    result = {"features": features, "steps": steps, "scaled_steps": scaled_steps,
              "feature_delta": delta, "feature_derivative": derivative,
              "mean_abs_derivative": np.abs(derivative).mean(axis=0),
              "delta_standard_error": delta_se,
              "mean_delta_standard_error": delta_se.mean(axis=0), "blocks": {}}
    for name, indices in blocks.items():
        if not len(indices):
            continue
        norms = np.linalg.norm(derivative[..., indices], axis=-1)
        result["blocks"][name] = {
            "n_features": len(indices), "derivative_l2": norms,
            "mean_l2": norms.mean(axis=0), "median_l2": np.median(norms, axis=0),
            "mean_rms": (norms / np.sqrt(len(indices))).mean(axis=0),
            "mean_delta_noise_l2": np.linalg.norm(delta_se[..., indices], axis=-1).mean(axis=0),
        }
    return result


def save_result(directory, name, result):
    """JSON stores scalars/metadata; numeric arrays go to a pickle-free NPZ.

    JSON array entries point to NPZ keys. Undefined scalar statistics become null;
    array NaNs (e.g. undefined Pearson correlations) remain explicit in the NPZ.
    """
    arrays = {}

    def pack(value, key):
        if isinstance(value, dict):
            return {k: pack(v, f"{key}.{k}" if key else k) for k, v in value.items()}
        if hasattr(value, "detach"):
            value = as_numpy(value)
        if isinstance(value, (tuple, list)):
            if value and all(hasattr(v, "shape") for v in value):
                value = np.stack([as_numpy(v) for v in value])
            else:
                return [pack(v, f"{key}.{i}") for i, v in enumerate(value)]
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return pack(value.item(), key)
            if value.dtype.hasobject:
                raise ValueError(f"Object arrays are not supported in validation output: {key}")
            arrays[key] = value
            return {"npz_key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    metadata = pack(result, "")
    directory = Path(directory)
    np.savez_compressed(directory / f"{name}.npz", **arrays)
    with (directory / f"{name}.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)


def report_result(test, result, names, feature_names, blocks):
    """Console summaries; the complete numeric results are saved separately."""
    if test == "consistency":
        print("Mean resimulation diff:", as_numpy(result["mean_diff"]).item())
        print("Median resimulation diff:", as_numpy(result["median_diff"]).item())
    elif test == "width":
        print("Width = sample standard deviation. Physical noise width is std(10**log10_noise).")
        for prefix, labels in (("", names), ("physical_", result["physical_parameter_names"])):
            print("Coordinates:", "checkpoint" if not prefix else "physical")
            print("parameter mean_width median_width std_width mean/prior_span median/prior_span std/prior_span")
            for j, name in enumerate(labels):
                keys = ("mean_width_per_param", "median_width_per_param", "std_width_per_param",
                        "mean_width_normalized_per_param", "median_width_normalized_per_param",
                        "std_width_normalized_per_param")
                print(name, *(f"{result[prefix + k][j]:.6g}" for k in keys))
    elif test == "recovery":
        print("Checkpoint coordinates; log10_weight_noise stays log10. Errors use posterior means.")
        print("training_row parameter true mean median abs_error abs_error/prior_span")
        for i, row in enumerate(result["observation_indices"]):
            for j, name in enumerate(names):
                keys = ("theta_true", "theta_means", "theta_medians", "abs_errors_per_param",
                        "normalized_abs_errors_per_param")
                print(row, name, *(f"{result[k][i, j]:.6g}" for k in keys))
        print("parameter MAE normalized_MAE RMSE normalized_RMSE Pearson_r")
        for j, name in enumerate(names):
            keys = ("mean_abs_error_per_param", "mean_normalized_abs_error_per_param",
                    "rmse_per_param", "normalized_rmse_per_param", "pearson_r_per_param")
            print(name, *(f"{result[k][j]:.6g}" for k in keys))
        print("Pearson r is undefined (NaN) for fewer than 3 rows or constant true/estimated columns.")
    elif test == "coverage":
        print("Marginal equal-tailed intervals; in-sample training-pair coverage, not independent SBC.")
        print("A small sample cannot establish calibration.")
        print("parameter nominal empirical n_observations")
        for k, level in enumerate(result["levels"]):
            for j, name in enumerate(names):
                print(name, f"{level:.2f}", f"{result['coverage_per_param'][k, j]:.6g}", result["n_observations"])
    elif test == "predictive":
        print("Baseline:", result["baseline_method"])
        print("Distances are L2 in saved normalized feature coordinates; no pass/fail threshold.")
        print("Posterior distances:", result["posterior_summary"])
        print("Random baseline distances:", result["baseline_summary"])
        print("Pooled median posterior / baseline:", result["median_distance_ratio"])
        print("training_row posterior_mean posterior_median baseline_mean baseline_median median_ratio")
        for i, row in enumerate(result["observation_indices"]):
            keys = ("per_observation_mean", "per_observation_median", "baseline_per_observation_mean",
                    "baseline_per_observation_median", "per_observation_median_ratio")
            print(row, *(f"{result[k][i]:.6g}" for k in keys))
        for block, values in result["blocks"].items():
            print("[BLOCK PREDICTIVE]", block, "n_features", values["n_features"],
                  "posterior", values["posterior_summary"], "baseline", values["baseline_summary"],
                  "median_ratio", values["median_distance_ratio"])
    elif test == "sensitivity":
        print("Forward one-parameter finite perturbations in checkpoint coordinates.")
        print("Feature derivative = normalized feature change / fraction of prior range changed.")
        print("Stochastic repeats use unchanged simulator RNG; noise SE describes repeat variability.")
        if result["repeats"] < 2:
            print("Noise SE is unavailable (NaN) with fewer than two repeats.")
        print("This is forward sensitivity, not proof of unique posterior information or identifiability.")
        top = result["top_features"]
        for j, name in enumerate(names):
            print("[SENSITIVITY PARAMETER]", name)
            for f in np.argsort(-result["mean_abs_derivative"][j])[:top]:
                block = next(b for b, ix in blocks.items() if f in ix)
                print("[FEATURE]", f, feature_names[f], "block", block,
                      "mean_abs_derivative", result["mean_abs_derivative"][j, f],
                      "mean_delta_noise_SE", result["mean_delta_standard_error"][j, f])
            for block, values in result["blocks"].items():
                print("[BLOCK SENSITIVITY]", block, "n_features", values["n_features"],
                      "mean_L2", values["mean_l2"][j], "median_L2", values["median_l2"][j],
                      "mean_RMS", values["mean_rms"][j], "mean_delta_noise_L2", values["mean_delta_noise_l2"][j])
