"""Shared checkpoint/posterior loading, without MPI or simulator imports."""
import numpy as np
import torch
from tvbgpu.analysis import sbi_features as features
from tvbgpu.analysis import posterior_validation as validation

MCMC_PARAMETERS = dict(num_chains=2, warmup_steps=20, thin=1, init_strategy="resample")


def load_checkpoint(path):
    # Keep saved training arrays on CPU; only the estimator is moved for inference.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a trainer checkpoint dictionary")
    version = checkpoint.get("feature_pipeline_version")
    if version != features.FEATURE_PIPELINE_VERSION:
        raise ValueError(f"Checkpoint feature pipeline {version!r} is incompatible with "
                         f"{features.FEATURE_PIPELINE_VERSION!r}; use a compatible checkpoint. "
                         "Missing provenance cannot be inferred or repaired during inference.")
    features.validate_checkpoint_features(checkpoint)
    validation.parameter_metadata(checkpoint)
    validation.feature_block_indices(checkpoint)
    if checkpoint["theta_dim"] != len(validation.PARAMETER_ORDER):
        raise ValueError("Checkpoint theta_dim disagrees with parameter order")
    keep = validation.as_numpy(checkpoint["feature_keep"])
    if keep.dtype != np.bool_ or keep.ndim != 1:
        raise ValueError("feature_keep must be a one-dimensional boolean mask")
    if tuple(checkpoint["signal_block_order"]) != features.SIGNAL_BLOCK_ORDER:
        raise ValueError("Signal block order mismatch")
    if tuple(checkpoint["alpha_feature_names"]) != features.ALPHA_FEATURE_KEYS:
        raise ValueError("Alpha feature order mismatch")
    return checkpoint


def build_posterior(checkpoint, device):
    """Retain the validator's MCMC construction and settings exactly."""
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    prior = BoxUniform(
        low=torch.as_tensor(checkpoint["prior_low"], dtype=torch.float32, device=device),
        high=torch.as_tensor(checkpoint["prior_high"], dtype=torch.float32, device=device))
    estimator = checkpoint["density_estimator"].to(device)
    inference = SNPE(prior=prior, device=device)
    posterior = inference.build_posterior(estimator, sample_with="mcmc",
                                         mcmc_parameters=dict(MCMC_PARAMETERS))
    return posterior, estimator, prior


@torch.no_grad()
def draw_posteriors(posterior, xs, n_samples):
    """Draw independently for each observation; preserve (observation, draw, parameter)."""
    draws = []
    for i, x in enumerate(xs):
        samples = posterior.sample((n_samples,), x=x, show_progress_bars=False)
        validation.require_finite(f"posterior samples[{i}]", samples)
        draws.append(samples.detach().cpu())
    return torch.stack(draws)
