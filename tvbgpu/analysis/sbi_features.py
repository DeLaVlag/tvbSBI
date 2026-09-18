"""Canonical projected/empirical EEG features for the seven-parameter SBI pipeline.

Signal input and output convention: (batch, channels, time). PCA models, feature
selection and final SBI statistics are fitted on training simulations elsewhere;
this module's signal extraction never fits them.
"""
from dataclasses import asdict, dataclass
import logging
import numpy as np
import torch

FEATURE_PIPELINE_VERSION = 'projected_eeg_time_zscore_logF_v2'
SIGNAL_BLOCK_ORDER = ('dfa', 'lya', 'fc_flat', 'dfa_curve_flat',
                      'lya_curve_flat', 'pli_flat', 'aecc_flat', 'alpha')
POST_PCA_BLOCK_ORDER = ('dfa', 'lya', 'fc_pca', 'dfa_curve_pca',
                        'lya_curve_pca', 'pli_pca', 'aecc_pca', 'alpha')
ALPHA_FEATURE_KEYS = (
    'alpha_hypersync_precuneus_ACC', 'alpha_power_global_mean',
    'alpha_power_precuneus_mean', 'alpha_power_ACC_mean', 'alpha_fc_DMN_mean',
)

@dataclass(frozen=True)
class FeatureConfig:
    fs: float = 500.0
    n_channels: int = 61
    standardization_eps: float = 1e-8
    alpha_band: tuple = (8.0, 12.0)
    broadband: tuple = (1.0, 45.0)
    alpha_fc_method: str = 'corr'
    precuneus_idx: tuple = (10, 11)
    acc_idx: tuple = (5, 6)
    dmn_idx: tuple = (5, 6, 10, 11, 20, 21)

    def metadata(self):
        return asdict(self)


def validate_eeg(eeg, config=FeatureConfig()):
    """Reject non-(batch, channels, time) input and nonfinite signals."""
    shape = tuple(eeg.shape)
    if len(shape) != 3 or shape[0] < 1 or shape[1] != config.n_channels or shape[2] <= int(dfa_window_sizes().max()):
        raise ValueError(f'Expected EEG (batch, {config.n_channels} channels, time>500); got {shape}. '
                         'A (batch, time, channels) array must be transposed explicitly.')
    if not np.isfinite(np.asarray(eeg)).all():
        raise ValueError('EEG contains NaN or Inf before feature extraction')
    return shape


def preprocess_eeg(eeg, config=FeatureConfig()):
    """Temporal z-score independently for every (batch, channel).

    Uses NumPy population std over axis=-1 and denominator std + 1e-8, matching
    the previous re-simulation safeguard. Near-constant channels approach zero;
    no row is standardized against other simulations.
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    validate_eeg(eeg, config)
    mean = eeg.mean(axis=-1, keepdims=True)
    std = eeg.std(axis=-1, keepdims=True)
    result = (eeg - mean) / (std + config.standardization_eps)
    if not np.isfinite(result).all():
        raise ValueError('EEG temporal standardization produced NaN or Inf')
    return np.ascontiguousarray(result, dtype=np.float32)


def dfa_window_sizes():
    """Preserve the active SBI DFA window selection (based on N=2000)."""
    n = 2000
    floats = np.logspace(np.log10(10), np.log10(n // 4), 20)
    sizes = np.unique(np.round(floats).astype(np.int32))
    sizes = sizes[(sizes >= 10) & (sizes <= n // 4)]
    sizes = sizes[np.array([n // w >= 4 for w in sizes])]
    return np.ascontiguousarray(np.sort(sizes), dtype=np.int32)


def _compute_dfa(eeg, logger, rank):
    from tvbgpu.analysis.gpu_dfa2 import computeDFA_gpu
    alphas, log_s, log_F = computeDFA_gpu(eeg, logger, rank, dfa_window_sizes())
    return alphas, log_s, log_F


def _compute_lya(eeg, logger, rank):
    from tvbgpu.analysis.gpu_lya_3 import computeLYA_gpu
    return computeLYA_gpu(eeg, logger, rank, 8, 8, 16, 80, .001)


def _interpolate_curve(curve):
    """Preserve the existing finite-window interpolation/fallback policy."""
    curve = np.asarray(curve, dtype=float)
    out = np.empty_like(curve)
    t = np.arange(curve.shape[-1])
    for i in range(curve.shape[0]):
        for r in range(curve.shape[1]):
            row = curve[i, r]
            valid = np.isfinite(row)
            if valid.sum() == 0:
                out[i, r] = 0.0
            elif valid.sum() == 1:
                out[i, r] = row[valid][0]
            else:
                out[i, r] = np.interp(t, t[valid], row[valid])
    return out


def _to_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _zscore_fc(fc, eps=1e-8):
    """Existing off-diagonal FC z-score, retained before FC PCA."""
    n = fc.shape[-1]
    fc = fc.clone()
    diagonal = torch.eye(n, device=fc.device, dtype=torch.bool).unsqueeze(0)
    masked = fc.masked_fill(diagonal, float('nan'))
    mean = torch.nanmean(masked, dim=(1, 2), keepdim=True)
    diff = masked - mean
    variance = torch.nanmean(diff * diff, dim=(1, 2), keepdim=True)
    return (diff / torch.sqrt(variance + eps)).masked_fill(diagonal, 0.0)


def extract_signal_features(eeg, config=FeatureConfig(), logger=None, rank=0):
    """Preprocess EEG once, then extract the deterministic pre-PCA SBI blocks.

    Returns arrays aligned on the batch axis. `alpha_valid` records validity
    before the alpha implementation replaces invalid values with zero.
    """
    logger = logger or logging.getLogger(__name__)
    x = preprocess_eeg(eeg, config)
    batch, channels, _ = x.shape
    alphas, log_s, log_F = _compute_dfa(x, logger, rank)
    lya, divergence = _compute_lya(x, logger, rank)
    # Explicit contract: the PCA curve is log_F, never log_s.
    log_F = _interpolate_curve(log_F)
    log_s = _interpolate_curve(log_s)

    from tvbgpu.analysis.fc_analysis import compute_fc
    from tvbgpu.analysis.alpha_feat_analysis import compute_alpha_features
    fc = compute_fc(x, method='corr')
    pli = compute_fc(x, method='pli', fs=config.fs, band=config.alpha_band)
    aecc = compute_fc(x, method='aecc', fs=config.fs, band=config.alpha_band)
    fc = _to_numpy(_zscore_fc(fc))
    pli, aecc = _to_numpy(pli), _to_numpy(aecc)
    upper = np.triu_indices(channels, k=1)
    alpha_features, alpha_valid = compute_alpha_features(
        ts=np.transpose(x, (0, 2, 1)), fs=config.fs,
        precuneus_idx=config.precuneus_idx, acc_idx=config.acc_idx,
        dmn_idx=config.dmn_idx, ALPHA_FEATURE_KEYS=ALPHA_FEATURE_KEYS,
        fc_method=config.alpha_fc_method, alpha_band=config.alpha_band,
        broadband=config.broadband,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    alpha = _to_numpy(torch.stack([alpha_features[key].reshape(-1)
                                   for key in ALPHA_FEATURE_KEYS], dim=1))
    blocks = {
        'dfa': np.nan_to_num(_to_numpy(alphas), nan=0., posinf=1e6, neginf=-1e6),
        'lya': np.nan_to_num(_to_numpy(lya), nan=0., posinf=1e6, neginf=-1e6),
        'fc_flat': fc[:, upper[0], upper[1]],
        'dfa_curve_flat': np.nan_to_num(log_F, nan=0., posinf=1e6, neginf=-1e6).reshape(batch, -1),
        'lya_curve_flat': np.nan_to_num(_to_numpy(divergence), nan=0., posinf=1e6, neginf=-1e6).reshape(batch, -1),
        'pli_flat': pli[:, upper[0], upper[1]],
        'aecc_flat': aecc[:, upper[0], upper[1]],
        'alpha': alpha,
    }
    for name in SIGNAL_BLOCK_ORDER:
        block = np.asarray(blocks[name])
        if block.ndim != 2 or block.shape[0] != batch:
            raise ValueError(f'{name} must have shape (batch, features), got {block.shape}')
        blocks[name] = np.ascontiguousarray(block, dtype=np.float32)
    blocks['alpha_valid'] = _to_numpy(alpha_valid).astype(bool)
    return blocks


def post_pca_blocks(signal_blocks, pca_models):
    """Apply already-fitted PCA objects; never call fit or fit_transform."""
    mapping = {'fc_pca': 'fc_flat', 'dfa_curve_pca': 'dfa_curve_flat',
               'lya_curve_pca': 'lya_curve_flat', 'pli_pca': 'pli_flat',
               'aecc_pca': 'aecc_flat'}
    result = {'dfa': signal_blocks['dfa'], 'lya': signal_blocks['lya'],
              'alpha': signal_blocks['alpha']}
    for output, source in mapping.items():
        model = pca_models[output]
        if signal_blocks[source].shape[1] != model.n_features_in_:
            raise ValueError(f'{source} width {signal_blocks[source].shape[1]} '
                             f'does not match saved {output} PCA width {model.n_features_in_}')
        result[output] = model.transform(signal_blocks[source])
    return result


def assemble_post_pca(blocks):
    """Return full post-PCA vector, names, and block slices in canonical order."""
    names, boundaries, arrays = [], {}, []
    offset = 0
    for block_name in POST_PCA_BLOCK_ORDER:
        values = np.asarray(blocks[block_name], dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f'{block_name} must be two-dimensional, got {values.shape}')
        count = values.shape[1]
        boundaries[block_name] = slice(offset, offset + count)
        names.extend(ALPHA_FEATURE_KEYS if block_name == 'alpha'
                     else [f'{block_name}_{i + 1}' for i in range(count)])
        arrays.append(values)
        offset += count
    if len({array.shape[0] for array in arrays}) != 1:
        raise ValueError('Post-PCA blocks have different batch sizes')
    return np.concatenate(arrays, axis=1), names, boundaries


def apply_saved_transform(full, checkpoint):
    """Apply saved feature_keep and SBI normalization to a full post-PCA vector."""
    keep = np.asarray(checkpoint['feature_keep'], dtype=bool)
    mean = _to_numpy(checkpoint['x_mean']).reshape(-1)
    std = _to_numpy(checkpoint['x_std']).reshape(-1)
    if full.shape[1] != keep.size or int(keep.sum()) != mean.size or mean.size != std.size:
        raise ValueError('Saved feature_keep/x_mean/x_std do not match full vector')
    return (np.asarray(full)[:, keep] - mean) / std


def validate_checkpoint_features(checkpoint):
    """Reject inconsistent saved feature ordering, dimensions, mask, and statistics."""
    full_names = list(checkpoint['feature_names_full'])
    names = list(checkpoint['feature_names'])
    keep = np.asarray(checkpoint['feature_keep'], dtype=bool)
    mean = _to_numpy(checkpoint['x_mean']).reshape(-1)
    std = _to_numpy(checkpoint['x_std']).reshape(-1)
    if (len(full_names) != checkpoint['x_full_dim'] or keep.size != len(full_names)
            or len(names) != checkpoint['x_dim'] or int(keep.sum()) != len(names)
            or mean.size != len(names) or std.size != len(names)):
        raise ValueError('Checkpoint feature dimensions, mask, names, or normalization statistics disagree')
    if [name for name, retained in zip(full_names, keep) if retained] != names:
        raise ValueError('Checkpoint feature names disagree with feature_keep ordering')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Checkpoint x_mean/x_std contain invalid values')
    if tuple(checkpoint['feature_block_order']) != POST_PCA_BLOCK_ORDER:
        raise ValueError('Checkpoint feature block order disagrees with current pipeline')
    config = FeatureConfig(**checkpoint['feature_config'])
    preprocessing = checkpoint['preprocessing_config']
    if (preprocessing['input_axes'] != 'batch_channels_time'
            or not preprocessing['projection_before_standardization']
            or preprocessing['standardization_axis'] != 'time'
            or preprocessing['standardization_eps'] != config.standardization_eps):
        raise ValueError('Checkpoint EEG preprocessing disagrees with current pipeline')
    return config
