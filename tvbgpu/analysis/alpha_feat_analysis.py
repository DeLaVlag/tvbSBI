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


import torch
import torch.fft
import numpy as np
import mne
from tvbgpu.analysis.fc_analysis import _analytic_signal
from tvbgpu.analysis.fc_analysis import _pli_from_analytic

# ---------- Helpers ----------
def _bandpass_fft(x, fs, low, high):
    """Bandpass filter with FFT masking. Input: (E, T, C)."""
    E, T, C = x.shape
    freqs = torch.fft.fftfreq(T, d=1/fs).to(x.device)
    Xf = torch.fft.fft(x, dim=1)
    mask = (freqs >= low) & (freqs <= high)
    mask |= (freqs <= -low) & (freqs >= -high)
    Xf = Xf * mask[:, None]
    return torch.fft.ifft(Xf, dim=1).real

def _pli_old(fr_band):
    """PLI connectivity, input (E, T, C) → output (E, C, C)."""
    E, T, C = fr_band.shape
    F = torch.fft.fft(fr_band, dim=1)
    H = torch.zeros_like(F)
    H[:, 0] = 1
    if T % 2 == 0:
        H[:, 1:T//2] = 2
        H[:, T//2] = 1
    else:
        H[:, 1:(T+1)//2] = 2
    analytic = torch.fft.ifft(F * H, dim=1)
    phases = torch.angle(analytic)                           # (E, T, C)
    dphi = phases.unsqueeze(2) - phases.unsqueeze(3)         # (E, T, C, C)
    pli = torch.abs(torch.mean(torch.sign(torch.sin(dphi)), dim=1))
    return pli

def _corr_pld(fr_band, eps=1e-8):
    """Correlation connectivity, input (E, T, C) → output (E, C, C)."""
    E, T, C = fr_band.shape
    x = (fr_band - fr_band.mean(dim=1, keepdim=True)) / (fr_band.std(dim=1, keepdim=True) + eps)
    FC = torch.matmul(x.transpose(1, 2), x) / (T - 1)        # (E, C, C)
    d = torch.diagonal(FC, dim1=1, dim2=2).unsqueeze(-1)
    FC = FC / torch.sqrt(d @ d.transpose(-2, -1) + eps)
    return FC

def _pli(fr_band: torch.Tensor, batch_size=10) -> torch.Tensor:
    """PLI connectivity, input (E, T, C) → output (E, C, C)."""
    device = fr_band.device
    dtype = fr_band.dtype

    # Convert (N, R, T) → (N, T, R)
    fr_band = fr_band.permute(0, 2, 1).contiguous()

    E, T, C = fr_band.shape

    # Process in batches to avoid OOM
    all_pli = []

    for start in range(0, E, batch_size):
        end = min(start + batch_size, E)
        batch = fr_band[start:end]
        batch_size_actual = batch.shape[0]

        # Hilbert transform via FFT
        F = torch.fft.fft(batch, dim=1)
        H = torch.zeros_like(F)
        H[:, 0] = 1
        if T % 2 == 0:
            H[:, 1:T//2] = 2
            H[:, T//2] = 1
        else:
            H[:, 1:(T+1)//2] = 2
        analytic = torch.fft.ifft(F * H, dim=1)
        phases = torch.angle(analytic)   # (batch, T, C)

        # Compute PLI incrementally to avoid large tensors
        pli_batch = torch.zeros((batch_size_actual, C, C), device=device, dtype=dtype)
        for i in range(C):
            for j in range(i+1, C):
                dphi = phases[:, :, i] - phases[:, :, j]   # (batch, T)
                val = torch.abs(torch.mean(torch.sign(torch.sin(dphi)), dim=1))  # (batch,)
                pli_batch[:, i, j] = pli_batch[:, j, i] = val

        # Move to CPU and clean up GPU memory
        all_pli.append(pli_batch.cpu())
        del batch, F, H, analytic, phases, pli_batch
        torch.cuda.empty_cache()

    # Concatenate all batches and move back to original device
    return torch.cat(all_pli, dim=0).to(device)

    # Hilbert transform via FFT
    F = torch.fft.fft(fr_band, dim=1)
    H = torch.zeros_like(F)
    H[:, 0] = 1
    if T % 2 == 0:
        H[:, 1:T//2] = 2
        H[:, T//2] = 1
    else:
        H[:, 1:(T+1)//2] = 2
    analytic = torch.fft.ifft(F * H, dim=1)
    phases = torch.angle(analytic)   # (E, T, C)

    # Instead of building (E, T, C, C), compute incrementally
    pli = torch.zeros((E, C, C), device=fr_band.device, dtype=fr_band.dtype)
    for i in range(C):
        for j in range(i+1, C):
            dphi = phases[:, :, i] - phases[:, :, j]   # (E, T)
            val = torch.abs(torch.mean(torch.sign(torch.sin(dphi)), dim=1))  # (E,)
            pli[:, i, j] = pli[:, j, i] = val

    return pli


@torch.no_grad()
def _corr(
    x: torch.Tensor,
    eps: float = 1e-8,
    batch_size: int = 16,
) -> torch.Tensor:
    """
    Batched Pearson correlation across time.

    Args:
        x:
            Tensor with shape (N, R, T).
        eps:
            Minimum standard deviation used for numerical stability.
        batch_size:
            Number of simulations processed simultaneously.

    Returns:
        Correlation matrices with shape (N, R, R).
    """
    if x.ndim != 3:
        raise ValueError(
            f"Expected x shape (N, R, T), got {tuple(x.shape)}"
        )

    device = x.device
    N, R, T = x.shape

    if T < 2:
        raise ValueError(
            "At least two time points are required for correlation"
        )

    correlations = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # Shape (batch, R, T).
        batch = x[start:end]

        centered = (
            batch
            - batch.mean(dim=-1, keepdim=True)
        )

        # Population covariance: denominator T.
        covariance = torch.matmul(
            centered,
            centered.transpose(-1, -2),
        ) / T

        # Must use the same population denominator as covariance.
        std = centered.std(
            dim=-1,
            unbiased=False,
        ).clamp_min(eps)

        denominator = (
            std.unsqueeze(2)
            * std.unsqueeze(1)
        )

        corr = covariance / denominator

        # Constant channels produce zero correlation.
        corr = torch.nan_to_num(
            corr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Remove minor floating-point deviations outside [-1, 1].
        corr = corr.clamp(-1.0, 1.0)

        correlations.append(corr.cpu())

        del (
            batch,
            centered,
            covariance,
            std,
            denominator,
            corr,
        )

    return torch.cat(
        correlations,
        dim=0,
    ).to(device)

def _band_power(fr_band):
    """Band power per channel. Input: (E, T, C) → output (E, C)."""
    return (fr_band ** 2).mean(dim=1)

def _get_data(fromfile=True):

    if fromfile:
        # filepointer = ('/tsd/p3139/data/durable/AI-Mind-data-2025-03-eBRAIN-Health-subset/2025-01-31_14-43-20-897314/'
        #                'building/1-379/1-379-1-Z/sensors/1-379-1-Z_1-EO_eeg.fif')

        filepointer = ('/home/michiel/Documents/Repos/tvb_apptainer/tvbgpu/data/alpha_test_eeg.fif')

        data = mne.read_epochs(filepointer).get_data()

    else:

        data = np.random.randn(2, 68, 2000).astype(np.float32)  # example

    return data


# ---------- Main function ----------
def compute_alpha_features(
    ts,
    fs,
    precuneus_idx,
    acc_idx,
    dmn_idx,
    ALPHA_FEATURE_KEYS,
    fc_method="corr",
    alpha_band=(8.0, 12.0),
    broadband=(1.0, 45.0),
    device="cuda",
):
    """
    Parameters
    ----------
    ts:
        Shape (N, T, R).
    """
    ts = torch.as_tensor(
        ts,
        dtype=torch.float32,
        device=device,
    )

    if ts.ndim != 3:
        raise ValueError(
            f"Expected ts shape (N, T, R), got {tuple(ts.shape)}"
        )

    # _bandpass_fft and _corr expect (N, R, T).
    x = ts.permute(0, 2, 1).contiguous()

    N, R, T = x.shape
    eps = 1e-8

    # Remove temporal mean independently for every channel.
    x = x - x.mean(dim=-1, keepdim=True)

    x_alpha = _bandpass_fft(
        x,
        fs,
        alpha_band[0],
        alpha_band[1],
    )

    broadband_high = min(
        broadband[1],
        0.999 * fs / 2.0,
    )

    x_broadband = _bandpass_fft(
        x,
        fs,
        broadband[0],
        broadband_high,
    )

    # Power over the temporal dimension.
    alpha_power = x_alpha.square().mean(dim=-1)
    broadband_power = x_broadband.square().mean(dim=-1)

    alpha_relative = (
        alpha_power
        / broadband_power.clamp_min(eps)
    )

    if fc_method == "pli":
        z_alpha = _analytic_signal(x_alpha)
        fc_alpha = _pli_from_analytic(
            z_alpha,
            # eps=eps,
            simulation_chunk=16,
            pair_chunk=128,
        )
    elif fc_method == "corr":
        fc_alpha = _corr(
            x_alpha,
            eps=eps,
            batch_size=16,
        )
    else:
        raise ValueError(
            "fc_method must be 'pli' or 'corr'"
        )

    p = torch.as_tensor(
        precuneus_idx,
        device=device,
        dtype=torch.long,
    )
    a = torch.as_tensor(
        acc_idx,
        device=device,
        dtype=torch.long,
    )
    dmn = torch.as_tensor(
        dmn_idx,
        device=device,
        dtype=torch.long,
    )

    if p.numel() == 0:
        raise ValueError("precuneus_idx cannot be empty")

    if a.numel() == 0:
        raise ValueError("acc_idx cannot be empty")

    if dmn.numel() < 2:
        raise ValueError(
            "dmn_idx must contain at least two channels "
            "because the checkpoint expects five alpha features"
        )

    for name, indices in {
        "precuneus_idx": p,
        "acc_idx": a,
        "dmn_idx": dmn,
    }.items():
        if indices.min() < 0 or indices.max() >= R:
            raise IndexError(
                f"{name} contains an index outside the "
                f"available range 0–{R - 1}"
            )

    fc_pa = (
        fc_alpha
        .index_select(1, p)
        .index_select(2, a)
    )

    fc_dmn = (
        fc_alpha
        .index_select(1, dmn)
        .index_select(2, dmn)
    )

    upper_dmn = torch.triu_indices(
        dmn.numel(),
        dmn.numel(),
        offset=1,
        device=device,
    )

    features = {
        "alpha_hypersync_precuneus_ACC":
            fc_pa.mean(dim=(1, 2)),

        "alpha_power_global_mean":
            alpha_relative.mean(dim=1),

        "alpha_power_precuneus_mean":
            alpha_relative.index_select(1, p).mean(dim=1),

        "alpha_power_ACC_mean":
            alpha_relative.index_select(1, a).mean(dim=1),

        "alpha_fc_DMN_mean":
            fc_dmn[
                :,
                upper_dmn[0],
                upper_dmn[1],
            ].mean(dim=1),
    }

    feature_matrix = torch.stack(
        [features[key] for key in ALPHA_FEATURE_KEYS],
        dim=1,
    )

    valid_mask = torch.isfinite(feature_matrix).all(dim=1)

    features = {
        name: torch.nan_to_num(
            value,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for name, value in features.items()
    }

    return features, valid_mask


def compute_fc_2(data: torch.Tensor, method: str = "corr", fs: float = None, band: tuple = None) -> torch.Tensor:
    """
    Compute functional connectivity for each node pair.

    Args:
        data: torch.Tensor or numpy array of shape (N, R, T)
            N = number of epochs / simulations
            R = regions / channels
            T = time points
        method: 'corr' or 'pli'
            Functional connectivity metric.
        fs: float, optional
            Sampling frequency (Hz), required if using bandpass.
        band: (low, high), optional
            Frequency band for filtering before PLI (e.g. (8,12)).

    Returns:
        FC: torch.Tensor of shape (N, R, R)
    """
    assert method in ("corr", "pli"), "method must be 'corr' or 'pli'"

    # Convert to torch tensor if numpy array
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # Optional bandpass
    if band is not None and fs is not None:
        data = _bandpass_fft(data, fs, band[0], band[1])

    # Compute FC
    if method == "corr":
        FC = _corr(data)   # expects (N, R, T) → (N, R, R)
    else:
        FC = _pli(data)    # expects (N, R, T) → (N, R, R)

    # Enforce diagonal convention
    if method == "corr":
        FC.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    else:
        FC.diagonal(dim1=-2, dim2=-1).fill_(0.0)

    return FC



if __name__ == '__main__':

    fs = 500  # Hz

    eeg = _get_data(fromfile=True)
    # eeg = eeg[:60, :60, :2000]
    print(eeg.shape)

    FC_corr = compute_fc_2(eeg, method="pli")
    print('FC_corr.shape', FC_corr.shape)
    print('FC_corr.shape', FC_corr)

    compute_af = 0
    if compute_af:
        precuneus_idx = [10, 11]  # example channels
        acc_idx = [5, 6]
        dmn_idx = [5, 6, 10, 11, 20, 21]

        features = compute_alpha_features(
            eeg, fs,
            precuneus_idx=precuneus_idx,
            acc_idx=acc_idx,
            dmn_idx=dmn_idx,
            fc_method="corr"
        )
        # features["alpha_hypersync_precuneus_ACC"] -> (B,) mean PLI between precuneus & ACC
        # features["alpha_power_global_mean"]       -> (B,) global alpha power
        # features["alpha_fc_DMN_mean"]             -> (B,) mean alpha-band FC within DMN (if dmn_idx provided)
        # fc_alpha                                  -> (B, M, M) full alpha-band FC matrix (PLI or corr)
        # power_alpha                                -> (B, M) per-region alpha power (un-normalized)

        print('alpha_hypersync_precuneus_ACC', features["alpha_hypersync_precuneus_ACC"])
        print("alpha_power_global_mean", features["alpha_power_global_mean"])
        print("alpha_fc_DMN_mean", features["alpha_fc_DMN_mean"])

        hyper_ratio = features["alpha_hypersync_precuneus_ACC"] / features["alpha_fc_DMN_mean"]
        print('hyper_ratio', hyper_ratio)

        # Tells you who talks to whom in alpha band.
        # print('fc_alpha', fc_alpha.shape)
        # # how strong the oscillations are regionally
        # print("power_alpha", power_alpha.shape)
