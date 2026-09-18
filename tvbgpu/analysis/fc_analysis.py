import numpy as np
import torch


def compute_fc(
        data: torch.Tensor,
        method: str = "corr",
        fs: float = None,
        band: tuple = None,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute batched functional connectivity.

    Args:
        data: shape (N, R, T)
            N = simulations / epochs
            R = regions / channels
            T = time points
        method: 'corr', 'pli', or 'aecc'
        fs: sampling frequency, required if band is used
        band: optional frequency band, e.g. (8, 12)
        eps: numerical stability

    Returns:
        FC: shape (N, R, R)
    """
    assert method in ("corr", "pli", "aecc")

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = data.float().to(device)

    if band is not None:
        assert fs is not None, "fs is required when band is used"
        x = _bandpass_fft(x, fs, band[0], band[1])

    if method == "corr":
        fc = _corr(x, eps=eps)
        fc.diagonal(dim1=-2, dim2=-1).fill_(1.0)

    elif method == "pli":
        z = _analytic_signal(x)
        fc = _pli_from_analytic(z, eps=eps)
        fc.diagonal(dim1=-2, dim2=-1).fill_(0.0)

    elif method == "aecc":
        z = _analytic_signal(x)
        fc = _aecc_from_analytic(z, eps=eps)
        fc.diagonal(dim1=-2, dim2=-1).fill_(0.0)

    return fc


def _bandpass_fft(
        x: torch.Tensor,
        fs: float,
        low: float,
        high: float,
) -> torch.Tensor:
    """
    FFT bandpass for data of shape (N, R, T).
    """
    T = x.shape[-1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs).to(x.device)

    Xf = torch.fft.rfft(x, dim=-1)
    mask = (freqs >= low) & (freqs <= high)

    Xf = Xf * mask[None, None, :]
    y = torch.fft.irfft(Xf, n=T, dim=-1)

    return y


def _analytic_signal(x: torch.Tensor) -> torch.Tensor:
    """
    Torch Hilbert transform via FFT.

    Args:
        x: real tensor, shape (N, R, T)

    Returns:
        analytic signal z: complex tensor, shape (N, R, T)
    """
    T = x.shape[-1]
    Xf = torch.fft.fft(x, dim=-1)

    h = torch.zeros(T, device=x.device, dtype=x.dtype)

    if T % 2 == 0:
        h[0] = 1.0
        h[T // 2] = 1.0
        h[1:T // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(T + 1) // 2] = 2.0

    z = torch.fft.ifft(Xf * h[None, None, :], dim=-1)
    return z


def _corr(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Batched Pearson correlation.

    Args:
        x: shape (N, R, T)

    Returns:
        corr: shape (N, R, R)
    """
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / (x.std(dim=-1, keepdim=True, unbiased=False) + eps)

    T = x.shape[-1]
    fc = torch.matmul(x, x.transpose(-1, -2)) / T

    return fc


@torch.no_grad()
def _pli_from_analytic(
    z: torch.Tensor,
    eps: float = 1e-8,
    simulation_chunk: int = 16,
    pair_chunk: int = 128,
) -> torch.Tensor:
    """
    Memory-efficient batched Phase Lag Index.

    Args:
        z:
            Analytic signal, complex tensor with shape (N, R, T).
        eps:
            Retained for API compatibility. Currently unused.
        simulation_chunk:
            Number of simulations processed simultaneously.
        pair_chunk:
            Number of unique region pairs processed simultaneously.

    Returns:
        PLI tensor with shape (N, R, R).
    """
    if z.ndim != 3:
        raise ValueError(
            f"Expected z with shape (N, R, T), got {tuple(z.shape)}"
        )

    if not torch.is_complex(z):
        raise TypeError(
            f"Expected complex analytic signal, got dtype={z.dtype}"
        )

    N, R, T = z.shape
    device = z.device

    # This tensor is only (N, R, T), rather than (N, R, R, T).
    phase = torch.angle(z)

    pli = torch.zeros(
        (N, R, R),
        dtype=phase.dtype,
        device=device,
    )

    # Unique pairs above the diagonal.
    pair_indices = torch.triu_indices(
        R,
        R,
        offset=1,
        device=device,
    )
    region_i = pair_indices[0]
    region_j = pair_indices[1]

    for sim_start in range(0, N, simulation_chunk):
        sim_end = min(sim_start + simulation_chunk, N)
        phase_sim = phase[sim_start:sim_end]

        for pair_start in range(0, region_i.numel(), pair_chunk):
            pair_end = min(
                pair_start + pair_chunk,
                region_i.numel(),
            )

            ii = region_i[pair_start:pair_end]
            jj = region_j[pair_start:pair_end]

            # Shape:
            # (simulation_chunk, pair_chunk, T)
            phase_diff = (
                phase_sim[:, ii, :]
                - phase_sim[:, jj, :]
            )

            pair_pli = (
                torch.sin(phase_diff)
                .sign()
                .mean(dim=-1)
                .abs()
            )

            # PLI is symmetric.
            pli[sim_start:sim_end, ii, jj] = pair_pli
            pli[sim_start:sim_end, jj, ii] = pair_pli

            del phase_diff, pair_pli

    return pli


@torch.no_grad()
def _aecc_from_analytic(
    z: torch.Tensor,
    eps: float = 1e-8,
    simulation_chunk: int = 16,
    pair_chunk: int = 128,
) -> torch.Tensor:
    """
    Memory-efficient corrected Amplitude Envelope Correlation
    using pairwise orthogonalization.

    Args:
        z:
            Analytic signal, complex tensor with shape (N, R, T).
        eps:
            Numerical stability constant.
        simulation_chunk:
            Number of simulations processed simultaneously.
        pair_chunk:
            Number of unique region pairs processed simultaneously.

    Returns:
        AECc tensor with shape (N, R, R).

    Notes:
        For every region pair (i, j), both directional correlations
        are calculated:

            corr(|z_i|, |z_j orthogonalized to z_i|)
            corr(|z_j|, |z_i orthogonalized to z_j|)

        The two values are averaged to obtain the symmetric AECc.
    """
    if z.ndim != 3:
        raise ValueError(
            f"Expected z with shape (N, R, T), got {tuple(z.shape)}"
        )

    if not torch.is_complex(z):
        raise TypeError(
            f"Expected complex analytic signal, got dtype={z.dtype}"
        )

    N, R, T = z.shape
    device = z.device

    # Shape (N, R, T), which is manageable.
    amp = torch.abs(z)

    aecc = torch.zeros(
        (N, R, R),
        dtype=amp.dtype,
        device=device,
    )

    # Only calculate unique off-diagonal region pairs.
    pair_indices = torch.triu_indices(
        R,
        R,
        offset=1,
        device=device,
    )
    region_i = pair_indices[0]
    region_j = pair_indices[1]

    for sim_start in range(0, N, simulation_chunk):
        sim_end = min(sim_start + simulation_chunk, N)

        z_sim = z[sim_start:sim_end]
        amp_sim = amp[sim_start:sim_end]

        for pair_start in range(0, region_i.numel(), pair_chunk):
            pair_end = min(
                pair_start + pair_chunk,
                region_i.numel(),
            )

            ii = region_i[pair_start:pair_end]
            jj = region_j[pair_start:pair_end]

            # Each has shape:
            # (simulation_chunk, pair_chunk, T)
            zi = z_sim[:, ii, :]
            zj = z_sim[:, jj, :]

            amp_i = amp_sim[:, ii, :]
            amp_j = amp_sim[:, jj, :]

            # -----------------------------------------------------
            # Direction 1: z_j orthogonalized with respect to z_i
            # -----------------------------------------------------
            cross_ji = zj * torch.conj(zi)
            cross_ji.div_(amp_i.clamp_min(eps))

            amp_j_orth = torch.abs(torch.imag(cross_ji))

            # Add a singleton matrix dimension so the existing
            # _pairwise_corr_4d function can be reused.
            aecc_ij = _pairwise_corr_4d(
                amp_i.unsqueeze(2),
                amp_j_orth.unsqueeze(2),
                eps=eps,
            ).squeeze(2)

            del cross_ji, amp_j_orth

            # -----------------------------------------------------
            # Direction 2: z_i orthogonalized with respect to z_j
            # -----------------------------------------------------
            cross_ij = zi * torch.conj(zj)
            cross_ij.div_(amp_j.clamp_min(eps))

            amp_i_orth = torch.abs(torch.imag(cross_ij))

            aecc_ji = _pairwise_corr_4d(
                amp_j.unsqueeze(2),
                amp_i_orth.unsqueeze(2),
                eps=eps,
            ).squeeze(2)

            del cross_ij, amp_i_orth

            # Directional correction.
            pair_aecc = 0.5 * (aecc_ij + aecc_ji)

            aecc[sim_start:sim_end, ii, jj] = pair_aecc
            aecc[sim_start:sim_end, jj, ii] = pair_aecc

            del (
                zi,
                zj,
                amp_i,
                amp_j,
                aecc_ij,
                aecc_ji,
                pair_aecc,
            )

    return aecc


def _pairwise_corr_4d(
        a: torch.Tensor,
        b: torch.Tensor,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Correlation over last dimension.

    Args:
        a: shape broadcastable to (N, R, R, T)
        b: shape broadcastable to (N, R, R, T)

    Returns:
        corr: shape (N, R, R)
    """
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)

    numerator = torch.mean(a * b, dim=-1)
    denominator = (
            torch.sqrt(torch.mean(a * a, dim=-1)) *
            torch.sqrt(torch.mean(b * b, dim=-1)) +
            eps
    )

    return numerator / denominator