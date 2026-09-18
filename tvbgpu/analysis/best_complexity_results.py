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
import numpy as np
import heapq

def prepare_features(dfa_data, lya_data):
    # Ensure numpy arrays
    dfa_data = np.array(dfa_data, dtype=np.float32)
    lya_data = np.array(lya_data, dtype=np.float32)

    # Reshape DFA to (B, 10) if it's 1D
    # if dfa_data.ndim == 1:
    #     dfa_data = dfa_data.reshape(-1, 1)
    #
    # # Reshape Lya to (B, 1) if it's 1D
    # if lya_data.ndim == 1:
    #     lya_data = lya_data.reshape(-1, 1)

    print('dfa_data', dfa_data.shape)
    print('lya_data', lya_data.shape)

    # Check number of samples/channels (first dimension) only
    # if dfa_data.shape[0] != lya_data.shape[0]:
    #     raise ValueError(
    #         f"Sample count mismatch: DFA has {dfa_data.shape[0]}, Lya has {lya_data.shape[0]}"
    #     )

    # Convert to torch tensors
    dfa_tensor = torch.from_numpy(dfa_data)
    lya_tensor = torch.from_numpy(lya_data)

    # Concatenate along the feature dimension -> shape (B, 11)
    full_batch = torch.cat([dfa_tensor, lya_tensor], dim=1)

    return full_batch

def find_matches(dfa_eeg, lya_eeg, dfa_tvb, lya_tvb, tvb_params, batch_size=512):
    """
    Matches EEG and TVB features.
    """
    eeg_features = prepare_features(dfa_eeg, lya_eeg)
    tvb_features = prepare_features(dfa_tvb, lya_tvb)  # TVB usually has no epochs

    # Compute distances in batches
    matches = []
    for start in range(0, eeg_features.shape[0], batch_size):
        end = start + batch_size
        batch = eeg_features[start:end]  # (B, n_features_total)
        dist = torch.cdist(batch, tvb_features)  # (B, TVB_count)
        best_dist, best_idx = torch.min(dist, dim=1)
        for b, idx in zip(best_dist, best_idx):
            matches.append((float(b), tvb_params[idx]))

    return matches


def find_matches_bare(eeg_dfa_np, eeg_lya_np, tvb_dfa_np, tvb_lya_np, tvb_params_np, batch_size=512):
    """
    Find top 10 TVB simulations that best match EEG DFA and Lya using GPU and PyTorch.

    Parameters:
    - eeg_dfa_np: shape (n_channels, 10)
    - eeg_lya_np: shape (n_channels,)
    - tvb_dfa_np: shape (n_sim, n_channels, 10)
    - tvb_lya_np: shape (n_sim, n_channels)
    - tvb_params_np: shape (n_sim, n_params)

    Returns:
    - List of top 10 (distance, param_vector, tvb_dfa_sim, tvb_lya_sim)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Flatten EEG vector
    eeg_vector = torch.from_numpy(
        np.concatenate([eeg_dfa_np.flatten(), eeg_lya_np.flatten()])
    ).float().to(device)  # shape (n_channels * 11,)

    # Convert TVB data
    tvb_dfa = torch.from_numpy(tvb_dfa_np).float().to(device)    # (n_sim, n_ch, 10)
    tvb_lya = torch.from_numpy(tvb_lya_np).float().to(device)    # (n_sim, n_ch)
    tvb_params = torch.from_numpy(tvb_params_np)                # keep on CPU

    # isit?
    n_sim = tvb_dfa.shape[0]
    flat_results = []

    # Process in batches
    for start in range(0, n_sim, batch_size):
        end = min(start + batch_size, n_sim)

        # Extract batch
        dfa_batch = tvb_dfa[start:end]                      # (B, n_ch, 10)
        lya_batch = tvb_lya[start:end]                      # (B, n_ch)
        dfa_flat = dfa_batch.view(dfa_batch.size(0), -1)    # (B, n_ch*10)
        lya_flat = lya_batch                                # (B, n_ch)
        full_batch = torch.cat([dfa_flat, lya_flat], dim=1) # (B, n_ch*11)

        # Compute L2 distance to EEG vector
        diff = full_batch - eeg_vector[None, :]
        dist = torch.norm(diff, dim=1)                      # (B,)

        # Store distance and index
        for i in range(dist.shape[0]):
            flat_results.append((dist[i].item(), start + i))

    # Sort and get top 10
    # flat_results.sort(key=lambda x: x[0])
    # top10 = flat_results[:10]
    # top10 = flat_results

    # Package all output
    results = []
    for dist_val, idx in flat_results:
        param_vec = tvb_params[idx].numpy()
        sim_dfa = tvb_dfa_np[idx]
        sim_lya = tvb_lya_np[idx]
        results.append((dist_val, param_vec, sim_dfa, sim_lya))

    return results


def find_matches_with_alpha_haveabetterone(
    eeg_dfa_np, eeg_lya_np,
    eeg_features,              # dict with EEG alpha features (numpy arrays, averaged across epochs)
    tvb_dfa_np, tvb_lya_np,
    tvb_features,              # dict with TVB alpha features (same keys as eeg_features, shape (n_sim,))
    tvb_params_np,
    batch_size=512,
    weights=None
):
    """
    Find top TVB simulations that best match EEG DFA, Lya, and alpha-band features.

    Parameters
    ----------
    eeg_dfa_np : np.ndarray
        Shape (n_channels, 10)
    eeg_lya_np : np.ndarray
        Shape (n_channels,)
    eeg_features : dict
        Must include numpy arrays (averaged EEG features):
          - "alpha_power_global_mean" (scalar)
          - "alpha_power_precuneus_mean" (scalar)
          - "alpha_power_ACC_mean" (scalar)
          - "alpha_fc_DMN_mean" (scalar)
          - "alpha_hypersync_precuneus_ACC" (scalar)
    tvb_dfa_np : np.ndarray
        Shape (n_sim, n_channels, 10)
    tvb_lya_np : np.ndarray
        Shape (n_sim, n_channels)
    tvb_features : dict
        Same keys as eeg_features, values shape (n_sim,)
    tvb_params_np : np.ndarray
        Shape (n_sim, n_params)
    batch_size : int
        Batch size for GPU
    weights : dict or None
        Feature weights, default gives stronger weight to DFA/FC.

    Returns
    -------
    results : list of tuples
        (distance, param_vector, sim_dfa, sim_lya, sim_features)
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------- Weights ---------
    if weights is None:
        weights = {
            "dfa_lya": 0.6,
            "alpha_power_global_mean": 0.1,
            "alpha_power_precuneus_mean": 0.05,
            "alpha_power_ACC_mean": 0.05,
            "alpha_fc_DMN_mean": 0.1,
            "alpha_hypersync_precuneus_ACC": 0.1,
        }

    # --------- Flatten EEG DFA+Lya ---------
    eeg_vector = torch.from_numpy(
        np.concatenate([eeg_dfa_np.flatten(), eeg_lya_np.flatten()])
    ).float().to(device)  # (n_ch*11,)

    # Normalize EEG feature values to torch tensors
    # Meaning here over total epochs. correct?
    eeg_feat = {k: torch.tensor(np.mean(v), dtype=torch.float32, device=device)
                for k, v in eeg_features.items()}

    # --------- Convert TVB data ---------
    tvb_dfa = torch.from_numpy(tvb_dfa_np).float().to(device)    # (n_sim, n_ch, 10)
    tvb_lya = torch.from_numpy(tvb_lya_np).float().to(device)    # (n_sim, n_ch)
    tvb_params = torch.from_numpy(tvb_params_np)                # keep CPU for easier return

    # TVB feature dict to torch
    tvb_feat = {k: torch.from_numpy(v).float().to(device) for k, v in tvb_features.items()}

    n_sim = tvb_dfa.shape[0]
    flat_results = []

    # --------- Process in batches ---------
    for start in range(0, n_sim, batch_size):
        end = min(start + batch_size, n_sim)

        # DFA + Lya part
        dfa_batch = tvb_dfa[start:end]                   # (B, n_ch, 10)
        lya_batch = tvb_lya[start:end]                   # (B, n_ch)
        dfa_flat = dfa_batch.view(dfa_batch.size(0), -1) # (B, n_ch*10)
        lya_flat = lya_batch                             # (B, n_ch)
        full_batch = torch.cat([dfa_flat, lya_flat], dim=1) # (B, n_ch*11)

        diff_dfa_lya = full_batch - eeg_vector[None, :]
        dist_dfa_lya = torch.norm(diff_dfa_lya, dim=1)   # (B,)

        # print("tvb_feat", tvb_feat[key][start:end].shape)
        # print("eeg_feat", eeg_feat[key].shape)

        # Alpha features part
        dist_alpha = torch.zeros(end - start, device=device)
        for key, w in weights.items():
            if key == "dfa_lya":
                continue
            diff_feat = tvb_feat[key][start:end] - eeg_feat[key]
            dist_alpha += w * torch.abs(diff_feat)       # L1 distance weighted

        # Combine distances
        total_dist = weights["dfa_lya"] * dist_dfa_lya + dist_alpha

        # Store results
        for i in range(total_dist.shape[0]):
            idx = start + i
            param_vec = tvb_params[idx].numpy()
            sim_dfa = tvb_dfa_np[idx]
            sim_lya = tvb_lya_np[idx]
            sim_feats = {k: v[idx].item() for k, v in tvb_features.items()}
            flat_results.append((total_dist[i].item(), param_vec, sim_dfa, sim_lya, sim_feats))

        del dfa_batch, lya_batch, dfa_flat, lya_flat, full_batch, diff_dfa_lya, dist_dfa_lya, dist_alpha, total_dist
        # for key in tvb_feat:
        #     del tvb_feat[key][start:end]  # optional if tvb_feat is huge; else skip
        torch.cuda.empty_cache()

    del eeg_vector
    # for key in eeg_feat:
    #     del eeg_feat[key]
    del tvb_dfa, tvb_lya
    # for key in tvb_feat:
    #     del tvb_feat[key]
    torch.cuda.empty_cache()

    # Sort and return
    flat_results.sort(key=lambda x: x[0])
    return flat_results[:10]  # top 10


def prepare_alpha_features(alpha_hypersync_precuneus_ACC,
                           alpha_power_global_mean,
                           alpha_fc_DMN_mean,
                           fc_alpha=None,
                           power_alpha=None,
                           is_eeg=True):
    """
    Prepare alpha features for EEG or TVB to feed into matching function.

    Parameters
    ----------
    alpha_hypersync_precuneus_ACC : np.ndarray
        Shape (n_sim,) for TVB or (n_epochs,) for EEG.
    alpha_power_global_mean : np.ndarray
        Same as above.
    alpha_fc_DMN_mean : np.ndarray
        Same as above.
    fc_alpha : np.ndarray or None
        Optional, shape (n_sim, n_ch, n_ch) or (n_epochs, n_ch, n_ch).
    power_alpha : np.ndarray or None
        Optional, shape (n_sim, n_ch) or (n_epochs, n_ch).
    is_eeg : bool
        If True, averages across epochs → scalars for matching.
        If False, keeps array shape (n_sim,) for TVB.

    Returns
    -------
    features : dict
        Dictionary with keys expected by find_matches_with_alpha:
          - alpha_power_global_mean
          - alpha_power_precuneus_mean
          - alpha_power_ACC_mean
          - alpha_fc_DMN_mean
          - alpha_hypersync_precuneus_ACC
    """
    if is_eeg:
        # Collapse to scalars
        features = {
            "alpha_power_global_mean": float(np.mean(alpha_power_global_mean)),
            "alpha_power_precuneus_mean": float(np.mean(power_alpha[:, 20:22])) if power_alpha is not None else 0.0,
            "alpha_power_ACC_mean": float(np.mean(power_alpha[:, 10:12])) if power_alpha is not None else 0.0,
            "alpha_fc_DMN_mean": float(np.mean(alpha_fc_DMN_mean)),
            "alpha_hypersync_precuneus_ACC": float(np.mean(alpha_hypersync_precuneus_ACC)),
        }
    else:
        # Keep per-simulation arrays
        features = {
            "alpha_power_global_mean": np.asarray(alpha_power_global_mean),
            "alpha_power_precuneus_mean": np.mean(power_alpha[:, 20:22], axis=1) if power_alpha is not None else np.zeros_like(alpha_power_global_mean),
            "alpha_power_ACC_mean": np.mean(power_alpha[:, 10:12], axis=1) if power_alpha is not None else np.zeros_like(alpha_power_global_mean),
            "alpha_fc_DMN_mean": np.asarray(alpha_fc_DMN_mean),
            "alpha_hypersync_precuneus_ACC": np.asarray(alpha_hypersync_precuneus_ACC),
        }

    return features

def find_matches_with_alpha(
    eeg_dfa_np, eeg_lya_np,
    eeg_features,              # dict with EEG alpha features (numpy arrays or lists)
    tvb_dfa_np, tvb_lya_np,
    tvb_features,              # dict with TVB alpha features (numpy arrays, shape (n_sim,))
    tvb_params_np,
    batch_size=512,
    weights=None
):
    """
    Find top TVB simulations that best match EEG DFA, Lyapunov, and alpha-band features.

    Parameters
    ----------
    eeg_dfa_np : np.ndarray, shape (n_channels, n_windows)
    eeg_lya_np : np.ndarray, shape (n_channels,)
    eeg_features : dict of np.ndarray or list
        Keys: alpha_power_global_mean, alpha_power_precuneus_mean, alpha_power_ACC_mean,
              alpha_fc_DMN_mean, alpha_hypersync_precuneus_ACC
        Values can be arrays per epoch; will be averaged to scalars internally
    tvb_dfa_np : np.ndarray, shape (n_sim, n_channels, n_windows)
    tvb_lya_np : np.ndarray, shape (n_sim, n_channels)
    tvb_features : dict of np.ndarray, each shape (n_sim,)
    tvb_params_np : np.ndarray, shape (n_sim, n_params)
    batch_size : int
    weights : dict or None
        Example:
        {
            "dfa_lya": 0.6,
            "alpha_power_global_mean": 0.1,
            "alpha_power_precuneus_mean": 0.05,
            "alpha_power_ACC_mean": 0.05,
            "alpha_fc_DMN_mean": 0.1,
            "alpha_hypersync_precuneus_ACC": 0.1,
        }

    Returns
    -------
    top_matches : list of tuples
        (distance, param_vector, sim_dfa, sim_lya, sim_features)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Prepare EEG vectors ----------
    eeg_dfa = torch.as_tensor(eeg_dfa_np, dtype=torch.float32, device=device)
    eeg_lya = torch.as_tensor(eeg_lya_np, dtype=torch.float32, device=device)

    # Flatten DFA + Lya
    eeg_vector = torch.cat([eeg_dfa.flatten(), eeg_lya.flatten()])

    # Average EEG alpha features to scalars
    # eeg_feat = {k: torch.as_tensor(np.mean(v), dtype=torch.float32, device=device)
    #             for k, v in eeg_features.items()}

    eeg_feat = {}
    for k, v in eeg_features.items():
        if isinstance(v, torch.Tensor):
            eeg_feat[k] = v.float().mean().to(device)
        else:
            eeg_feat[k] = torch.tensor(np.mean(v), dtype=torch.float32, device=device)

    # ---------- Convert TVB data ----------
    tvb_dfa = torch.as_tensor(tvb_dfa_np, dtype=torch.float32, device=device)
    tvb_lya = torch.as_tensor(tvb_lya_np, dtype=torch.float32, device=device)
    tvb_params = torch.as_tensor(tvb_params_np, dtype=torch.float32)  # keep CPU if desired

    tvb_feat = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in tvb_features.items()}

    n_sim = tvb_dfa.shape[0]

    # ---------- Default weights ----------
    if weights is None:
        weights = {
            "dfa_lya": 0.6,
            "alpha_power_global_mean": 0.1,
            "alpha_power_precuneus_mean": 0.05,
            "alpha_power_ACC_mean": 0.05,
            "alpha_fc_DMN_mean": 0.1,
            "alpha_hypersync_precuneus_ACC": 0.1,
        }

    flat_results = []

    # ---------- Process in batches ----------
    for start in range(0, n_sim, batch_size):
        end = min(start + batch_size, n_sim)
        bsize = end - start

        # --- DFA + Lya ---
        dfa_batch = tvb_dfa[start:end]          # (B, n_ch, n_windows)
        lya_batch = tvb_lya[start:end]          # (B, n_ch)
        dfa_flat = dfa_batch.view(bsize, -1)    # (B, n_ch * n_windows)
        lya_flat = lya_batch                     # (B, n_ch)
        full_batch = torch.cat([dfa_flat, lya_flat], dim=1)  # (B, n_ch*11)
        diff_dfa_lya = full_batch - eeg_vector[None, :]
        dist_dfa_lya = torch.norm(diff_dfa_lya, dim=1)       # (B,)

        # --- Alpha features ---
        dist_alpha = torch.zeros(bsize, device=device)
        for key, w in weights.items():
            if key == "dfa_lya":
                continue
            diff_feat = tvb_feat[key][start:end] - eeg_feat[key]  # scalar broadcast
            dist_alpha += w * torch.abs(diff_feat)

        # --- Total distance ---
        total_dist = weights.get("dfa_lya", 0.6) * dist_dfa_lya + dist_alpha

        # --- Store results ---
        for i in range(bsize):
            idx = start + i
            sim_feats = {k: v[idx].item() for k, v in tvb_features.items()}
            flat_results.append((
                total_dist[i].item(),
                tvb_params[idx].numpy(),
                tvb_dfa_np[idx],
                tvb_lya_np[idx],
                sim_feats
            ))

        # Free GPU memory
        del dfa_batch, lya_batch, dfa_flat, lya_flat, full_batch, diff_dfa_lya, dist_dfa_lya, dist_alpha, total_dist
        torch.cuda.empty_cache()

    # Sort and return top 10 matches
    flat_results.sort(key=lambda x: x[0])
    return flat_results[:10]


def find_matches_with_fc_notoptimizedformemory(
    eeg_dfa_np, eeg_lya_np,
    eeg_fc_np,                  # shape (entries, entries)
    tvb_dfa_np, tvb_lya_np,
    tvb_fc_np,                  # shape (n_sim, entries, entries)
    tvb_params_np,
    batch_size=512,
    weights=None
):
    """
    Find top TVB simulations that best match EEG DFA, Lyapunov, and FC matrices.

    Parameters
    ----------
    eeg_dfa_np : np.ndarray, shape (n_channels, n_windows)
    eeg_lya_np : np.ndarray, shape (n_channels,)
    eeg_fc_np : np.ndarray, shape (entries, entries)
    tvb_dfa_np : np.ndarray, shape (n_sim, n_channels, n_windows)
    tvb_lya_np : np.ndarray, shape (n_sim, n_channels)
    tvb_fc_np : np.ndarray, shape (n_sim, entries, entries)
    tvb_params_np : np.ndarray, shape (n_sim, n_params)
    batch_size : int
    weights : dict or None
        Example:
        {
            "dfa_lya": 0.6,
            "fc": 0.4,
        }

    Returns
    -------
    top_matches : list of tuples
        (distance, param_vector, sim_dfa, sim_lya, sim_fc)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Convert to GPU ----------
    eeg_dfa = torch.as_tensor(eeg_dfa_np, dtype=torch.float32, device=device)
    eeg_lya = torch.as_tensor(eeg_lya_np, dtype=torch.float32, device=device)
    eeg_fc = torch.as_tensor(eeg_fc_np, dtype=torch.float32, device=device)

    tvb_dfa = torch.as_tensor(tvb_dfa_np, dtype=torch.float32, device=device)
    tvb_lya = torch.as_tensor(tvb_lya_np, dtype=torch.float32, device=device)
    tvb_fc = torch.as_tensor(tvb_fc_np, dtype=torch.float32, device=device)
    tvb_params = torch.as_tensor(tvb_params_np, dtype=torch.float32)

    n_sim = tvb_dfa.shape[0]

    # ---------- Default weights ----------
    if weights is None:
        weights = {"dfa_lya": 0.6, "fc": 0.4}

    # ---------- Flatten EEG DFA + Lya ----------
    eeg_vector = torch.cat([eeg_dfa.flatten(), eeg_lya.flatten()])

    # ---------- Prepare FC mask (upper triangle only to avoid redundancy) ----------
    triu_idx = torch.triu_indices(eeg_fc.shape[0], eeg_fc.shape[1], offset=1)
    eeg_fc_vec = eeg_fc[triu_idx]

    flat_results = []

    # ---------- Process in batches ----------
    for start in range(0, n_sim, batch_size):
        end = min(start + batch_size, n_sim)
        bsize = end - start

        # --- DFA + Lya distance ---
        dfa_batch = tvb_dfa[start:end]        # (B, n_ch, n_win)
        lya_batch = tvb_lya[start:end]        # (B, n_ch)
        dfa_flat = dfa_batch.view(bsize, -1)
        lya_flat = lya_batch
        full_batch = torch.cat([dfa_flat, lya_flat], dim=1)
        diff_dfa_lya = full_batch - eeg_vector[None, :]
        dist_dfa_lya = torch.norm(diff_dfa_lya, dim=1)

        # --- FC similarity (Pearson correlation) ---
        fc_batch = tvb_fc[start:end]          # (B, entries, entries)
        fc_vec = fc_batch[:, triu_idx[0], triu_idx[1]]  # (B, N_edges)

        # normalize to zero mean / unit variance
        fc_vec_norm = (fc_vec - fc_vec.mean(dim=1, keepdim=True)) / (fc_vec.std(dim=1, keepdim=True) + 1e-8)
        eeg_fc_norm = (eeg_fc_vec - eeg_fc_vec.mean()) / (eeg_fc_vec.std() + 1e-8)

        fc_corr = torch.sum(fc_vec_norm * eeg_fc_norm[None, :], dim=1) / (fc_vec_norm.shape[1] - 1)
        fc_dist = 1 - fc_corr  # correlation distance

        # --- Total distance ---
        total_dist = weights["dfa_lya"] * dist_dfa_lya + weights["fc"] * fc_dist

        # --- Store results ---
        for i in range(bsize):
            idx = start + i
            flat_results.append((
                total_dist[i].item(),
                tvb_params[idx].cpu().numpy(),
                tvb_dfa_np[idx],
                tvb_lya_np[idx],
                tvb_fc_np[idx]
            ))

        del dfa_batch, lya_batch, fc_batch, fc_vec, fc_vec_norm
        torch.cuda.empty_cache()

    # flat_results.sort(key=lambda x: x[0])
    # return flat_results[:10]
    return flat_results


def find_matches_with_fc(
    eeg_dfa_np, eeg_lya_np,
    eeg_fc_np,                  # shape (1, entries, entries)
    tvb_dfa_np, tvb_lya_np,
    tvb_fc_np,                  # shape (n_sim, entries, entries)
    tvb_params_np,
    myrank,
    batch_size=32,
    top_k=10,
    weights=None
):
    """
    Find top TVB simulations that best match EEG DFA, Lyapunov, and FC matrices.
    Memory-efficient version: only keeps top_k matches and uses optimized FC correlation.
    """
    import heapq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Convert DFA/Lya ----------
    eeg_dfa = torch.as_tensor(eeg_dfa_np, dtype=torch.float32, device=device)
    eeg_lya = torch.as_tensor(eeg_lya_np, dtype=torch.float32, device=device)

    tvb_dfa = torch.as_tensor(tvb_dfa_np, dtype=torch.float32, device=device)
    tvb_lya = torch.as_tensor(tvb_lya_np, dtype=torch.float32, device=device)
    tvb_params = torch.as_tensor(tvb_params_np, dtype=torch.float32)

    # FC
    eeg_fc = torch.as_tensor(eeg_fc_np, dtype=torch.float32, device=device)   # (1, C, C)
    tvb_fc = torch.as_tensor(tvb_fc_np, dtype=torch.float32, device=device)   # (N, C, C)

    n_sim = tvb_dfa.shape[0]
    entries = eeg_fc.shape[1]         # <-- FIX HERE (was shape[0])

    if weights is None:
        weights = {"dfa_lya": 0.6, "fc": 0.4}

    # Flatten EEG DFA+Lya
    eeg_vector = torch.cat([eeg_dfa.flatten(), eeg_lya.flatten()])

    # ---------- Correct upper-triangle indices ----------
    triu_idx = torch.triu_indices(entries, entries, offset=1)   # (2, N_edges)

    # eeg_fc_vec = eeg_fc[0, triu_idx[0], triu_idx[1]]            # (N_edges,) used to take only the first, now median
    eeg_fc_vec = eeg_fc[triu_idx[0], triu_idx[1]]
    eeg_fc_centered = eeg_fc_vec - eeg_fc_vec.mean()
    eeg_fc_norm = torch.norm(eeg_fc_centered)

    # ---------- Heap ----------
    top_k_heap = []

    # ---------- Loop ----------
    for start in range(0, n_sim, batch_size):
        end = min(start + batch_size, n_sim)
        bsize = end - start

        # DFA + LYA
        dfa_batch = tvb_dfa[start:end].view(bsize, -1)
        lya_batch = tvb_lya[start:end]
        full_batch = torch.cat([dfa_batch, lya_batch], dim=1)
        diff_dfa_lya = full_batch - eeg_vector[None, :]
        dist_dfa_lya = torch.norm(diff_dfa_lya, dim=1)

        # FC (vectorized upper triangle)
        fc_batch = tvb_fc[start:end]                    # (B, C, C)
        fc_vec = fc_batch[:, triu_idx[0], triu_idx[1]]  # (B, N_edges)

        fc_vec_centered = fc_vec - fc_vec.mean(dim=1, keepdim=True)

        dot = torch.sum(fc_vec_centered * eeg_fc_centered[None, :], dim=1)
        fc_norms = torch.norm(fc_vec_centered, dim=1)

        fc_corr = dot / (fc_norms * eeg_fc_norm + 1e-8)
        fc_dist = 1 - fc_corr

        # Total distance
        total_dist = weights["dfa_lya"] * dist_dfa_lya + weights["fc"] * fc_dist

        # Push into heap
        for i in range(bsize):
            idx = start + i
            score = total_dist[i].item()

            if len(top_k_heap) < top_k:
                heapq.heappush(top_k_heap, (-score, idx))
            else:
                if -score > top_k_heap[0][0]:
                    heapq.heapreplace(top_k_heap, (-score, idx))

        del dfa_batch, lya_batch, fc_batch, fc_vec, fc_vec_centered, dot, fc_norms, fc_dist
        torch.cuda.empty_cache()

    # ---------- Extract results ----------
    top_matches = []
    top_indices = [idx for _, idx in sorted(top_k_heap, reverse=True)]

    # final reconstruction for dashboard
    eeg_fc_centered_cpu = eeg_fc_centered.cpu()
    eeg_fc_norm_cpu = eeg_fc_norm.cpu()

    for idx in top_indices:
        tvb_fc_vec = torch.as_tensor(tvb_fc_np[idx][triu_idx[0], triu_idx[1]])
        # tvb_fc_centered = tvb_fc_vec - tvb_fc_vec.mean()
        tvb_fc_centered = (tvb_fc_vec - tvb_fc_vec.mean()).cpu()
        fc_dist = 1 - (torch.sum(tvb_fc_centered * eeg_fc_centered_cpu) /
                       (torch.norm(tvb_fc_centered) * eeg_fc_norm_cpu + 1e-8))

        dfa_lya_dist = torch.norm(
            torch.cat([
                torch.as_tensor(tvb_dfa_np[idx]).flatten(),
                torch.as_tensor(tvb_lya_np[idx])
            ]) - eeg_vector.cpu()
        )

        total = weights["dfa_lya"] * dfa_lya_dist + weights["fc"] * fc_dist
        gidx = myrank * tvb_params_np.shape[0] + idx
        # print('gidx', gidx)

        top_matches.append((
            float(total),
            tvb_params_np[idx],
            tvb_dfa_np[idx],
            tvb_lya_np[idx],
            tvb_fc_np[idx],
            gidx
        ))

    return top_matches




if __name__ == '__main__':

    def test_lyadfa_func():
        '''
        eeg_dfa    # shape (n_channels, 10)
        eeg_lya    # shape (n_channels,)
        tvb_dfa    # shape (n_sim, n_channels, 10)
        tvb_lya    # shape (n_sim, n_channels)
        tvb_params # shape (n_sim, n_params)
        '''

        # =====================================
        # 🔧 Step 1: Generate random test data
        # =====================================
        np.random.seed(42)

        n_sim = 5000  # number of TVB simulations
        n_channels = 64  # EEG channels
        n_windows = 10  # DFA window sizes
        n_params = 6  # TVB parameter space dimensionality

        # EEG DFA & Lya: average over epochs
        eeg_dfa = np.random.rand(n_channels, n_windows)
        eeg_lya = np.random.rand(n_channels)

        # TVB simulations
        tvb_dfa = np.random.rand(n_sim, n_channels, n_windows)
        tvb_lya = np.random.rand(n_sim, n_channels)
        tvb_params = np.random.rand(n_sim, n_params)

        print('eeg_dfa', eeg_dfa.shape)
        print('eeg_lya', eeg_lya.shape)
        print('tvb_dfa', tvb_dfa.shape)
        print('tvb_lya', tvb_lya.shape)
        print('tvb_params', tvb_params.shape)

        # =====================================
        # 🔍 Step 2: Run GPU batch matching
        # =====================================
        # results = find_matches_bare(eeg_dfa, eeg_lya, tvb_dfa, tvb_lya, tvb_params)
        # test for 0 distance?
        results = find_matches_bare(tvb_dfa[1], tvb_lya[1], tvb_dfa, tvb_lya, tvb_params)

        # =====================================
        # 🖨️ Step 3: Print results
        # =====================================
        for rank, (dist, params, dfa, lya) in enumerate(results, 1):
            print(f"[{rank}] Distance = {dist:.4f} | Params = {params}")


    def prepare_alpha_features_for_match(features_dict, is_eeg=True):
        """
        Prepare alpha features (from compute_alpha_features_eeg/ tvb) for matching.

        Parameters
        ----------
        features_dict : dict
            Dict with keys:
              - "alpha_power_global_mean"
              - "alpha_power_precuneus_mean"
              - "alpha_power_ACC_mean"
              - "alpha_fc_DMN_mean"
              - "alpha_hypersync_precuneus_ACC"
            Values are np.ndarray from compute_alpha_features_*.
        is_eeg : bool
            If True → collapse to scalars (average across epochs).
            If False → keep arrays (per simulation).

        Returns
        -------
        prepared : dict
            Ready for find_matches_with_alpha
        """
        if is_eeg:
            prepared = {k: float(np.mean(v)) for k, v in features_dict.items()}
        else:
            prepared = {k: np.asarray(v) for k, v in features_dict.items()}

        return prepared


    def test_with_alpha():

        n_sim = 5000  # number of TVB simulations
        n_channels = 64  # EEG channels
        n_windows = 10  # DFA window sizes
        n_params = 6  # TVB parameter space dimensionality

        eeg_dfa = np.random.rand(n_channels, n_windows)
        eeg_lya = np.random.rand(n_channels)

        # TVB simulations
        tvb_dfa = np.random.rand(n_sim, n_channels, n_windows)
        tvb_lya = np.random.rand(n_sim, n_channels)
        tvb_params = np.random.rand(n_sim, n_params)

        # EEG (averaged across epochs)
        eeg_features = {
            "alpha_power_global_mean": 0.0087,
            "alpha_power_precuneus_mean": 0.0091,
            "alpha_power_ACC_mean": 0.0082,
            "alpha_fc_DMN_mean": 0.17,
            "alpha_hypersync_precuneus_ACC": 0.16,
        }

        # TVB (precomputed per simulation, shape (n_sim,))
        tvb_features = {
            "alpha_power_global_mean": np.random.rand(n_sim),
            "alpha_power_precuneus_mean": np.random.rand(n_sim),
            "alpha_power_ACC_mean": np.random.rand(n_sim),
            "alpha_fc_DMN_mean": np.random.rand(n_sim),
            "alpha_hypersync_precuneus_ACC": np.random.rand(n_sim),
        }

        print("eegfeatshape", eeg_features)
        print("tvbfeatshape", tvb_features)

        matches = find_matches_with_alpha(
            eeg_dfa, eeg_lya,
            eeg_features,
            tvb_dfa, tvb_lya,
            tvb_features,
            tvb_params,
            batch_size=256
        )

        for dist, params, dfa, lya, feats in matches:
            print("Dist:", dist, "Params:", params, "Alpha FC DMN:", feats["alpha_fc_DMN_mean"])


    def test_preparefeat():

        eeg_features = prepare_alpha_features(
            alpha_hypersync_precuneus_ACC=np.array([0.609, 0.871, 0.936]),
            alpha_power_global_mean=np.array([1.18e-07, 4.49e-09, 4.24e-09]),
            alpha_fc_DMN_mean=np.array([0.519, 0.871, 0.936]),
            fc_alpha=np.random.rand(3, 61, 61),
            power_alpha=np.random.rand(3, 61),
            is_eeg=True
        )
        print(eeg_features)

        tvb_features = prepare_alpha_features(
            alpha_hypersync_precuneus_ACC=np.random.rand(5000),
            alpha_power_global_mean=np.random.rand(5000),
            alpha_fc_DMN_mean=np.random.rand(5000),
            fc_alpha=np.random.rand(5000, 61, 61),
            power_alpha=np.random.rand(5000, 61),
            is_eeg=False
        )
        print({k: v.shape for k, v in tvb_features.items()})

    # test_lyadfa_func()
    test_with_alpha()
    # test_preparefeat()