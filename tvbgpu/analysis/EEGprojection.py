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


import numpy as np
import scipy.io
import torch

def sequential_EEGproj():
    # 1. Load TVB simulated data (your own or from Zenodo example)
    # tvb_data = np.load("my_simulated_data.npy")  # shape: (68, n_times)
    tvb_data = np.random.randn(16, 68, 1000)
    # OR load from their example:
    # mat = scipy.io.loadmat("QL_BOLD_regiontimecourse.mat")
    # tvb_data = mat["BOLD_timecourse"].T  # adjust depending on variable name

    # 2. Load the precomputed projection matrix
    proj_mat = scipy.io.loadmat("../data/QL_20120814_ProjectionMatrix.mat")
    gain_matrix = proj_mat["ProjectionMatrix"]  # shape: (n_channels, 68)

    print('gain_matrix', gain_matrix.shape)

    # RegionMapping.txt: one integer per vertex ∈ [1, 68] (or maybe 0-indexed)
    region_mapping = np.loadtxt("../data/QL_20120814_RegionMapping.txt", dtype=int)  # shape: (14981,)

    print('region_mapping', region_mapping.shape)

    # Choose one sample to project (e.g., simulation 0, condition 0)
    tvb_sample = tvb_data[0]  # shape: (68, 1000)

    print('tvb_sample', tvb_sample.shape)
    # print('tvb_sample', tvb_sample[41])

    # Map TVB 68-region activity to 14981 vertices
    vertex_data = np.zeros((14981, 1000))
    for vertex_idx in range(14981):
        region_idx = region_mapping[vertex_idx]
        if region_idx > 0 and region_idx <= 68:
            vertex_data[vertex_idx] = tvb_sample[region_idx - 1]

    print('vertex_data', vertex_data.shape)

    # 3. Project TVB data to EEG
    eeg_projected = gain_matrix @ vertex_data  # shape: (n_channels, n_times)
    print('eeg_projected', eeg_projected.shape)

    # 4. Save for analysis
    # np.save("simulated_eeg.npy", eeg_projected)

def parallel_EEGproj(tvb_data, region_mapping_np, gain_matrix_np):

    '''input shape (simulations, regions, timesteps) '''

    # num_vertices = len(region_mapping)
    num_regions = 68

    print(region_mapping.min())

    # Build one-hot matrix: shape (14981, 68)
    region_map_onehot = torch.nn.functional.one_hot(
        torch.tensor(region_mapping_np), num_classes=num_regions
    ).float()  # shape: (14981, 68)

    # Transpose to (68, 14981) if you want region → vertex, but keep (14981, 68) for einsum

    # Step 1: Upsample to vertex space (N, 14981, T)
    vertex_data = torch.einsum('vr, nrt -> nvt', region_map_onehot, tvb_data)

    # Step 2: Load projection matrix: shape (61, 14981)
    gain_matrix = torch.tensor(gain_matrix_np, dtype=torch.float32)

    # Step 3: Project to EEG: (N, 61, T)
    eeg_data = torch.einsum('cv, nvt -> nct', gain_matrix, vertex_data)

    eeg_data_cpu = eeg_data.cpu().numpy()
    print(eeg_data_cpu.shape)

    # eeg_data now has shape (N, 61, T) — all EEG simulations!

    return eeg_data_cpu

def parallel_EEGproj_datatotorch(tvb_data_np, region_mapping_np, gain_matrix_np):

    '''input shape (simulations, regions, timesteps) '''

    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tvb_data = torch.tensor(tvb_data_np, dtype=torch.float32).to(device)
    tvb_data = torch.from_numpy(tvb_data_np).to(device)
    gain_matrix = torch.tensor(gain_matrix_np, dtype=torch.float32).to(device)  # (61, 14981)
    region_mapping = torch.tensor(region_mapping_np, dtype=torch.int64).clamp(min=0)

    # num_vertices = len(region_mapping)
    num_regions = 68

    # print(region_mapping.min())

    # Build one-hot matrix: shape (14981, 68)
    # region_map_onehot = torch.nn.functional.one_hot(
    #     torch.tensor(region_mapping), num_classes=num_regions
    # ).float()  # shape: (14981, 68)

    region_map_onehot = torch.nn.functional.one_hot(region_mapping, num_classes=num_regions).float().to(device)  # (14981, 68)

    # Transpose to (68, 14981) if you want region → vertex, but keep (14981, 68) for einsum

    # Step 1: Upsample to vertex space (N, 14981, T)
    vertex_data = torch.einsum('vr, nrt -> nvt', region_map_onehot, tvb_data)

    # Step 2: Load projection matrix: shape (61, 14981)
    # gain_matrix = torch.tensor(gain_matrix_np, dtype=torch.float32)

    # Step 3: Project to EEG: (N, 61, T)
    eeg_data = torch.einsum('cv, nvt -> nct', gain_matrix, vertex_data)

    eeg_data_cpu = eeg_data.cpu().numpy()
    print(eeg_data_cpu.shape)

    # eeg_data now has shape (N, 61, T) — all EEG simulations!

    return eeg_data_cpu


def parallel_EEGproj_batched(tvb_data_np, region_mapping_np, gain_matrix_np, batch_size=4):

    ''' tvb_data_np: shape (N, 68, T) '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_regions = 68
    num_vertices = len(region_mapping_np)
    num_channels = gain_matrix_np.shape[0]
    N, R, T = tvb_data_np.shape

    # ✅ Ensure all input NumPy arrays are float32
    tvb_data_np = tvb_data_np.astype(np.float32)
    gain_matrix_np = gain_matrix_np.astype(np.float32)

    # Preallocate output
    eeg_all = np.zeros((N, num_channels, T), dtype=np.float32)

    # Prepare shared tensors
    region_mapping = torch.tensor(region_mapping_np, dtype=torch.int64).clamp(min=0)
    region_map_onehot = torch.nn.functional.one_hot(region_mapping, num_classes=num_regions).float().to(device)
    gain_matrix = torch.tensor(gain_matrix_np, dtype=torch.float32).to(device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # ✅ Slice and convert batch to torch.float32
        tvb_batch = torch.from_numpy(tvb_data_np[start:end]).float().to(device)

        # Vertex upsampling
        vertex_data = torch.einsum('vr, nrt -> nvt', region_map_onehot, tvb_batch)

        # EEG projection
        eeg_batch = torch.einsum('cv, nvt -> nct', gain_matrix, vertex_data)

        # Back to NumPy on CPU
        eeg_all[start:end] = eeg_batch.cpu().numpy()

        # Free memory between batches
        del tvb_batch, vertex_data, eeg_batch
        torch.cuda.empty_cache()

    # print(eeg_all.shape)



    return eeg_all


if __name__ == '__main__':

    # Simulated TVB data: shape (N, 68, T)
    N, R, T = 16, 68, 1000
    # tvb_data = torch.randn(N, R, T)

    tvb_data = np.random.randn(N, R, T)

    region_mapping = np.loadtxt("../data/QL_20120814_RegionMapping.txt", dtype=int)
    gain_matrix_np = scipy.io.loadmat("../data/QL_20120814_ProjectionMatrix.mat")["ProjectionMatrix"]

    # parallel_EEGproj(tvb_data, region_mapping, gain_matrix_np)
    # parallel_EEGproj_datatotorch(tvb_data, region_mapping, gain_matrix_np)
    parallel_EEGproj_batched(tvb_data, region_mapping, gain_matrix_np, batch_size=2)