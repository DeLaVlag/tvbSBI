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
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import os

#making sure the cubin dir exist in the container
os.makedirs("/tmp/pycuda_cache", exist_ok=True)

import mne

def computeDFA_gpu(data, logger, mpirank, window_sizes):

    # Define the CUDA kernel for DFA computation
    mod = SourceModule(r"""
    #include <math.h>

    extern "C" __global__
    void computeDFA(
        float *data,
        int num_steps,
        int num_regions,
        int *window_sizes,
        int num_window_sizes,
        int nsims,
        float *dfa_log_s,
        float *dfa_log_F,
        float *alphas
    )
    {
        int sim_idx    = blockIdx.x * blockDim.x + threadIdx.x;
        int region_idx = blockIdx.y * blockDim.y + threadIdx.y;

        if (sim_idx >= nsims || region_idx >= num_regions) return;

        // int base_data = sim_idx * num_regions * num_steps
        //              + region_idx;
                      
        int base_data = sim_idx * num_regions * num_steps
              + region_idx * num_steps;

        int base_out  = sim_idx * num_regions * num_window_sizes
                      + region_idx * num_window_sizes;

        for (int window_idx = 0; window_idx < num_window_sizes; ++window_idx) {

            int s = window_sizes[window_idx];
            if (s < 4 || s >= num_steps) {
                // dfa_log_s[base_out + window_idx] = NAN;
                // dfa_log_F[base_out + window_idx] = NAN;
                // alphas   [base_out + window_idx] = NAN;
                continue;
            }

            int num_windows = num_steps / s;
            if (num_windows < 2) {
                // dfa_log_s[base_out + window_idx] = NAN;
                // dfa_log_F[base_out + window_idx] = NAN;
                // alphas   [base_out + window_idx] = NAN;
                continue;
            }

            float fluct_sum = 0.f;

            for (int w = 0; w < num_windows; ++w) {

                int start = w * s;

                float mean = 0.f;
                for (int i = 0; i < s; ++i)
                    // mean += data[base_data + (start + i) * num_regions];
                    mean += data[base_data + start + i];
                mean /= s;

                float var = 0.f;
                for (int i = 0; i < s; ++i) {
                    // float d = data[base_data + (start + i) * num_regions] - mean;
                    float d = data[base_data + start + i] - mean;
                    var += d * d;
                }

                fluct_sum += var / s;
            }

            float F = sqrtf(fluct_sum / num_windows);

            if (F > 1e-20f) {
                float log_s = logf((float)s);
                float log_F = logf(F);

                dfa_log_s[base_out + window_idx] = log_s;
                dfa_log_F[base_out + window_idx] = log_F;
                alphas   [base_out + window_idx] = log_F / log_s;
            } // else {
              //  dfa_log_s[base_out + window_idx] = 0.f;
              //  dfa_log_F[base_out + window_idx] = 0.f;
              //  alphas   [base_out + window_idx] = 0.f;
            // }
        }
    }
    """, options=["--use_fast_math"])

    # Get the computeDFA kernel function
    computeDFA_kernel = mod.get_function("computeDFA")

    # Host function to invoke the PyCUDA kernel

    # min_window_size = 10  # Minimum window size
    # max_window_size = int(data.shape[2]/2)  # Maximum window size (half of the time series length)
    # min_window_size = 4  # Minimum window size
    # max_window_size = 100 # Maximum window size (half of the time series length)
    # num_windows = 10
    # window_sizes = np.unique(np.logspace(np.log10(min_window_size), np.log10(max_window_size), num=num_windows, dtype=np.int32))

    # window_sizes = np.array([4, 8, 16, 32, 64], dtype=np.int32)
    # window_sizes = np.array([ 4,  7, 15, 31, 62], dtype=np.int32)

    window_sizes = np.asarray(window_sizes, dtype=np.int32)
    nsims, num_regions, num_steps = data.shape
    num_window_sizes = len(window_sizes)

    # Allocate memory for the results
    alphas = np.zeros((nsims, num_regions, num_window_sizes), dtype=np.float32)
    # dfa_log_s = np.zeros((nsims, num_regions, num_window_sizes), dtype=np.float32)
    # dfa_log_F = np.zeros((nsims, num_regions, num_window_sizes), dtype=np.float32)

    dfa_log_s = np.full(
        (nsims, num_regions, num_window_sizes),
        np.nan,
        dtype=np.float32,
    )

    dfa_log_F = np.full(
        (nsims, num_regions, num_window_sizes),
        np.nan,
        dtype=np.float32,
    )

    # Flatten the data for the GPU
    data_flat = data.ravel().astype(np.float32)

    # Allocate device memory
    data_gpu = cuda.mem_alloc(data_flat.nbytes)
    window_sizes_gpu = cuda.mem_alloc(window_sizes.nbytes)
    alphas_gpu = cuda.mem_alloc(alphas.nbytes)
    dfa_log_s_gpu = cuda.mem_alloc(dfa_log_s.nbytes)
    dfa_log_F_gpu = cuda.mem_alloc(dfa_log_F.nbytes)

    # Copy data to the device (GPU)
    cuda.memcpy_htod(data_gpu, data_flat)
    cuda.memcpy_htod(window_sizes_gpu, window_sizes)
    cuda.memcpy_htod(dfa_log_s_gpu, dfa_log_s)
    cuda.memcpy_htod(dfa_log_F_gpu, dfa_log_F)

    # Define thread and block size
    block_size = (8, 8, 1)  # 8 threads for simulations and 8 for regions
    grid_size = ((nsims + block_size[0] - 1) // block_size[0], (num_regions + block_size[1] - 1) // block_size[1])

    if mpirank == 0 and logger != None:
        # print("\n")
        logger.debug("DFA nsims %d", nsims)
        logger.debug("DFA num_regions %d", num_regions)
        logger.debug("DFA num_steps %d", num_steps)
        logger.debug('DFA Window_sizesdata %s', window_sizes)
        logger.debug("DFA data shape %s", data.shape)
        logger.debug("DFA kenrlel grid_size %s", grid_size)
        # print("\n")

    # Launch the kernel
    computeDFA_kernel(data_gpu, np.int32(num_steps), np.int32(num_regions),
                      window_sizes_gpu, np.int32(num_window_sizes), np.int32(nsims), dfa_log_s_gpu, dfa_log_F_gpu, alphas_gpu,
                      block=block_size, grid=grid_size)

    # Copy the result back to host
    cuda.memcpy_dtoh(alphas, alphas_gpu)
    cuda.memcpy_dtoh(dfa_log_s, dfa_log_s_gpu)
    cuda.memcpy_dtoh(dfa_log_F, dfa_log_F_gpu)

    data_gpu.free()
    window_sizes_gpu.free()
    alphas_gpu.free()
    dfa_log_s_gpu.free()
    dfa_log_F_gpu.free()

    # slope, intercept = np.polyfit(dfa_log_s, dfa_log_F, 1)
    # alphas = dfa_log_F / dfa_log_s

    alphas = compute_dfa_alphas(dfa_log_s, dfa_log_F)

    # return np.median(alphas, axis=2)  # Take median across the window sizes
    return alphas, dfa_log_s, dfa_log_F

def compute_dfa_alphas_nonnans(log_s, log_F):
    """
    Compute DFA scaling exponent alpha by linear regression:
    slope of log(F) vs log(s).

    Parameters
    ----------
    log_s : ndarray, shape (n_sims, n_regions, n_windows)
    log_F : ndarray, shape (n_sims, n_regions, n_windows)

    Returns
    -------
    alphas : ndarray, shape (n_sims, n_regions)
    """

    # sort by scale (important for numerical stability)
    order = np.argsort(log_s, axis=-1)
    log_s = np.take_along_axis(log_s, order, axis=-1)
    log_F = np.take_along_axis(log_F, order, axis=-1)

    # compute linear regression slope analytically
    x = log_s
    y = log_F
    n = x.shape[-1]

    sum_x  = np.sum(x, axis=-1)
    sum_y  = np.sum(y, axis=-1)
    sum_xx = np.sum(x * x, axis=-1)
    sum_xy = np.sum(x * y, axis=-1)

    numerator   = n * sum_xy - sum_x * sum_y
    denominator = n * sum_xx - sum_x * sum_x

    alphas = numerator / denominator

    return alphas

def compute_dfa_alphas(log_s, log_F, min_points=5):
    """
    Compute DFA alpha as slope of log_F vs log_s.
    Handles NaNs safely.

    log_s: (n_sims, n_regions, n_windows)
    log_F: (n_sims, n_regions, n_windows)

    returns:
        alphas: (n_sims, n_regions)
    """

    log_s = np.asarray(log_s, dtype=np.float32)
    log_F = np.asarray(log_F, dtype=np.float32)

    valid = np.isfinite(log_s) & np.isfinite(log_F)

    n = np.sum(valid, axis=-1)

    x = np.where(valid, log_s, 0.0)
    y = np.where(valid, log_F, 0.0)

    sum_x  = np.sum(x, axis=-1)
    sum_y  = np.sum(y, axis=-1)
    sum_xx = np.sum(x * x, axis=-1)
    sum_xy = np.sum(x * y, axis=-1)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_xx - sum_x * sum_x

    alphas = numerator / denominator

    alphas = np.where(
        (n >= min_points) & np.isfinite(alphas) & (np.abs(denominator) > 1e-12),
        alphas,
        np.nan,
    )

    return alphas.astype(np.float32)


# Example usage
if __name__ == "__main__":

    def generate_test_data(num_sims, num_regions, num_steps):
        # Generate data for each target DFA exponent
        data_close_to_1 = np.cumsum(np.random.randn(num_sims, num_regions, num_steps),
                                    axis=2)  # Random walk (Brownian motion)
        data_close_to_0 = np.random.randn(num_sims, num_regions, num_steps)  # White noise
        data_close_to_minus_1 = np.array([
            np.sin(2 * np.pi * (i + 1) * np.arange(num_steps) / num_steps) for i in range(num_sims * num_regions)
        ]).reshape(num_sims, num_regions, num_steps)  # Sine waves with different frequencies

        return data_close_to_1, data_close_to_0, data_close_to_minus_1


    # Test the computeDFA_gpu function with generated data
    def test_computeDFA():
        num_sims, num_regions, num_steps = 32, 62, 250  # Define the shape of the test data

        # Generate test data
        data_close_to_1, data_close_to_0, data_close_to_minus_1 = generate_test_data(num_sims, num_regions, num_steps)

        # Compute DFA for each test case
        alpha_1 = computeDFA_gpu(data_close_to_1)
        alpha_0 = computeDFA_gpu(data_close_to_0)
        alpha_minus_1 = computeDFA_gpu(data_close_to_minus_1)

        # Print the results for inspection

        # Random Walk (Brownian Motion, H ≈ 0.5)
        # A random walk also has a Hurst exponent near 0.5.
        # However, because each step is based on a cumulative sum of white noise,
        # it has some apparent "momentum."
        print("DFA exponents close to 1 (random walk):", alpha_1)
        # White noise is a purely random process with no memory or persistence;
        # it has a Hurst exponent close to 0.5.
        print("DFA exponents close to 0 (white noise):", alpha_0)
        # Replaced by generate_anti_correlated_signal function
        print("DFA exponents close to -1 (sine wave):", alpha_minus_1)


    def generate_anti_correlated_signal(num_sims, num_regions, num_steps, theta=0.5, mu=0, sigma=0.5):
        """
        Generate anti-correlated signals using an Ornstein-Uhlenbeck process for DFA testing.

        Parameters:
        - num_sims: Number of simulations
        - num_regions: Number of regions
        - num_steps: Length of time series
        - theta: Speed of mean reversion
        - mu: Long-term mean
        - sigma: Volatility (controls amplitude of fluctuations)

        Returns:
        - data: Anti-correlated signal array with shape (num_sims, num_regions, num_steps)
        """
        dt = 1  # Discrete time step
        data = np.zeros((num_sims, num_regions, num_steps), dtype=np.float32)

        for sim in range(num_sims):
            for region in range(num_regions):
                x = np.zeros(num_steps, dtype=np.float32)
                x[0] = np.random.normal(mu, sigma)  # Start with random initial value

                # Generate the OU process
                for t in range(1, num_steps):
                    x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * np.random.normal()

                data[sim, region, :] = x

        return data

    # test_computeDFA()

    # EEG data shape (65, 129, 4001)
    # FIle example 1-102-1-G
    # 1- site
    # 102- subjective
    # 1- which visit
    # G- random letter no meaning
    #
    # EO Eyes open
    # EC Closed
    #
    # num_sims = 65
    # num_regions = 129
    # num_steps = 4001
    # anti_correlated_data = generate_anti_correlated_signal(num_sims, num_regions, num_steps, theta=1)
    # print(anti_correlated_data)

    # load EEG data
    def EEG_file():
        filepointer = ('/tsd/p3139/data/durable/AI-Mind-data-2025-03-eBRAIN-Health-subset/2025-01-31_14-43-20-897314/'
                       'building/1-379/1-379-1-Z/sensors/1-379-1-Z_1-EO_eeg.fif')
        epochs = mne.read_epochs(filepointer).get_data().transpose(0,2,1)

        return epochs

    n_channels_forsim = 61  # berlin subjec
    readfile = 'random'
    if readfile == 'fif':

        data_root = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
        filepointer = data_root + 'alpha_eeg_like.fif'
        data = mne.read_epochs(filepointer).pick_types(eeg=True).get_data()

        data = data[:30, :n_channels_forsim, :]

    elif readfile == 'random':

        data = np.random.rand(10, n_channels_forsim, 2000).astype(np.float32)

    elif readfile == 'tvb':

        here = os.path.dirname(os.path.abspath(__file__))
        data = np.load(here + "/../output/LB_tavg222222_5k.npy")[:,0].T

    elif readfile == 'antic':

        num_sims = 10
        num_regions = n_channels_forsim
        num_steps = 2000
        data = generate_anti_correlated_signal(num_sims, num_regions, num_steps)


    print('data.shape', data.shape)

    def check_nans(label, arr):
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0:
            print(f"⚠️ {label}: {nan_count} NaNs detected")
        if inf_count > 0:
            print(f"⚠️ {label}: {inf_count} Infs detected")
        if nan_count == 0 and inf_count == 0:
            print(f"✅ {label}: No NaNs or Infs")


    check_nans("dfanans", data[:,:,1000:])


    nsims = 256
    num_steps = 4001
    num_regions = 68
    window_sizes = np.array([4, 8, 16, 32, 64, 128, 256], dtype=np.int32)


    # # Call the PyCUDA DFA computation
    alphas, log_s, log_F = computeDFA_gpu(data, None, 0, window_sizes)
    # print(log_F)

    # log_s = np.mean(log_s, axis=(0,1))
    # log_F = np.mean(log_F, axis=(0,1))
    # alphas2 = np.polyfit(log_s, log_F, 1)[0]

    print('log_s', log_s.shape)
    print('log_F', log_F.shape)
    print("Alphas shape:", alphas.shape)


# from tvbgpu.plotting.plot_dashbord import *
    #
    # plot_dfa_loglog(alphas, log_s, log_F, 9)


