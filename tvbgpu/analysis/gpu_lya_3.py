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


import time

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os



def computeLYA_gpu(timeseries, logger, mpirank, emb_dim=3, lag=1, min_tsep=2, trajectory_len=20, tau=1):

    nsims, nregions, ntimesteps = timeseries.shape  # your timeseries: shape (nsims, nregions, ntimesteps)
    logger.debug('Lya timeseries shape %s', timeseries.shape)
    logger.debug('Trajectory_len %d', trajectory_len)

    # compute m
    m = int(ntimesteps - (emb_dim - 1) * lag)
    logger.debug('m %d', m)
    assert m > 0 and trajectory_len <= m
    nslices = nsims * nregions

    # ---------- build embeddings on host ----------
    # emb_data shape: (nslices, m, emb_dim) flattened as contiguous float32
    emb_data = np.zeros((nslices, m, emb_dim), dtype=np.float32)
    for s in range(nsims):
        for r in range(nregions):
            base_idx = s * nregions + r
            x = timeseries[s, r, :]  # 1D array length ntimesteps
            # delay embedding
            for i in range(m):
                for d in range(emb_dim):
                    emb_data[base_idx, i, d] = x[i + d * lag]

    # flatten to 1D contiguous
    emb_data_flat = emb_data.ravel().astype(np.float32)

    # ---------- compile module ----------
    cuda_source = """
        
        #include <math.h>
        #include <float.h>
        
        extern "C" {
        
            __device__ inline float rowwise_euclidean_sq(const float* a, const float* b, int emb_dim) {
                float sum = 0.0f;
                for (int d = 0; d < emb_dim; ++d) {
                    float diff = a[d] - b[d];
                    sum += diff * diff;
                }
                return sum;
            }
            
            // --- Kernel 1: tiled nearest-neighbour using shared memory ---
            // One block per (sim,region)
            // blockDim.x = TILE (threads per block)
            // shared mem size = TILE * emb_dim * sizeof(float)
            __global__ void nn_tiled_shared(
                const float* emb_data,   // shape: (nslices * m * emb_dim), where nslices = nsims * nregions
                int* nn_indices,         // output: shape nslices * m (int)
                int nslices, int m, int emb_dim,
                int min_tsep
            ) {
                int sim_region_idx = blockIdx.x + gridDim.x * blockIdx.y; // allow 2D grid mapping if necessary
                if (sim_region_idx >= nslices) return;
            
                const float* emb_base = emb_data + (size_t)sim_region_idx * (size_t)m * (size_t)emb_dim;
                int* nn_base = nn_indices + (size_t)sim_region_idx * (size_t)m;
            
                const int TILE = blockDim.x;
                extern __shared__ float s_tile_j[]; // TILE * emb_dim floats
            
                // Each thread will handle multiple i by striding over i_base tiles
                for (int i_base = 0; i_base < m; i_base += TILE) {
                    int i = i_base + threadIdx.x;
                    if (i >= m) continue;
            
                    // load vec_i into registers for reuse across j-tiles
                    // careful: emb_dim may be dynamic; use small local array if emb_dim is small.
                    // We'll load into local array by manual loop
                    // Note: For large emb_dim this will use registers; if emb_dim too large, reduce TILE.
                    // Copy vec_i into regs
                    extern __shared__ float dummy[]; // no-op to avoid compiler warnings when emb_dim zero (not used)
                    // We'll use stack array with dynamic indexing (compilers may unroll if emb_dim small)
                    // Because emb_dim is runtime, we load on the fly inside j-tile loop but read vec_i once per j-tile:
                    float vec_i_local[64]; // assume emb_dim <= 64; if not, reduce TILE or adjust.
                    // Guard against large emb_dim
                    int D = emb_dim;
                    if (D > 64) { /* fall back to direct loads; still works */ }
                    for (int d = 0; d < D && d < 64; ++d) {
                        vec_i_local[d] = emb_base[(size_t)i * emb_dim + d];
                    }
            
                    float my_min_sq = FLT_MAX;
                    int my_min_idx = -1;
            
                    // scan all j tiles
                    for (int j_base = 0; j_base < m; j_base += TILE) {
                        // cooperative load j-tile into shared memory
                        for (int p = threadIdx.x; p < TILE; p += blockDim.x) {
                            int j_idx = j_base + p;
                            float* dst = s_tile_j + (size_t)p * emb_dim;
                            if (j_idx < m) {
                                const float* src = emb_base + (size_t)j_idx * emb_dim;
                                for (int d = 0; d < emb_dim; ++d) dst[d] = src[d];
                            } else {
                                for (int d = 0; d < emb_dim; ++d) dst[d] = 0.0f;
                            }
                        }
                        __syncthreads();
            
                        // compute distances from vec_i to rows in shared tile
                        for (int p = 0; p < TILE; ++p) {
                            int j_idx = j_base + p;
                            if (j_idx >= m) break;
                            int dt = i - j_idx;
                            if (dt < 0) dt = -dt;
                            if (dt <= min_tsep) continue; // Theiler exclusion
                            // compute squared distance using shared memory row
                            const float* vec_j_s = s_tile_j + (size_t)p * emb_dim;
                            float sum = 0.0f;
                            // try to use vec_i_local if emb_dim <= 64
                            if (emb_dim <= 64) {
                                for (int d = 0; d < emb_dim; ++d) {
                                    float diff = vec_i_local[d] - vec_j_s[d];
                                    sum += diff * diff;
                                }
                            } else {
                                // fallback: load vec_i directly
                                const float* vec_i_ptr = emb_base + (size_t)i * emb_dim;
                                for (int d = 0; d < emb_dim; ++d) {
                                    float diff = vec_i_ptr[d] - vec_j_s[d];
                                    sum += diff * diff;
                                }
                            }
                            if (sum < my_min_sq) {
                                my_min_sq = sum;
                                my_min_idx = j_idx;
                            }
                        }
                        __syncthreads();
                    } // end j tiles
            
                    nn_base[i] = my_min_idx;
                } // end i_base
            }
            
            // --- Kernel 2: divergence and regression from precomputed nn_indices ---
            // One thread per slice (sim_region_idx) - similar to your original compute_lyapunov but
            // it uses squared distances and 0.5f * logf(dist2) to get log(dist).
            __global__ void divergence_from_nn(
                const float* emb_data,    // nslices * m * emb_dim
                const int* nn_indices,    // nslices * m
                float* div_traj,          // nslices * trajectory_len
                float* results,           // nslices
                float* divergence_curves, // nslices
                int nslices, int m, int emb_dim,
                int trajectory_len, float tau
            ) {
                int sim_region_idx = blockIdx.x + gridDim.x * blockIdx.y;
                if (sim_region_idx >= nslices) return;
            
                const float* emb_base = emb_data + (size_t)sim_region_idx * (size_t)m * (size_t)emb_dim;
                const int* nn_base = nn_indices + (size_t)sim_region_idx * (size_t)m;
                float* div_base = div_traj + (size_t)sim_region_idx * (size_t)trajectory_len;
            
                int ntraj = m - trajectory_len + 1;
            
                for (int k = 0; k < trajectory_len; ++k) {
                    float sum_log = 0.0f;
                    int count = 0;
                    for (int i = 0; i < ntraj; ++i) {
                        int j = nn_base[i];
                        if (j >= 0 && (i + k) < m && (j + k) < m) {
                            const float* vi = emb_base + (size_t)(i + k) * emb_dim;
                            const float* vj = emb_base + (size_t)(j + k) * emb_dim;
                            float dist2 = rowwise_euclidean_sq(vi, vj, emb_dim);
                            if (dist2 > 1e-20f) {
                                // log(dist) = 0.5 * log(dist^2)
                                sum_log += 0.5f * logf(dist2);
                                count++;
                            }
                        }
                    }
                    div_base[k] = (count > 0) ? (sum_log / count) : -FLT_MAX;
                    divergence_curves[(sim_region_idx * trajectory_len) + k] = div_base[k];
                }
            
                // linear regression (slope of <ln d(k)> vs k)
                float sum_k = 0.0f, sum_div = 0.0f, sum_kdiv = 0.0f, sum_k2 = 0.0f;
                int valid = 0;
                for (int k = 0; k < trajectory_len; ++k) {
                    float v = div_base[k];
                    if (isfinite(v)) {
                        sum_k += (float)k;
                        sum_div += v;
                        sum_kdiv += (float)k * v;
                        sum_k2 += (float)k * (float)k;
                        valid++;
                    }
                }
                float denom = (valid * sum_k2 - sum_k * sum_k);
                float slope = (denom != 0.0f) ? (valid * sum_kdiv - sum_k * sum_div) / denom : 0.0f;
                results[sim_region_idx] = slope / tau;
            }
        
        } // extern "C"
    
        """
    mod = SourceModule(cuda_source)

    nn_kernel = mod.get_function("nn_tiled_shared")
    div_kernel = mod.get_function("divergence_from_nn")

    # ---------- allocate device memory ----------
    emb_data_gpu = cuda.mem_alloc(emb_data_flat.nbytes)
    cuda.memcpy_htod(emb_data_gpu, emb_data_flat)

    nn_indices_gpu = cuda.mem_alloc(nslices * m * np.int32().nbytes)
    # initialize to -1
    nn_indices_init = -np.ones((nslices, m), dtype=np.int32)
    cuda.memcpy_htod(nn_indices_gpu, nn_indices_init)

    div_traj_gpu = cuda.mem_alloc(nslices * trajectory_len * np.float32().nbytes)
    # optional initialize
    cuda.memset_d32(int(div_traj_gpu), 0x80000000, nslices * trajectory_len)  # set to NaN-like (optional)

    n_total = nslices * trajectory_len
    divergence_curves_gpu = cuda.mem_alloc(nslices * trajectory_len * np.float32().nbytes)
    # initialize to very negative values (same as before)
    cuda.memset_d32(int(divergence_curves_gpu), 0x80000000, n_total)

    results_gpu = cuda.mem_alloc(nslices * np.float32().nbytes)
    # optional zero
    cuda.memset_d32(int(results_gpu), 0, nslices)

    # ---------- launch nn_tiled_shared ----------
    TILE = 32  # tune: 16, 32, 64 depending on emb_dim and shared mem
    block = (TILE, 1, 1)
    # grid mapping: use grid.x = nslices, grid.y = 1 (we used sim_region_idx = blockIdx.x + gridDim.x*blockIdx.y)
    # but easier: set grid.x=nslices and grid.y=1
    grid_x = int(nslices)
    grid = (grid_x, 1, 1)

    shared_bytes = TILE * emb_dim * np.dtype(np.float32).itemsize
    # call
    nn_kernel(
        emb_data_gpu, nn_indices_gpu,
        np.int32(nslices), np.int32(m), np.int32(emb_dim),
        np.int32(min_tsep),
        block=block, grid=grid, shared=shared_bytes
    )

    # ---------- launch divergence kernel ----------
    # use same grid layout
    div_kernel(
        emb_data_gpu, nn_indices_gpu, div_traj_gpu, results_gpu, divergence_curves_gpu,
        np.int32(nslices), np.int32(m), np.int32(emb_dim),
        np.int32(trajectory_len), np.float32(tau),
        block=(1,1,1), grid=(grid_x,1,1)
    )

    # ---------- copy results back ----------
    results = np.empty(nslices, dtype=np.float32)
    cuda.memcpy_dtoh(results, results_gpu)
    results = results.reshape((nsims, nregions))

    divergence_curves = np.empty(n_total, dtype=np.float32)
    cuda.memcpy_dtoh(divergence_curves, divergence_curves_gpu)
    divergence_curves = divergence_curves.reshape((nsims, nregions, trajectory_len))

    # div_traj if needed
    div_traj = np.empty((nslices, trajectory_len), dtype=np.float32)
    cuda.memcpy_dtoh(div_traj, div_traj_gpu)
    div_traj = div_traj.reshape((nsims, nregions, trajectory_len))

    return results, divergence_curves

if __name__ == "__main__":

    # timeseries = np.random.rand(60, 61, 2000).astype(np.float32)
    filename = '/../output/randomtestdata.npy'  # + timestring
    here = os.path.dirname(os.path.abspath(__file__))
    timeseries = np.load(here + filename, allow_pickle=True)

    emb_dim = 8
    lag = 8
    min_tsep = 16
    trajectory_len = 80
    tau = 0.001  # sampling period in seconds (example)

    logger = None
    mpirank = 0

    tic = time.time()
    results, divergence_curves = computeLYA_gpu(np.ascontiguousarray(timeseries), logger, mpirank, emb_dim, lag, min_tsep, trajectory_len, tau)
    toc = time.time()
    print('time for analy %.2f' % (toc - tic))

    print("results (lambda):", results[0][0])

    from tvbgpu.plotting.plot_dashbord import *

    plot_lya_divergence(divergence_curves)

