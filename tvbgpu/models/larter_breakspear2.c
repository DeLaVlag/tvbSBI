/******************************************************************************
 * ============================================================
 * Project:     TVB EEG Pipeline / HPC Simulation Framework
 * Author:      Michiel van der Vlag
 * Institution: Forschungszentrum Juelich (FZJ)
 * Created:     2024 for EBRAINS-Health
 *
 * Description:
 * This project contains simulation pipelines, containerized
 * environments (Apptainer), and analysis workflows for
 * large-scale brain network modeling and data-driven
 * computational neuroscience. The framework focuses on
 * fitting EEG data to TVB simulations to augment and enhance
 * EEG-based machine learning pipelines.
 * ============================================================
 ******************************************************************************/


#ifndef NH
#define NH nh
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define STATELIM 20

#include <curand_kernel.h>
#include <curand.h>
#include <stdbool.h>
#include <math.h>

__device__ __forceinline__
void compute_derivatives(
    float Vx, float Wx, float Zx, float c0,
    float &dVx, float &dWx, float &dZx,
    float tau_K,
    // constants
    float gCa, float gK, float gL, float gNa,
    float TK, float TCa, float TNa,
    float VCa, float VK, float VL, float VNa,
    float d_K, float d_Na, float d_Ca,
    float aei, float aie, float b, float C,
    float ane, float ani, float aee,
    float Iext, float VT, float d_V,
    float ZT, float d_Z,
    float QV_max, float QZ_max,
    float t_scale,
    float rNMDA
)
{
    float arg;

    arg = (Vx - TCa) / d_Ca; arg = fminf(fmaxf(arg, -20.0f), 20.0f);
    float m_Ca = 0.5f * (1.0f + tanhf(arg));

    arg = (Vx - TNa) / d_Na; arg = fminf(fmaxf(arg, -20.0f), 20.0f);
    float m_Na = 0.5f * (1.0f + tanhf(arg));

    arg = (Vx - TK) / d_K; arg = fminf(fmaxf(arg, -20.0f), 20.0f);
    float m_K = 0.5f * (1.0f + tanhf(arg));

    arg = (Vx - VT) / d_V; arg = fminf(fmaxf(arg, -20.0f), 20.0f);
    float QV = 0.5f * QV_max * (1.0f + tanhf(arg));

    arg = (Zx - ZT) / d_Z; arg = fminf(fmaxf(arg, -20.0f), 20.0f);
    float QZ = 0.5f * QZ_max * (1.0f + tanhf(arg));

    float lc_0 = 0.0f;

    dVx = t_scale * (
        - (gCa + (1.0f - C) * (rNMDA * aee) * (QV + lc_0)
                + C * rNMDA * aee * c0) * m_Ca * (Vx - VCa)
        - gK * Wx * (Vx - VK)
        - gL * (Vx - VL)
        - (gNa * m_Na + (1.0f - C) * aee * (QV + lc_0)
                         + C * aee * c0) * (Vx - VNa)
        - aie * Zx * QZ
        + ane * Iext
    );

    dWx = t_scale * (aei * (m_K - Wx)) / tau_K;

    dZx = t_scale * b * (ani * Iext + aei * Vx * QV);
}

__device__ float wrap_it_V(float V) {
    if (V < -STATELIM) V = -STATELIM;
    else if (V > STATELIM) V = STATELIM;
    return V;
}

__device__ float wrap_it_W(float W) {
    if (W < -STATELIM) W = -STATELIM;
    else if (W > STATELIM) W = STATELIM;
    return W;
}

__device__ float wrap_it_Z(float Z) {
    if (Z < -STATELIM) Z = -STATELIM;
    else if (Z > STATELIM) Z = STATELIM;
    return Z;
}

__global__ void larter_breakspear(
        unsigned int i_step, unsigned int n_node, unsigned int nh, unsigned int n_step, unsigned int n_work_items,
        float dt, float conduct_speed, int my_rank,
        float * __restrict__ weights_pwi,
        float * __restrict__ lengths,
        float * __restrict__ params_pwi,
        float * __restrict__ state_pwi,
        float * __restrict__ tavg_pwi)
{
    const unsigned int id = (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    const unsigned int size = n_work_items;
    if (id >= size) return;

    float *weights = weights_pwi;

#define params(i_par) (params_pwi[(size * (i_par)) + id])
#define state(time, i_node) (state_pwi[((time) * 4 * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

    const float global_coupling = params(0);
    const float nsig = params(1);
    const float rNMDA = params(2);
    const float global_speed = params(3);
    const float tau_K = params(4);
    const float phi = params(5);
    const float d_V = params(6);

    float V = 0.0f, W = 0.0f, Z = 0.0f;
    float gCa=1.0f, gK=2.0f, gL=0.5f, gNa=6.7f;
    float TK=0.0f, TCa=-0.01f, TNa=0.3f;
    float VCa=1.0f, VK=-0.7f, VL=-0.5f, VNa=0.53f;
    float d_K=0.3f, d_Na=0.15f, d_Ca=0.15f;
    float aei=2.0f, aie=2.0f, b=0.1f, C=0.1f, ane=1.0f, ani=0.4f, aee=0.4f;
    float Iext=0.3f, VT=0.0f, ZT=0.0f, d_Z=0.65f;
    float QV_max=1.0f, QZ_max=1.0f, t_scale=1.0f;

    const float rec_speed_dt = 1.0f / global_speed / dt;

    curandState crndst;
    curand_init(id * 1234567u + (unsigned int)clock64(), 0, 0, &crndst);

    float etaV = 0;
    const float tau_eta = 0.05f;
    const float sigma_eta = sqrt(nsig);
    const float theta = 1.0f / tau_eta;

    for (unsigned int i_node = 0; i_node < n_node; i_node++) {
        tavg(i_node) = 0.0f;
        if (i_step == 0) {
            state(0, i_node + 0*n_node) = 0.001f*curand_normal(&crndst);
            state(0, i_node + 1*n_node) = 0.001f*curand_normal(&crndst);
            state(0, i_node + 2*n_node) = 0.001f*curand_normal(&crndst);
            state(0, i_node + 3*n_node) = 0.0f;
        }
    }

    int euler_integrated = 0;
    int heun_integrated = 1;

    for (unsigned int t = i_step; t < (i_step + n_step); t++) {
        for (int i_node = 0; i_node < n_node; i_node++) {
            float c_pop0 = 0.0f;

            V = wrap_it_V(state((t)%nh, i_node + 0*n_node));
            W = wrap_it_W(state((t)%nh, i_node + 1*n_node));
            Z = wrap_it_Z(state((t)%nh, i_node + 2*n_node));
            etaV = state((t)%nh, i_node + 3*n_node);

            unsigned int i_n = i_node * n_node;
            for (unsigned int j_node=0; j_node<n_node; j_node++) {
                float wij = weights[i_n+j_node];
                if (wij == 0.0f) continue;
                int dij_i = (int)(lengths[i_n+j_node]*rec_speed_dt + 0.5f);
                if (dij_i >= nh) dij_i = nh-1;
                float V_j = state((t-dij_i+nh)%nh, j_node + 0*n_node);
                c_pop0 += wij*V_j;
            }
            c_pop0 *= global_coupling;

            etaV += dt * (-theta*etaV) + sqrtf(2.0f*theta*dt)*sigma_eta*curand_normal(&crndst);

            /* ------------------- HEUN ------------------- */
            if (heun_integrated==1) {
                float k1V,k1W,k1Z, k2V,k2W,k2Z;

                compute_derivatives(V,W,Z,c_pop0,k1V,k1W,k1Z,
                    tau_K, gCa,gK,gL,gNa,TK,TCa,TNa,VCa,VK,VL,VNa,d_K,d_Na,d_Ca,
                    aei,aie,b,C,ane,ani,aee,Iext,VT,d_V,ZT,d_Z,QV_max,QZ_max,t_scale,rNMDA);

                float Vtmp=V + dt*k1V;
                float Wtmp=W + dt*k1W;
                float Ztmp=Z + dt*k1Z;

                compute_derivatives(Vtmp,Wtmp,Ztmp,c_pop0,k2V,k2W,k2Z,
                    tau_K, gCa,gK,gL,gNa,TK,TCa,TNa,VCa,VK,VL,VNa,d_K,d_Na,d_Ca,
                    aei,aie,b,C,ane,ani,aee,Iext,VT,d_V,ZT,d_Z,QV_max,QZ_max,t_scale,rNMDA);

                V = fmaf(0.5f*dt, k1V+k2V, V) + etaV;
                W = fmaf(0.5f*dt, k1W+k2W, W);
                Z = fmaf(0.5f*dt, k1Z+k2Z, Z);

                if (!isfinite(V)) V=0.0f;
                if (!isfinite(W)) W=0.0f;
                if (!isfinite(Z)) Z=0.0f;
            }

            /* ------------------- EULER ------------------- */
            else if (euler_integrated==1) {
                float k1V,k1W,k1Z;
                compute_derivatives(V,W,Z,c_pop0,k1V,k1W,k1Z,
                    tau_K, gCa,gK,gL,gNa,TK,TCa,TNa,VCa,VK,VL,VNa,d_K,d_Na,d_Ca,
                    aei,aie,b,C,ane,ani,aee,Iext,VT,d_V,ZT,d_Z,QV_max,QZ_max,t_scale,rNMDA);

                V += dt*k1V + etaV;
                W += dt*k1W;
                Z += dt*k1Z;

                if (!isfinite(V)) V=0.0f;
                if (!isfinite(W)) W=0.0f;
                if (!isfinite(Z)) Z=0.0f;
            }

            /* wrap after integration */
            V = wrap_it_V(V);
            W = wrap_it_W(W);
            Z = wrap_it_Z(Z);

            state((t+1)%nh,i_node+0*n_node)=V;
            state((t+1)%nh,i_node+1*n_node)=W;
            state((t+1)%nh,i_node+2*n_node)=Z;
            state((t+1)%nh,i_node+3*n_node)=etaV;

            tavg(i_node) += V / n_step;
        }
    }
}



// defaults from Stefan 2007, cf tvb/analyzers/fmri_balloon.py
#define TAU_S 0.65f
#define TAU_F 0.41f
#define TAU_O 0.98f
#define ALPHA 0.32f
#define TE 0.04f
#define V0 4.0f
#define E0 0.4f
#define EPSILON 0.5f
#define NU_0 40.3f
#define R_0 25.0f

#define RECIP_TAU_S (1.0f / TAU_S)
#define RECIP_TAU_F (1.0f / TAU_F)
#define RECIP_TAU_O (1.0f / TAU_O)
#define RECIP_ALPHA (1.0f / ALPHA)
#define RECIP_E0 (1.0f / E0)

// "derived parameters"
#define k1 (4.3f * NU_0 * E0 * TE)
#define k2 (EPSILON * R_0 * E0 * TE)
#define k3 (1.0f - EPSILON)

__global__ void bold_update(int n_node, float dt,
                      // bold.shape = (4, n_nodes, n_threads)
            float * __restrict__ bold_state,
                      // nrl.shape = (n_nodes, n_threads)
            float * __restrict__ neural_state,
                      // out.shape = (n_nodes, n_threads)
            float * __restrict__ out)
{
    const unsigned int it = (gridDim.x * blockDim.x * threadIdx.y) + threadIdx.x;
    const unsigned int nt = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

    int var_stride = n_node * nt;
    for (int i_node=0; i_node < n_node; i_node++)
    {
        float *node_bold = bold_state + i_node * nt + it;

        float s = node_bold[0 * var_stride];
        float f = node_bold[1 * var_stride];
        float v = node_bold[2 * var_stride];
        float q = node_bold[3 * var_stride];

        float x = neural_state[i_node * nt + it];

        float ds = x - RECIP_TAU_S * s - RECIP_TAU_F * (f - 1.0f);
        float df = s;
        float dv = RECIP_TAU_O * (f - pow(v, RECIP_ALPHA));
        float dq = RECIP_TAU_O * (f * (1.0f - pow(1.0f - E0, 1.0f / f))
                * RECIP_E0 - pow(v, RECIP_ALPHA) * (q / v));

        s += dt * ds;
        f += dt * df;
        v += dt * dv;
        q += dt * dq;

        node_bold[0 * var_stride] = s;
        node_bold[1 * var_stride] = f;
        node_bold[2 * var_stride] = v;
        node_bold[3 * var_stride] = q;

        out[i_node * nt + it] = V0 * (    k1 * (1.0f - q    )
                                        + k2 * (1.0f - q / v)
                                        + k3 * (1.0f -     v) );
    } // i_node
} // kernel
