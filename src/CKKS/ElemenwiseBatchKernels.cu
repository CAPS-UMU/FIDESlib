//
// Created by carlosad on 27/09/24.
//

#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/Rescale.cuh"

namespace FIDESlib {
namespace CKKS {
__global__ void mult1AddMult23Add4_(const __grid_constant__ int primeid_init, void** l, void** l1, void** l2, void** l3,
                                    void** l4) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T aux = ((T*)l4[blockIdx.y])[idx];
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx], ((T*)l1[blockIdx.y])[idx], primeid);
        //res = modadd(res, aux, primeid);
        res = modadd(res, modmult<algo>(((T*)l2[blockIdx.y])[idx], ((T*)l3[blockIdx.y])[idx], primeid), primeid);
        res = modmult(res, C_.P[primeid], primeid);
        T opt = modadd(res, aux, primeid);
        ((T*)l4[blockIdx.y])[idx] = opt;
    } else {
        using T = uint32_t;
    }
}

__global__ void mult1Add2_(const __grid_constant__ int primeid_init, void** l, void** l1, void** l2) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T aux = ((T*)l2[blockIdx.y])[idx];
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx], ((T*)l1[blockIdx.y])[idx], primeid);
        res = modmult(res, C_.P[primeid], primeid);
        ((T*)l2[blockIdx.y])[idx] = modadd(res, aux, primeid);
    } else {
        using T = uint32_t;
        T aux = ((T*)l2[blockIdx.y])[idx];
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx], ((T*)l1[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modadd(res, aux, primeid);
    }
}

template <typename T>
__device__ __forceinline__ void addMult__(T* l, const T* l1, const T* l2, const int primeid) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    l[idx] = modadd(l[idx], modmult<algo>(l1[idx], l2[idx], primeid), primeid);
}

template <typename T>
__global__ void addMult_(T* l, const T* l1, const T* l2, const __grid_constant__ int primeid) {
    addMult__<T>(l, l1, l2, primeid);
}

__global__ void addMult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

    //    if (threadIdx.x + blockDim.x * blockIdx.x == 0)
    //        printf("%d %d\n", primeid_init + blockIdx.y, primeid);
    if (ISU64(primeid)) {
        addMult__<uint64_t>((uint64_t*)l[blockIdx.y], (uint64_t*)l1[blockIdx.y], (uint64_t*)l2[blockIdx.y], primeid);
    } else {
        addMult__<uint32_t>((uint32_t*)l[blockIdx.y], (uint32_t*)l1[blockIdx.y], (uint32_t*)l2[blockIdx.y], primeid);
    }
}

__global__ void Mult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    //    if (idx == 0)
    //        printf("%d %d\n", primeid_init + blockIdx.y, primeid);
    if (ISU64(primeid)) {
        ((uint64_t*)l[blockIdx.y])[idx] =
            modmult<ALGO_BARRETT>(((uint64_t*)l1[blockIdx.y])[idx], ((uint64_t*)l2[blockIdx.y])[idx], primeid);
    } else {
        ((uint32_t*)l[blockIdx.y])[idx] =
            modmult<ALGO_BARRETT>(((uint32_t*)l1[blockIdx.y])[idx], ((uint32_t*)l2[blockIdx.y])[idx], primeid);
    }
}

__global__ void square_(void** l, void** l1, const __grid_constant__ int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (ISU64(primeid)) {
        uint64_t in = ((uint64_t*)l1[blockIdx.y])[idx];
        ((uint64_t*)l[blockIdx.y])[idx] = modmult<ALGO_BARRETT>(in, in, primeid);
    } else {
        uint32_t in = ((uint64_t*)l1[blockIdx.y])[idx];
        ((uint32_t*)l[blockIdx.y])[idx] = modmult<ALGO_BARRETT>(in, in, primeid);
    }
};

__global__ void binomial_square_fold_(void** c0_res, void** c2_key_switched_0, void** c1, void** c2_key_switched_1,
                                      const __grid_constant__ int primeid_init) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

    if (ISU64(primeid)) {
        uint64_t in2_0 = ((uint64_t*)c2_key_switched_0[blockIdx.y])[idx];
        uint64_t in2_1 = ((uint64_t*)c2_key_switched_1[blockIdx.y])[idx];
        uint64_t in0 = ((uint64_t*)c0_res[blockIdx.y])[idx];
        uint64_t ok = modadd(modmult<ALGO_BARRETT>(in0, in0, primeid), in2_0, primeid);
        ((uint64_t*)c0_res[blockIdx.y])[idx] = ok;
        uint64_t in1 = ((uint64_t*)c1[blockIdx.y])[idx];
        uint64_t aux = modmult<ALGO_BARRETT>(in0, in1, primeid);
        uint64_t aux2 = modadd(aux, aux, primeid);
        ok = modadd(aux2, in2_1, primeid);
        ((uint64_t*)c1[blockIdx.y])[idx] = ok;
    } else {
    }
}

__global__ void broadcastLimb0_(void** a) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = blockIdx.y + 1;
    if (ISU64(primeid) && ISU64(0)) {
        uint64_t in = ((uint64_t*)a[0])[idx];
        SwitchModulus(in, 0, primeid);
        ((uint64_t*)a[primeid])[idx] = in;
    }
}

__global__ void copy_(void** a, void** b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (ISU64(blockIdx.y)) {
        ((uint64_t*)b[blockIdx.y])[idx] = ((uint64_t*)a[blockIdx.y])[idx];
    } else {
        ((uint32_t*)b[blockIdx.y])[idx] = ((uint32_t*)a[blockIdx.y])[idx];
    }
}

template <ALGO algo>
__global__ void Scalar_mult_(void** a, const uint64_t* b, const __grid_constant__ int primeid_init,
                             const uint64_t* shoup_mu) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] =
            modmult<algo>(((uint64_t*)a[blockIdx.y])[idx], b[blockIdx.y], primeid, shoup_mu ? shoup_mu[blockIdx.y] : 0);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] = modmult<algo>(((uint32_t*)a[blockIdx.y])[idx], (uint32_t)b[blockIdx.y],
                                                        primeid, (uint32_t)(shoup_mu ? shoup_mu[blockIdx.y] : 0));
    }
}

__global__ void eval_linear_w_sum_(const __grid_constant__ int n, void** a, void*** bs, uint64_t* w) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = blockIdx.y;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        uint64_t res = modmult<algo>(((uint64_t*)(bs[0])[blockIdx.y])[idx], w[blockIdx.y], primeid);
        for (int i = 1; i < n; ++i) {
            uint64_t temp =
                modmult<algo>(((uint64_t*)(bs[i])[blockIdx.y])[idx], w[i * gridDim.y + blockIdx.y], primeid);
            res = modadd(res, temp, primeid);
        }
        ((uint64_t*)a[blockIdx.y])[idx] = res;
    }
}

}  // namespace CKKS
}  // namespace FIDESlib

#define YY(algo)                                                 \
    template __global__ void FIDESlib::CKKS::Scalar_mult_<algo>( \
        void** a, const uint64_t* b, const __grid_constant__ int primeid_init, const uint64_t* shoup_mu);
#include "ntt_types.inc"
#undef YY

template __global__ void FIDESlib::CKKS::addMult_<uint64_t>(uint64_t* l, const uint64_t* l1, const uint64_t* l2,
                                                            const __grid_constant__ int primeid);

template __global__ void FIDESlib::CKKS::addMult_<uint32_t>(uint32_t* l, const uint32_t* l1, const uint32_t* l2,
                                                            const __grid_constant__ int primeid);
