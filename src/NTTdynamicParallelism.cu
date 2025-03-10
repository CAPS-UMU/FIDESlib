//
// Created by carlosad on 21/1ALGO_NATIVE/24.
//
#include <cassert>

#include "ConstantsGPU.cuh"
#include "NTTdynamicParallelism.cuh"
namespace FIDESlib {

template <typename T, ALGO algo, NTT_MODE mode>
__global__ void device_launch_NTT(const __grid_constant__ int logN, const __grid_constant__ int primeid, T* dat, T* aux,
                                  T* res, const __grid_constant__ T q_L_inv, const T* pt,
                                  const __grid_constant__ int primeid_rescale) {
    assert(primeid >= 0);
    assert(primeid_rescale >= 0);
    assert(blockDim.x == 1);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.x == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    assert(dat != nullptr);
    assert(aux != nullptr);
    assert(res != nullptr);

    constexpr int M = sizeof(T) == 8 ? 4 : 8;
    const int size = 1 << logN;

    const int bytes_per_thread = sizeof(T) * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
    {
        const int auxlog = (((logN + 1) >> 1) - 1);
        dim3 blockDim = 1U << auxlog;
        dim3 gridDim{(uint32_t)size >> (auxlog + (sizeof(T) == 8U ? 3 : 4))};
        const int bytes = (int)blockDim.x * bytes_per_thread;
        FIDESlib::NTT_<T, false, algo, mode>
            <<<gridDim, blockDim, bytes, cudaStreamTailLaunch>>>(dat, primeid, aux, nullptr);
    }

    {
        const int auxlog = (((logN - 0) >> 1) - 1);
        dim3 blockDim = 1U << auxlog;
        dim3 gridDim{(uint32_t)size >> (auxlog + (sizeof(T) == 8U ? 3 : 4))};
        const int bytes = (int)blockDim.x * bytes_per_thread;
        FIDESlib::NTT_<T, true, algo, mode><<<gridDim, blockDim, bytes, cudaStreamTailLaunch>>>(aux, primeid, res, pt);
    }
}

#define VV(T, algo, mode)                                                                        \
    template __global__ void device_launch_NTT<T, algo, mode>(                                   \
        int __grid_constant__ logN, const int __grid_constant__ primeid, T* dat, T* aux, T* res, \
        const T __grid_constant__ q_L_inv, const T* pt, const int __grid_constant__ primeid_rescale);
#include "ntt_types.inc"
#undef VV

template <ALGO algo, NTT_MODE mode>
__global__ void device_launch_batch_NTT(const __grid_constant__ int logN, const __grid_constant__ int primeid_init,
                                        void** dat, void** aux, void** res) {
    assert(primeid_init >= 0);
    assert(blockDim.x == 1);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(primeid_init + gridDim.x <= C_.L + C_.K);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);
    assert(dat != nullptr);
    assert(aux != nullptr);
    assert(res != nullptr);

    const int primeid = primeid_init + blockIdx.x;
    if (ISU64(primeid)) {
        /*
        device_launch_NTT<uint64_t, algo, mode><<<1, 1>>>(logN, primeid, (uint64_t *) dat[blockIdx.x],
                                                          (uint64_t *) aux[blockIdx.x],
                                                          (uint64_t *) res[blockIdx.x], 0,
                                                          nullptr, 0);
                                                          */
        using T = uint64_t;
        constexpr int M = sizeof(T) == 8 ? 4 : 8;
        const int size = 1 << logN;

        const int bytes_per_thread = sizeof(T) * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
        {
            const int auxlog = (((logN + 1) >> 1) - 1);
            dim3 blockDim = 1U << auxlog;
            dim3 gridDim{(uint32_t)size >> (auxlog + (sizeof(T) == 8U ? 3 : 4))};
            const int bytes = (int)blockDim.x * bytes_per_thread;
            FIDESlib::NTT_<T, false, algo, mode><<<gridDim, blockDim, bytes>>>((uint64_t*)dat[blockIdx.x], primeid,
                                                                               (uint64_t*)aux[blockIdx.x], nullptr);
        }

        {
            const int auxlog = (((logN - 0) >> 1) - 1);
            dim3 blockDim = 1U << auxlog;
            dim3 gridDim{(uint32_t)size >> (auxlog + (sizeof(T) == 8U ? 3 : 4))};
            const int bytes = (int)blockDim.x * bytes_per_thread;
            FIDESlib::NTT_<T, true, algo, mode><<<gridDim, blockDim, bytes>>>((uint64_t*)aux[blockIdx.x], primeid,
                                                                              (uint64_t*)res[blockIdx.x], nullptr);
        }
    } else {
        /*
        device_launch_NTT<uint32_t, algo, mode><<<1, 1>>>(logN, primeid, (uint32_t *) dat[blockIdx.x],
                                                          (uint32_t *) aux[blockIdx.x],
                                                          (uint32_t *) res[blockIdx.x], 0,
                                                          nullptr, 0);
        */
        using T = uint32_t;
        constexpr int M = sizeof(T) == 8 ? 4 : 8;
        const int size = 1 << logN;

        const int bytes_per_thread = sizeof(T) * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
        {
            const int auxlog = (((logN + 1) >> 1) - 1);
            dim3 blockDim = 1U << auxlog;
            dim3 gridDim{(uint32_t)size >> (auxlog + (sizeof(T) == 8U ? 3 : 4))};
            const int bytes = (int)blockDim.x * bytes_per_thread;
            FIDESlib::NTT_<T, false, algo, mode><<<gridDim, blockDim, bytes>>>((uint32_t*)dat[blockIdx.x], primeid,
                                                                               (uint32_t*)aux[blockIdx.x], nullptr);
        }

        {
            const int auxlog = (((logN - 0) >> 1) - 1);
            dim3 blockDim = 1U << auxlog;
            dim3 gridDim{(uint32_t)size >> (auxlog + (sizeof(T) == 8U ? 3 : 4))};
            const int bytes = (int)blockDim.x * bytes_per_thread;
            FIDESlib::NTT_<T, true, algo, mode><<<gridDim, blockDim, bytes>>>((uint32_t*)aux[blockIdx.x], primeid,
                                                                              (uint32_t*)res[blockIdx.x], nullptr);
        }
    }
}

#define YYY(algo, mode)                                                                                                \
    template __global__ void device_launch_batch_NTT<algo, mode>(const __grid_constant__ int logN,                     \
                                                                 const __grid_constant__ int primeid_init, void** dat, \
                                                                 void** aux, void** res);

#include "ntt_types.inc"
#undef YYY

template <typename T, ALGO algo>
__global__ void device_launch_INTT(const __grid_constant__ int logN, const __grid_constant__ int primeid, T* dat,
                                   T* aux, T* res) {
    assert(primeid >= 0);
    assert(blockDim.x == 1);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.x == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);

    assert(dat != nullptr);
    assert(aux != nullptr);
    assert(res != nullptr);

    constexpr int M = sizeof(T) == 8 ? 4 : 8;
    const int size = 1 << logN;

    {
        dim3 blockDim{(uint32_t)(1 << ((logN) / 2 - 1))};
        dim3 gridDim{size / blockDim.x / 2 / M};
        const int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
        FIDESlib::INTT_<T, false, algo><<<gridDim, blockDim, bytes>>>(dat, primeid, aux);
    }

    {
        dim3 blockDim = (1 << ((logN + 1) / 2 - 1));
        dim3 gridDim = {size / blockDim.x / 2 / M};
        const int bytes = sizeof(T) * blockDim.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
        FIDESlib::INTT_<T, true, algo><<<gridDim, blockDim, bytes>>>(aux, primeid, res);
    }
}

#define Y(T, algo)                                                                   \
    template __global__ void device_launch_INTT<T, algo>(int __grid_constant__ logN, \
                                                         const int __grid_constant__ primeid, T* dat, T* aux, T* res);
#include "ntt_types.inc"
#undef Y

template <ALGO algo>
__global__ void device_launch_batch_INTT(const __grid_constant__ int logN, const __grid_constant__ int primeid_init,
                                         void** dat, void** aux, void** res) {
    assert(primeid_init >= 0);
    assert(blockDim.x == 1);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(primeid_init + gridDim.x <= C_.L + C_.K);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);
    assert(dat != nullptr);
    assert(aux != nullptr);
    assert(res != nullptr);

    const int primeid = primeid_init + blockIdx.x;
    if (ISU64(primeid)) {
        device_launch_INTT<uint64_t, algo><<<1, 1>>>(logN, primeid, (uint64_t*)dat[blockIdx.x],
                                                     (uint64_t*)aux[blockIdx.x], (uint64_t*)res[blockIdx.x]);
    } else {
        device_launch_INTT<uint32_t, algo><<<1, 1>>>(logN, primeid, (uint32_t*)dat[blockIdx.x],
                                                     (uint32_t*)aux[blockIdx.x], (uint32_t*)res[blockIdx.x]);
    }
}

#define YY(algo)                                                                                                  \
    template __global__ void device_launch_batch_INTT<algo>(const __grid_constant__ int logN,                     \
                                                            const __grid_constant__ int primeid_init, void** dat, \
                                                            void** aux, void** res);
#include "ntt_types.inc"
#undef YY
}  // namespace FIDESlib