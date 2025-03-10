//
// Created by seyda on 9/14/24.
//

#ifndef FIDESLIB_ROTATION_CUH
#define FIDESLIB_ROTATION_CUH

#include <iostream>
#include "ConstantsGPU.cuh"
#include "Math.cuh"

namespace FIDESlib::CKKS {

template <typename T>
__device__ __forceinline__ void automorph__(T* a, T* a_rot, const int n, const int n_bits, const int index,
                                            const int br) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    // const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Index shuffle:

    //uint32_t m    = n << 1;  // cyclOrder
    //uint32_t logm = std::round(log2(m));
    //uint32_t logn = std::round(log2(n));
    //for (uint32_t j = 0; j < n; j++) {
    //  uint32_t jTmp    = ((j << 1) + 1);
    //  usint idx        = ((jTmp * k) - (((jTmp * k) >> logm) << logm)) >> 1;
    //  usint jrev       = ReverseBits(j, logn);
    //  usint idxrev     = ReverseBits(idx, logn);
    //  (*precomp)[jrev] = idxrev;
    //}

    uint32_t jTmp = ((j << 1) + 1);
    uint32_t rotIndex = (((uint64_t)jTmp * index) - ((((uint64_t)jTmp * index) >> (n_bits + 1)) << (n_bits + 1))) >> 1;

    //printf("(%d %d) ", j, rotIndex);
    // Bit reversal:
    if (br == 1) {
        j = __brev(j) >> (32 - n_bits);
        rotIndex = __brev(rotIndex) >> (32 - n_bits);
    }
    a_rot[j] = a[rotIndex];
}

template <typename T>
__global__ void automorph_(T* a, T* a_rot, const int index, const int br);

__global__ void automorph_multi_(void** a, void** a_rot, const int k, const int br);

template <typename T>
__global__ void automorph_multi_ct(T**** a, T**** a_rot, const int n, const int n_bits, const int k,
                                   const int limb_count, const int ct_count, const int br);
}  // namespace FIDESlib::CKKS

#endif  //FIDESLIB_ROTATION_CUH