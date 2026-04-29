//
// Created by seyda on 9/14/24.
//

#ifndef FIDESLIB_ROTATION_CUH
#define FIDESLIB_ROTATION_CUH

#include "ConstantsGPU.cuh"
#include "Math.cuh"
#include <iostream>

namespace FIDESlib::CKKS {

__device__ __forceinline__ uint32_t automorph_slot(const int n_bits, const int index, const uint32_t slot) {
	uint32_t j = slot;

	j = __brev(j) >> (32 - n_bits);

	uint32_t jTmp	  = (j << 1) + 1;
	uint32_t rotIndex = ((jTmp * index) & ((1 << (n_bits + 1)) - 1)) >> 1;

	// Bit reversal:
	rotIndex = __brev(rotIndex) >> (32 - n_bits);

	return rotIndex;
}

template <typename T> __device__ __forceinline__ void automorph__(T* a, T* a_rot, const int n, const int n_bits, const int index, const int br) {
	uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t rotIndex = automorph_slot(n_bits, index, j);

	a_rot[rotIndex] = a[j];
}

template <typename T> __global__ void automorph_(T* a, T* a_rot, const int index, const int br);

__global__ void automorph_multi_(void** a, void** a_rot, const int k, const int br, const int primeid_init);

} // namespace FIDESlib::CKKS

#endif // FIDESLIB_ROTATION_CUH