//
// Created by carlosad on 4/04/24.
//
#include "ModMult.cuh"

namespace FIDESlib {

template <typename T, ALGO algo> __global__ void mult_(T* a, const T* b, const int primeid, const __grid_constant__ int elem_b) {
	int idx             = threadIdx.x + blockIdx.x * blockDim.x;
	const int logstride = __ffs(blockDim.x * gridDim.x / elem_b) - 1;
	const int bidx      = idx >> logstride;

	a[idx] = modmult<algo>(a[idx], b[bidx], primeid);
}

template <typename T, ALGO algo> __global__ void mult_(T* a, const T* b, const T* c, const int primeid, const __grid_constant__ int elem_c) {
	int idx             = threadIdx.x + blockIdx.x * blockDim.x;
	const int logstride = __ffs(blockDim.x * gridDim.x / elem_c) - 1;
	const int cidx      = idx >> logstride;

	a[idx] = modmult<algo>(b[idx], c[cidx], primeid);
}

template <typename T, ALGO algo> __global__ void scalar_mult_(T* a, const T b, const int primeid, const T shoup_mu) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	a[idx]  = modmult<algo>(a[idx], b, primeid, shoup_mu);
}

#define Y(T, algo) template __global__ void mult_<T, algo>(T * a, const T* b, const int primeid, const __grid_constant__ int elem_b);

#include "ntt_types.inc"

#undef Y

#define Y(T, algo) template __global__ void mult_<T, algo>(T * a, const T* b, const T* c, const int primeid, const __grid_constant__ int elem_c);

#include "ntt_types.inc"

#undef Y

#define Y(T, algo) template __global__ void scalar_mult_<T, algo>(T * a, const T b, const int primeid, const T shoup_mu);

#include "ntt_types.inc"

#undef Y
} // namespace FIDESlib