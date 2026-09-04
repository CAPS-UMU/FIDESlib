//
// Created by carlosad on 25/03/24.
//
#include "AddSub.cuh"
#include "ModMult.cuh"
#include <cassert>

namespace FIDESlib {

template <typename T> __global__ void add_(T* a, const T* b, const int primeId, const __grid_constant__ int elem_b) {
	const int idx       = (blockIdx.x * blockDim.x + threadIdx.x);
	const int logstride = __ffs(blockDim.x * gridDim.x / elem_b) - 1;
	const int bidx      = idx >> logstride;
	// if(threadIdx.x == 0 && blockIdx.x == 0) printf("Prime %d: %lu ", primeId, p_prime);
	//   if(threadIdx.x == 0 && blockIdx.x == 0) printf("Size: %d", blockDim.x * gridDim.x);
	a[idx] = modadd(a[idx], b[bidx], primeId);
}

template __global__ void add_(uint64_t* a, const uint64_t* b, const int primeId, const __grid_constant__ int elem_b);

template __global__ void add_(uint32_t* a, const uint32_t* b, const int primeId, const __grid_constant__ int elem_b);

template <typename T> __global__ void sub_(T* a, const T* b, const int primeId, const __grid_constant__ int elem_b) {
	const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
	const int logstride = __ffs(blockDim.x * gridDim.x / elem_b) - 1;
	const int bidx      = idx >> logstride;
	// if(threadIdx.x == 0 && blockIdx.x == 0) printf("Prime %d: %lu ", primeId, p_prime);
	a[idx] = modsub(a[idx], b[bidx], primeId);
}

template __global__ void sub_(uint64_t* a, const uint64_t* b, const int primeId, const __grid_constant__ int elem_b);

template __global__ void sub_(uint32_t* a, const uint32_t* b, const int primeId, const __grid_constant__ int elem_b);

__global__ void add_(void** a, void** b, const __grid_constant__ int primeid_init, const __grid_constant__ int elem_b) {

	const int primeid   = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
	const int logstride = __ffs(blockDim.x * gridDim.x / elem_b) - 1;
	const int bidx      = idx >> logstride;

	if (ISU64(primeid)) {
		((uint64_t*)a[blockIdx.y])[idx] = modadd(((uint64_t*)a[blockIdx.y])[idx], ((uint64_t*)b[blockIdx.y])[bidx], primeid);
	} else {
		((uint32_t*)a[blockIdx.y])[idx] = modadd(((uint32_t*)a[blockIdx.y])[idx], ((uint32_t*)b[blockIdx.y])[bidx], primeid);
	}
}

__global__ void add_scale_p_b_(void** a, void** b, const int primeid_init, const __grid_constant__ int elem_b) {

	const int primeid   = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
	const int logstride = __ffs(blockDim.x * gridDim.x / elem_b) - 1;
	const int bidx      = idx >> logstride;

	if (ISU64(primeid)) {
		uint64_t tmp                    = ((uint64_t*)b[blockIdx.y])[bidx];
		tmp                             = modmult<ALGO_SHOUP>(tmp, C_.P[primeid], primeid, C_.P_shoup[primeid]);
		((uint64_t*)a[blockIdx.y])[idx] = modadd(((uint64_t*)a[blockIdx.y])[idx], tmp, primeid);
	} else {
		uint32_t tmp                    = ((uint64_t*)b[blockIdx.y])[bidx];
		tmp                             = modmult<ALGO_SHOUP>(tmp, (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
		((uint32_t*)a[blockIdx.y])[idx] = modadd(((uint32_t*)a[blockIdx.y])[idx], tmp, primeid);
	}
}

__global__ void add_scale_p_a_(void** a, void** b, const int primeid_init, const __grid_constant__ int elem_b) {

	const int primeid   = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
	const int logstride = __ffs(blockDim.x * gridDim.x / elem_b) - 1;
	const int bidx      = idx >> logstride;

	if (ISU64(primeid)) {
		uint64_t tmp                    = ((uint64_t*)a[blockIdx.y])[idx];
		tmp                             = modmult<ALGO_SHOUP>(tmp, C_.P[primeid], primeid, C_.P_shoup[primeid]);
		((uint64_t*)a[blockIdx.y])[idx] = modadd(((uint64_t*)b[blockIdx.y])[bidx], tmp, primeid);
	} else {
		uint32_t tmp                    = ((uint64_t*)a[blockIdx.y])[idx];
		tmp                             = modmult<ALGO_SHOUP>(tmp, (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
		((uint32_t*)a[blockIdx.y])[idx] = modadd(((uint32_t*)b[blockIdx.y])[bidx], tmp, primeid);
	}
}

__global__ void sub_(void** a, void** b, const int primeid_init, const __grid_constant__ int elem_b) {
	const int primeid   = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
	const int logstride = __ffs(blockDim.x * gridDim.x / elem_b) - 1;
	const int bidx      = idx >> logstride;

	if (ISU64(primeid)) {
		((uint64_t*)a[blockIdx.y])[idx] = modsub(((uint64_t*)a[blockIdx.y])[idx], ((uint64_t*)b[blockIdx.y])[bidx], primeid);
	} else {
		((uint32_t*)a[blockIdx.y])[idx] = modsub(((uint32_t*)a[blockIdx.y])[idx], ((uint32_t*)b[blockIdx.y])[bidx], primeid);
	}
}

__global__ void add_(void** a, void** b, void** c, const int primeid_init, const __grid_constant__ int elem_c) {
	const int primeid   = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
	const int logstride = __ffs(blockDim.x * gridDim.x / elem_c) - 1;
	const int cidx      = idx >> logstride;

	if (ISU64(primeid)) {
		((uint64_t*)a[blockIdx.y])[idx] = modadd(((uint64_t*)b[blockIdx.y])[idx], ((uint64_t*)c[blockIdx.y])[cidx], primeid);
	} else {
		((uint32_t*)a[blockIdx.y])[idx] = modadd(((uint32_t*)b[blockIdx.y])[idx], ((uint32_t*)c[blockIdx.y])[cidx], primeid);
	}
}

__global__ void sub_(void** a, void** b, void** c, const int primeid_init, const __grid_constant__ int elem_c) {
	const int primeid   = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx       = blockIdx.x * blockDim.x + threadIdx.x;
	const int logstride = __ffs(blockDim.x * gridDim.x / elem_c) - 1;
	const int cidx      = idx >> logstride;

	if (ISU64(primeid)) {
		((uint64_t*)a[blockIdx.y])[idx] = modsub(((uint64_t*)b[blockIdx.y])[idx], ((uint64_t*)c[blockIdx.y])[cidx], primeid);
	} else {
		((uint32_t*)a[blockIdx.y])[idx] = modsub(((uint32_t*)b[blockIdx.y])[idx], ((uint32_t*)c[blockIdx.y])[cidx], primeid);
	}
}

__global__ void scalar_add_(void** a, uint64_t* b, const int primeid_init) {
	const int idx     = threadIdx.x + blockIdx.x * blockDim.x;
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

	if (ISU64(primeid)) {
		((uint64_t*)a[blockIdx.y])[idx] = modadd(((uint64_t*)a[blockIdx.y])[idx], b[primeid], primeid);
	} else {
		((uint32_t*)a[blockIdx.y])[idx] = modadd(((uint32_t*)a[blockIdx.y])[idx], (uint32_t)b[primeid], primeid);
	}
}

__global__ void scalar_sub_(void** a, uint64_t* b, const int primeid_init) {
	const int idx     = threadIdx.x + blockIdx.x * blockDim.x;
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

	if (ISU64(primeid)) {
		((uint64_t*)a[blockIdx.y])[idx] = modsub(((uint64_t*)a[blockIdx.y])[idx], b[primeid], primeid);
	} else {
		((uint32_t*)a[blockIdx.y])[idx] = modsub(((uint32_t*)a[blockIdx.y])[idx], (uint32_t)b[primeid], primeid);
	}
}

} // namespace FIDESlib