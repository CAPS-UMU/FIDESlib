//
// Created by carlosad on 25/03/24.
//
#include <cassert>
#include "AddSub.cuh"

namespace FIDESlib {

template <typename T>
__global__ void add_(T* a, const T* b, const int primeId) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    //if(threadIdx.x == 0 && blockIdx.x == 0) printf("Prime %d: %lu ", primeId, p_prime);
    //  if(threadIdx.x == 0 && blockIdx.x == 0) printf("Size: %d", blockDim.x * gridDim.x);
    a[idx] = modadd(a[idx], b[idx], primeId);
}

template __global__ void add_(uint64_t* a, const uint64_t* b, const int primeId);

template __global__ void add_(uint32_t* a, const uint32_t* b, const int primeId);

template <typename T>
__global__ void sub_(T* a, const T* b, const int primeId) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if(threadIdx.x == 0 && blockIdx.x == 0) printf("Prime %d: %lu ", primeId, p_prime);
    a[idx] = modsub(a[idx], b[idx], primeId);
}

template __global__ void sub_(uint64_t* a, const uint64_t* b, const int primeId);

template __global__ void sub_(uint32_t* a, const uint32_t* b, const int primeId);

__global__ void add_(void** a, void** b, const int primeid_init) {

    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] =
            modadd(((uint64_t*)a[blockIdx.y])[idx], ((uint64_t*)b[blockIdx.y])[idx], primeid);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] =
            modadd(((uint32_t*)a[blockIdx.y])[idx], ((uint32_t*)b[blockIdx.y])[idx], primeid);
    }
}

__global__ void sub_(void** a, void** b, const int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] =
            modsub(((uint64_t*)a[blockIdx.y])[idx], ((uint64_t*)b[blockIdx.y])[idx], primeid);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] =
            modsub(((uint32_t*)a[blockIdx.y])[idx], ((uint32_t*)b[blockIdx.y])[idx], primeid);
    }
}

__global__ void add_(void** a, void** b, void** c, const int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] =
            modadd(((uint64_t*)b[blockIdx.y])[idx], ((uint64_t*)c[blockIdx.y])[idx], primeid);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] =
            modadd(((uint32_t*)b[blockIdx.y])[idx], ((uint32_t*)c[blockIdx.y])[idx], primeid);
    }
}

__global__ void sub_(void** a, void** b, void** c, const int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] =
            modsub(((uint64_t*)b[blockIdx.y])[idx], ((uint64_t*)c[blockIdx.y])[idx], primeid);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] =
            modsub(((uint32_t*)b[blockIdx.y])[idx], ((uint32_t*)c[blockIdx.y])[idx], primeid);
    }
}

__global__ void scalar_add_(void** a, uint64_t* b, const int primeid_init) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] = modadd(((uint64_t*)a[blockIdx.y])[idx], b[blockIdx.y], primeid);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] = modadd(((uint32_t*)a[blockIdx.y])[idx], (uint32_t)b[blockIdx.y], primeid);
    }
}

__global__ void scalar_sub_(void** a, uint64_t* b, const int primeid_init) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] = modsub(((uint64_t*)a[blockIdx.y])[idx], b[blockIdx.y], primeid);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] = modsub(((uint32_t*)a[blockIdx.y])[idx], (uint32_t)b[blockIdx.y], primeid);
    }
}

}  // namespace FIDESlib