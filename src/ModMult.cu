//
// Created by carlosad on 4/04/24.
//
#include "ModMult.cuh"


namespace FIDESlib {

    template<typename T, ALGO algo>
    __global__ void mult_(T *a, const T *b, const int primeid) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        a[idx] = modmult<algo>(a[idx], b[idx], primeid);
    }

    template<typename T, ALGO algo>
    __global__ void mult_(T *a, const T *b, const T *c, const int primeid) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        a[idx] = modmult<algo>(b[idx], c[idx], primeid);
    }

    template<typename T, ALGO algo>
    __global__ void scalar_mult_(T *a, const T b, const int primeid, const T shoup_mu) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        a[idx] = modmult<algo>(a[idx], b, primeid, shoup_mu);
    }


#define Y(T, algo) \
    template \
    __global__ void mult_<T, algo>(T *a, const T *b, const int primeid);

#include "ntt_types.inc"

#undef Y


#define Y(T, algo) \
    template \
    __global__ void mult_<T, algo>(T *a, const T *b, const T *c, const int primeid);

#include "ntt_types.inc"

#undef Y

#define Y(T, algo) \
    template \
    __global__ void scalar_mult_<T, algo>(T *a, const T b, const int primeid, const T shoup_mu);

#include "ntt_types.inc"

#undef Y
}