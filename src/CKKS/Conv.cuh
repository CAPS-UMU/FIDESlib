//
// Created by carlosad on 24/03/24.
//

#ifndef FIDESLIB_CKKS_CONV_CUH
#define FIDESLIB_CKKS_CONV_CUH

#include "ModMult.cuh"

namespace FIDESlib::CKKS {
template <typename T> __global__ void conv1_(T* a, const T q_hat_inv, const int primeid);

/**
 * Dynamic shared memory should be sizeof(T) * (K + blockDim.y) * blockDim.y
 * @tparam T
 * @param a
 * @param typea
 * @param n
 * @param b
 * @param G_
 * @param typeb
 * @param m
 * @param L
 */
template <ALGO algo = ALGO_SHOUP>
__global__ void
ModDown2(void** __restrict__ a, const __grid_constant__ int n, void** __restrict__ b, const __grid_constant__ int primeid_init, const Global::Globals* Globals);

template <ALGO algo = ALGO_SHOUP>
__global__ void
ModDown3(void** __restrict__ a, const __grid_constant__ int n, void** __restrict__ b, const __grid_constant__ int primeid_init, const Global::Globals* Globals);

template <ALGO algo = ALGO_SHOUP>
__global__ void
DecompAndModUpConv(void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b, const int __grid_constant__ d, const Global::Globals* Globals);

template <ALGO algo = ALGO_SHOUP>
__global__ void
DecompAndModUpConv_spec(void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b, const int __grid_constant__ d, const Global::Globals* Globals);

template <ALGO algo = ALGO_SHOUP>
__global__ void
DecompAndModUpConv_spec2(void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b, const int __grid_constant__ d, const Global::Globals* Globals);

} // namespace FIDESlib::CKKS
#endif // FIDESLIB_CKKS_CONV_CUH
