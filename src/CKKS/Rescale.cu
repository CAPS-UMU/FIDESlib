//
// Created by carlosad on 10/09/24.
//
#include "CKKS/Rescale.cuh"

namespace FIDESlib::CKKS{

    template <typename T>
    __global__ void SwitchModulus(const T * src, const int __grid_constant__ o_primeid,
                                  T * res, const int __grid_constant__ n_primeid){
        int pid = blockIdx.x * blockDim.x + threadIdx.x;
        T a = src[pid];
        SwitchModulus(a, o_primeid, n_primeid);
        res[pid] = a;
    }

    template
    __global__ void SwitchModulus<uint32_t>(const uint32_t * src, const int __grid_constant__ o_primeid,
                                            uint32_t * res, const int __grid_constant__ n_primeid);

    template
    __global__ void SwitchModulus<uint64_t>(const uint64_t * src, const int __grid_constant__ o_primeid,
                                            uint64_t * res, const int __grid_constant__ n_primeid);
}