//
// Created by carlosad on 21/10/24.
//

#ifndef GPUCKKS_NTTDYNAMICPARALLELISM_CUH
#define GPUCKKS_NTTDYNAMICPARALLELISM_CUH

#include "NTT.cuh"

namespace FIDESlib {
    template<typename T, ALGO algo, NTT_MODE mode>
    __global__ void device_launch_NTT(const int __grid_constant__ logN, const int __grid_constant__ primeid,
                                      T *dat, T *aux, T *res, const T __grid_constant__ q_L_inv,
                                      const T *pt, const int __grid_constant__ primeid_rescale);

    template<typename T, ALGO algo>
    __global__ void
    device_launch_INTT(const __grid_constant__ int logN, const __grid_constant__ int primeid, const T *dat, T *aux,
                       T *res);

    template<ALGO algo, NTT_MODE mode>
    __global__ void
    device_launch_batch_NTT(const __grid_constant__ int logN, const __grid_constant__ int primeid_init, void **dat,
                            void **aux, void **res);


    template<ALGO algo>
    __global__ void
    device_launch_batch_INTT(const __grid_constant__ int logN,
                             const __grid_constant__ int primeid_init, void **dat, void **aux, void **res);

}
#endif //GPUCKKS_NTTDYNAMICPARALLELISM_CUH
