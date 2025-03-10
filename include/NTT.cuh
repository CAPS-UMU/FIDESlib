//
// Created by carlosad on 4/04/24.
//

#ifndef FIDESLIB_NTT_CUH
#define FIDESLIB_NTT_CUH

#include <cinttypes>
#include "ModMult.cuh"

namespace FIDESlib {

/* Utility function, no real use other than testing. */
template <typename T>
__global__ void Bit_Reverse(T* dat, uint32_t N);

/* Get pointer to kernel, needed for explicit Cuda Graph construction. */
void* get_NTT_reference(bool second);

// ------------------------------------- INTT ----------------------------------------
/** Kernel fusions */
enum INTT_MODE { INTT_NONE, INTT_MULT_AND_SAVE, INTT_MULT_AND_ACC, INTT_ROTATE_AND_SAVE, INTT_SQUARE_AND_SAVE };

template <typename T, bool second = true, ALGO algo = ALGO_SHOUP, INTT_MODE mode = INTT_NONE>
__global__ void INTT_(T* __restrict__ dat, const int __grid_constant__ primeid, T* __restrict__ res,
                      const T* __restrict__ dat2 = nullptr, T* __restrict__ res0 = nullptr,
                      T* __restrict__ res1 = nullptr, const T* __restrict__ kska = nullptr,
                      const T* __restrict__ kskb = nullptr, T* __restrict__ c0 = nullptr,
                      const T* __restrict__ c0tilde = nullptr);

template <bool second, ALGO algo, INTT_MODE mode>
__global__ void INTT_(void** __restrict__ dat, const int __grid_constant__ primeid_init, void** __restrict__ res,
                      void** __restrict__ dat2 = nullptr, void** __restrict__ res0 = nullptr,
                      void** __restrict__ res1 = nullptr, void** __restrict__ kska = nullptr,
                      void** __restrict__ kskb = nullptr, void** __restrict__ c0 = nullptr,
                      void** __restrict__ c0tilde = nullptr);

// ------------------------------------- NTT ----------------------------------------
/** Kernel fusions */
enum NTT_MODE { NTT_NONE, NTT_RESCALE, NTT_MULTPT, NTT_MODDOWN, NTT_KSK_DOT, NTT_KSK_DOT_ACC };

template <typename T, bool second = true, ALGO algo = ALGO_SHOUP, NTT_MODE mode = NTT_NONE>
__global__ void NTT_(T* __restrict__ dat, const int __grid_constant__ primeid, T* __restrict__ res,
                     const T* __restrict__ pt = nullptr, const int __grid_constant__ primeid_rescale = -1,
                     T* __restrict__ res2 = nullptr, const T* __restrict__ kskb = nullptr);

template <bool second, ALGO algo, NTT_MODE mode>
__global__ void NTT_(void** __restrict__ dat, const int __grid_constant__ primeid_init, void** __restrict__ res,
                     void** __restrict__ pt, const int __grid_constant__ primeid_rescale, void** __restrict__ res2,
                     void** __restrict__ kskb);

// ------------------------------------- 1D NTT version ----------------------------------------

template <typename T, int WARP_SIZE = 32>
__global__ void NTT_1D(T* dat, const T* psi_dat, const int __grid_constant__ N, const int __grid_constant__ primeid,
                       const int __grid_constant__ logN);

template <typename T, int WARP_SIZE = 32>
__global__ void INTT_1D(T* dat, const T* psi_dat, const int __grid_constant__ N, const int __grid_constant__ primeid,
                        const T N_inv, const int __grid_constant__ logN);
}  // namespace FIDESlib

#endif  //FIDESLIB_NTT_CUH
