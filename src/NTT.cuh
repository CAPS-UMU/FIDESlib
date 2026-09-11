//
// Created by carlosad on 4/04/24.
//

#ifndef FIDESLIB_NTT_CUH
#define FIDESLIB_NTT_CUH

#include "CKKS/forwardDefs.cuh"
#include "ConstantsGPU.cuh"
#include <cinttypes>

namespace FIDESlib {

struct FusedIterationsParams {
	struct __align__(128) AtomicCounter {
		uint32_t n = 0;
		uint32_t pad[(128 - sizeof(uint32_t)) / sizeof(uint32_t)];
	};

	AtomicCounter counters[MAXP];

	struct Conf {
		dim3 grid;
		dim3 block;
	};

	Conf first;
	Conf second;
};

template <uint32_t M, bool second> uint32_t NTT_block_dim_X(int logN) {
	return second ?
		(uint32_t)(1 << ((logN + (logN > 13 ? 0 : 0)) / 2 - 1)) :
		(uint32_t)(1 << ((logN + 1 + (logN > 13 ? 0 : 0)) / 2 - 1));
}

template <uint32_t M, bool second> uint32_t NTT_grid_dim_X(int logN) {
	uint32_t blockDimX = NTT_block_dim_X<M, second>(logN);
	return second ? (1 << logN) / blockDimX / 2 / M : (1 << logN) / blockDimX / 2 / M;
}

template <uint32_t M, bool second> uint32_t INTT_block_dim_X(int logN) {
	return second ?
		(uint32_t)(1 << ((logN + 1 + (logN > 13 ? 0 : 0)) / 2 - 1)) :
		(uint32_t)(1 << ((logN + (logN > 13 ? 0 : 0)) / 2 - 1));
}

template <uint32_t M, bool second> uint32_t INTT_grid_dim_X(int logN) {
	uint32_t blockDimX = INTT_block_dim_X<M, second>(logN);
	return second ? (1 << logN) / blockDimX / 2 / M : (1 << logN) / blockDimX / 2 / M;
}

template <uint32_t M, typename T, ALGO algo> uint32_t NTT_shmem(uint32_t blockDimX) {
	return sizeof(T) * blockDimX * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
}

/* Utility function, no real use other than testing. */
template <typename T> __global__ void Bit_Reverse(T* dat, uint32_t N);

/* Get pointer to kernel, needed for explicit Cuda Graph construction. */
void* get_NTT_reference(bool second);

// ------------------------------------- INTT ----------------------------------------
/** Kernel fusions */
enum INTT_MODE { INTT_NONE, INTT_MULT_AND_SAVE, INTT_MULT_AND_ACC, INTT_ROTATE_AND_SAVE, INTT_SQUARE_AND_SAVE };

template <typename T, bool second = true, ALGO algo = ALGO_SHOUP, INTT_MODE mode = INTT_NONE> __global__ void INTT_(const Global::Globals* Globals,
	T* __restrict__ dat,
	const int __grid_constant__ primeid,
	T* __restrict__ res,
	const T* __restrict__ dat2    = nullptr,
	T* __restrict__ res0          = nullptr,
	T* __restrict__ res1          = nullptr,
	const T* __restrict__ kska    = nullptr,
	const T* __restrict__ kskb    = nullptr,
	T* __restrict__ c0            = nullptr,
	const T* __restrict__ c0tilde = nullptr);

template <bool second, ALGO algo, INTT_MODE mode> __global__ void INTT_(const Global::Globals* Globals,
                                                                        void** __restrict__ dat,
                                                                        const int __grid_constant__ primeid_init,
                                                                        void** __restrict__ res,
                                                                        void** __restrict__ dat2    = nullptr,
                                                                        void** __restrict__ res0    = nullptr,
                                                                        void** __restrict__ res1    = nullptr,
                                                                        void** __restrict__ kska    = nullptr,
                                                                        void** __restrict__ kskb    = nullptr,
                                                                        void** __restrict__ c0      = nullptr,
                                                                        void** __restrict__ c0tilde = nullptr);

template <typename T, ALGO algo = ALGO_SHOUP, INTT_MODE mode = INTT_NONE> __global__ void INTT_1D_(const Global::Globals* Globals,
                                                                                                   T* __restrict__ dat,
                                                                                                   const int __grid_constant__ primeid,
                                                                                                   T* __restrict__ res,
                                                                                                   const T* __restrict__ dat2    = nullptr,
                                                                                                   T* __restrict__ res0          = nullptr,
                                                                                                   T* __restrict__ res1          = nullptr,
                                                                                                   const T* __restrict__ kska    = nullptr,
                                                                                                   const T* __restrict__ kskb    = nullptr,
                                                                                                   T* __restrict__ c0            = nullptr,
                                                                                                   const T* __restrict__ c0tilde = nullptr);

template <ALGO algo, INTT_MODE mode> __global__ void INTT_1D_(const Global::Globals* Globals,
                                                              void** __restrict__ dat,
                                                              const int __grid_constant__ primeid_init,
                                                              void** __restrict__ res,
                                                              void** __restrict__ dat2    = nullptr,
                                                              void** __restrict__ res0    = nullptr,
                                                              void** __restrict__ res1    = nullptr,
                                                              void** __restrict__ kska    = nullptr,
                                                              void** __restrict__ kskb    = nullptr,
                                                              void** __restrict__ c0      = nullptr,
                                                              void** __restrict__ c0tilde = nullptr);

// ------------------------------------- NTT ----------------------------------------
/** Kernel fusions */
enum NTT_MODE { NTT_NONE, NTT_RESCALE, NTT_MULTPT, NTT_MODDOWN, NTT_KSK_DOT, NTT_KSK_DOT_ACC };

template <typename T, bool second = true, ALGO algo = ALGO_SHOUP, NTT_MODE mode = NTT_NONE> __global__ void NTT_(const Global::Globals* Globals,
	T* __restrict__ dat,
	const int __grid_constant__ primeid,
	T* __restrict__ res,
	const T* __restrict__ pt                    = nullptr,
	const int __grid_constant__ pt_elem         = -1,
	const int __grid_constant__ primeid_rescale = -1,
	T* __restrict__ res2                        = nullptr,
	const T* __restrict__ kskb                  = nullptr);

template <bool second, ALGO algo, NTT_MODE mode> __global__ void NTT_(const Global::Globals* Globals,
                                                                      void** __restrict__ dat,
                                                                      const int __grid_constant__ primeid_init,
                                                                      void** __restrict__ res,
                                                                      void** __restrict__ pt                      = nullptr,
                                                                      const __grid_constant__ int pt_elem         = -1,
                                                                      const int __grid_constant__ primeid_rescale = -1,
                                                                      void** __restrict__ res2                    = nullptr,
                                                                      void** __restrict__ kskb                    = nullptr);

template <typename T, ALGO algo = ALGO_SHOUP, NTT_MODE mode = NTT_NONE> __global__ void NTT_1D_(const Global::Globals* Globals,
                                                                                                T* __restrict__ dat,
                                                                                                const int __grid_constant__ primeid,
                                                                                                T* __restrict__ res,
                                                                                                const T* __restrict__ pt                    = nullptr,
                                                                                                const int __grid_constant__ pt_elem         = -1,
                                                                                                const int __grid_constant__ primeid_rescale = -1,
                                                                                                T* __restrict__ res2                        = nullptr,
                                                                                                const T* __restrict__ kskb                  = nullptr);

template <ALGO algo, NTT_MODE mode> __global__ void NTT_1D_(const Global::Globals* Globals,
                                                            void** __restrict__ dat,
                                                            const int __grid_constant__ primeid_init,
                                                            void** __restrict__ res,
                                                            void** __restrict__ pt                      = nullptr,
                                                            const __grid_constant__ int pt_elem         = -1,
                                                            const int __grid_constant__ primeid_rescale = -1,
                                                            void** __restrict__ res2                    = nullptr,
                                                            void** __restrict__ kskb                    = nullptr);

// ------------------------------------- 1D NTT version ----------------------------------------

template <typename T, int WARP_SIZE = 32> __global__ void
NTT_1D(const Global::Globals* Globals,
       T* dat,
       const T* psi_dat,
       const int __grid_constant__ N,
       const int __grid_constant__ primeid,
       const int __grid_constant__ logN);

template <typename T, int WARP_SIZE = 32> __global__ void
INTT_1D(const Global::Globals* Globals,
        T* dat,
        const T* psi_dat,
        const int __grid_constant__ N,
        const int __grid_constant__ primeid,
        const T N_inv,
        const int __grid_constant__ logN);
} // namespace FIDESlib

#endif // FIDESLIB_NTT_CUH
