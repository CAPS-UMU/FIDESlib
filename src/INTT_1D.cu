//
// Created by carlosad on 9/9/26.
//

#include "AddSub.cuh"
#include "CKKS/Rescale.cuh"
#include "ConstantsGPU.cuh"
#include "ModMult.cuh"
#include "NTT.cuh"

// #include <cooperative_groups.h>
#include <cassert>
#include <iostream>

#include "NTTfusions.cuh"
#include "NTThelper.cuh"

namespace FIDESlib {
using Scheme = CKKS::Scheme;

template <typename T, ALGO algo, int M> __device__ __forceinline__ void backward_negacyclic_scale_1D(
	char* buffer,
	const int primeid,
	T* psi,
	T* psi_shoup,
	const Global::Globals* Globals,
	const int logn) {

	const int tid = threadIdx.x;

	{
		const uint32_t logBD = __clz(blockDim.x) + 1;
		// Now, try to load this from bit-reversed psi array
		uint32_t pos1 = tid;
		pos1          = __brev(pos1);
		pos1 >>= (logBD);

		assert(pos1 < blockDim.x);

		// INTT_1D ouput is normal-ordered so we just undo psi_inv array for the higher exponent bits [1, ..., logN], then multiply by root for exponent bit 0, and multiply by
		T aux = psi[pos1];

		//const T root = C_.inv_root[primeid];
		//T root_shoup;
		//if constexpr (algo == ALGO_SHOUP)
		//	root_shoup = C_.inv_root_shoup[primeid];

		const T root = ((T*)G_->inv_psi[primeid])[(1 << (logn - 1))];
		T root_shoup = 0;
		if constexpr (algo == ALGO_SHOUP)
			root_shoup = ((T*)G_->inv_psi_shoup[primeid])[(1 << (logn - 1))];

		aux    = modmult<ALGO_BARRETT>(aux, 1 << (C_.logN - logn), primeid);
		aux    = modmult<ALGO_SHOUP>(aux, (T)C_.N_inv[primeid], primeid, (T)C_.N_inv_shoup[primeid]);
		T aux2 = modmult<algo>(aux, root, primeid, root_shoup);

		for (int i = 0; i < M; i += 1) {

			A(i)[tid << 1]       = modmult<FIDESlib::ALGO_BARRETT>(A(i)[tid << 1], aux, primeid);
			A(i)[(tid << 1) + 1] = modmult<FIDESlib::ALGO_BARRETT>(A(i)[(tid << 1) + 1], aux2, primeid);
		}
	}
}

template <typename T, ALGO algo, INTT_MODE mode> __device__ __forceinline__ void INTT_1D__(const Global::Globals* Globals,
                                                                                           const T* __restrict__ dat,
                                                                                           const int primeid,
                                                                                           T* __restrict__ res,
                                                                                           const T* __restrict__ dat2,
                                                                                           T* __restrict__ res0,
                                                                                           T* __restrict__ res1,
                                                                                           const T* __restrict__ kska,
                                                                                           const T* __restrict__ kskb,
                                                                                           T* __restrict__ c0,
                                                                                           const T* __restrict__ c0tilde) {
	const int tid = threadIdx.x;
	extern __shared__ char buffer[];
	constexpr int M = 1;

	T* psi       = &(((T*)buffer)[blockDim.x * 2 * M]);
	T* psi_shoup = &(((T*)buffer)[blockDim.x * (2 * M + (algo == ALGO_SHOUP))]);

	assert(((uint64_t)dat & 0b1111ul) == 0);
	assert(((uint64_t)res & 0b1111ul) == 0);
	assert(((uint64_t)psi & 0b1111ul) == 0);
	assert(((uint64_t)A(0) & 0b1111ul) == 0);
	assert(((uint64_t)A(1) & 0b1111ul) == 0);
	assert(((uint64_t)A(2) & 0b1111ul) == 0);
	assert(((uint64_t)A(3) & 0b1111ul) == 0);
	assert(G_ != nullptr);
	assert(G_->inv_psi[primeid] != nullptr);
	assert(((uint64_t)G_->inv_psi[primeid] & 0b1111ul) == 0);
	assert(G_->inv_psi_shoup[primeid] != nullptr);
	assert(((uint64_t)G_->inv_psi_shoup[primeid] & 0b1111ul) == 0);

	const int j          = tid << 1;
	const uint32_t logBD = 32 - __clz(blockDim.x);
	const int N          = 2 * M * blockDim.x * gridDim.x;
	const int logN       = __ffs(N) - 1;
	assert(1 << logN == N);

	{

		psi[tid] = ((T*)G_->inv_psi[primeid])[tid];
		if constexpr (algo == ALGO_SHOUP)
			psi_shoup[tid] = ((T*)G_->inv_psi_shoup[primeid])[tid];

		if constexpr (mode == INTT_MULT_AND_SAVE) {
			mult_and_save_fusion<T, algo, M>(buffer, logBD, j, primeid, (T*)dat, (T*)dat2, (T*)c0, (T*)dat, (T*)kska, (T*)kskb, (T*)c0, (T*)c0tilde);
		} else if constexpr (mode == INTT_MULT_AND_ACC) {
			mult_and_acc_fusion<T, algo, M>(buffer, logBD, j, primeid, (T*)dat, (T*)dat2, (T*)c0, (T*)dat, (T*)kska, (T*)kskb, (T*)c0, (T*)c0tilde);
		} else if constexpr (mode == INTT_ROTATE_AND_SAVE) {
			rotate_and_save_fusion<T, algo, M>(buffer, logBD, j, primeid, (T*)dat, (T*)c0, (T*)dat, (T*)kska, (T*)kskb, (T*)c0);
		} else if constexpr (mode == INTT_SQUARE_AND_SAVE) {
			square_and_save_fusion<T, algo, M>(buffer, logBD, j, primeid, (T*)dat, (T*)c0, (T*)dat, (T*)kska, (T*)kskb, (T*)c0);
		} else {
			for (int i = 0; i < M; ++i) {
				if constexpr (sizeof(T) == 8) {
					int4 aux;
					aux                     = ((int4*)dat)[OFFSET_2T(i)];
					((int4*)(A(i)))[j >> 1] = aux;
				} else {
					int2 aux;
					aux                     = ((int2*)dat)[OFFSET_2T(i)];
					((int2*)(A(i)))[j >> 1] = aux;
				}
			}
		}

		/*
		if (OFFSET_2T(0) == 0 && primeid == 0)
			printf("INTT load %lu %lu\n", A(0)[0], A(0)[1]);
*/
	}
	__syncthreads();

	int m            = 1;
	int maskPsi      = (blockDim.x - 1);
	uint32_t log_psi = 0;

	for (; m < blockDim.x; m <<= 1, maskPsi &= (maskPsi << 1), ++log_psi) {
		if (m >= warpSize)
			__syncthreads();
		else
			__syncwarp();

		const int mask = m - 1;
		const int j1   = (mask & tid) | (((~mask) << 1) & (tid << 1));
		const int j2   = j1 | m;

		const int psiid = (tid & maskPsi) >> log_psi;

		const T psiaux = psi[psiid];
		T psiaux_shoup;
		if constexpr (algo == 3)
			psiaux_shoup = psi_shoup[psiid];

		for (int i = 0; i < M; ++i) {
			T& a0 = A(i)[j1];
			T& a1 = A(i)[j2];
			if constexpr (algo == 3) {
				GS_butterfly<T, algo>(a0, a1, psiaux, primeid, psiaux_shoup);
			} else {
				GS_butterfly<T, algo>(a0, a1, psiaux, primeid);
			}
		}
	}

	__syncthreads();
	for (int i = 0; i < M; ++i) {
		T aux[2];
		aux[0]        = A(i)[tid];
		aux[1]        = A(i)[tid + m];
		A(i)[tid]     = modadd(aux[0], aux[1], primeid);
		A(i)[tid + m] = modsub(aux[0], aux[1], primeid);
	}

	// Obs: Almacenamos el array transpuesto ambas veces
	// Idea: calcular full_psi en función de ambos arrays psi
	// Idea: incluir N_inv en full_psi

	if constexpr (NEGACYCLIC) {
		backward_negacyclic_scale_1D<T, algo, M>(buffer, primeid, psi, psi_shoup, Globals, logN);
	}

	{
		__syncthreads();
		if constexpr (sizeof(T) == 8) {
			((int4*)res)[tid] = ((int4*)buffer)[tid];
		} else {
			((int2*)res)[tid] = ((int2*)buffer)[tid];
		}
	}
}

template <typename T, ALGO algo, INTT_MODE mode> __global__ void INTT_1D_(const Global::Globals* Globals,
                                                                          T* __restrict__ dat,
                                                                          const int __grid_constant__ primeid,
                                                                          T* __restrict__ res,
                                                                          const T* __restrict__ dat2,
                                                                          T* __restrict__ res0,
                                                                          T* __restrict__ res1,
                                                                          const T* __restrict__ kska,
                                                                          const T* __restrict__ kskb,
                                                                          T* __restrict__ c0,
                                                                          const T* __restrict__ c0tilde) {
	INTT_1D__<T, algo, mode>(Globals, dat, primeid, res, dat2, res0, res1, kska, kskb, c0, c0tilde);
}

#define WWWW(T, algo, mode)                                                          \
template __global__ void INTT_1D_<T, algo, mode>(const Global::Globals* Globals, \
T* __restrict__ dat,                                                                \
const int __grid_constant__ primeid,                                                \
T* __restrict__ res,                                                                \
const T* __restrict__ dat2,                                                         \
T* __restrict__ res0,                                                               \
T* __restrict__ res1,                                                               \
const T* __restrict__ kska,                                                         \
const T* __restrict__ kskb,                                                         \
T* __restrict__ c0,                                                                 \
const T* __restrict__ c0tilde);

#include "ntt_types.inc"

#undef WWWW

template <ALGO algo, INTT_MODE mode> __global__ void INTT_1D_(const Global::Globals* Globals,
                                                              void** __restrict__ dat,
                                                              const int __grid_constant__ primeid_init,
                                                              void** __restrict__ res,
                                                              void** __restrict__ dat2,
                                                              void** __restrict__ res0,
                                                              void** __restrict__ res1,
                                                              void** __restrict__ kska,
                                                              void** __restrict__ kskb,
                                                              void** __restrict__ c0,
                                                              void** __restrict__ c0tilde) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	if (ISU64(primeid)) {
		INTT_1D__<uint64_t, algo, mode>(Globals,
		                                (uint64_t*)dat[blockIdx.y],
		                                primeid,
		                                (uint64_t*)res[blockIdx.y],
		                                dat2 ? (uint64_t*)dat2[blockIdx.y] : nullptr,
		                                res0 ? (uint64_t*)res0[blockIdx.y] : nullptr,
		                                res1 ? (uint64_t*)res1[blockIdx.y] : nullptr,
		                                kska ? (uint64_t*)kska[blockIdx.y] : nullptr,
		                                kskb ? (uint64_t*)kskb[blockIdx.y] : nullptr,
		                                c0 ? (uint64_t*)c0[blockIdx.y] : nullptr,
		                                c0tilde ? (uint64_t*)c0tilde[blockIdx.y] : nullptr);
	} else {
		INTT_1D__<uint32_t, algo, mode>(Globals,
		                                (uint32_t*)dat[blockIdx.y],
		                                primeid,
		                                (uint32_t*)res[blockIdx.y],
		                                dat2 ? (uint32_t*)dat2[blockIdx.y] : nullptr,
		                                res0 ? (uint32_t*)res0[blockIdx.y] : nullptr,
		                                res1 ? (uint32_t*)res1[blockIdx.y] : nullptr,
		                                kska ? (uint32_t*)kska[blockIdx.y] : nullptr,
		                                kskb ? (uint32_t*)kskb[blockIdx.y] : nullptr,
		                                c0 ? (uint32_t*)c0[blockIdx.y] : nullptr,
		                                c0tilde ? (uint32_t*)c0tilde[blockIdx.y] : nullptr);
	}
}

#define WWW(algo, mode)                                                         \
template __global__ void INTT_1D_<algo, mode>(const Global::Globals* Globals, \
void** __restrict__ dat,                                                         \
const int __grid_constant__ primeid_init,                                        \
void** __restrict__ res,                                                         \
void** __restrict__ dat2,                                                        \
void** __restrict__ res0,                                                        \
void** __restrict__ res1,                                                        \
void** __restrict__ kska,                                                        \
void** __restrict__ kskb,                                                        \
void** __restrict__ c0,                                                          \
void** __restrict__ c0tilde);

#include "ntt_types.inc"

#undef WWW
}