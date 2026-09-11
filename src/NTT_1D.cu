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

template <typename T, ALGO algo, int M> __device__ __forceinline__ void forward_negacyclic_scale_1D(
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
		assert(tid == 1 ? pos1 == blockDim.x / 2 : true);

		// INTT_1D ouput is normal-ordered so we just undo psi_inv array for the higher exponent bits [2, ..., logN], then multiply by root for exponent bit 0,
		// and multiply by root^2 for exponent bit 1
		T aux = psi[pos1];
		T aux_shoup;
		if constexpr (algo == ALGO_SHOUP)
			aux_shoup = psi_shoup[pos1];

		//const T root = C_.inv_root[primeid];
		//T root_shoup;
		//if constexpr (algo == ALGO_SHOUP)
		//	root_shoup = C_.inv_root_shoup[primeid];

		const T root = ((T*)G_->psi[primeid])[(1 << (logn - 1))];
		T root_shoup;
		if constexpr (algo == ALGO_SHOUP)
			root_shoup = ((T*)G_->psi_shoup[primeid])[(1 << (logn - 1))];

		for (int i = 0; i < M; i += 1) {
			T temp               = modmult<algo>(A(i)[tid << 1], aux, primeid, aux_shoup);
			A(i)[tid << 1]       = temp;
			T aux2               = modmult<algo>(A(i)[(tid << 1) + 1], aux, primeid, aux_shoup);
			A(i)[(tid << 1) + 1] = modmult(aux2, root, primeid, root_shoup);
		}
	}
}

template <typename T, ALGO algo, NTT_MODE mode> __device__ __forceinline__ void
NTT_1D__(const Global::Globals* Globals,
         T* __restrict__ dat,
         const int primeid,
         T* __restrict__ res,
         const T* __restrict__ pt,
         const int pt_elem,
         const int primeid_rescale,
         T* __restrict__ res2,
         const T* __restrict__ kskb) {
	const int tid = threadIdx.x;
	const int j   = tid << 1;
	extern __shared__ char buffer[];
	constexpr int M = 1;
	const int N     = 2 * M * blockDim.x * gridDim.x;
	const int logN  = __ffs(N) - 1;
	assert(1 << logN == N);

	T* psi        = &(((T*)buffer)[blockDim.x * 2 * M]);
	T* psi_barret = (T*)(buffer + sizeof(T) * blockDim.x * (2 * M + (algo == ALGO_SHOUP)));

	assert(((uint64_t)dat & 0b1111ul) == 0);
	assert(((uint64_t)res & 0b1111ul) == 0);
	assert(((uint64_t)psi & 0b1111ul) == 0);
	assert(((uint64_t)A(0) & 0b1111ul) == 0);
	assert(((uint64_t)A(1) & 0b1111ul) == 0);
	assert(((uint64_t)A(2) & 0b1111ul) == 0);
	assert(((uint64_t)A(3) & 0b1111ul) == 0);

	assert(G_ != nullptr);
	assert(G_->psi[primeid] != nullptr);
	assert(((uint64_t)G_->psi[primeid] & 0b1111ul) == 0);
	assert(G_->psi_shoup[primeid] != nullptr);
	assert(((uint64_t)G_->psi_shoup[primeid] & 0b1111ul) == 0);
#ifdef COOPERATIVE_GROUPS
	cg::grid_group grid = cg::this_grid();
	for (int second_ = 0; second_ < 2; ++second_) {
#endif
	{
		psi[tid] = ((T*)G_->psi[primeid])[tid];
		if constexpr (algo == ALGO_SHOUP)
			psi_barret[tid] = ((T*)G_->psi_shoup[primeid])[tid];

		if constexpr (sizeof(T) == 8) {
			((int4*)buffer)[tid] = ((int4*)dat)[tid];
		} else {
			((int2*)buffer)[tid] = ((int2*)dat)[tid];
		}
	}

	if constexpr (1) {
		if constexpr (mode == NTT_RESCALE || mode == NTT_MULTPT) {
			assert(primeid_rescale >= 0);
			for (int i = 0; i < M; i += 1) {
				CKKS::SwitchModulus(A(i)[(tid << 1)], primeid_rescale, primeid);
				CKKS::SwitchModulus(A(i)[(tid << 1) + 1], primeid_rescale, primeid);
			}
		}
		__syncthreads();

		if constexpr (NEGACYCLIC) {
			forward_negacyclic_scale_1D<T, algo, M>(buffer, primeid, psi, psi_barret, Globals, logN);
		}

		__syncthreads();

		int m                = blockDim.x;
		int maskPsi          = m;
		const uint64_t logBD = 32 - __clz(blockDim.x) + (sizeof(T) == 8 ? 3 : 2);

		// Iteración 0 optimizada.`
		for (int i = 0; i < M; i += 1) {
			/*
	T *A = (T *) (buffer + (i << (logBD)));
	T aux[2];
	aux[0] = A[tid];
	aux[1] = A[tid | m];
	A[tid] = modadd(aux[0], aux[1], primeid);
	A[tid | m] = modsub(aux[0], aux[1], primeid);
	 */
			// T aux[2]; // esto causa error de alineamiento lol
			T aux0        = A(i)[tid];
			T aux1        = A(i)[tid + m];
			A(i)[tid]     = modadd(aux0, aux1, primeid);
			A(i)[tid + m] = modsub(aux0, aux1, primeid);
		}

		m >>= 1;
		maskPsi |= (maskPsi >> 1);
		int log_psi = __ffs(blockDim.x) - 2; // Ojo al logaritmo.

		for (; m >= 1 /*warpSize*/; m >>= 1, log_psi--, maskPsi |= (maskPsi >> 1)) {
			const int mask        = m - 1;
			int j1                = (mask & tid) | ((~mask & tid) << 1);
			int j2                = j1 + m;
			const int psiid       = (tid & maskPsi) >> log_psi;
			const T psiaux        = psi[psiid];
			const T psiaux_barret = psi_barret[psiid];

			if (m >= warpSize)
				__syncthreads();
			else
				__syncwarp();

			for (int i = 0; i < M; i += 1) {
				T* A = (T*)(buffer + (i << (logBD)));

				T& aux1 = A[j1];
				T& aux2 = A[j2];
				if constexpr (algo == 3) {
					CT_butterfly<T, algo>(aux1, aux2, psiaux, primeid, psiaux_barret);
				} else {
					CT_butterfly<T, algo>(aux1, aux2, psiaux, primeid);
				}
			}
		}

		{
			if constexpr (mode == NTT_RESCALE) {
				rescale_fusion<T, algo, M>(buffer, logBD, j, primeid, primeid_rescale, res, Globals);
			}
			if constexpr (mode == NTT_MULTPT) {
				multpt_fusion<T, algo, M>(buffer, logBD, j, primeid, primeid_rescale, res, pt, Globals, pt_elem);
			}
			if constexpr (mode == NTT_MODDOWN) {
				moddown_fusion<T, algo, M>(buffer, logBD, j, primeid, res);
			}

			if constexpr (mode == NTT_KSK_DOT) {
				ksk_dot_fusion<T, algo, M>(buffer, logBD, j, primeid, res, res2, pt, kskb);
			} else if constexpr (mode == NTT_KSK_DOT_ACC) {
				ksk_dot_acc_fusion<T, algo, M>(buffer, logBD, j, primeid, res, res2, pt, kskb);
			} else {
				for (int i = 0; i < M; i += 1) {
					const T* A = (T*)(buffer + (i << (logBD)));
					if constexpr (sizeof(T) == 8) {
						((int4*)res)[tid] = ((int4*)A)[tid];
					} else {
						((int2*)res)[tid] = ((int2*)A)[tid];
					}
				}
			}
		}
	}
}


template <typename T, ALGO algo, NTT_MODE mode> __global__ void NTT_1D_(const Global::Globals* Globals,
                                                                        T* __restrict__ dat,
                                                                        const int __grid_constant__ primeid,
                                                                        T* __restrict__ res,
                                                                        const T* __restrict__ pt,
                                                                        const int __grid_constant__ pt_elem,
                                                                        const int __grid_constant__ primeid_rescale,
                                                                        T* __restrict__ res2,
                                                                        const T* __restrict__ kskb) {
	NTT_1D__<T, algo, mode>(Globals, dat, primeid, res, pt, pt_elem, primeid_rescale, res2, kskb);
}

#define VV(T, algo, mode)                                                                   \
template __global__ void FIDESlib::NTT_1D_<T, algo, mode>(const Global::Globals* Globals, \
T* __restrict__ dat,                                                                         \
const int __grid_constant__ primeid,                                                         \
T* __restrict__ res,                                                                         \
const T* __restrict__ pt,                                                                    \
const __grid_constant__ int pt_elem, \
const int __grid_constant__ primeid_rescale, \
T*__restrict__ res2, \
const T* __restrict__ kskb \
);
#include "ntt_types.inc"
#undef VV


template <ALGO algo, NTT_MODE mode> __global__ void NTT_1D_(const Global::Globals* Globals,
                                                            void** __restrict__ dat,
                                                            const int __grid_constant__ primeid_init,
                                                            void** __restrict__ res,
                                                            void** __restrict__ pt,
                                                            const __grid_constant__ int pt_elem,
                                                            const int __grid_constant__ primeid_rescale,
                                                            void** __restrict__ res2,
                                                            void** __restrict__ kskb) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	assert(primeid >= 0 && primeid < MAXP);
	if (ISU64(primeid)) {
		NTT_1D__<uint64_t, algo, mode>(Globals,
		                               (mode == NTT_RESCALE || mode == NTT_MULTPT) ? (uint64_t*)dat[0] : (uint64_t*)dat[blockIdx.y],
		                               primeid,
		                               (uint64_t*)res[blockIdx.y],
		                               pt ? (uint64_t*)pt[blockIdx.y] : nullptr,
		                               pt_elem,
		                               primeid_rescale,
		                               res2 ? (uint64_t*)res2[blockIdx.y] : nullptr,
		                               kskb ? (uint64_t*)kskb[blockIdx.y] : nullptr);

	} else {
		NTT_1D__<uint32_t, algo, mode>(Globals,
		                               (mode == NTT_RESCALE || mode == NTT_MULTPT) ? (uint32_t*)dat[0] : (uint32_t*)dat[blockIdx.y],
		                               primeid,
		                               (uint32_t*)res[blockIdx.y],
		                               pt ? (uint32_t*)pt[blockIdx.y] : nullptr,
		                               pt_elem,
		                               primeid_rescale,
		                               res2 ? (uint32_t*)res2[blockIdx.y] : nullptr,
		                               kskb ? (uint32_t*)kskb[blockIdx.y] : nullptr);
	}
}

#define YYY(algo, mode)                                                       \
template __global__ void NTT_1D_<algo, mode>(const Global::Globals* Globals, \
void** __restrict__ dat,                                                        \
const int __grid_constant__ primeid_init,                                       \
void** __restrict__ res,                                                        \
void** __restrict__ pt,                                                         \
const __grid_constant__ int pt_elem, \
const int __grid_constant__ primeid_rescale,                                    \
void** __restrict__ res2,                                                       \
void** __restrict__ kskb);
#include "ntt_types.inc"

#undef YYY
}