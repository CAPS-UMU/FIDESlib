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

// namespace cg = cooperative_groups;

namespace FIDESlib {

using Scheme = CKKS::Scheme;

template <typename T, ALGO algo, int M> __device__ __forceinline__ void backward_negacyclic_scale(
	char* buffer,
	const int primeid,
	T* psi,
	T* psi_shoup,
	const Global::Globals* Globals,
	const int logn) {

	const int tid = threadIdx.x;

	if constexpr (0) {
		// High bandwidth
		for (int i = 0; i < M; i += 1) {
			A(i)
				[tid] = modmult<ALGO_BARRETT>(A(i)[tid], ((T*)G_->inv_psi_no[primeid])[tid * (gridDim.x * M) + M * blockIdx.x + i], primeid);
			A(i)
				[tid + blockDim.x] =
				modmult<ALGO_BARRETT>(A(i)[tid + blockDim.x],
				                      ((T*)G_->inv_psi_no[primeid])[(tid + blockDim.x) * (gridDim.x * M) + M * blockIdx.x + i],
				                      primeid);
			A(i)[tid]              = modmult<ALGO_SHOUP>(A(i)[tid], C_.N, primeid, C_.N_shoup[primeid]);
			A(i)[tid + blockDim.x] = modmult<ALGO_SHOUP>(A(i)[tid + blockDim.x], C_.N, primeid, C_.N_shoup[primeid]);
		}
	} else if constexpr (0) {
		// Now, try to load this from bit-reversed psi array TODO
		T aux[2] = { ((T*)G_->psi_no[primeid])[tid * (2 * blockDim.x) + M * blockIdx.x],
		             ((T*)G_->psi_no[primeid])[(tid + blockDim.x) * (2 * blockDim.x) + M * blockIdx.x] };
		T root = C_.root;
		for (int i = 0; i < M; i += 1) {
			if (i > 0) {
				aux[0] = modmult<4>(aux[0], root, primeid);
				aux[1] = modmult<4>(aux[1], root, primeid);
			}
			A(i)[tid]              = modmult<4>(A(i)[tid], aux[0], primeid);
			A(i)[tid + blockDim.x] = modmult<4>(A(i)[tid + blockDim.x], aux[1], primeid);
		}
	} else {
		const uint32_t logBD = __clz(blockDim.x);
		// Now, try to load this from bit-reversed psi array
		uint32_t pos1 = tid & (~1);
		pos1          = __brev(pos1);
		pos1 >>= (logBD);

		T aux_3 = ((T*)G_->inv_psi_no[primeid])[((tid & 1) * (gridDim.x * M) + M * blockIdx.x) << (C_.logN - logn)];

		aux_3 = modmult<ALGO_SHOUP>(aux_3, C_.N, primeid, C_.N_shoup[primeid]); // TODO: optimize this somehow

		T aux;
		if constexpr (algo == ALGO_SHOUP) {
			aux = modmult<algo>(aux_3, psi[pos1], primeid, psi_shoup[pos1]);
		} else {
			aux = modmult<algo>(aux_3, psi[pos1], primeid);
		}
		//const T root = C_.inv_root[primeid];
		//T root_shoup;
		//if constexpr (algo == ALGO_SHOUP)
		//	root_shoup = C_.inv_root_shoup[primeid];

		const T root = ((T*)G_->inv_psi[primeid])[(1 << (logn - 1))];
		T root_shoup;
		if constexpr (algo == ALGO_SHOUP)
			root_shoup = ((T*)G_->inv_psi_shoup[primeid])[(1 << (logn - 1))];

		T fourth_root = psi[1];
		T fourth_root_shoup;
		if constexpr (algo == ALGO_SHOUP)
			fourth_root_shoup = psi_shoup[1];

		if constexpr (algo == ALGO_SHOUP) {
			assert(fourth_root_shoup == ((T*)G_->inv_psi_shoup[primeid])[1]);
		}
		assert(fourth_root == ((T*)G_->inv_psi[primeid])[1]);

		// fourth_root = ((T*)G_::psi[primeid])[1];

		for (int i = 0; i < M; i += 1) {
			if (i > 0) {
				if constexpr (algo == ALGO_SHOUP) {
					aux = modmult<algo>(aux, root, primeid, root_shoup);
				} else {
					aux = modmult<algo>(aux, root, primeid);
				}
			}

			T aux2;
			if constexpr (algo == ALGO_SHOUP) {
				aux2 = modmult<algo>(aux, fourth_root, primeid, fourth_root_shoup);
			} else {
				aux2 = modmult<algo>(aux, fourth_root, primeid);
			}

			A(i)[tid]              = modmult<FIDESlib::ALGO_BARRETT>(A(i)[tid], aux, primeid);
			A(i)[tid + blockDim.x] = modmult<FIDESlib::ALGO_BARRETT>(A(i)[tid + blockDim.x], aux2, primeid);
		}
	}
}

template <typename T, bool second, ALGO algo, INTT_MODE mode> __device__ __forceinline__ void INTT__(const Global::Globals* Globals,
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
	constexpr int M = sizeof(T) == 8 ? 4 : 8;

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

		if constexpr (mode == INTT_MULT_AND_SAVE && !second) {
			mult_and_save_fusion<T, algo, M>(buffer, logBD, j, primeid, (T*)dat, (T*)dat2, (T*)c0, (T*)dat, (T*)kska, (T*)kskb, (T*)c0, (T*)c0tilde);
		} else if constexpr (mode == INTT_MULT_AND_ACC && !second) {
			mult_and_acc_fusion<T, algo, M>(buffer, logBD, j, primeid, (T*)dat, (T*)dat2, (T*)c0, (T*)dat, (T*)kska, (T*)kskb, (T*)c0, (T*)c0tilde);
		} else if constexpr (mode == INTT_ROTATE_AND_SAVE && !second) {
			rotate_and_save_fusion<T, algo, M>(buffer, logBD, j, primeid, (T*)dat, (T*)c0, (T*)dat, (T*)kska, (T*)kskb, (T*)c0);
		} else if constexpr (mode == INTT_SQUARE_AND_SAVE && !second) {
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

	if constexpr (second) {
		for (int i = 0; i < M; ++i) {
			T psi_aux[2];
			{
				// index = j* bit_reverse(k, auxWidth), where j := blockIdx.x & k := 2*threadIdx.x + 1/0
				const uint32_t logBD       = 32 - __clz(blockDim.x);
				const uint32_t mask_lo_exp = (((N) >> 1) | ((N >> (logBD)) - 1));
				const uint32_t clzN        = __clz(N) + 2;
				const uint32_t block_pos   = (blockIdx.x * M + i);

				for (int k = 0; k < 2; ++k) {

					uint32_t br_j      = __brev(j + k) >> (32 - logBD);
					uint32_t exp       = block_pos * (br_j);
					uint32_t hi_exp_br = __brev(exp << clzN) & (blockDim.x - 1);
					uint32_t lo_exp    = exp & mask_lo_exp;

					if constexpr (algo == 3) {
						psi_aux[k] = modmult<algo>(((T*)G_->inv_psi_no[primeid])[(lo_exp << 1) << (C_.logN - logN)],
						                           psi[hi_exp_br],
						                           primeid,
						                           psi_shoup[hi_exp_br]);
					} else {
						psi_aux[k] = modmult<algo>(psi[hi_exp_br], ((T*)G_->inv_psi_no[primeid])[(lo_exp << 1) << (C_.logN - logN)], primeid);
					}

					if (C_.logN - logN) {
						psi_aux[k] = modmult<ALGO_BARRETT>(psi_aux[k], 1 << (C_.logN - logN), primeid);
					}
				}
			}

			if constexpr (algo == FIDESlib::ALGO_SHOUP) {
				A(i)[j]     = modmult<FIDESlib::ALGO_BARRETT>(A(i)[j], psi_aux[0], primeid);
				A(i)[j + 1] = modmult<FIDESlib::ALGO_BARRETT>(A(i)[j + 1], psi_aux[1], primeid);
			} else {
				A(i)[j]     = modmult<algo>(A(i)[j], psi_aux[0], primeid);
				A(i)[j + 1] = modmult<algo>(A(i)[j + 1], psi_aux[1], primeid);
			}
		}
	}

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

	if constexpr (sizeof(T) == 8 && second && NEGACYCLIC) {
		backward_negacyclic_scale<T, algo, M>(buffer, primeid, psi, psi_shoup, Globals, logN);
	}

	{
		__syncthreads();
		/*
		if (OFFSET_2T(0) == 0 && primeid == 0)
			printf("INTT write %lu %lu\n", A(0)[0], A(0)[1]);
		*/
		if constexpr (sizeof(T) == 8) {
			const int col_init = j & ~2;
			for (int i = 0; i < M; ++i) {
				int4 aux;
				const int pos_trasp = (M * gridDim.x) * (col_init + i) + M * blockIdx.x + (j & 2);
				const int pos_res   = (col_init + i);
				assert(pos_trasp < gridDim.x * 2 * blockDim.x * M);
				((T*)&aux)[0] = A((j & 2))[pos_res];
				((T*)&aux)[1] = A((j & 2) + 1)[pos_res];

				((int4*)res)[pos_trasp >> 1] = aux;
			}
		} else {
			const int col_init = j & ~2;
			for (int i = 0; i < M / 2; ++i) {
				int4 aux;
				const int pos_trasp = (M / 2) * ((gridDim.x) * (col_init + i) + blockIdx.x) + (j & 2);
				const int pos_res   = (col_init + i);
				assert(pos_trasp < gridDim.x * 2 * blockDim.x * M);
				aux.x                        = A(2 * (j & 2))[pos_res];
				aux.y                        = A(2 * (j & 2) + 1)[pos_res];
				aux.z                        = A(2 * (j & 2) + 2)[pos_res];
				aux.w                        = A(2 * (j & 2) + 3)[pos_res];
				((int4*)res)[pos_trasp >> 2] = aux;
			}
		}
	}
}


template <typename T, bool second, ALGO algo, INTT_MODE mode> __global__ void INTT_(const Global::Globals* Globals,
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

	INTT__<T, second, algo, mode>(Globals, dat, primeid, res, dat2, res0, res1, kska, kskb, c0, c0tilde);
}

#define W(T, second, algo, mode)                                                          \
template __global__ void INTT_<T, second, algo, mode>(const Global::Globals* Globals, \
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

#undef W

template <bool second, ALGO algo, INTT_MODE mode> __global__ void INTT_(const Global::Globals* Globals,
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
		INTT__<uint64_t, second, algo, mode>(Globals,
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
		INTT__<uint32_t, second, algo, mode>(Globals,
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

#define WW(second, algo, mode)                                                         \
template __global__ void INTT_<second, algo, mode>(const Global::Globals* Globals, \
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

#undef WW
}