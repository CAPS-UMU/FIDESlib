//
// Created by carlosad on 4/04/24.
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

constexpr bool NEGACYCLIC = true;
// constexpr bool FUSEITERATIONS = true;

using Scheme = CKKS::Scheme;

// #define COOPERATIVE_GROUPS 1
template <typename T, ALGO algo, int M> __device__ __forceinline__ void forward_negacyclic_scale(
	char* buffer,
	const int primeid,
	T* psi,
	T* psi_shoup,
	const Global::Globals* Globals,
	const int logn) {

	const int tid = threadIdx.x;

	 {
		const uint32_t logBD = __clz(blockDim.x);
		// Now, try to load this from bit-reversed psi array
		uint32_t pos1 = tid & (~1);
		pos1          = __brev(pos1);
		pos1 >>= (logBD);

		// Normal order coefficients get scaled by root^(idx)
		// root^idx = root * (psi^{blockDim.x * (idx/(blockDim.x))}) * psi^{idx % (blockDim.x)}

		// root ^ (4*blockIdx.x + 4*gridDim.x*(threadIdx.x %2))
		T aux_3 = ((T*)G_->psi_no[primeid])[((tid & 1) * (gridDim.x * M) + M * blockIdx.x) << (C_.logN - logn)];

		T aux;
		if constexpr (algo == ALGO_SHOUP) {
			aux = modmult<algo>(aux_3, psi[pos1], primeid, psi_shoup[pos1]);
		} else {
			aux = modmult<algo>(aux_3, psi[pos1], primeid);
		}

		//const T root = C_.root[primeid];
		//T root_shoup;
		//if constexpr (algo == ALGO_SHOUP)
		//	root_shoup = C_.root_shoup[primeid];

		const T root = ((T*)G_->psi[primeid])[(1 << (logn - 1))];
		T root_shoup;
		if constexpr (algo == ALGO_SHOUP)
			root_shoup = ((T*)G_->psi_shoup[primeid])[(1 << (logn - 1))];

		const T fourth_root = psi[1];
		T fourth_root_shoup;
		if constexpr (algo == ALGO_SHOUP)
			fourth_root_shoup = psi_shoup[1];

		for (int i = 0; i < M; i += 1) {
			if (i > 0) {
				// aux = modmult<4>(aux, root, primeid);
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

template <typename T, bool second, ALGO algo, NTT_MODE mode> __device__ __forceinline__ void
NTT__(const Global::Globals* Globals,
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
	constexpr int M = sizeof(T) == 8 ? 4 : 8;
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
			T temp[2][2];
			const int col_init = j & ~2;
			for (int i = 0; i < M; i += 1) {
				int4 aux;
				const int pos_transp = M * gridDim.x * (col_init + i) + M * blockIdx.x + (j & 2);
				const int pos_res    = (col_init + i);
				//((int4*)aux)[0] = ((int4*)dat)[pos_transp >> 1];
				aux = ((int4*)dat)[pos_transp >> 1];
				// aux[0] = dat[pos_transp];
				// aux[1] = dat[pos_transp + 1];
				if constexpr (1) {
					A(j & 2)[pos_res]       = ((uint64_t*)&aux)[0];
					A((j & 2) + 1)[pos_res] = ((uint64_t*)&aux)[1];
				} else {
					temp[0][i & 1] = ((uint64_t*)&aux)[0];
					temp[1][i & 1] = ((uint64_t*)&aux)[1];
					if (i & 1) {
						((int4*)A((j & 2)))[(col_init + i) >> 1]     = ((int4*)temp[0])[0];
						((int4*)A((j & 2) + 1))[(col_init + i) >> 1] = ((int4*)temp[1])[0];
					}
				}
			}

			// if(blockIdx.x == 0)
			//     printf("tid: %d, colinit / 4: %d\n", tid, col_init >> 2);
		} else {
			const int col_init = j & ~2;
			int4 temp[4];
			for (int i = 0; i < M / 2; i += 1) {
				int4 aux;
				const int pos_transp = (M / 2) * (gridDim.x * (col_init + i) + blockIdx.x) + (j & 2);
				//                   const int pos_res = (col_init + i);
				aux               = ((int4*)dat)[pos_transp >> 1];
				((T*)&temp[0])[i] = aux.x;
				((T*)&temp[1])[i] = aux.y;
				((T*)&temp[2])[i] = aux.z;
				((T*)&temp[3])[i] = aux.w;
			}
			((int4*)A(2 * (j & 2)))[col_init >> 2]     = temp[0];
			((int4*)A(2 * (j & 2)) + 1)[col_init >> 2] = temp[1];
			((int4*)A(2 * (j & 2)) + 2)[col_init >> 2] = temp[2];
			((int4*)A(2 * (j & 2)) + 3)[col_init >> 2] = temp[3];
		}

		__syncthreads();
	}

	if constexpr (1) {
		if constexpr (mode == NTT_RESCALE || mode == NTT_MULTPT) {
			if constexpr (!second) {
				assert(primeid_rescale >= 0);
				for (int i = 0; i < M; i += 1) {
					CKKS::SwitchModulus(A(i)[tid], primeid_rescale, primeid);
					CKKS::SwitchModulus(A(i)[tid + blockDim.x], primeid_rescale, primeid);
				}
			}
		}

		if constexpr (sizeof(T) == 8 && !second && NEGACYCLIC) {
			forward_negacyclic_scale<T, algo, M>(buffer, primeid, psi, psi_barret, Globals, logN);
		}

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

		if constexpr (0) {
			if constexpr (1) {
				assert(m <= warpSize);
				for (; m >= 1; m >>= 1, log_psi--, maskPsi |= (maskPsi >> 1)) {
					const int psiid       = (tid & maskPsi) >> log_psi;
					const T psiaux        = psi[psiid];
					const T psiaux_barret = psi_barret[psiid];
					const int mask        = m - 1;
					int j1                = (mask & tid) | ((~mask & tid) << 1);
					int j2                = j1 + m;

					__syncwarp();
					/*
			if (tid & (warpSize >> 1)) {
				j1 += m;
				j2 -= m;
			}
			*/

					for (int i = 0; i < M; i += 1) {
						/*
				T aux1 = A(i)[j1];
				T aux2 = A(i)[j2];
				if (tid & (warpSize >> 1)) swap(aux1, aux2);
				CT_butterfly<T, algo>(aux1, aux2, psiaux, primeid);
				if (tid & (warpSize >> 1)) swap(aux1, aux2);
				A(i)[j1] = aux1;
				A(i)[j2] = aux2;
				*/
						T* A    = (T*)(buffer + (i << (logBD)));
						T& aux1 = A[j1];
						T& aux2 = A[j2];
						if constexpr (algo == 3) {
							CT_butterfly<T, algo>(aux1, aux2, psiaux, primeid, psiaux_barret);
						} else {
							CT_butterfly<T, algo>(aux1, aux2, psiaux, primeid);
						}
					}
				}
			} else if constexpr (0) {
				assert(m == warpSize);

				const int mask = m - 1;
				int j1         = (mask & tid) | ((~mask & tid) << 1);
				int j2         = j1 + m;

				for (int i = 0; i < M; i += 1) {

					int log_psi_ = log_psi;
					int maskPsi_ = maskPsi;

					T aux1, aux2;
					aux1 = A(i)[j1];
					aux2 = A(i)[j2];

					for (int m_ = m; m_ >= 1; m_ >>= 1, log_psi_--, maskPsi_ |= (maskPsi_ >> 1)) {
						const int psiid = (tid & maskPsi_) >> log_psi_;
						const T& psiaux = psi[psiid];

						if (m_ != warpSize) {
							if (tid & m_) {
								swap(aux1, aux2);
							}
							A(i)[tid] = aux2;
							//__shfl_xor_sync(0xFFFFFFFF, aux2, m_, m_ << 1);
							__syncwarp();
							aux2 = A(i)[tid ^ m_];
							if (tid & m_) {
								swap(aux1, aux2);
							}
						}

						CT_butterfly<T, algo>(aux1, aux2, psiaux, primeid);
					}

					T temp[2] = { aux1, aux2 };
					if (sizeof(T) == 8) {
						((int4*)A(i))[tid] = ((int4*)temp)[0];
					} else {
						((int2*)A(i))[tid] = ((int2*)temp)[0];
					}
				}
			} else {
				assert(m == warpSize);

				const int mask = m - 1;
				int j1         = (mask & tid) | ((~mask & tid) << 1);
				int j2         = j1 + m;

				for (int i = 0; i < M; i += 1) {

					int log_psi_ = log_psi;
					int maskPsi_ = maskPsi;

					T aux1, aux2;
					aux1 = A(i)[j1];
					aux2 = A(i)[j2];
					for (int m_ = m; m_ >= 1; m_ >>= 1, log_psi_--, maskPsi_ |= (maskPsi_ >> 1)) {
						const int psiid = (tid & maskPsi_) >> log_psi_;
						const T& psiaux = psi[psiid];

						if (m_ != warpSize) {
							if (tid & m_) {
								swap(aux1, aux2);
							}
							A(i)[tid] = aux2;
							//__shfl_xor_sync(0xFFFFFFFF, aux2, m_, m_ << 1);
							__syncwarp();
							aux2 = A(i)[tid ^ m_];
							if (tid & m_) {
								swap(aux1, aux2);
							}
						}

						CT_butterfly<T, algo>(aux1, aux2, psiaux, primeid);
					}

					T temp[2] = { aux1, aux2 };
					if (sizeof(T) == 8) {
						((int4*)A(i))[tid] = ((int4*)temp)[0];
					} else {
						((int2*)A(i))[tid] = ((int2*)temp)[0];
					}
				}
			}
		}

		// Idea: calcular full_psi en función de ambos arrays psi
		if (!second) {

			for (int i = 0; i < M; i += 1) {
				const T* A = (T*)(buffer + (i << (logBD)));
				int4 aux;

				{
					// Low bandwidth
					// index = j* bit_reverse(k, auxWidth), where j := blockIdx.x & k := 2*threadIdx.x + 1/0
					const uint32_t logBD       = 32 - __clz(blockDim.x);
					const uint32_t mask_lo_exp = (((N) >> 1) | ((N >> (logBD)) - 1));
					const uint32_t clzN        = __clz(N) + 2;
					const uint32_t block_pos   = (blockIdx.x * M + i);

					for (int k = 0; k < 2; ++k) {

						uint32_t br_j      = __brev(j + k) >> (32 - logBD);
						uint32_t exp       = block_pos * (br_j);
						uint32_t hi_exp_br = __brev(exp << clzN) & ((blockDim.x - 1));
						uint32_t lo_exp    = exp & mask_lo_exp;

						// printf("j: %d, br_j: %d, logBD: %d, exp: %o, hi_exp_br: %o, lo_exp: %o\n", j + k, br_j, logBD,
						//        exp, hi_exp_br, lo_exp);

						// assert(hi_exp_br == __brev((exp >> logBD) >> (33 - logBD)));
						//  assert(((T*)G_::psi_no[primeid])[exp] == ((T*)G_::psi_middle_scale[primeid])[OFFSET_T(i) + k]);
						// assert(((T*)G_::psi[primeid])[hi_exp_br] == psi[hi_exp_br]);

						// assert(modmult<ALGO_NATIVE>(psi[hi_exp_br], ((T*)G_::psi_no[primeid])[lo_exp * 2], primeid) ==
						//        ((T*)G_::psi_no[primeid])[2 * exp]);

						// assert(aux[k] == ((T*)G_::psi_middle_scale[primeid])[OFFSET_T(i) + k]);

						if constexpr (algo == 3) {
							((T*)&aux)[k] = modmult<algo>(((T*)G_->psi_no[primeid])[(lo_exp * 2) << (C_.logN - logN)],
							                              psi[hi_exp_br],
							                              primeid,
							                              psi_barret[hi_exp_br]);
						} else {
							((T*)&aux)[k] = modmult<algo>(psi[hi_exp_br],
							                              ((T*)G_->psi_no[primeid])[(lo_exp * 2) << (C_.logN - logN)],
							                              primeid);
						}
					}
				}

				if constexpr (algo == ALGO_SHOUP) {
					((T*)&aux)[0] = modmult<ALGO_BARRETT>(A[j], (T)((T*)&aux)[0], primeid);
					((T*)&aux)[1] = modmult<ALGO_BARRETT>(A[j + 1], (T)((T*)&aux)[1], primeid);
				} else {
					((T*)&aux)[0] = modmult<algo>(A[j], (T)((T*)&aux)[0], primeid);
					((T*)&aux)[1] = modmult<algo>(A[j + 1], (T)((T*)&aux)[1], primeid);
				}

				if constexpr (sizeof(T) == 8) {
					((int4*)res)[OFFSET_2T(i)] = aux;
				} else {
					((int2*)res)[OFFSET_2T(i)] = ((int2*)&aux)[0];
				}
			}

		} else {

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
						((int4*)res)[OFFSET_2T(i)] = ((int4*)A)[tid];
					} else {
						((int2*)res)[OFFSET_2T(i)] = ((int2*)A)[tid];
					}
				}
			}
		}
	}
#ifdef COOPERATIVE_GROUPS
	grid.sync();
	swap(res, dat);
	}
#endif
}

template <typename T, bool second, ALGO algo, NTT_MODE mode> __global__ void NTT_(const Global::Globals* Globals,
                                                                                  T* __restrict__ dat,
                                                                                  const int __grid_constant__ primeid,
                                                                                  T* __restrict__ res,
                                                                                  const T* __restrict__ pt,
                                                                                  const __grid_constant__ int pt_elem,
                                                                                  const int __grid_constant__ primeid_rescale,
                                                                                  T* __restrict__ res2,
                                                                                  const T* __restrict__ kskb) {
	NTT__<T, second, algo, mode>(Globals, dat, primeid, res, pt, pt_elem, primeid_rescale, res2, kskb);
}

#define V(T, second, algo, mode)                                                                   \
	template __global__ void FIDESlib::NTT_<T, second, algo, mode>(const Global::Globals* Globals, \
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
#undef V

template <bool second, ALGO algo, NTT_MODE mode> __global__ void NTT_(const Global::Globals* Globals,
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
		NTT__<uint64_t, second, algo, mode>(Globals,
		                                    (mode == NTT_RESCALE || mode == NTT_MULTPT) && !second ? (uint64_t*)dat[0] : (uint64_t*)dat[blockIdx.y],
		                                    primeid,
		                                    (uint64_t*)res[blockIdx.y],
		                                    pt ? (uint64_t*)pt[blockIdx.y] : nullptr,
		                                    pt_elem,
		                                    primeid_rescale,
		                                    res2 ? (uint64_t*)res2[blockIdx.y] : nullptr,
		                                    kskb ? (uint64_t*)kskb[blockIdx.y] : nullptr);

	} else {
		NTT__<uint32_t, second, algo, mode>(Globals,
		                                    (mode == NTT_RESCALE || mode == NTT_MULTPT) && !second ? (uint32_t*)dat[0] : (uint32_t*)dat[blockIdx.y],
		                                    primeid,
		                                    (uint32_t*)res[blockIdx.y],
		                                    pt ? (uint32_t*)pt[blockIdx.y] : nullptr,
		                                    pt_elem,
		                                    primeid_rescale,
		                                    res2 ? (uint32_t*)res2[blockIdx.y] : nullptr,
		                                    kskb ? (uint32_t*)kskb[blockIdx.y] : nullptr);
	}
}

#define VVV(second, algo, mode)                                                       \
	template __global__ void NTT_<second, algo, mode>(const Global::Globals* Globals, \
	  void** __restrict__ dat,                                                        \
	  const int __grid_constant__ primeid_init,                                       \
	  void** __restrict__ res,                                                        \
	  void** __restrict__ pt,                                                         \
	const __grid_constant__ int pt_elem, \
	  const int __grid_constant__ primeid_rescale,                                    \
	  void** __restrict__ res2,                                                       \
	  void** __restrict__ kskb);
#include "ntt_types.inc"

#undef VVV

template <typename T, int WARP_SIZE> __global__ void
NTT_1D(const Global::Globals* Globals,
       T* dat,
       const T* psi_dat,
       const int __grid_constant__ N,
       const int __grid_constant__ primeid,
       const int __grid_constant__ logN) {

	extern __shared__ char buffer[];
	T* psi = &(((T*)buffer)[blockDim.x * 8]);
	T* aux = &(((T*)buffer)[0]);

	int m                = N / 2;
	uint32_t log_psi_ext = logN - 1;
	int maskPsi          = m;
	// printf("m: %d\n", m);
	/*
		for(;m > blockDim.x; m >>= 1, maskPsi = (maskPsi << 1) | 1){
			__syncthreads();
			for(int tile = 0; tile < N/2; tile += blockDim.x){
				const int tid = tile + threadIdx.x;
				const int mask = m - 1;
				int j1 = (mask & tid) | ((~mask & tid) << 1);
				int j2 = j1 | m;

				const T & psiaux = psi_dat[tid & maskPsi];
				T &aux1 = dat[j1];
				T &aux2 = dat[j2];
				CT_butterfly<T, 4>(aux1, aux2, psiaux, primeid);
			}
		}
 */

	for (int tile = 0; tile < N / 2; tile += blockDim.x) {
		int m            = blockDim.x;
		int maskPsi2     = maskPsi;
		uint32_t log_psi = log_psi_ext;
		__syncthreads();
		aux[threadIdx.x]     = dat[2 * tile + threadIdx.x];
		aux[threadIdx.x + m] = dat[2 * tile + threadIdx.x + m];
		// printf("hola\n");
		//  todo una mascara para calcular j1 y otra para calcular la psi
		for (; m >= 1; m >>= 1, maskPsi2 = (maskPsi2 >> 1) | maskPsi2, --log_psi) {
			// if (m >= WARP_SIZE)
			__syncthreads();
			const int tid  = threadIdx.x;
			const int mask = m - 1;
			int j1         = (mask & tid) | ((~mask & tid) << 1);
			int j2         = j1 | m;

			const int psiid = ((tid + tile) & maskPsi2) >> log_psi;
			const T& psiaux = ((T*)G_->psi[primeid])[psiid];

			T& aux1 = aux[j1];
			T& aux2 = aux[j2];

			// printf("m: %d, j1: %d, j2: %d, psi: %d, psi_id: %d\n", m, j1, j2, (int) psiaux, (tid + tile) & maskPsi2);
			// printf("m: %d, j1: %d, j2: %d, psi: %d, psi_id: %d, a1: %d, a2: %d\n", m, j1, j2, (int)psiaux, psiid,
			//       (int)aux1, (int)aux2);
			CT_butterfly<T, ALGO_NATIVE>(aux1, aux2, psiaux, primeid);

			// printf("m: %d, j1: %d, j2: %d, psi: %d, psi_id: %d, a1: %d, a2: %d\n", m, j1, j2, (int)psiaux, psiid,
			//        (int)aux1, (int)aux2);
		}
		__syncthreads();
		if constexpr (sizeof(T) == 8) {
			((int4*)dat)[tile + threadIdx.x] = ((int4*)aux)[threadIdx.x];
		} else {
			((int2*)dat)[tile + threadIdx.x] = ((int2*)aux)[threadIdx.x];
		}
	}
}

template __global__ void NTT_1D<uint64_t>(const Global::Globals* Globals,
                                          uint64_t* __restrict__ dat,
                                          const uint64_t* __restrict__psi_dat,
                                          const int __grid_constant__ N,
                                          const int __grid_constant__ primeid,
                                          const int __grid_constant__ logN);

template <typename T, int WARP_SIZE> __global__ void
INTT_1D(const Global::Globals* Globals,
        T* dat,
        const T* psi_dat,
        const int __grid_constant__ N,
        const int __grid_constant__ primeid,
        const T N_inv,
        const int __grid_constant__ logN) {

	extern __shared__ char buffer[];
	// T* psi = &(((T*)buffer)[blockDim.x * 8]);
	T* aux = &(((T*)buffer)[0]);

	uint32_t log_psi_ext = 0;
	// int m = 1;
	int maskPsi = blockDim.x - 1;
	// printf("m: %d\n", m);

	for (int tile = 0; tile < N / 2; tile += blockDim.x) {
		int m            = 1;
		int maskPsi2     = maskPsi;
		uint32_t log_psi = log_psi_ext;
		__syncthreads();
		aux[threadIdx.x]              = dat[2 * tile + threadIdx.x];
		aux[threadIdx.x + blockDim.x] = dat[2 * tile + threadIdx.x + blockDim.x];
		// printf("hola\n");
		//  todo una mascara para calcular j1 y otra para calcular la psi
		for (; m <= blockDim.x; m <<= 1, maskPsi2 = (maskPsi2 << 1) & maskPsi2, ++log_psi) {

			// if (m >= WARP_SIZE)
			__syncthreads();
			const int tid  = threadIdx.x;
			const int mask = m - 1;
			int j1         = (mask & tid) | ((~mask & tid) << 1);
			int j2         = j1 | m;

			const int psiid = ((tid + tile) & maskPsi2) >> log_psi;
			const T& psiaux = ((T*)G_->psi[primeid])[psiid];

			T& aux1 = aux[j1];
			T& aux2 = aux[j2];

			// printf("m: %d, j1: %d, j2: %d, psi: %d, psi_id: %d\n", m, j1, j2, (int) psiaux, (tid + tile) & maskPsi2);
			//  printf("m: %d, j1: %d, j2: %d, psi: %d, psi_id: %d, a1: %d, a2: %d\n", m, j1, j2, (int)psiaux, psiid,
			//       (int)aux1, (int)aux2);
			GS_butterfly<T, ALGO_NATIVE>(aux1, aux2, psiaux, primeid);

			//  printf("m: %d, j1: %d, j2: %d, psi: %d, psi_id: %d, a1: %d, a2: %d\n", m, j1, j2, (int)psiaux, psiid,
			//       (int)aux1, (int)aux2);
		}
		__syncthreads();
		if constexpr (sizeof(T) == 8) {
			aux[threadIdx.x] = modmult<ALGO_NATIVE>(aux[threadIdx.x], N_inv, primeid);
			printf("aux: %d\n", (int)aux[threadIdx.x]);
			aux[threadIdx.x + blockDim.x] = modmult<ALGO_NATIVE>(aux[threadIdx.x + blockDim.x], N_inv, primeid);
			printf("aux: %d\n", (int)aux[threadIdx.x + blockDim.x]);
			__syncthreads();
			((int4*)dat)[tile + threadIdx.x] = ((int4*)aux)[threadIdx.x];
		} else {
			((int2*)dat)[tile + threadIdx.x] = ((int2*)aux)[threadIdx.x];
		}
	}
}

template __global__ void INTT_1D(const Global::Globals* Globals,
                                 uint64_t* __restrict__ dat,
                                 const uint64_t* __restrict__ psi_dat,
                                 const int __grid_constant__ N,
                                 const int __grid_constant__ primeid,
                                 const uint64_t N_inv,
                                 const int __grid_constant__ logN);

__global__ void test_kernel() {
}

void* get_NTT_reference(bool second) {
	if (second)
		return (void*)NTT_<uint64_t, true, ALGO_BARRETT>;
	else
		return (void*)NTT_<uint64_t, false, ALGO_BARRETT>;
}
} // namespace FIDESlib