//
// Created by carlosad on 4/04/24.
//
#include "AddSub.cuh"
#include "CKKS/Conv.cuh"
#include "ModMult.cuh"

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>

namespace FIDESlib::CKKS {

template <typename T> __global__ void conv1_(T* a, const T q_hat_inv, const int primeid) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	a[idx] = modmult(a[idx], q_hat_inv, primeid);
}

template __global__ void conv1_(uint32_t* a, const uint32_t q_hat_inv, const int primeid);

template __global__ void conv1_(uint64_t* a, const uint64_t q_hat_inv, const int primeid);

constexpr bool USING_CONSTANTS_TABLE = 0;

template <ALGO algo>
__global__ void
ModDown2(void** __restrict__ a, const __grid_constant__ int n, void** __restrict__ b, const __grid_constant__ int primeid_init, const Global::Globals* Globals) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	extern __shared__ char shared_mem[];

	uint64_t* buff = &((uint64_t*)shared_mem)[0];

	for (int i = threadIdx.y; i < C_.K; i += blockDim.y) {
		int primeid = i + C_.L;
		if constexpr (USING_CONSTANTS_TABLE) { // using constants table
			constexpr ALGO algo_ = algo == ALGO_SHOUP ? ALGO_BARRETT : algo;
			if (ISU64(primeid)) {
				buff[tid + blockDim.x * i] = modmult<algo_>(((uint64_t*)(b[i]))[idx], TABLE64(C_.L, C_.L + i), C_.L + i);
			} else {
				buff[tid + blockDim.x * i] = modmult<algo_>(((uint32_t*)b[i])[idx], (uint32_t)TABLE32(C_.L, C_.L + i), C_.L + i);
			}
		} else {
			if constexpr (algo != 3) {
				if (ISU64(primeid)) {
					buff[tid + blockDim.x * i] = modmult<algo>(((uint64_t*)(b[i]))[idx], G_->ModDown_pre_scale[primeid], primeid);
				} else {
					buff[tid + blockDim.x * i] = modmult<algo>((uint64_t)((uint32_t*)b[i])[idx], G_->ModDown_pre_scale[primeid], primeid);
				}
			} else {
				if (ISU64(primeid)) {
					buff[tid + blockDim.x * i] = modmult<algo>(((uint64_t*)(b[i]))[idx], G_->ModDown_pre_scale[primeid], primeid, G_->ModDown_pre_scale_shoup[primeid]);
				} else {
					buff[tid + blockDim.x * i] =
					  modmult<algo>((uint64_t)((uint32_t*)b[i])[idx], G_->ModDown_pre_scale[primeid], primeid, G_->ModDown_pre_scale_shoup[primeid]);
				}
			}
			/*
				if (idx == 0) {
					printf("Pre Scale from primeid:%d: %lu ", primeid,
						   G_::ModDown_pre_scale[primeid]);
					for (int i_ = 0; i_ < 2; ++i_) {
						printf("%lu ", buff[tid + i_ + blockDim.x * i]);
					}
					printf("\n");
				}
*/
		}
	}
	__syncthreads();

	for (int j = threadIdx.y; j < n; j += blockDim.y) {

		// if (idx == 0) printf("Matrix to %d: ", j);
		int primeid = C_.primeid_flattened[primeid_init + j];
		if constexpr (1) {
			__uint128_t res = 0;
			for (int i = 0; i < C_.K; ++i) {
				res = res + (__uint128_t)buff[i * blockDim.x + tid] * G_->ModDown_matrix[MODDOWN_MATRIX(i, primeid)];
			}

			// TODO use better reduction
			if (!ISU64(primeid)) {
				((uint32_t*)a[j])[idx] = (uint32_t)modreduce<ALGO_NATIVE>(res, primeid);
			} else {
				((uint64_t*)a[j])[idx] = (uint64_t)modreduce<ALGO_NATIVE>(res, primeid);
			}
		} else {
			uint64_t res = 0;
			for (int i = 0; i < C_.K; ++i) {
				if constexpr (USING_CONSTANTS_TABLE) { // using constants table
					constexpr ALGO algo_ = algo == ALGO_SHOUP ? ALGO_BARRETT : algo;
					uint64_t aux		 = modmult<algo_>(buff[i * blockDim.x + tid], (uint64_t)TABLE64(C_.L + i, j), j);
					// res = modadd(res, aux, j);
				} else {
					if constexpr (algo != 3) {

						uint64_t aux = modmult<algo>(buff[i * blockDim.x + tid], G_->ModDown_matrix[MODDOWN_MATRIX(i, j)], j);

						res = modadd(res, aux, j);

					} else {
						uint64_t aux =
						  modmult<algo>(buff[i * blockDim.x + tid], G_->ModDown_matrix[MODDOWN_MATRIX(i, j)], j, G_->ModDown_matrix_shoup[MODDOWN_MATRIX(i, j)]);
						res = modadd(res, aux, j);
					}
				}

				if (idx == 0)
					printf("%lu ", G_->ModDown_matrix[MODDOWN_MATRIX(i, j)]);
			}

			if (!ISU64(j)) {
				((uint32_t*)a[j])[idx] = (uint32_t)res;
			} else {
				((uint64_t*)a[j])[idx] = (uint64_t)res;
			}
		}
		/*
			if (idx == 0)
				printf("\n");
*/
	}
}

#define YY(algo)                             \
	template __global__ void ModDown2<algo>( \
	  void** __restrict__ a, const __grid_constant__ int n, void** __restrict__ b, const __grid_constant__ int primeid_init, const Global::Globals* Globals);

#include "ntt_types.inc"

#undef YY

template <ALGO algo>
__global__ void
ModDown3(void** __restrict__ a, const __grid_constant__ int n, void** __restrict__ b, const __grid_constant__ int primeid_init, const Global::Globals* Globals) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	extern __shared__ char shared_mem[];
	uint64_t* buff = ((uint64_t*)shared_mem);
	// constexpr int BLOCK_THREADS = 128;

	auto pipe = cuda::make_pipeline();

	// assert(BLOCK_THREADS == blockDim.x * blockDim.y);
	assert(blockDim.y == 2);
	// assert(n_d_n != 0);
	// assert(n_d_n <= 15);
	int n_half = (C_.K + 1) / 2;

	{
		constexpr int STAGES = 1;
		// Fill the pipeline
		for (int i_ = 0; i_ < STAGES && i_ < n_half; i_ += 1) {
			const int pos = i_ + n_half * threadIdx.y;

			const int primeid = pos + C_.L;
			pipe.producer_acquire();
			if (pos < C_.K) {
				if (ISU64(primeid)) {
					assert(b[pos] != nullptr);
					void* __restrict__ peer_ptr = b[pos];
					cuda::memcpy_async(reinterpret_cast<ulonglong2*>(buff) + (blockDim.x * pos) + tid,
					  reinterpret_cast<ulonglong2*>(peer_ptr) + (blockIdx.x * blockDim.x) + tid,
					  sizeof(ulonglong2),
					  pipe);

				} else {
				}
			}
			pipe.producer_commit();
		}

		for (int i_ = 0; i_ < n_half; i_ += 1) {
			const int pos	  = i_ + n_half * threadIdx.y;
			const int primeid = pos + C_.L;
			if (i_ + STAGES < n_half && pos + STAGES < C_.K) {
				pipe.producer_acquire();
				// const int primeid = C_.primeid_digit_from[d][pos + STAGES];

				if (ISU64(primeid)) {
					if (pos + STAGES < C_.K) {
						assert(b[pos + STAGES] != nullptr);
						void* __restrict__ peer_ptr = b[pos + STAGES];

						cuda::memcpy_async(reinterpret_cast<ulonglong2*>(buff) + (blockDim.x * (pos + STAGES)) + tid,
						  reinterpret_cast<ulonglong2*>(peer_ptr) + (blockIdx.x * blockDim.x) + tid,
						  sizeof(ulonglong2),
						  pipe);
					}
				} else {
				}
				pipe.producer_commit();
			}

			pipe.consumer_wait();

			if (pos < C_.K) {
				if constexpr (algo != 3) {
					if (ISU64(primeid)) {

						ulonglong2 data								= ((ulonglong2*)buff)[tid + blockDim.x * pos];
						data.x										= modmult<algo>(data.x, G_->ModDown_pre_scale[primeid], primeid);
						data.y										= modmult<algo>(data.y, G_->ModDown_pre_scale[primeid], primeid);
						((ulonglong2*)buff)[tid + blockDim.x * pos] = data;
					} else {
						buff[tid + blockDim.x * pos] = modmult<algo>(buff[tid + blockDim.x * pos], G_->ModDown_pre_scale[primeid], primeid);
					}
				} else {
					if (ISU64(primeid)) {
						ulonglong2 data = ((ulonglong2*)buff)[tid + blockDim.x * pos];
						data.x			= modmult<algo>(data.x, G_->ModDown_pre_scale[primeid], primeid, G_->ModDown_pre_scale_shoup[primeid]);

						data.y = modmult<algo>(data.y, G_->ModDown_pre_scale[primeid], primeid, G_->ModDown_pre_scale_shoup[primeid]);
						((ulonglong2*)buff)[tid + blockDim.x * pos] = data;
					} else {
						buff[tid + blockDim.x * pos] =
						  modmult<algo>(buff[tid + blockDim.x * pos], G_->ModDown_pre_scale[primeid], primeid, G_->ModDown_pre_scale_shoup[primeid]);
					}
				}
			}
			pipe.consumer_release();
		}
	}

	__syncthreads();
	for (int j_ = threadIdx.y; j_ < n; j_ += blockDim.y) {

		int primeid = C_.primeid_flattened[primeid_init + j_];

		__uint128_t resx = 0, resy = 0;
		for (int i_ = 0; i_ < C_.K; ++i_) {
			ulonglong2 data = ((ulonglong2*)buff)[i_ * blockDim.x + tid];
			// assert(MODUPIDX_MATRIX(n - 1, d, i_, primeid_j) < 64 * 64 * 64 * 8);
			resx = resx + (__uint128_t)data.x * G_->ModDown_matrix[MODDOWN_MATRIX(i_, primeid)];
			resy = resy + (__uint128_t)data.y * G_->ModDown_matrix[MODDOWN_MATRIX(i_, primeid)];
		}

		assert(a[j_] != nullptr);
		if (!ISU64(primeid)) {
			((uint32_t*)a[j_])[idx] = (uint32_t)modreduce<ALGO_NATIVE>(resx, primeid);
		} else {
			ulonglong2 res			  = { modreduce<ALGO_NATIVE>(resx, primeid), modreduce<ALGO_NATIVE>(resy, primeid) };
			((ulonglong2*)a[j_])[idx] = res;
		}
	}
}

#define YY(algo)                             \
	template __global__ void ModDown3<algo>( \
	  void** __restrict__ a, const __grid_constant__ int n, void** __restrict__ b, const __grid_constant__ int primeid_init, const Global::Globals* Globals);

#include "ntt_types.inc"

#undef YY

template <ALGO algo>
__global__ void
DecompAndModUpConv(void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b, const int __grid_constant__ d, const Global::Globals* Globals) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	extern __shared__ char shared_mem[];
	uint64_t* buff = ((uint64_t*)shared_mem);
	/*
		if (threadIdx.y == 0 && idx == 0) {
			for (int j = d; j < d + 1; ++j) {
				for (int k = 0; k < 64; ++k) {
					printf("%d %d:", j, k);
					for (int l = 0; l < 64; ++l) {
						printf("%lu ", G_::DecompAndModUp_pre_scale[MODUPIDX_SCALE(j, k, l)]);
					}
					printf("\n");
				}
			}

			for (int i = 0; i < 64; ++i) {
				for (int j = d; j < d + 1; ++j) {
					for (int k = 0; k < 64; ++k) {
						printf("%d %d %d:", i, j, k);
						for (int l = 0; l < 64; ++l) {
							printf("%lu ", G_::DecompAndModUp_matrix[MODUPIDX_MATRIX(i, j, k, l)]);
						}
						printf("\n");
					}
				}
			}

		}
*/

	const int n_d_n = C_.num_primeid_digit_from[d][n - 1];
	// assert(n_d_n != 0);
	for (int i_ = threadIdx.y; i_ < n_d_n; i_ += blockDim.y) {
		const int primeid = C_.primeid_digit_from[d][i_];
		const int pos	  = i_; // C_.pos_in_digit[d][primeid];
		assert(a[i_] != nullptr);
		if constexpr (algo != 3) {
			if (ISU64(primeid)) {
				void* __restrict__ peer_ptr = a[pos]; // Load pointer once
				uint64_t val				= __ldg(&(reinterpret_cast<uint64_t*>(peer_ptr))[idx]);

				buff[tid + blockDim.x * i_] =
				  modmult<algo>(/*((uint64_t*)(a[pos]))[idx]*/ val, G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
			} else {
				buff[tid + blockDim.x * i_] =
				  modmult<algo>((uint64_t)((uint32_t*)a[pos])[idx], G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
			}
		} else {
			if (ISU64(primeid)) {
				void* __restrict__ peer_ptr = a[pos]; // Load pointer once
				uint64_t val				= __ldg(&(reinterpret_cast<uint64_t*>(peer_ptr))[idx]);

				buff[tid + blockDim.x * i_] = modmult<algo>(/*((uint64_t*)(a[pos]))[idx]*/ val,
				  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
				  primeid,
				  G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
			} else {
				buff[tid + blockDim.x * i_] = modmult<algo>((uint64_t)((uint32_t*)a[pos])[idx],
				  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
				  primeid,
				  G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
			}
		}
		/*
			if (i_ == 0 && idx == 0) {
				printf("Pre Scale from d=%d, n_d_n-1=%d, i_:%d, primeid=%d, index:%d: %lu ", d, n_d_n - 1, i_, primeid,
					   MODUPIDX_SCALE(d, n_d_n - 1, primeid),
					   G_::DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
				for (int i = 0; i < 8; ++i) {
					printf("%lu ", buff[tid + i + blockDim.x * i_]);
				}
				printf("\n");
			}
*/
	}

	__syncthreads();

	// assert(C_.num_primeid_digit_to[d][n - 1] != 0);
	for (int j_ = threadIdx.y; j_ < C_.num_primeid_digit_to[d][n - 1]; j_ += blockDim.y) {
		// if (j_ == 0 && idx == 0) printf("Matrix to %d: ", j_);
		const int primeid_j = C_.primeid_digit_to[d][j_];
		if (primeid_j < n || primeid_j >= C_.L) {

			if constexpr (1) {

				__uint128_t res = 0;
				for (int i_ = 0; i_ < n_d_n; ++i_) {

					const int primeid = C_.primeid_digit_from[d][i_];

					assert(MODUPIDX_MATRIX(n - 1, d, i_, primeid_j) < 64 * 64 * 64 * 8);
					res = res + (__uint128_t)buff[i_ * blockDim.x + tid] * G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j)];

					if (0) {
						uint64_t aux = G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j)];
						printf("(%d, %d, %d, %d, %lu)", i_, primeid, primeid_j, MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j), aux);
					}
				}

				assert(b[j_] != nullptr);
				if (!ISU64(primeid_j)) {
					((uint32_t*)b[j_])[idx] = (uint32_t)modreduce<ALGO_NATIVE>(res, primeid_j);
				} else {
					((uint64_t*)b[j_])[idx] = (uint64_t)modreduce<ALGO_NATIVE>(res, primeid_j);
				}
			} else {
				uint64_t res = 0;
				for (int i_ = 0; i_ < n_d_n; ++i_) {
					assert(MODUPIDX_MATRIX(n - 1, d, i_, primeid_j) < 64 * 64 * 64 * 8);

					if constexpr (algo != 3) {
						uint64_t aux = modmult<algo>(buff[i_ * blockDim.x + tid], G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, i_, primeid_j)], primeid_j);
						res = modadd(res, aux, primeid_j);
					} else {
						uint64_t aux = modmult<algo>(buff[i_ * blockDim.x + tid],
						  G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, i_, primeid_j)],
						  primeid_j,
						  G_->DecompAndModUp_matrix_shoup[MODUPIDX_MATRIX(n - 1, d, i_, primeid_j)]);
						res			 = modadd(res, aux, primeid_j);
					}
					/*
					if (j_ == 0 && idx == 0)
						printf("%lu ", G_::DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, i_, primeid_j)]);
*/
				}

				assert(b[j_] != nullptr);
				if (!ISU64(primeid_j)) {
					((uint32_t*)b[j_])[idx] = (uint32_t)res;
				} else {
					((uint64_t*)b[j_])[idx] = (uint64_t)res;
				}
			}
		}
		/*
			if (j_ == 0 && idx == 0)
				printf("\n");
		*/
	}
}

#define YY(algo)                                       \
	template __global__ void DecompAndModUpConv<algo>( \
	  void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b, const int __grid_constant__ d, const Global::Globals* Globals);
#include "ntt_types.inc"

// ---- PTX named barrier wrappers ----

// bar.sync <barrier_id>, <thread_count>
//   Blocking: thread arrives AND waits for all <thread_count> threads.
__device__ __forceinline__ void named_bar_sync(int barrier_id, int thread_count) {
	asm volatile("bar.sync %0, %1;" ::"r"(barrier_id), "r"(thread_count));
}

// bar.arrive <barrier_id>, <thread_count>
//   Non-blocking: thread signals arrival but does NOT wait.
__device__ __forceinline__ void named_bar_arrive(int barrier_id, int thread_count) {
	asm volatile("bar.arrive %0, %1;" ::"r"(barrier_id), "r"(thread_count));
}

template <ALGO algo>
__global__ void
DecompAndModUpConv_spec(void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b, const int __grid_constant__ d, const Global::Globals* Globals) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	extern __shared__ char shared_mem[];
	uint64_t* buff				= ((uint64_t*)shared_mem);
	constexpr int BLOCK_THREADS = 192;

	assert(BLOCK_THREADS == blockDim.x * blockDim.y);
	assert(blockDim.y == 3);
	const int n_d_n = C_.num_primeid_digit_from[d][n - 1];
	// assert(n_d_n != 0);
	// assert(n_d_n <= 15);
	int n_half = (n_d_n + 1) / 2;
	if (threadIdx.y == 2) {
		const int warp_id = tid & (warpSize - 1);
		const int idx	  = (blockIdx.x * blockDim.x >> 1) + warp_id;
		int i_			  = 0;

		for (; i_ < (n_half); i_ += 1) {
			const int pos	  = i_ + n_half * (tid >= 32); // C_.pos_in_digit[d][primeid];
			const int primeid = C_.primeid_digit_from[d][pos];

			if (ISU64(primeid)) {
				if (pos < n_d_n) {
					assert(a[pos] != nullptr);
					void* __restrict__ peer_ptr										= a[pos]; // Load pointer once
					ulonglong2 val													= __ldg(&(reinterpret_cast<ulonglong2*>(peer_ptr))[idx]);
					reinterpret_cast<ulonglong2*>(buff)[warp_id + (warpSize * pos)] = val;
					// if (blockIdx.x == 10 && blockIdx.y == 0 && i_ == 0) {
					//	printf("idx: %d buffpos: %d\n", idx, warp_id + (warpSize * pos));
					// }
				}
			} else {
			}
			if (i_ > 0) {
				named_bar_sync(2 /*(i_ >> 1) % 15 + 1*/, BLOCK_THREADS);
			}
			named_bar_arrive(1 /*(i_ >> 1) % 15 + 1*/, BLOCK_THREADS);
		}
		named_bar_sync(2 /*(i_ >> 1) % 15 + 1*/, BLOCK_THREADS);
	} else {
		for (int i_ = 0; i_ < n_half; i_ += 1) {
			const int pos	  = i_ + n_half * threadIdx.y; // C_.pos_in_digit[d][primeid];
			const int primeid = C_.primeid_digit_from[d][pos];

			// named_bar_arrive((i_ >> 1) % 15 + 1, BLOCK_THREADS);
			named_bar_sync(1 /*(i_ >> 1) % 15 + 1*/, BLOCK_THREADS);
			named_bar_arrive(2 /*(i_ >> 1) % 15 + 1*/, BLOCK_THREADS);

			if (pos < n_d_n) {
				assert(a[pos] != nullptr);
				if constexpr (algo != 3) {
					if (ISU64(primeid)) {
						// void* __restrict__ peer_ptr = a[pos]; // Load pointer once
						// uint64_t val				= __ldg(&(reinterpret_cast<uint64_t*>(peer_ptr))[idx]);

						buff[tid + blockDim.x * pos] = modmult<algo>(
						  /*((uint64_t*)(a[pos]))[idx]*/ buff[tid + blockDim.x * pos], G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
					} else {
						buff[tid + blockDim.x * pos] =
						  modmult<algo>((uint64_t)((uint32_t*)a[pos])[idx], G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
					}
				} else {
					if (ISU64(primeid)) {
						// void* __restrict__ peer_ptr = a[pos]; // Load pointer once
						// uint64_t val				= __ldg(&(reinterpret_cast<uint64_t*>(peer_ptr))[idx]);

						buff[tid + blockDim.x * pos] = modmult<algo>(buff[tid + blockDim.x * pos] /*((uint64_t*)(a[pos]))[idx]
																								   */
						  ,
						  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
						  primeid,
						  G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
					} else {
						buff[tid + blockDim.x * pos] = modmult<algo>((uint64_t)((uint32_t*)a[pos])[idx],
						  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
						  primeid,
						  G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
					}
				}
			}
			// if ((i_ >> 1) + 1 < (n_d_n >> 1))
		}
	}

	__syncthreads();
	for (int j_ = threadIdx.y; j_ < C_.num_primeid_digit_to[d][n - 1]; j_ += blockDim.y) {
		const int primeid_j = C_.primeid_digit_to[d][j_];
		if (primeid_j < n || primeid_j >= C_.L) {

			__uint128_t res = 0;
			for (int i_ = 0; i_ < n_d_n; ++i_) {

				const int primeid = C_.primeid_digit_from[d][i_];

				assert(MODUPIDX_MATRIX(n - 1, d, i_, primeid_j) < 64 * 64 * 64 * 8);
				res = res + (__uint128_t)buff[i_ * blockDim.x + tid] * G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j)];
			}

			assert(b[j_] != nullptr);
			if (!ISU64(primeid_j)) {
				((uint32_t*)b[j_])[idx] = (uint32_t)modreduce<ALGO_NATIVE>(res, primeid_j);
			} else {
				((uint64_t*)b[j_])[idx] = (uint64_t)modreduce<ALGO_NATIVE>(res, primeid_j);
			}
		}
	}
}

#undef YY
#define YY(algo)                                            \
	template __global__ void DecompAndModUpConv_spec<algo>( \
	  void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b, const int __grid_constant__ d, const Global::Globals* Globals);
#include "ntt_types.inc"

#if 0
template <ALGO algo>
__global__ void
DecompAndModUpConv_spec2(void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b, const int __grid_constant__ d, const Global::Globals* Globals) {
	namespace cg = cooperative_groups;

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	extern __shared__ char shared_mem[];
	uint64_t* buff				= ((uint64_t*)shared_mem);
	constexpr int BLOCK_THREADS = 128;

	auto pipe = cuda::make_pipeline();

	assert(BLOCK_THREADS == blockDim.x * blockDim.y);
	assert(blockDim.y == 2);
	const int n_d_n = C_.num_primeid_digit_from[d][n - 1];
	// assert(n_d_n != 0);
	// assert(n_d_n <= 15);
	int n_half = (n_d_n + 1) / 2;

	{
		constexpr int STAGES = 1;
		// Fill the pipeline
		for (int i_ = 0; i_ < STAGES && i_ < n_half; i_ += 1) {
			const int pos = i_ + n_half * threadIdx.y;

			pipe.producer_acquire();
			if (pos < n_d_n) {
				const int primeid = C_.primeid_digit_from[d][pos];
				if (ISU64(primeid)) {
					assert(a[pos] != nullptr);
					void* __restrict__ peer_ptr = a[pos];
					cuda::memcpy_async(reinterpret_cast<uint64_t*>(buff) + (blockDim.x * pos) + tid,
					  reinterpret_cast<uint64_t*>(peer_ptr) + (blockIdx.x * blockDim.x) + tid,
					  sizeof(uint64_t),
					  pipe);

				} else {
				}
			}
			pipe.producer_commit();
		}

		for (int i_ = 0; i_ < n_half; i_ += 1) {
			const int pos = i_ + n_half * threadIdx.y;

			if (i_ + STAGES < n_half && pos + STAGES < n_d_n) {
				pipe.producer_acquire();
				const int primeid = C_.primeid_digit_from[d][pos + STAGES];
				if (ISU64(primeid)) {
					if (pos + STAGES < n_d_n) {
						assert(a[pos + STAGES] != nullptr);
						void* __restrict__ peer_ptr = a[pos + STAGES];

						cuda::memcpy_async(reinterpret_cast<uint64_t*>(buff) + (blockDim.x * (pos + STAGES)) + tid,
						  reinterpret_cast<uint64_t*>(peer_ptr) + (blockIdx.x * blockDim.x) + tid,
						  sizeof(uint64_t),
						  pipe);
					}
				} else {
				}
				pipe.producer_commit();
			}

			pipe.consumer_wait();
			if (pos < n_d_n) {

				const int primeid = C_.primeid_digit_from[d][pos];
				assert(a[pos] != nullptr);
				if constexpr (algo != 3) {
					if (ISU64(primeid)) {

						buff[tid + blockDim.x * pos] =
						  modmult<algo>(buff[tid + blockDim.x * pos], G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
					} else {
						buff[tid + blockDim.x * pos] =
						  modmult<algo>(buff[tid + blockDim.x * pos], G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
					}
				} else {
					if (ISU64(primeid)) {

						buff[tid + blockDim.x * pos] = modmult<algo>(buff[tid + blockDim.x * pos],
						  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
						  primeid,
						  G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
					} else {
						buff[tid + blockDim.x * pos] = modmult<algo>(buff[tid + blockDim.x * pos],
						  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
						  primeid,
						  G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
					}
				}
			}
			pipe.consumer_release();
		}
	}

	__syncthreads();
	for (int j_ = threadIdx.y; j_ < C_.num_primeid_digit_to[d][n - 1]; j_ += blockDim.y) {
		const int primeid_j = C_.primeid_digit_to[d][j_];
		if (primeid_j < n || primeid_j >= C_.L) {

			__uint128_t res = 0;
			for (int i_ = 0; i_ < n_d_n; ++i_) {

				const int primeid = C_.primeid_digit_from[d][i_];

				assert(MODUPIDX_MATRIX(n - 1, d, i_, primeid_j) < 64 * 64 * 64 * 8);
				res = res + (__uint128_t)buff[i_ * blockDim.x + tid] * G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j)];
			}

			assert(b[j_] != nullptr);
			if (!ISU64(primeid_j)) {
				((uint32_t*)b[j_])[idx] = (uint32_t)modreduce<ALGO_NATIVE>(res, primeid_j);
			} else {
				((uint64_t*)b[j_])[idx] = (uint64_t)modreduce<ALGO_NATIVE>(res, primeid_j);
			}
		}
	}
}

#else
template <ALGO algo>
__global__ void
DecompAndModUpConv_spec2(void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b, const int __grid_constant__ d, const Global::Globals* Globals) {
	namespace cg = cooperative_groups;

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	extern __shared__ char shared_mem[];
	uint64_t* buff = ((uint64_t*)shared_mem);
	// constexpr int BLOCK_THREADS = 128;

	auto pipe = cuda::make_pipeline();

	// assert(BLOCK_THREADS == blockDim.x * blockDim.y);
	assert(blockDim.y == 2);
	const int n_d_n = C_.num_primeid_digit_from[d][n - 1];
	// assert(n_d_n != 0);
	// assert(n_d_n <= 15);
	int n_half = (n_d_n + 1) / 2;

	{
		constexpr int STAGES = 1;
		// Fill the pipeline
		for (int i_ = 0; i_ < STAGES && i_ < n_half; i_ += 1) {
			const int pos = i_ + n_half * threadIdx.y;

			pipe.producer_acquire();
			if (pos < n_d_n) {
				const int primeid = C_.primeid_digit_from[d][pos];
				if (ISU64(primeid)) {
					assert(a[pos] != nullptr);
					void* __restrict__ peer_ptr = a[pos];
					cuda::memcpy_async(reinterpret_cast<ulonglong2*>(buff) + (blockDim.x * pos) + tid,
					  reinterpret_cast<ulonglong2*>(peer_ptr) + (blockIdx.x * blockDim.x) + tid,
					  sizeof(ulonglong2),
					  pipe);

				} else {
				}
			}
			pipe.producer_commit();
		}

		for (int i_ = 0; i_ < n_half; i_ += 1) {
			const int pos = i_ + n_half * threadIdx.y;

			if (i_ + STAGES < n_half && pos + STAGES < n_d_n) {
				pipe.producer_acquire();
				const int primeid = C_.primeid_digit_from[d][pos + STAGES];
				if (ISU64(primeid)) {
					if (pos + STAGES < n_d_n) {
						assert(a[pos + STAGES] != nullptr);
						void* __restrict__ peer_ptr = a[pos + STAGES];

						cuda::memcpy_async(reinterpret_cast<ulonglong2*>(buff) + (blockDim.x * (pos + STAGES)) + tid,
						  reinterpret_cast<ulonglong2*>(peer_ptr) + (blockIdx.x * blockDim.x) + tid,
						  sizeof(ulonglong2),
						  pipe);
					}
				} else {
				}
				pipe.producer_commit();
			}

			pipe.consumer_wait();
			if (pos < n_d_n) {

				const int primeid = C_.primeid_digit_from[d][pos];
				assert(a[pos] != nullptr);
				if constexpr (algo != 3) {
					if (ISU64(primeid)) {

						ulonglong2 data = ((ulonglong2*)buff)[tid + blockDim.x * pos];
						data.x			= modmult<algo>(data.x, G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
						data.y			= modmult<algo>(data.y, G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
						((ulonglong2*)buff)[tid + blockDim.x * pos] = data;
					} else {
						buff[tid + blockDim.x * pos] =
						  modmult<algo>(buff[tid + blockDim.x * pos], G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
					}
				} else {
					if (ISU64(primeid)) {
						ulonglong2 data = ((ulonglong2*)buff)[tid + blockDim.x * pos];
						data.x			= modmult<algo>(data.x,
						   G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
						   primeid,
						   G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);

						data.y										= modmult<algo>(data.y,
						   G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
						   primeid,
						   G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
						((ulonglong2*)buff)[tid + blockDim.x * pos] = data;
					} else {
						buff[tid + blockDim.x * pos] = modmult<algo>(buff[tid + blockDim.x * pos],
						  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
						  primeid,
						  G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
					}
				}
			}
			pipe.consumer_release();
		}
	}

	__syncthreads();
	for (int j_ = threadIdx.y; j_ < C_.num_primeid_digit_to[d][n - 1]; j_ += blockDim.y) {
		const int primeid_j = C_.primeid_digit_to[d][j_];
		if (primeid_j < n || primeid_j >= C_.L) {

			__uint128_t resx = 0, resy = 0;
			for (int i_ = 0; i_ < n_d_n; ++i_) {

				const int primeid = C_.primeid_digit_from[d][i_];

				ulonglong2 data = ((ulonglong2*)buff)[i_ * blockDim.x + tid];
				assert(MODUPIDX_MATRIX(n - 1, d, i_, primeid_j) < 64 * 64 * 64 * 8);
				resx = resx + (__uint128_t)data.x * G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j)];
				resy = resy + (__uint128_t)data.y * G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j)];
			}

			assert(b[j_] != nullptr);
			if (!ISU64(primeid_j)) {
				((uint32_t*)b[j_])[idx] = (uint32_t)modreduce<ALGO_NATIVE>(resx, primeid_j);
			} else {
				ulonglong2 res			  = { modreduce<ALGO_NATIVE>(resx, primeid_j), modreduce<ALGO_NATIVE>(resy, primeid_j) };
				((ulonglong2*)b[j_])[idx] = res; //(uint64_t)modreduce<ALGO_NATIVE>(res, primeid_j);
			}
		}
	}
}
#endif
#if 0
template <ALGO algo>
__global__ void
DecompAndModUpConv_spec2(void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b, const int __grid_constant__ d, const Global::Globals* Globals) {
	namespace cg = cooperative_groups;

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	extern __shared__ char shared_mem[];
	uint64_t* buff				= ((uint64_t*)shared_mem);
	constexpr int BLOCK_THREADS = 128;

	auto pipe = cuda::make_pipeline();

	assert(BLOCK_THREADS == blockDim.x * blockDim.y);
	assert(blockDim.y == 2);
	const int n_d_n = C_.num_primeid_digit_from[d][n - 1];
	// assert(n_d_n != 0);
	// assert(n_d_n <= 15);
	int n_half = (n_d_n + 1) / 2;

	{
		constexpr int STAGES = 1;
		// Fill the pipeline
		for (int i_ = 0; i_ < STAGES && i_ < n_half; i_ += 1) {
			const int pos = i_ + n_half * threadIdx.y;

			pipe.producer_acquire();
			if (pos < n_d_n) {
				const int primeid = C_.primeid_digit_from[d][pos];
				if (ISU64(primeid)) {
					assert(a[pos] != nullptr);
					void* __restrict__ peer_ptr = a[pos];
					cuda::memcpy_async(reinterpret_cast<ulonglong4*>(buff) + (blockDim.x * pos) + tid,
					  reinterpret_cast<ulonglong4*>(peer_ptr) + (blockIdx.x * blockDim.x) + tid,
					  sizeof(ulonglong4),
					  pipe);

				} else {
				}
			}
			pipe.producer_commit();
		}

		for (int i_ = 0; i_ < n_half; i_ += 1) {
			const int pos = i_ + n_half * threadIdx.y;

			if (i_ + STAGES < n_half && pos + STAGES < n_d_n) {
				pipe.producer_acquire();
				const int primeid = C_.primeid_digit_from[d][pos + STAGES];
				if (ISU64(primeid)) {
					if (pos + STAGES < n_d_n) {
						assert(a[pos + STAGES] != nullptr);
						void* __restrict__ peer_ptr = a[pos + STAGES];

						cuda::memcpy_async(reinterpret_cast<ulonglong4*>(buff) + (blockDim.x * (pos + STAGES)) + tid,
						  reinterpret_cast<ulonglong4*>(peer_ptr) + (blockIdx.x * blockDim.x) + tid,
						  sizeof(ulonglong4),
						  pipe);
					}
				} else {
				}
				pipe.producer_commit();
			}

			pipe.consumer_wait();
			if (pos < n_d_n) {

				const int primeid = C_.primeid_digit_from[d][pos];
				assert(a[pos] != nullptr);
				if constexpr (algo != 3) {
					if (ISU64(primeid)) {

						ulonglong4 data = ((ulonglong4*)buff)[tid + blockDim.x * pos];
						data.x			= modmult<algo>(data.x, G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
						data.y			= modmult<algo>(data.y, G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
						data.z			= modmult<algo>(data.z, G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
						data.w			= modmult<algo>(data.w, G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
						((ulonglong4*)buff)[tid + blockDim.x * pos] = data;
					} else {
						buff[tid + blockDim.x * pos] =
						  modmult<algo>(buff[tid + blockDim.x * pos], G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)], primeid);
					}
				} else {
					if (ISU64(primeid)) {
						ulonglong4 data = ((ulonglong4*)buff)[tid + blockDim.x * pos];
						data.x			= modmult<algo>(data.x,
						   G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
						   primeid,
						   G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);

						data.y = modmult<algo>(data.y,
						  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
						  primeid,
						  G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
						data.z = modmult<algo>(data.z,
						  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
						  primeid,
						  G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);

						data.w										= modmult<algo>(data.w,
						   G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
						   primeid,
						   G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
						((ulonglong4*)buff)[tid + blockDim.x * pos] = data;
					} else {
						buff[tid + blockDim.x * pos] = modmult<algo>(buff[tid + blockDim.x * pos],
						  G_->DecompAndModUp_pre_scale[MODUPIDX_SCALE(d, n_d_n - 1, primeid)],
						  primeid,
						  G_->DecompAndModUp_pre_scale_shoup[MODUPIDX_SCALE(d, n_d_n - 1, primeid)]);
					}
				}
			}
			pipe.consumer_release();
		}
	}

	__syncthreads();
	for (int j_ = threadIdx.y; j_ < C_.num_primeid_digit_to[d][n - 1]; j_ += blockDim.y) {
		const int primeid_j = C_.primeid_digit_to[d][j_];
		if (primeid_j < n || primeid_j >= C_.L) {

			__uint128_t resx = 0, resy = 0, resz = 0, resw = 0;
			for (int i_ = 0; i_ < n_d_n; ++i_) {

				const int primeid = C_.primeid_digit_from[d][i_];

				ulonglong4 data = ((ulonglong4*)buff)[i_ * blockDim.x + tid];
				assert(MODUPIDX_MATRIX(n - 1, d, i_, primeid_j) < 64 * 64 * 64 * 8);
				resx = resx + (__uint128_t)data.x * G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j)];
				resy = resy + (__uint128_t)data.y * G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j)];
				resz = resz + (__uint128_t)data.z * G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j)];
				resw = resw + (__uint128_t)data.w * G_->DecompAndModUp_matrix[MODUPIDX_MATRIX(n - 1, d, primeid /*i_*/, primeid_j)];
			}

			assert(b[j_] != nullptr);
			if (!ISU64(primeid_j)) {
				((uint32_t*)b[j_])[idx] = (uint32_t)modreduce<ALGO_NATIVE>(resx, primeid_j);
			} else {
				ulonglong4 res			  = { modreduce<ALGO_NATIVE>(resx, primeid_j),
							   modreduce<ALGO_NATIVE>(resy, primeid_j),
							   modreduce<ALGO_NATIVE>(resz, primeid_j),
							   modreduce<ALGO_NATIVE>(resw, primeid_j) };
				((ulonglong4*)b[j_])[idx] = res; //(uint64_t)modreduce<ALGO_NATIVE>(res, primeid_j);
			}
		}
	}
}
#endif

#undef YY
#define YY(algo)                                             \
	template __global__ void DecompAndModUpConv_spec2<algo>( \
	  void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b, const int __grid_constant__ d, const Global::Globals* Globals);
#include "ntt_types.inc"

#undef YY
} // namespace FIDESlib::CKKS
