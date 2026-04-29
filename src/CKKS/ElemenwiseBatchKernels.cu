//
// Created by carlosad on 27/09/24.
//

#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/Rescale.cuh"
#include "Rotation.cuh"

#include <cooperative_groups.h>
#include <cuda/barrier>
namespace cg = cooperative_groups;

namespace FIDESlib ::CKKS {
__global__ void mult1AddMult23Add4_(const __grid_constant__ int primeid_init, void** l, void** l1, void** l2, void** l3, void** l4) {
	const int primeid	= C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx		= threadIdx.x + blockDim.x * blockIdx.x;
	constexpr ALGO algo = ALGO_BARRETT;

	if (ISU64(primeid)) {
		using T					 = uint64_t;
		T aux					 = ((T*)l4[blockIdx.y])[idx];
		T res					 = modmult<algo>(((T*)l[blockIdx.y])[idx], ((T*)l1[blockIdx.y])[idx], primeid);
		res						 = modadd(res, aux, primeid);
		res						 = modadd(res, modmult<algo>(((T*)l2[blockIdx.y])[idx], ((T*)l3[blockIdx.y])[idx], primeid), primeid);
		((T*)l[blockIdx.y])[idx] = res;
	} else {
		using T = uint32_t;
	}
}

__global__ void multnomoddownend_(const __grid_constant__ int primeid_init, void** c1, void** c0, void** bc0, void** bc1, void** in, void** aux) {
	const int primeid	= C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx		= threadIdx.x + blockDim.x * blockIdx.x;
	constexpr ALGO algo = ALGO_BARRETT;

	if (ISU64(primeid)) {
		using T = uint64_t;
		T aux0	= ((T*)in[blockIdx.y])[idx];
		T res = modmult<ALGO_SHOUP>(modmult<algo>(((T*)c1[blockIdx.y])[idx], ((T*)bc0[blockIdx.y])[idx], primeid), C_.P[primeid], primeid, C_.P_shoup[primeid]);
		res	  = modadd(res, aux0, primeid);
		res	  = modadd(
			res, modmult<ALGO_SHOUP>(modmult<algo>(((T*)c0[blockIdx.y])[idx], ((T*)bc1[blockIdx.y])[idx], primeid), C_.P[primeid], primeid, C_.P_shoup[primeid]), primeid);
		((T*)c1[blockIdx.y])[idx] = res;
		aux0					  = ((T*)aux[blockIdx.y])[idx];
		res = modmult<ALGO_SHOUP>(modmult<algo>(((T*)c0[blockIdx.y])[idx], ((T*)bc0[blockIdx.y])[idx], primeid), C_.P[primeid], primeid, C_.P_shoup[primeid]);
		res = modadd(res, aux0, primeid);

		((T*)c0[blockIdx.y])[idx] = res;
	} else {
		using T = uint32_t;
	}
}

__global__ void mult1Add2_(const __grid_constant__ int primeid_init, void** l, void** l1, void** l2) {
	const int primeid	= C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx		= threadIdx.x + blockDim.x * blockIdx.x;
	constexpr ALGO algo = ALGO_BARRETT;

	if (ISU64(primeid)) {
		using T					 = uint64_t;
		T aux					 = ((T*)l2[blockIdx.y])[idx];
		T res					 = modmult<algo>(((T*)l[blockIdx.y])[idx], ((T*)l1[blockIdx.y])[idx], primeid);
		((T*)l[blockIdx.y])[idx] = modadd(res, aux, primeid);
	} else {
		using T					 = uint32_t;
		T aux					 = ((T*)l2[blockIdx.y])[idx];
		T res					 = modmult<algo>(((T*)l[blockIdx.y])[idx], ((T*)l1[blockIdx.y])[idx], primeid);
		((T*)l[blockIdx.y])[idx] = modadd(res, aux, primeid);
	}
}

template <typename T> __device__ __forceinline__ void addMult__(T* l, const T* l1, const T* l2, const int primeid) {
	const int idx		= threadIdx.x + blockDim.x * blockIdx.x;
	constexpr ALGO algo = ALGO_BARRETT;

	l[idx] = modadd(l[idx], modmult<algo>(l1[idx], l2[idx], primeid), primeid);
}

template <typename T> __global__ void addMult_(T* l, const T* l1, const T* l2, const __grid_constant__ int primeid) {
	addMult__<T>(l, l1, l2, primeid);
}

__global__ void addMult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

	//    if (threadIdx.x + blockDim.x * blockIdx.x == 0)
	//        printf("%d %d\n", primeid_init + blockIdx.y, primeid);
	if (ISU64(primeid)) {
		addMult__<uint64_t>((uint64_t*)l[blockIdx.y], (uint64_t*)l1[blockIdx.y], (uint64_t*)l2[blockIdx.y], primeid);
	} else {
		addMult__<uint32_t>((uint32_t*)l[blockIdx.y], (uint32_t*)l1[blockIdx.y], (uint32_t*)l2[blockIdx.y], primeid);
	}
}

__global__ void Mult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = threadIdx.x + blockDim.x * blockIdx.x;

	//    if (idx == 0)
	//        printf("%d %d\n", primeid_init + blockIdx.y, primeid);
	if (ISU64(primeid)) {
		((uint64_t*)l[blockIdx.y])[idx] = modmult<ALGO_BARRETT>(((uint64_t*)l1[blockIdx.y])[idx], ((uint64_t*)l2[blockIdx.y])[idx], primeid);
	} else {
		((uint32_t*)l[blockIdx.y])[idx] = modmult<ALGO_BARRETT>(((uint32_t*)l1[blockIdx.y])[idx], ((uint32_t*)l2[blockIdx.y])[idx], primeid);
	}
}

__global__ void square_(void** l, void** l1, const __grid_constant__ int primeid_init) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = threadIdx.x + blockDim.x * blockIdx.x;

	if (ISU64(primeid)) {
		uint64_t in						= ((uint64_t*)l1[blockIdx.y])[idx];
		((uint64_t*)l[blockIdx.y])[idx] = modmult<ALGO_BARRETT>(in, in, primeid);
	} else {
		uint32_t in						= ((uint64_t*)l1[blockIdx.y])[idx];
		((uint32_t*)l[blockIdx.y])[idx] = modmult<ALGO_BARRETT>(in, in, primeid);
	}
};

__global__ void binomial_square_fold_(void** c0_res, void** c2_key_switched_0, void** c1, void** c2_key_switched_1, const __grid_constant__ int primeid_init) {
	int idx			  = threadIdx.x + blockIdx.x * blockDim.x;
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

	if (ISU64(primeid)) {
		uint64_t in2_0						 = ((uint64_t*)c2_key_switched_0[blockIdx.y])[idx];
		uint64_t in2_1						 = ((uint64_t*)c2_key_switched_1[blockIdx.y])[idx];
		uint64_t in0						 = ((uint64_t*)c0_res[blockIdx.y])[idx];
		uint64_t ok							 = modadd(modmult<ALGO_BARRETT>(in0, in0, primeid), in2_0, primeid);
		((uint64_t*)c0_res[blockIdx.y])[idx] = ok;
		uint64_t in1						 = ((uint64_t*)c1[blockIdx.y])[idx];
		uint64_t aux						 = modmult<ALGO_BARRETT>(in0, in1, primeid);
		uint64_t aux2						 = modadd(aux, aux, primeid);
		ok									 = modadd(aux2, in2_1, primeid);
		((uint64_t*)c1[blockIdx.y])[idx]	 = ok;
	} else {
	}
}

__global__ void broadcastLimb0_(void** a) {
	int idx			  = threadIdx.x + blockIdx.x * blockDim.x;
	const int primeid = blockIdx.y + 1;
	if (ISU64(primeid) && ISU64(0)) {
		uint64_t in = ((uint64_t*)a[0])[idx];
		SwitchModulus(in, 0, primeid);
		((uint64_t*)a[primeid])[idx] = in;
	}
}

__global__ void broadcastLimb0_mgpu(void** a, const __grid_constant__ int primeid_init, void** limb0) {
	int idx			  = threadIdx.x + blockIdx.x * blockDim.x;
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	if (ISU64(primeid) && ISU64(0)) {
		uint64_t in = ((uint64_t*)limb0[0])[idx];
		SwitchModulus(in, 0, primeid);
		((uint64_t*)a[blockIdx.y])[idx] = in;
	}
}

__global__ void copy_(void** a, void** b) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (ISU64(blockIdx.y)) {
		((uint64_t*)b[blockIdx.y])[idx] = ((uint64_t*)a[blockIdx.y])[idx];
	} else {
		((uint32_t*)b[blockIdx.y])[idx] = ((uint32_t*)a[blockIdx.y])[idx];
	}
}

__global__ void copy1D_(void* a, void* b) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	((uint64_t*)b)[idx] = ((uint64_t*)a)[idx];
}

template <ALGO algo> __global__ void Scalar_mult_(void** a, const uint64_t* b, const __grid_constant__ int primeid_init, const uint64_t* shoup_mu) {
	int idx			  = threadIdx.x + blockIdx.x * blockDim.x;
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

	if (ISU64(primeid)) {
		((uint64_t*)a[blockIdx.y])[idx] = modmult<algo>(((uint64_t*)a[blockIdx.y])[idx], b[primeid], primeid, shoup_mu ? shoup_mu[primeid] : 0);
	} else {
		((uint32_t*)a[blockIdx.y])[idx] = modmult<algo>(((uint32_t*)a[blockIdx.y])[idx], (uint32_t)b[primeid], primeid, (uint32_t)(shoup_mu ? shoup_mu[primeid] : 0));
	}
}

__global__ void eval_linear_w_sum_(const __grid_constant__ int n, void** a, void*** bs, uint64_t* w, const __grid_constant__ int primeid_init) {
	int idx				= threadIdx.x + blockIdx.x * blockDim.x;
	const int primeid	= C_.primeid_flattened[primeid_init + blockIdx.y];
	constexpr ALGO algo = ALGO_BARRETT;

	{
		uint64_t res = modmult<algo>(((uint64_t*)(bs[0])[blockIdx.y])[idx], w[primeid], primeid);
		for (int i = 1; i < n; ++i) {
			uint64_t temp = modmult<algo>(((uint64_t*)(bs[i])[blockIdx.y])[idx], w[i * MAXP + primeid], primeid);
			res			  = modadd(res, temp, primeid);
		}
		((uint64_t*)a[blockIdx.y])[idx] = res;
	}
}

__global__ void fusedDotKSK_2_(void** out1, void** sout1, void** out2, void** sout2, void*** digits, int num_d, int id, int num_special, int init) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;

	const int blky = blockIdx.y + init;
	// num_special = C_.K;
	int primeid;
	if (blky < num_special) {
		primeid = C_.primeid_digit_to[0][blky];
	} else {
		primeid = C_.primeid_partition[id][blky - num_special];
	}

	const int primeid_digit = C_.primeid_digit[primeid];

	int pos_dec = blky - num_special;

	/*
	int i = 0;
	bool decomp = (i == primeid_digit);
	int pos = C_.pos_in_digit[i][primeid];

	uint64_t aux1, aux2;

	uint64_t in = ((uint64_t*)digits[0 + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
	aux1 = modmult<ALGO_BARRETT>(in, ((uint64_t*)digits[C_.dnum + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx],
								 primeid);
	aux2 = modmult<ALGO_BARRETT>(
		in, ((uint64_t*)digits[2 * C_.dnum + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);

	for (i = 1; i < num_d; ++i) {
		decomp = (i == primeid_digit);
		pos = C_.pos_in_digit[i][primeid];
		in = ((uint64_t*)digits[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
		uint64_t add1 = modmult<ALGO_BARRETT>(
			in, ((uint64_t*)digits[C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
		uint64_t add2 = modmult<ALGO_BARRETT>(
			in, ((uint64_t*)digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
		aux1 = modadd(aux1, add1, primeid);
		aux2 = modadd(aux2, add2, primeid);
	}
	*/
	uint64_t aux1, aux2;

	for (int i = 0; i < num_d; ++i) {
		bool decomp = (i == primeid_digit);
		int pos		= C_.pos_in_digit[i][primeid];

		// printf("Digit %d: in: %p\n", i, digits);
		// printf("Digit %d: in: %p, kska: %p, kskb: %p\n", i, digits[i + decomp * 3 * C_.dnum],
		//        digits[C_.dnum + i + decomp * 3 * C_.dnum], digits[2 * C_.dnum + i + decomp * 3 * C_.dnum]);

		uint64_t in	  = ((uint64_t*)digits[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
		uint64_t add1 = modmult<ALGO_BARRETT>(in, ((uint64_t*)digits[C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
		uint64_t add2 = modmult<ALGO_BARRETT>(in, ((uint64_t*)digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);

		if (i == 0) {
			aux1 = add1;
			aux2 = add2;
		} else {
			aux1 = modadd(aux1, add1, primeid);
			aux2 = modadd(aux2, add2, primeid);
		}
	}

	if (primeid < C_.L) {
		((uint64_t*)out1[pos_dec])[idx] = aux1;
		((uint64_t*)out2[pos_dec])[idx] = aux2;
	} else {
		((uint64_t*)sout1[primeid - C_.L])[idx] = aux1;
		((uint64_t*)sout2[primeid - C_.L])[idx] = aux2;
	}
}

constexpr bool PRINT = false;

__global__ void hoistedRotateDotKSK_2_(void*** din1,
  void** c0,
  void*** out1,
  void*** sout1,
  void*** out2,
  void*** sout2,
  const int n,
  const int* indexes,
  void*** digits,
  int num_d,
  int id,
  int num_special,
  int init,
  void** sc0,
  bool c0_modup) {
	const int idx  = threadIdx.x + blockIdx.x * blockDim.x;
	const int blky = blockIdx.y + init;

	const int primeid = (blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];

	const int primeid_digit = C_.primeid_digit[primeid];
	const int pos_dec		= blky - num_special;

	extern __shared__ char buffer[];

	uint64_t* in1 = ((uint64_t*)buffer) + num_d * threadIdx.x;

	for (int i = 0; i < num_d; ++i) {
		bool decomp = (i == primeid_digit);
		int pos		= C_.pos_in_digit[i][primeid];
		in1[i]		= ((uint64_t*)din1[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];

		if (PRINT && idx == 0 && blky == 6)
			printf("In %d : %lu\n", i, in1[i]);
	}

	uint64_t in2 = 0;

	if (c0_modup || primeid < C_.L) {
		in2 = primeid < C_.L ? ((uint64_t*)c0[pos_dec])[idx] : ((uint64_t*)sc0[primeid - C_.L])[idx];
		if (!c0_modup && primeid < C_.L)
			in2 = modmult<ALGO_SHOUP>(in2, C_.P[primeid], primeid, C_.P_shoup[primeid]);
	}

	if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 0)
		printf("In c0: %lu\n", in2);

	for (int j = 0; j < n; ++j) {
		uint64_t aux1, aux2;
		int offset = j * 3 * 2 * C_.dnum;

		for (int i = 0; i < num_d; ++i) {
			bool decomp = (i == primeid_digit);
			int pos		= C_.pos_in_digit[i][primeid];
			uint64_t add1 = modmult<ALGO_BARRETT>(in1[i], ((uint64_t*)digits[offset + C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);

			if (PRINT && idx == 0 && blky == 6)
				printf("kska %d : %lu\n", i, ((uint64_t*)digits[offset + C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx]);

			if (PRINT && idx == 0 && blky == 6)
				printf("add1 %d : %lu\n", i, add1);
			uint64_t add2 = modmult<ALGO_BARRETT>(in1[i], ((uint64_t*)digits[offset + 2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);

			if (PRINT && idx == 0 && blky == 6)
				printf("kskb %d : %lu\n", i, ((uint64_t*)digits[offset + 2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx]);
			if (PRINT && idx == 0 && blky == 6)
				printf("add2 %d : %lu\n", i, add2);

			if (i == 0) {
				aux1 = add1;
				if (c0_modup || primeid < C_.L) {
					aux2 = modadd(in2, add2, primeid);
				} else {
					aux2 = add2;
				}
			} else {
				aux1 = modadd(aux1, add1, primeid);
				aux2 = modadd(aux2, add2, primeid);
			}
			if (PRINT && idx == 0 && blky == 6)
				printf("aux1 %d : %lu\n", i, aux1);
			if (PRINT && idx == 0 && blky == 6)
				printf("aux2 %d : %lu\n", i, aux2);
		}

		if (PRINT && idx == 0 && blky == 6)
			printf("%d : %d %d %d %lu\n", j, indexes[j], C_.logN, automorph_slot(C_.logN, indexes[j], idx), aux1);
		if (PRINT && idx == 0 && blky == 6)
			printf("%d : %d %d %d %lu\n", j, indexes[j], C_.logN, automorph_slot(C_.logN, indexes[j], idx), aux2);

		uint32_t out_idx = automorph_slot(C_.logN, indexes[j], idx);
		// uint32_t out_idx = idx;
		if (primeid < C_.L) {
			((uint64_t*)out1[j][pos_dec])[out_idx] = aux1;
			((uint64_t*)out2[j][pos_dec])[out_idx] = aux2;
		} else {
			((uint64_t*)sout1[j][primeid - C_.L])[out_idx] = aux1;
			((uint64_t*)sout2[j][primeid - C_.L])[out_idx] = aux2;
		}
	}
}

__global__ void hoistedRotateDotKSKBatched___(void*** c1,
  void*** din1,
  void*** c0,
  void*** sc0,
  void*** out1,
  void*** sout1,
  void*** out2,
  void*** sout2,
  const int n,
  const int* indexes,
  void*** digits,
  int num_d,
  int id,
  int num_special,
  int init_,
  bool c0_modup) {
	// cg::thread_block tb = cg::this_thread_block();
	const int idx  = threadIdx.x + blockIdx.x * blockDim.x;
	const int blky = blockIdx.y + init_;

	const int primeid = (blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];

	const int primeid_digit = C_.primeid_digit[primeid];
	const int pos_dec		= blky - num_special;
	//__shared__ cuda::barrier<cuda::thread_scope_block> bar;
	extern __shared__ char buffer[];

	// Initialize barrier (single thread)
	// if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
	//    init(&bar, blockDim.x * blockDim.y * blockDim.z);
	//}
	//__syncthreads();

	int stride_in  = blockDim.x * blockDim.z;
	int stride_ksk = blockDim.x * blockDim.y;
	// uint64_t* in_base = ((uint64_t*)buffer);
	// uint64_t* ksk_base = ((uint64_t*)buffer) + num_d * stride_in;
	uint64_t* in1 = ((uint64_t*)buffer) + (threadIdx.x + blockDim.x * threadIdx.z);
	uint64_t* ksk = ((uint64_t*)buffer) + num_d * stride_in + (threadIdx.y * blockDim.x + threadIdx.x);

	if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 0)
		printf("in1: %p, ksk: %p\n", in1, ksk);

	// int n_elem = blockDim.x;
	/*
	for (int z = 0; z < blockDim.z; z++) {
		for (uint32_t i = 0; i < num_d; ++i) {
			bool decomp = (i == primeid_digit);
			int pos = C_.pos_in_digit[i][primeid];
			cuda::memcpy_async(
				tb, in_base + n_elem * z + stride_in * i,
				((uint64_t*)(decomp ? c1[z] : din1[num_d * z + i])[decomp ? pos_dec : pos]) + blockIdx.x * blockDim.x,
				cuda::aligned_size_t<16>(n_elem * sizeof(uint64_t)), bar);
		}
	}
*/
	/*
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		for (uint32_t i = 0; i < num_d; ++i) {
			bool decomp = (i == primeid_digit);
			int pos = C_.pos_in_digit[i][primeid];
			cuda::memcpy_async(
				in_base + n_elem * threadIdx.z + stride_in * i,
				((uint64_t*)(decomp ? c1[threadIdx.z] : din1[num_d * threadIdx.z + i])[decomp ? pos_dec : pos]) +
					blockIdx.x * blockDim.x,
				cuda::aligned_size_t<16>(n_elem * sizeof(uint64_t)), bar);
		}
	}
*/

	for (uint32_t i = threadIdx.y; i < num_d; i += blockDim.y) {
		bool decomp = (i == primeid_digit);
		int pos		= C_.pos_in_digit[i][primeid];

		in1[i * stride_in] = ((uint64_t*)(decomp ? c1[threadIdx.z] : din1[num_d * threadIdx.z + i])[decomp ? pos_dec : pos])[idx];

		if (PRINT && idx == 0 && blky == 6)
			printf("In %d : %lu\n", i, in1[i * stride_in]);
	}
	__syncthreads();
	uint64_t in2 = 0;
	if ((c0_modup && threadIdx.y == 1) || primeid < C_.L) {
		in2 = threadIdx.y == 1 ? ((primeid < C_.L) ? ((uint64_t*)c0[threadIdx.z][pos_dec])[idx] : ((uint64_t*)sc0[threadIdx.z][primeid - C_.L])[idx]) :
								 in1[stride_in * primeid_digit];
		if (!c0_modup || threadIdx.y == 0)
			in2 = modmult<ALGO_SHOUP>(in2, C_.P[primeid], primeid, C_.P_shoup[primeid]);
	}

	if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 1)
		printf("In c0: %lu\n", in2);

	for (int j = 0; j < n; ++j) {

		if (j > 0)
			__syncthreads();

		if (digits[(2 * j + threadIdx.y) * (num_d + 1)]) {
			uint64_t aux;
			for (int i = threadIdx.z; i < num_d; i += blockDim.z) {
				bool decomp = (i == primeid_digit);
				int pos		= C_.pos_in_digit[i][primeid];
				/*
			if (threadIdx.x == 0) {


				cuda::memcpy_async(
					ksk_base + n_elem * threadIdx.y + stride_ksk * i,
					((uint64_t*)
						 digits[(2 * j + threadIdx.y) * (num_d + 1) + (decomp ? num_d : i)][decomp ? pos_dec : pos]) +
						blockIdx.x * blockDim.x,
					cuda::aligned_size_t<16>(n_elem * sizeof(uint64_t)), bar);
			}*/
				uint64_t** ksk_from = (uint64_t**)digits[(2 * j + threadIdx.y) * (num_d + 1) + (decomp ? num_d : i)];

				ksk[i * stride_ksk] = ksk_from[decomp ? pos_dec : pos][idx];

				if (PRINT && idx == 0 && blky == 6)
					printf("ksk %d : %lu\n", i, ksk[i * stride_ksk]);
			}
			// bar.arrive_and_wait();
			__syncthreads();

			for (int i = 0; i < num_d; ++i) {
				uint64_t add = modmult<ALGO_BARRETT>(in1[i * stride_in], ksk[i * stride_ksk], primeid);

				if (PRINT && idx == 0 && blky == 6)
					printf("add %d : %lu\n", i, add);
				if (i == 0) {
					if ((c0_modup || primeid < C_.L) && threadIdx.y == 1) {
						aux = modadd(in2, add, primeid);
					} else {
						aux = add;
					}
				} else {
					aux = modadd(aux, add, primeid);
				}
				if (PRINT && idx == 0 && blky == 6)
					printf("aux %d : %lu\n", i, aux);
			}

			if (PRINT && idx == 0 && blky == 6)
				printf("%d : %d %d %d %lu\n", j, indexes[j], C_.logN, automorph_slot(C_.logN, indexes[j], idx), aux);

			uint32_t out_idx = automorph_slot(C_.logN, indexes[j], idx);
			// uint32_t out_idx = idx;
			uint64_t* out = (primeid < C_.L) ?
			  (threadIdx.y == 0 ? (uint64_t*)out1[threadIdx.z * n + j][pos_dec] : (uint64_t*)out2[threadIdx.z * n + j][pos_dec]) :
			  (threadIdx.y == 0 ? (uint64_t*)sout1[threadIdx.z * n + j][primeid - C_.L] : (uint64_t*)sout2[threadIdx.z * n + j][primeid - C_.L]);

			out[out_idx] = aux;
		} else {

			uint64_t* out = (primeid < C_.L) ?
			  (threadIdx.y == 0 ? (uint64_t*)out1[threadIdx.z * n + j][pos_dec] : (uint64_t*)out2[threadIdx.z * n + j][pos_dec]) :
			  (threadIdx.y == 0 ? (uint64_t*)sout1[threadIdx.z * n + j][primeid - C_.L] : (uint64_t*)sout2[threadIdx.z * n + j][primeid - C_.L]);

			if (PRINT && idx == 0)
				printf("y: %d Primeid_digit: %d, in2: %lu %lu\n", blky, primeid_digit, in2, in1[stride_in * (primeid_digit + (primeid_digit == -1))]);

			out[idx] = in2;
		}
	}
}

/*
__global__ void hoistedRotateDotKSKBatched___(void*** c1, void*** din1, void*** c0, void*** sc0, void*** out1,
											  void*** sout1, void*** out2, void*** sout2, const int n,
											  const int* indexes, void*** digits, int num_d, int id, int num_special,
											  int init, bool c0_modup) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int blky = blockIdx.y + init;

	const int primeid =
		(blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];

	const int primeid_digit = C_.primeid_digit[primeid];
	const int pos_dec = blky - num_special;

	extern __shared__ char buffer[];

	int stride_in = blockDim.x * blockDim.z;
	int stride_ksk = blockDim.x * blockDim.y;
	uint64_t* in1 = ((uint64_t*)buffer) + (threadIdx.x + blockDim.x * threadIdx.z);
	uint64_t* ksk = ((uint64_t*)buffer) + num_d * stride_in + (threadIdx.x * blockDim.y + threadIdx.y);

	if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 0)
		printf("in1: %p, ksk: %p\n", in1, ksk);

	for (int i = threadIdx.y; i < num_d; i += blockDim.y) {
		bool decomp = (i == primeid_digit);
		int pos = C_.pos_in_digit[i][primeid];
		in1[i * stride_in] =
			((uint64_t*)(decomp ? c1[threadIdx.z] : din1[num_d * threadIdx.z + i])[decomp ? pos_dec : pos])[idx];

		if (PRINT && idx == 0 && blky == 6)
			printf("In %d : %lu\n", i, in1[i * stride_in]);
	}

	uint64_t in2 = 0;
	if ((c0_modup || primeid < C_.L) && threadIdx.y == 1) {
		in2 = ((uint64_t*)(primeid < C_.L ? c0 : sc0)[threadIdx.z][pos_dec])[idx];
		if (!c0_modup)
			in2 = modmult<ALGO_SHOUP>(in2, C_.P[primeid], primeid, C_.P_shoup[primeid]);
	}

	if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 1)
		printf("In c0: %lu\n", in2);

	for (int j = 0; j < n; ++j) {
		uint64_t aux;

		if (j > 0)
			__syncthreads();
		for (int i = threadIdx.z; i < num_d; i += blockDim.z) {
			bool decomp = (i == primeid_digit);
			int pos = C_.pos_in_digit[i][primeid];
			ksk[i * stride_ksk] =
				((uint64_t*)
					 digits[(2 * j + threadIdx.y) * (num_d + 1) + (decomp ? num_d : i)][decomp ? pos_dec : pos])[idx];

			if (PRINT && idx == 0 && blky == 6)
				printf("ksk %d : %lu\n", i, ksk[i * stride_ksk]);
		}
		__syncthreads();
		for (int i = 0; i < num_d; ++i) {
			uint64_t add = modmult<ALGO_BARRETT>(in1[i * stride_in], ksk[i * stride_ksk], primeid);

			if (PRINT && idx == 0 && blky == 6)
				printf("add %d : %lu\n", i, add);
			if (i == 0) {
				if ((c0_modup || primeid < C_.L) && threadIdx.y == 1) {
					aux = modadd(in2, add, primeid);
				} else {
					aux = add;
				}
			} else {
				aux = modadd(aux, add, primeid);
			}
			if (PRINT && idx == 0 && blky == 6)
				printf("aux %d : %lu\n", i, aux);
		}

		if (PRINT && idx == 0 && blky == 6)
			printf("%d : %d %d %d %lu\n", j, indexes[j], C_.logN, automorph_slot(C_.logN, indexes[j], idx), aux);

		uint32_t out_idx = automorph_slot(C_.logN, indexes[j], idx);
		//uint32_t out_idx = idx;
		uint64_t* out = (primeid < C_.L) ? (threadIdx.y == 0 ? (uint64_t*)out1[threadIdx.z * n + j][pos_dec]
															 : (uint64_t*)out2[threadIdx.z * n + j][pos_dec])
										 : (threadIdx.y == 0 ? (uint64_t*)sout1[threadIdx.z * n + j][primeid - C_.L]
															 : (uint64_t*)sout2[threadIdx.z * n + j][primeid - C_.L]);

		out[out_idx] = aux;
	}
}
*/

__global__ void dotProductPt_(void** c0, void** c1, void*** data, const size_t ptroffset, const int primeidInit, const int n) {
	int idx				= threadIdx.x + blockIdx.x * blockDim.x;
	const int primeid	= C_.primeid_flattened[primeidInit + blockIdx.y];
	constexpr ALGO algo = ALGO_BARRETT;

	uint64_t out0, out1;
	uint64_t in = ((uint64_t*)data[n * 2][ptroffset + blockIdx.y])[idx];
	if (PRINT && idx == 0 && blockIdx.y == 0)
		printf("LT:, b: %d, g:, in pt: %lu \n", 0, in);
	out0 = modmult<algo>(in, ((uint64_t*)data[0][ptroffset + blockIdx.y])[idx], primeid);

	if (PRINT && idx == 0 && blockIdx.y == 0)
		printf("LT: %d, b: %d, in c0: %lu \n", -1, 0, ((uint64_t*)data[0][ptroffset + blockIdx.y])[idx]);

	out1 = modmult<algo>(in, ((uint64_t*)data[n][ptroffset + blockIdx.y])[idx], primeid);

	if (PRINT && idx == 0 && blockIdx.y == 0)
		printf("LT: %d, b: %d, in c1: %lu \n", -1, 0, ((uint64_t*)data[n][ptroffset + blockIdx.y])[idx]);

	for (int i = 1; i < n; ++i) {
		in = ((uint64_t*)data[n * 2 + i][ptroffset + blockIdx.y])[idx];

		if (PRINT && idx == 0 && blockIdx.y == 0)
			printf("LT:, b: %d, g:, in pt: %lu \n", i, in);

		uint64_t aux0 = modmult<algo>(in, ((uint64_t*)data[i][ptroffset + blockIdx.y])[idx], primeid);

		if (PRINT && idx == 0 && blockIdx.y == 0)
			printf("LT: %d, b: %d, in c0: %lu \n", -1, i, ((uint64_t*)data[i][ptroffset + blockIdx.y])[idx]);

		uint64_t aux1 = modmult<algo>(in, ((uint64_t*)data[n + i][ptroffset + blockIdx.y])[idx], primeid);

		if (PRINT && idx == 0 && blockIdx.y == 0)
			printf("LT: %d, b: %d, in c1: %lu \n", -1, i, ((uint64_t*)data[n + i][ptroffset + blockIdx.y])[idx]);
		out0 = modadd(out0, aux0, primeid);
		out1 = modadd(out1, aux1, primeid);
	}
	if (PRINT && idx == 0 && blockIdx.y == 0)
		printf("LT: , g: , res: %lu \n", out0);
	if (PRINT && idx == 0 && blockIdx.y == 0)
		printf("LT: , g: , res: %lu \n", out1);
	((uint64_t*)c0[ptroffset + blockIdx.y])[idx] = out0;
	((uint64_t*)c1[ptroffset + blockIdx.y])[idx] = out1;
}

/*
__global__ void dotProductLtBatchedPt___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
										 const int batch, const int gStep, const int primeidInit, const int n) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int& b = blockDim.z;
	const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
	constexpr ALGO algo = ALGO_BARRETT;

	extern __shared__ char buffer[];

	// Shared required: (2*batch+1/2)*threads_per_block
	const int in_stride = blockDim.x * blockDim.y * blockDim.z;
	const int block_id = (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
	uint64_t* in = ((uint64_t*)buffer) + block_id;

	__uint128_t* acc_this_thread = (__uint128_t*)(((uint64_t*)buffer) + in_stride * batch) + block_id;
	uint64_t* pt = ((uint64_t*)buffer) + (batch + 2) * in_stride + (threadIdx.x + blockDim.x * threadIdx.z);

	const bool im_c0 = threadIdx.y == 0;
	const int b_idx = threadIdx.z;

	if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0) {
		printf("in: %lu, acc: %lu, pt: %lu\n", in, acc_this_thread, pt);
	}

	void*** inputs = im_c0 ? c0_in : c1_in;
	void*** outputs = im_c0 ? c0_out : c1_out;

	for (int k = blockIdx.z; k < n; k += gridDim.z) {
		for (int i = 0; i < batch; ++i) {
			in[in_stride * i] = ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx];
			if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0)
				printf("LT: %d, b: %d, in c0: %lu %lu %p %p\n", k, b_idx, in[in_stride * i],
					   ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
					   inputs[k * b * batch + i * b + b_idx], inputs);
			if (PRINT && idx == 0 && blockIdx.y == 0 && !im_c0)
				printf("LT: %d, b: %d, in c1: %lu %lu %p %p\n", k, b_idx, in[in_stride * i],
					   ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
					   inputs[k * b * batch + i * b + b_idx], inputs);
		}

		for (int j = 0; j < gStep; ++j) {
			void** pt_partition = pts[k * b * gStep + j * b + b_idx];
			if (pt_partition != nullptr) {
				if (!im_c0) {
					pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
					if (PRINT && idx == 0 && blockIdx.y == 0)
						printf("LT: %d, b: %d, g: %d, in pt: %lu %lu %p %p\n", k, b_idx, j, pt[0],
							   ((uint64_t*)pt_partition[blockIdx.y])[idx], pt_partition, pts);
				}
			}

			__syncwarp();
			for (int i = 0; i < batch; ++i) {
				if (pt_partition != nullptr) {
					acc_this_thread[0] = (__uint128_t)in[in_stride * i] * pt[0];
				} else {
					acc_this_thread[0] = 0;
				}

				if constexpr (0) {
					__syncthreads();
					if (threadIdx.z == 0) {
						__uint128_t res = acc_this_thread[0];
						for (int i = 1; i < blockDim.z; ++i) {
							res = res + acc_this_thread[i * blockDim.x * blockDim.y];
						}

						((uint64_t*)outputs[k * batch * gStep + i * gStep + j][blockIdx.y])[idx] =
							modreduce<ALGO_NATIVE>(res, primeid);
						if (PRINT && idx == 0 && blockIdx.y == 0)
							printf("LT: %d, g: %d, res: %lu \n", k, j, acc_this_thread[0]);
					}
					__syncthreads();
				} else {
					const int r_init = 1 << (32 - __clz(b - 1) - 1);
					int r = r_init;
					if (r > 0) {
						__syncthreads();
						if (threadIdx.z + r < b) {
							acc_this_thread[0] = acc_this_thread[0] + acc_this_thread[0 + r * blockDim.x * blockDim.y];
						}
					}

					r >>= 1;
					for (; r > 0; r >>= 1) {
						__syncthreads();
						if (threadIdx.z < r) {
							acc_this_thread[0] = acc_this_thread[0] + acc_this_thread[0 + r * blockDim.x * blockDim.y];
						}
					}
					if (threadIdx.z == 0) {

						((uint64_t*)outputs[k * batch * gStep + i * gStep + j][blockIdx.y])[idx] =
							modreduce<ALGO_NATIVE>(acc_this_thread[0], primeid);
						if (PRINT && idx == 0 && blockIdx.y == 0)
							printf("LT: %d, g: %d, res: %lu \n", k, j, acc_this_thread[0]);
					}
				}
			}
		}
	}
}*/

__global__ void
dotProductLtBatchedPt2___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts, const int bStep, const int gStep, const int primeidInit, const int n) {
	int idx = threadIdx.x + threadIdx.z * blockDim.x + blockIdx.x * blockDim.x * blockDim.z;
	// int b = blockDim.z;
	const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
	// constexpr ALGO algo = ALGO_BARRETT;

	extern __shared__ char buffer[];

	// Shared required: (2*batch+1/2)*threads_per_block
	const int in_stride = blockDim.x * blockDim.y * blockDim.z;
	const int block_id	= (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
	// uint64_t* in = ((uint64_t*)buffer) + block_id;
	// uint64_t in[6];

	// uint64_t* acc = ((uint64_t*)buffer) + in_stride * batch;
	uint64_t* acc_this_thread = ((uint64_t*)buffer) + block_id;
	// uint64_t* pt = acc + 2 * in_stride + (threadIdx.x + blockDim.x * threadIdx.z);

	// int r_init = 1 << (32 - __clz(b - 1) - 1);
	bool im_c0 = threadIdx.y == 0;
	// const int b_idx = threadIdx.z;

	// uint64_t pt;

	void*** inputs	= im_c0 ? c0_in : c1_in;
	void*** outputs = im_c0 ? c0_out : c1_out;

	for (int k = blockIdx.z; k < n; k += gridDim.z) {

		for (int i = 0; i < bStep; ++i) {
			uint64_t in = ((uint64_t*)inputs[k * bStep + i][blockIdx.y])[idx];

			for (int j = 0; j < gStep; ++j) {
				void** pt_partition = pts[k * bStep * gStep + j * bStep + i];

				uint64_t mult = 0;
				if (pt_partition != nullptr) {
					// pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
					mult = modmult<ALGO_BARRETT>(in, ((uint64_t*)pt_partition[blockIdx.y])[idx], primeid);
				}
				if (i == 0)
					acc_this_thread[j * in_stride] = mult;
				else
					acc_this_thread[j * in_stride] = modadd(acc_this_thread[j * in_stride], mult, primeid);
				if (i == bStep - 1)
					((uint64_t*)outputs[k * gStep + j][blockIdx.y])[idx] = acc_this_thread[j * in_stride];
			}
		}
	}
}

__global__ void
dotProductLtBatchedPt3___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts, const int bStep, const int gStep, const int primeidInit, const int n) {
	int idx = threadIdx.x + threadIdx.z * blockDim.x + blockIdx.x * blockDim.x * blockDim.z;
	// int b = blockDim.z;
	const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
	// constexpr ALGO algo = ALGO_BARRETT;

	extern __shared__ char buffer[];

	// Shared required: (2*batch+1/2)*threads_per_block
	const int in_stride = blockDim.x * blockDim.y * blockDim.z;
	const int block_id	= (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
	// uint64_t* in = ((uint64_t*)buffer) + block_id;
	// uint64_t in[6];

	// uint64_t* acc = ((uint64_t*)buffer) + in_stride * batch;
	__uint128_t* acc_this_thread = ((__uint128_t*)buffer) + block_id;
	// uint64_t* pt = acc + 2 * in_stride + (threadIdx.x + blockDim.x * threadIdx.z);

	// int r_init = 1 << (32 - __clz(b - 1) - 1);
	bool im_c0 = threadIdx.y == 0;
	// const int b_idx = threadIdx.z;

	// uint64_t pt;

	void*** inputs	= im_c0 ? c0_in : c1_in;
	void*** outputs = im_c0 ? c0_out : c1_out;

	for (int k = blockIdx.z; k < n; k += gridDim.z) {

		for (int i = 0; i < bStep; ++i) {
			uint64_t in = ((uint64_t*)inputs[k * bStep + i][blockIdx.y])[idx];

			for (int j = 0; j < gStep; ++j) {
				void** pt_partition = pts[k * bStep * gStep + j * bStep + i];

				__uint128_t mult = 0;
				if (pt_partition != nullptr) {
					// pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
					mult = (__uint128_t)in * (__uint128_t)((uint64_t*)pt_partition[blockIdx.y])[idx];
				}
				if (i == 0)
					acc_this_thread[j * in_stride] = mult;
				else
					acc_this_thread[j * in_stride] = acc_this_thread[j * in_stride] + mult;

				if (i == bStep - 1) {
					uint64_t res										 = modreduce<ALGO_NATIVE>(acc_this_thread[j * in_stride], primeid);
					((uint64_t*)outputs[k * gStep + j][blockIdx.y])[idx] = res;
				}
			}
		}
	}
}

__global__ void
dotProductLtBatchedPt___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts, const int batch, const int gStep, const int primeidInit, const int n) {
	int idx			  = threadIdx.x + blockIdx.x * blockDim.x;
	int b			  = blockDim.z;
	const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
	// constexpr ALGO algo = ALGO_BARRETT;

	extern __shared__ char buffer[];

	// Shared required: (2*batch+1/2)*threads_per_block
	const int in_stride = blockDim.x * blockDim.y * blockDim.z;
	const int block_id	= (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
	uint64_t* in		= ((uint64_t*)buffer) + block_id;
	// uint64_t in[6];

	uint64_t* acc			  = ((uint64_t*)buffer) + in_stride * batch;
	uint64_t* acc_this_thread = acc + block_id;
	uint64_t* pt			  = acc + 2 * in_stride + (threadIdx.x + blockDim.x * threadIdx.z);

	int r_init		= 1 << (32 - __clz(b - 1) - 1);
	bool im_c0		= threadIdx.y == 0;
	const int b_idx = threadIdx.z;

	if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0) {
		printf("in: %lu, acc: %lu, pt: %lu\n", in, acc_this_thread, pt);
	}

	if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0)
		printf("%d <- 2^{floor(log2(bStep))}\n", r_init);

	void*** inputs	= im_c0 ? c0_in : c1_in;
	void*** outputs = im_c0 ? c0_out : c1_out;

	for (int k = blockIdx.z; k < n; k += gridDim.z) {
		for (int i = 0; i < batch; ++i) {
			in[in_stride * i] = ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx];
			if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0)
				printf("LT: %d, b: %d, in c0: %lu %lu %p %p\n",
				  k,
				  b_idx,
				  in[/*in_stride * */ i],
				  ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
				  inputs[k * b * batch + i * b + b_idx],
				  inputs);
			if (PRINT && idx == 0 && blockIdx.y == 0 && !im_c0)
				printf("LT: %d, b: %d, in c1: %lu %lu %p %p\n",
				  k,
				  b_idx,
				  in[/*in_stride * */ i],
				  ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
				  inputs[k * b * batch + i * b + b_idx],
				  inputs);
		}

		for (int j = 0; j < gStep; ++j) {
			void** pt_partition = pts[k * b * gStep + j * b + b_idx];
			if (pt_partition != nullptr) {
				if (im_c0) {
					pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
					if (PRINT && idx == 0 && blockIdx.y == 0)
						printf("LT: %d, b: %d, g: %d, in pt: %lu %lu %p %p\n", k, b_idx, j, pt[0], ((uint64_t*)pt_partition[blockIdx.y])[idx], pt_partition, pts);
				}
			}

			__syncwarp();
			for (int i_0 = 0; i_0 < batch; i_0 += 2) {

				if (pt_partition != nullptr) {
					acc_this_thread[0] = modmult<ALGO_BARRETT>(in[in_stride * i_0], pt[0], primeid);
					if (i_0 + 1 < batch)
						acc_this_thread[in_stride] = modmult<ALGO_BARRETT>(in[in_stride * (i_0 + 1)], pt[0], primeid);
				} else {
					acc_this_thread[0]		   = 0;
					acc_this_thread[in_stride] = 0;
				}

				{
					int r = r_init;
					if (r > 0) {
						__syncthreads();
						if (threadIdx.z + r < b) {
							acc_this_thread[0] = modadd(acc_this_thread[0], acc_this_thread[0 + r * blockDim.x * blockDim.y], primeid);
						}
						if (threadIdx.z >= r) {
							acc_this_thread[in_stride] = modadd(acc_this_thread[in_stride], acc_this_thread[in_stride - r * blockDim.x * blockDim.y], primeid);
						}
					}

					r >>= 1;
					for (; r > 0; r >>= 1) {
						__syncthreads();
						if (threadIdx.z < r) {
							acc_this_thread[0] = modadd(acc_this_thread[0], acc_this_thread[0 + r * blockDim.x * blockDim.y], primeid);
						}
						if (threadIdx.z >= b - r) {
							acc_this_thread[in_stride] = modadd(acc_this_thread[in_stride], acc_this_thread[in_stride - r * blockDim.x * blockDim.y], primeid);
						}
					}
					if (threadIdx.z == 0) {
						((uint64_t*)outputs[k * batch * gStep + i_0 * gStep + j][blockIdx.y])[idx] = acc_this_thread[0];
						if (PRINT && idx == 0 && blockIdx.y == 0)
							printf("LT: %d, g: %d, res: %lu \n", k, j, acc_this_thread[0]);
					}
					if (threadIdx.z == b - 1) {
						if (i_0 + 1 < batch)
							((uint64_t*)outputs[k * batch * gStep + (i_0 + 1) * gStep + j][blockIdx.y])[idx] = acc_this_thread[in_stride];
					}
				}
			}
		}
	}
}

/*

__global__ void dotProductLtBatchedPt___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
										 const int batch, const int gStep, const int primeidInit, const int n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int b = blockDim.z;
	const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
	constexpr ALGO algo = ALGO_BARRETT;

	extern __shared__ char buffer[];

	// Shared required: (2*batch+1/2)*threads_per_block
	const int in_stride = blockDim.x * blockDim.y * blockDim.z;
	const int block_id = (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
	uint64_t* in = ((uint64_t*)buffer) + block_id;

	uint64_t* acc = ((uint64_t*)buffer) + in_stride * batch;
	uint64_t* acc_this_thread = acc + block_id;
	uint64_t* pt = acc + in_stride * batch + (threadIdx.x + blockDim.x * threadIdx.z);

	int r_init = 1 << (32 - __clz(b - 1) - 1);
	bool im_c0 = threadIdx.y == 0;
	const int b_idx = threadIdx.z;

	if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0) {
		printf("in: %lu, acc: %lu, pt: %lu\n", in, acc_this_thread, pt);
	}

	if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0)
		printf("%d <- 2^{floor(log2(bStep))}\n", r_init);

	void*** inputs = im_c0 ? c0_in : c1_in;
	void*** outputs = im_c0 ? c0_out : c1_out;

	for (int k = blockIdx.z; k < n; k += gridDim.z) {
		for (int i = 0; i < batch; ++i) {
			in[in_stride * i] = ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx];
			if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0)
				printf("LT: %d, b: %d, in c0: %lu %lu %p %p\n", k, b_idx, in[in_stride * i],
					   ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
					   inputs[k * b * batch + i * b + b_idx], inputs);
			if (PRINT && idx == 0 && blockIdx.y == 0 && !im_c0)
				printf("LT: %d, b: %d, in c1: %lu %lu %p %p\n", k, b_idx, in[in_stride * i],
					   ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
					   inputs[k * b * batch + i * b + b_idx], inputs);
		}

		for (int j = 0; j < gStep; ++j) {
			void** pt_partition = pts[k * b * gStep + j * b + b_idx];
			if (pt_partition != nullptr) {
				if (im_c0) {
					pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
					if (PRINT && idx == 0 && blockIdx.y == 0)
						printf("LT: %d, b: %d, g: %d, in pt: %lu %lu %p %p\n", k, b_idx, j, pt[0],
							   ((uint64_t*)pt_partition[blockIdx.y])[idx], pt_partition, pts);
				}
			}

			__syncwarp();
			for (int i = 0; i < batch; ++i) {
				if (pt_partition != nullptr) {
					acc_this_thread[in_stride * i] = modmult<ALGO_BARRETT>(in[in_stride * i], pt[0], primeid);
				} else {
					acc_this_thread[in_stride * i] = 0;
				}
			}

			int r = r_init;
			if (r > 0) {
				__syncthreads();
				if (threadIdx.z + r < b) {
					for (int i = 0; i < batch; ++i) {
						acc_this_thread[in_stride * i] =
							modadd(acc_this_thread[in_stride * i],
								   acc_this_thread[in_stride * i + r * blockDim.x * blockDim.y], primeid);
					}
				}
			}

			r >>= 1;
			for (; r > 0; r >>= 1) {
				__syncthreads();
				if (threadIdx.z < r) {
					for (int i = 0; i < batch; ++i) {
						acc_this_thread[in_stride * i] =
							modadd(acc_this_thread[in_stride * i],
								   acc_this_thread[in_stride * i + r * blockDim.x * blockDim.y], primeid);
					}
				}
			}
			if (threadIdx.z == 0) {

				for (int i = 0; i < batch; ++i) {
					((uint64_t*)outputs[k * batch * gStep + i * gStep + j][blockIdx.y])[idx] =
						acc_this_thread[in_stride * i];
					if (PRINT && idx == 0 && blockIdx.y == 0)
						printf("LT: %d, g: %d, res: %lu \n", k, j, acc_this_thread[in_stride * i]);
				}
			}
		}
	}
}

 */
__global__ void addScaleB_(void** a, void** b, void** c, const int primeid_init) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		uint64_t in						= modmult<ALGO_SHOUP>(((uint64_t*)b[blockIdx.y])[idx], C_.P[primeid], primeid, C_.P_shoup[primeid]);
		((uint64_t*)a[blockIdx.y])[idx] = modadd(in, ((uint64_t*)c[blockIdx.y])[idx], primeid);
	} else {
	}
}

__global__ void scaleByP_(void** a, const int primeid_init) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		uint64_t in						= modmult<ALGO_SHOUP>(((uint64_t*)a[blockIdx.y])[idx], C_.P[primeid], primeid, C_.P_shoup[primeid]);
		((uint64_t*)a[blockIdx.y])[idx] = in;
	} else {
	}
}

__global__ void add_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modadd(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
			}
		}
	} else {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modadd(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
			}
		}
	}
}

__global__ void add_reuse_scale_p_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
			load		  = modmult<ALGO_SHOUP>(load, C_.P[primeid], primeid, C_.P_shoup[primeid]);
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modadd(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
			}
		}
	} else {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
			load		  = modmult<ALGO_SHOUP>(load, (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modadd(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
			}
		}
	}
}

__global__ void sub_reuse_scale_p_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
			load		  = modmult<ALGO_SHOUP>(load, C_.P[primeid], primeid, C_.P_shoup[primeid]);
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modsub(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
			}
		}
	} else {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
			load		  = modmult<ALGO_SHOUP>(load, (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modsub(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
			}
		}
	}
}

__global__ void add_scale_p_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				uint64_t aux = modmult<ALGO_SHOUP>(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], C_.P[primeid], primeid, C_.P_shoup[primeid]);
				((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modadd(aux, load, primeid);
			}
		}
	} else {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
			load		  = modmult<ALGO_SHOUP>(load, (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				uint32_t aux = modmult<ALGO_SHOUP>(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
				((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modadd(aux, load, primeid);
			}
		}
	}
}

__global__ void sub_scale_p_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				uint64_t aux = modmult<ALGO_SHOUP>(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], C_.P[primeid], primeid, C_.P_shoup[primeid]);
				((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modsub(aux, load, primeid);
			}
		}
	} else {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
			load		  = modmult<ALGO_SHOUP>(load, (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				uint32_t aux = modmult<ALGO_SHOUP>(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
				((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modsub(aux, load, primeid);
			}
		}
	}
}

__global__ void copy_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = load;
			}
		}
	} else {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = load;
			}
		}
	}
}

__global__ void copy_reuse_negative_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
			load		  = load == 0 ? 0 : C_.primes[primeid] - load;
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = load;
			}
		}
	} else {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
			load		  = load == 0 ? 0 : C_.primes[primeid] - load;
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = load;
			}
		}
	}
}

__global__ void add_scalar_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint64_t load = ((uint64_t*)b)[i * MAXP + primeid];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modadd(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
			}
		}
	} else {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint32_t load = ((uint32_t*)b)[i * MAXP + primeid];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modadd(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
			}
		}
	}
}

__global__ void mult_scalar_reuse_b___(void*** a, void*** b, void*** b_shoup, const int primeid_init, const int n, const int its) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint64_t load		= ((uint64_t*)b)[i * MAXP + primeid];
			uint64_t load_shoup = ((uint64_t*)b_shoup)[i * MAXP + primeid];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid, load_shoup);
			}
		}
	} else {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint32_t load		= ((uint32_t*)b)[i * MAXP + primeid];
			uint32_t load_shoup = ((uint32_t*)b_shoup)[i * MAXP + primeid];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid, load_shoup);
			}
		}
	}
}

__global__ void sub_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modsub(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
			}
		}
	} else {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modsub(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
			}
		}
	}
}

__global__ void mult_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	if (ISU64(primeid)) {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modmult<ALGO_BARRETT>(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
			}
		}
	} else {
		for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
			uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
			for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
				((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modmult<ALGO_BARRETT>(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
			}
		}
	}
}

__global__ void binomialMult_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2, void** d0, void** d1) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = threadIdx.x + blockDim.x * blockIdx.x;
	// constexpr ALGO algo = ALGO_BARRETT;

	if (ISU64(primeid)) {
		using T = uint64_t;
		T d0in	= ((T*)(d0[blockIdx.y]))[idx];
		T d1in	= ((T*)(d1[blockIdx.y]))[idx];
		T c0in	= ((T*)(c0[blockIdx.y]))[idx];
		T c1in	= ((T*)(c1[blockIdx.y]))[idx];

		T aux0						= modmult<ALGO_BARRETT>(c0in, d0in, primeid);
		((T*)(c0[blockIdx.y]))[idx] = aux0;

		T aux1						= modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);
		((T*)(c1[blockIdx.y]))[idx] = aux1;

		T aux2						= modmult<ALGO_BARRETT>(c1in, d1in, primeid);
		((T*)(c2[blockIdx.y]))[idx] = aux2;

	} else {
		using T = uint32_t;
		T d0in	= ((T*)(d0[blockIdx.y]))[idx];
		T d1in	= ((T*)(d1[blockIdx.y]))[idx];
		T c0in	= ((T*)(c0[blockIdx.y]))[idx];
		T c1in	= ((T*)(c1[blockIdx.y]))[idx];

		T aux0						= modmult<ALGO_BARRETT>(c0in, d0in, primeid);
		((T*)(c0[blockIdx.y]))[idx] = aux0;

		T aux1						= modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);
		((T*)(c1[blockIdx.y]))[idx] = aux1;

		T aux2						= modmult<ALGO_BARRETT>(c1in, d1in, primeid);
		((T*)(c2[blockIdx.y]))[idx] = aux2;
	}
}

__global__ void binomialMultExtend_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2, void** d0, void** d1) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = threadIdx.x + blockDim.x * blockIdx.x;
	// constexpr ALGO algo = ALGO_BARRETT;

	if (ISU64(primeid)) {
		using T = uint64_t;
		T d0in	= ((T*)(d0[blockIdx.y]))[idx];
		T d1in	= ((T*)(d1[blockIdx.y]))[idx];
		T c0in	= ((T*)(c0[blockIdx.y]))[idx];
		T c1in	= ((T*)(c1[blockIdx.y]))[idx];

		T aux0						= modmult<ALGO_BARRETT>(c0in, d0in, primeid);
		((T*)(c0[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux0, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

		T aux1						= modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);
		((T*)(c1[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux1, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

		T aux2						= modmult<ALGO_BARRETT>(c1in, d1in, primeid);
		((T*)(c2[blockIdx.y]))[idx] = aux2;

	} else {
		using T = uint32_t;
		T d0in	= ((T*)(d0[blockIdx.y]))[idx];
		T d1in	= ((T*)(d1[blockIdx.y]))[idx];
		T c0in	= ((T*)(c0[blockIdx.y]))[idx];
		T c1in	= ((T*)(c1[blockIdx.y]))[idx];

		T aux0						= modmult<ALGO_BARRETT>(c0in, d0in, primeid);
		((T*)(c0[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux0, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

		T aux1						= modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);
		((T*)(c1[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux1, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

		T aux2						= modmult<ALGO_BARRETT>(c1in, d1in, primeid);
		((T*)(c2[blockIdx.y]))[idx] = aux2;
	}
}

__global__ void binomialSquare_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = threadIdx.x + blockDim.x * blockIdx.x;
	// constexpr ALGO algo = ALGO_BARRETT;

	if (ISU64(primeid)) {
		using T = uint64_t;
		T c0in	= ((T*)(c0[blockIdx.y]))[idx];
		T c1in	= ((T*)(c1[blockIdx.y]))[idx];

		T aux0						= modmult<ALGO_BARRETT>(c0in, c0in, primeid);
		((T*)(c0[blockIdx.y]))[idx] = aux0;

		T aux1						= modmult<ALGO_BARRETT>(c0in, c1in, primeid);
		((T*)(c1[blockIdx.y]))[idx] = modadd(aux1, aux1, primeid);

		T aux2						= modmult<ALGO_BARRETT>(c1in, c1in, primeid);
		((T*)(c2[blockIdx.y]))[idx] = aux2;

	} else {
		using T = uint32_t;
		T c0in	= ((T*)(c0[blockIdx.y]))[idx];
		T c1in	= ((T*)(c1[blockIdx.y]))[idx];

		T aux0						= modmult<ALGO_BARRETT>(c0in, c0in, primeid);
		((T*)(c0[blockIdx.y]))[idx] = aux0;

		T aux1						= modmult<ALGO_BARRETT>(c0in, c1in, primeid);
		((T*)(c1[blockIdx.y]))[idx] = modadd(aux1, aux1, primeid);

		T aux2						= modmult<ALGO_BARRETT>(c1in, c1in, primeid);
		((T*)(c2[blockIdx.y]))[idx] = aux2;
	}
}

__global__ void binomialSquareExtend_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = threadIdx.x + blockDim.x * blockIdx.x;
	// constexpr ALGO algo = ALGO_BARRETT;

	if (ISU64(primeid)) {
		using T = uint64_t;
		T c0in	= ((T*)(c0[blockIdx.y]))[idx];
		T c1in	= ((T*)(c1[blockIdx.y]))[idx];

		T aux0						= modmult<ALGO_BARRETT>(c0in, c0in, primeid);
		((T*)(c0[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux0, C_.P[primeid], primeid, C_.P_shoup[primeid]);

		T aux1						= modmult<ALGO_BARRETT>(c0in, c1in, primeid);
		((T*)(c1[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(modadd(aux1, aux1, primeid), C_.P[primeid], primeid, C_.P_shoup[primeid]);

		T aux2						= modmult<ALGO_BARRETT>(c1in, c1in, primeid);
		((T*)(c2[blockIdx.y]))[idx] = aux2;

	} else {
		using T = uint32_t;
		T c0in	= ((T*)(c0[blockIdx.y]))[idx];
		T c1in	= ((T*)(c1[blockIdx.y]))[idx];

		T aux0						= modmult<ALGO_BARRETT>(c0in, c0in, primeid);
		((T*)(c0[blockIdx.y]))[idx] = aux0;

		T aux1						= modmult<ALGO_BARRETT>(c0in, c1in, primeid);
		((T*)(c1[blockIdx.y]))[idx] = modadd(aux1, aux1, primeid);

		T aux2						= modmult<ALGO_BARRETT>(c1in, c1in, primeid);
		((T*)(c2[blockIdx.y]))[idx] = aux2;
	}
}

__global__ void
binomialDotProdBatched___(const __grid_constant__ int primeid_init, void*** c0, void*** c1, void*** d0, void*** d1, void*** c0_out, void*** c1_out, void*** c2_out, int its, int n, bool ext_in) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = threadIdx.x + blockDim.x * blockIdx.x;
	// constexpr ALGO algo = ALGO_BARRETT;

	if (ISU64(primeid)) {
		using T = uint64_t;
		for (int i = blockIdx.z; i < n; i += gridDim.z) {
			T acc0, acc1, acc2;
			for (int j = 0; j < its && c0[i * its + j] != nullptr; ++j) {
				T d0in = ((T*)(d0[i * its + j][blockIdx.y]))[idx];
				T d1in = ((T*)(d1[i * its + j][blockIdx.y]))[idx];

				if (ext_in) {
					// d0in = modmult<ALGO_SHOUP>(d0in, C_.P_inv[primeid], primeid, C_.P_inv_shoup[primeid]);
					// d1in = modmult<ALGO_SHOUP>(d1in, C_.P_inv[primeid], primeid, C_.P_inv_shoup[primeid]);
				}

				T c0in = ((T*)(c0[i * its + j][blockIdx.y]))[idx];
				T c1in = ((T*)(c1[i * its + j][blockIdx.y]))[idx];

				T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);

				T aux1 = modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);

				T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);

				if (j == 0) {
					acc0 = aux0;
					acc1 = aux1;
					acc2 = aux2;
				} else {
					acc0 = modadd(acc0, aux0, primeid);
					acc1 = modadd(acc1, aux1, primeid);
					acc2 = modadd(acc2, aux2, primeid);
				}
			}

			((T*)(c0_out[i][blockIdx.y]))[idx] = acc0;
			((T*)(c1_out[i][blockIdx.y]))[idx] = acc1;
			((T*)(c2_out[i][blockIdx.y]))[idx] = acc2;
		}
	} else {
		using T = uint32_t;
		for (int i = blockIdx.z; i < n; i += gridDim.z) {
			T acc0, acc1, acc2;
			for (int j = 0; j < its && c0[i * its + j] != nullptr; ++j) {

				T d0in = ((T*)(d0[i * its + j][blockIdx.y]))[idx];
				T d1in = ((T*)(d1[i * its + j][blockIdx.y]))[idx];

				if (ext_in) {
					d0in = modmult<ALGO_SHOUP>(d0in, (uint32_t)C_.P_inv[primeid], primeid, (uint32_t)C_.P_inv_shoup[primeid]);
					d1in = modmult<ALGO_SHOUP>(d1in, (uint32_t)C_.P_inv[primeid], primeid, (uint32_t)C_.P_inv_shoup[primeid]);
				}

				T c0in = ((T*)(c0[i * its + j][blockIdx.y]))[idx];
				T c1in = ((T*)(c1[i * its + j][blockIdx.y]))[idx];

				T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);

				T aux1 = modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);

				T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);

				if (j == 0) {
					acc0 = aux0;
					acc1 = aux1;
					acc2 = aux2;
				} else {
					acc0 = modadd(acc0, aux0, primeid);
					acc1 = modadd(acc1, aux1, primeid);
					acc2 = modadd(acc2, aux2, primeid);
				}
			}

			((T*)(c0_out[i][blockIdx.y]))[idx] = acc0;
			((T*)(c1_out[i][blockIdx.y]))[idx] = acc1;
			((T*)(c2_out[i][blockIdx.y]))[idx] = acc2;
		}
	}
}

__global__ void binomialDotProdSpecialBatched___(const __grid_constant__ int primeid_init,
  void*** c0,
  void*** c1,
  void*** d0,
  void*** d1,
  void*** pt1,
  void*** pt2,
  void*** c0_out,
  void*** c1_out,
  void*** c2_out,
  int its,
  int n) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	const int idx	  = threadIdx.x + blockDim.x * blockIdx.x;
	// constexpr ALGO algo = ALGO_BARRETT;

	if (ISU64(primeid)) {
		using T = uint64_t;
		for (int i = blockIdx.z; i < n; i += gridDim.z) {
			T acc0, acc1, acc2;
			for (int j = 0; j < its && c0[i * its + j] != nullptr; ++j) {
				T d0in = ((T*)(d0[i * its + j][blockIdx.y]))[idx];
				T d1in = ((T*)(d1[i * its + j][blockIdx.y]))[idx];
				T c0in = ((T*)(c0[i * its + j][blockIdx.y]))[idx];
				T c1in = ((T*)(c1[i * its + j][blockIdx.y]))[idx];

				{
					T pt1in = ((T*)(pt1[i * its + j][blockIdx.y]))[idx];
					T pt2in = ((T*)(pt2[i * its + j][blockIdx.y]))[idx];
					c0in	= modadd(modmult<ALGO_BARRETT>(c0in, pt1in, primeid), modmult<ALGO_BARRETT>(c0in, pt2in, primeid), primeid);
					c1in	= modadd(modmult<ALGO_BARRETT>(c1in, pt1in, primeid), modmult<ALGO_BARRETT>(c1in, pt2in, primeid), primeid);
				}

				T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);

				T aux1 = modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);

				T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);

				if (j == 0) {
					acc0 = aux0;
					acc1 = aux1;
					acc2 = aux2;
				} else {
					acc0 = modadd(acc0, aux0, primeid);
					acc1 = modadd(acc1, aux1, primeid);
					acc2 = modadd(acc2, aux2, primeid);
				}
			}

			((T*)(c0_out[i][blockIdx.y]))[idx] = acc0;
			((T*)(c1_out[i][blockIdx.y]))[idx] = acc1;
			((T*)(c2_out[i][blockIdx.y]))[idx] = acc2;
		}
	} else {
		using T = uint32_t;
		for (int i = blockIdx.z; i < n; i += gridDim.z) {
			T acc0, acc1, acc2;
			for (int j = 0; j < its && c0[i * its + j] != nullptr; ++j) {

				T d0in = ((T*)(d0[i * its + j][blockIdx.y]))[idx];
				T d1in = ((T*)(d1[i * its + j][blockIdx.y]))[idx];
				T c0in = ((T*)(c0[i * its + j][blockIdx.y]))[idx];
				T c1in = ((T*)(c1[i * its + j][blockIdx.y]))[idx];

				{
					T pt1in = ((T*)(pt1[i * its + j][blockIdx.y]))[idx];
					T pt2in = ((T*)(pt2[i * its + j][blockIdx.y]))[idx];
					c0in	= modadd(modmult<ALGO_BARRETT>(c0in, pt1in, primeid), modmult<ALGO_BARRETT>(c0in, pt2in, primeid), primeid);
					c1in	= modadd(modmult<ALGO_BARRETT>(c1in, pt1in, primeid), modmult<ALGO_BARRETT>(c1in, pt2in, primeid), primeid);
				}

				T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);

				T aux1 = modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);

				T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);

				if (j == 0) {
					acc0 = aux0;
					acc1 = aux1;
					acc2 = aux2;
				} else {
					acc0 = modadd(acc0, aux0, primeid);
					acc1 = modadd(acc1, aux1, primeid);
					acc2 = modadd(acc2, aux2, primeid);
				}
			}

			((T*)(c0_out[i][blockIdx.y]))[idx] = acc0;
			((T*)(c1_out[i][blockIdx.y]))[idx] = acc1;
			((T*)(c2_out[i][blockIdx.y]))[idx] = acc2;
		}
	}
}

} // namespace FIDESlib::CKKS

#define YY(algo) \
	template __global__ void FIDESlib::CKKS::Scalar_mult_<algo>(void** a, const uint64_t* b, const __grid_constant__ int primeid_init, const uint64_t* shoup_mu);
#include "ntt_types.inc"
#undef YY

template __global__ void FIDESlib::CKKS::addMult_<uint64_t>(uint64_t* l, const uint64_t* l1, const uint64_t* l2, const __grid_constant__ int primeid);

template __global__ void FIDESlib::CKKS::addMult_<uint32_t>(uint32_t* l, const uint32_t* l1, const uint32_t* l2, const __grid_constant__ int primeid);
