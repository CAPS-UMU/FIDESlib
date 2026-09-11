//
// Created by carlosad on 21/10/24.
//

#ifndef GPUCKKS_NTTHELPER_CUH
#define GPUCKKS_NTTHELPER_CUH

constexpr bool NEGACYCLIC = true;

namespace FIDESlib {
template <typename T> __device__ __inline__ void swap(T& a, T& b) {
	T c = a;
	a   = b;
	b   = c;
}

#define A(i) (((T*)buffer) + 2 * blockDim.x * (i))

#define OFFSET_T(i) ((blockDim.x * 2 * M) * blockIdx.x + 2 * blockDim.x * (i) + 2 * threadIdx.x)

#define OFFSET_2T(i) ((blockDim.x * M) * blockIdx.x + blockDim.x * (i) + threadIdx.x)

template <typename T, ALGO algo = ALGO_SHOUP> __device__ __forceinline__ void CT_butterfly(T& c, T& d, T psi, const int primeid, T shoup_psi = 2) {
	T a = c;
	T b = d;
	if constexpr (algo == 1) {

	} else if constexpr (algo == 2) {
		const uint64_t hi = __umul64hi(b, shoup_psi);
		b                 = b * psi - hi * C_.primes[primeid];
		d                 = a - b;
		c                 = a + b;
	} else if constexpr (algo == 3) {
		b = modmult<algo>(b, psi, primeid, shoup_psi);
		c = modadd(a, b, primeid);
		d = modsub(a, b, primeid);
	} else if constexpr (algo <= 5) {
		// assert(b < primes[primeid]);
		// assert(a < primes[primeid]);
		//  T baux = modmult<0>(b, psi, primeid);
		b = modmult<algo>(b, psi, primeid);
		// assert(psi < primes[primeid]);
		// assert(b < primes[primeid]);
		// assert(b == baux);
		c = modadd(a, b, primeid);
		d = modsub(a, b, primeid);
	}
}

template <typename T, ALGO algo = ALGO_SHOUP> __device__ __forceinline__ void GS_butterfly(T& c, T& d, T psi, const int primeid, T shoup_psi = 2) {
	T a = c;
	T b = d;
	if constexpr (algo == 1) {
	} else if constexpr (algo == 2) {
		d                 = a - b;
		c                 = a + b;
		const uint64_t hi = __umul64hi(d, shoup_psi);
		d                 = d * psi - hi * C_.primes[primeid];
	} else if constexpr (algo == 3) {
		c = modadd(a, b, primeid);
		b = modsub(a, b, primeid);
		d = modmult<algo>(b, psi, primeid, shoup_psi);
	} else if constexpr (algo <= 5) {
		//      assert(b < primes[primeid]);
		//      assert(a < primes[primeid]);
		c = modadd(a, b, primeid);
		b = modsub(a, b, primeid);
		//   T baux = modmult<0>(b, psi, primeid);
		d = modmult<algo>(b, psi, primeid);
		//     assert(psi < primes[primeid]);
		//     assert(d < primes[primeid]);
		//     assert(d == baux);
	}
}

template <typename T> __global__ void Bit_Reverse(T* dat, uint32_t N) {
	int idx         = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t br_idx = __brev(idx) >> (__clz(N) + 1);
	if (br_idx > idx) {
		T a         = dat[idx];
		T b         = dat[br_idx];
		dat[idx]    = b;
		dat[br_idx] = a;
	}
}

template __global__ void Bit_Reverse(uint32_t* dat, uint32_t N);

template __global__ void Bit_Reverse(uint64_t* dat, uint32_t N);

} // namespace FIDESlib
#endif // GPUCKKS_NTTHELPER_CUH
