//
// Created by carlosad on 25/3/26.
//
#include <errno.h>
#include <nccl.h>

#include <iomanip>

#include "AddSub.cuh"
#include "ModMult.cuh"
#include <CudaUtils.cuh>
#include <chrono>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <vector>

#if 0
#define CUDA_CHECK(call)                                                                                \
	do {                                                                                                \
		cudaError_t err__ = (call);                                                                     \
		if (err__ != cudaSuccess) {                                                                     \
			fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
			std::exit(EXIT_FAILURE);                                                                    \
		}                                                                                               \
	} while (0)

// Adjust this if you change the body of the loop
#ifndef OPS_PER_ITER
#define OPS_PER_ITER 8 // 4 int32 ALU ops per loop iteration
#endif

__global__ void int32_iops_kernel(int* out, int iters) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	int4 x	 = { tid, tid << 1, tid << 2, tid << 3 };
	int4 y	 = { tid << 4, tid << 5, tid << 6, tid << 7 };
	int4 acc = { 0, 0, 0, 0 };

	// Unroll to reduce loop overhead and help reach peak throughput
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = x.x + y.x;
		x.y = x.y + y.y;
		x.z = x.z + y.z;
		x.w = x.w + y.w;
		x.x = x.x + y.x;
		x.y = x.y + y.y;
		x.z = x.z + y.z;
		x.w = x.w + y.w;
	}

	// Prevent the compiler from optimizing the loop away
	out[tid] = x.x + x.y + x.z + x.w;
}

__global__ void int32_xor(int* out, int iters) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	int4 x	 = { tid, tid << 1, tid << 2, tid << 3 };
	int4 y	 = { tid << 4, tid << 5, tid << 6, tid << 7 };
	int4 acc = { 0, 0, 0, 0 };

	// Unroll to reduce loop overhead and help reach peak throughput
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = x.x ^ y.x;
		x.y = x.y ^ y.y;
		x.z = x.z ^ y.z;
		x.w = x.w ^ y.w;
		x.x = x.x + y.x;
		x.y = x.y + y.y;
		x.z = x.z + y.z;
		x.w = x.w + y.w;
	}

	// Prevent the compiler from optimizing the loop away
	out[tid] = x.x + x.y + x.z + x.w;
}

__global__ void int32_mult(int* out, int iters) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	int4 x	 = { tid, tid << 1, tid << 2, tid << 3 };
	int4 y	 = { tid << 4, tid << 5, tid << 6, tid << 7 };
	int4 z	 = { tid << 5, tid << 6, tid << 7, tid << 8 };
	int4 acc = { 0, 0, 0, 0 };

	// Unroll to reduce loop overhead and help reach peak throughput
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = x.x * y.x;
		x.y = x.y * y.y;
		x.z = x.z * y.z;
		x.w = x.w * y.w;
		z.x = z.x * y.x;
		z.y = z.y * y.y;
		z.z = z.z * y.z;
		z.w = z.w * y.w;
	}

	// Prevent the compiler from optimizing the loop away
	out[tid] = x.x + x.y + x.z + x.w + z.x + z.y + z.z + z.w;
}

__global__ void int32_add(int* out, int iters) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	int4 x	 = { tid, tid << 1, tid << 2, tid << 3 };
	int4 y	 = { tid << 4, tid << 5, tid << 6, tid << 7 };
	int4 z	 = { tid << 5, tid << 6, tid << 7, tid << 8 };
	int4 acc = { 0, 0, 0, 0 };

	// Unroll to reduce loop overhead and help reach peak throughput
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = x.x + y.x;
		x.y = x.y + y.y;
		x.z = x.z + y.z;
		x.w = x.w + y.w;
		z.x = z.x + y.x;
		z.y = z.y + y.y;
		z.z = z.z + y.z;
		z.w = z.w + y.w;
	}

	// Prevent the compiler from optimizing the loop away
	out[tid] = x.x + x.y + x.z + x.w + z.x + z.y + z.z + z.w;
}

__global__ void int32_mac(int* out, int iters) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	int4 x	 = { tid, tid << 1, tid << 2, tid << 3 };
	int4 y	 = { tid << 4, tid << 5, tid << 6, tid << 7 };
	int4 z	 = { tid << 5, tid << 6, tid << 7, tid << 8 };
	int4 t	 = { tid << 9, tid << 10, tid << 11, tid << 12 };
	int4 acc = { 0, 0, 0, 0 };

	// Unroll to reduce loop overhead and help reach peak throughput
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = x.x * t.x + y.x;
		x.y = x.y * t.y + y.y;
		x.z = x.z * t.z + y.z;
		x.w = x.w * t.w + y.w;
		z.x = z.x * t.x + y.x;
		z.y = z.y * t.y + y.y;
		z.z = z.z * t.z + y.z;
		z.w = z.w * t.w + y.w;
	}

	// Prevent the compiler from optimizing the loop away
	out[tid] = x.x + x.y + x.z + x.w + z.x + z.y + z.z + z.w;
}

__global__ void int64_add(int* out, int iters) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	longlong4 x = { tid, tid << 1, tid << 2, tid << 3 };
	longlong4 y = { tid << 4, tid << 5, tid << 6, tid << 7 };
	longlong4 z = { tid << 5, tid << 6, tid << 7, tid << 8 };

	// Unroll to reduce loop overhead and help reach peak throughput
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = x.x + y.x;
		x.y = x.y + y.y;
		x.z = x.z + y.z;
		x.w = x.w + y.w;
		z.x = z.x + y.x;
		z.y = z.y + y.y;
		z.z = z.z + y.z;
		z.w = z.w + y.w;
	}

	// Prevent the compiler from optimizing the loop away
	out[tid] = x.x + x.y + x.z + x.w + z.x + z.y + z.z + z.w;
}

__global__ void int64_mult(int* out, int iters) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	longlong4 x = { tid, tid << 1, tid << 2, tid << 3 };
	longlong4 y = { tid << 4, tid << 5, tid << 6, tid << 7 };
	longlong4 z = { tid << 5, tid << 6, tid << 7, tid << 8 };

	// Unroll to reduce loop overhead and help reach peak throughput
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = x.x * y.x;
		x.y = x.y * y.y;
		x.z = x.z * y.z;
		x.w = x.w * y.w;
		z.x = z.x * y.x;
		z.y = z.y * y.y;
		z.z = z.z * y.z;
		z.w = z.w * y.w;
	}

	// Prevent the compiler from optimizing the loop away
	out[tid] = x.x + x.y + x.z + x.w + z.x + z.y + z.z + z.w;
}

__global__ void int64_mac(int* out, int iters) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	longlong4 x = { tid, tid << 1, tid << 2, tid << 3 };
	longlong4 y = { tid << 4, tid << 5, tid << 6, tid << 7 };
	longlong4 z = { tid << 5, tid << 6, tid << 7, tid << 8 };
	longlong4 t = { tid << 9, tid << 10, tid << 11, tid << 12 };

	// Unroll to reduce loop overhead and help reach peak throughput
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = x.x * t.x + y.x;
		x.y = x.y * t.y + y.y;
		x.z = x.z * t.z + y.z;
		x.w = x.w * t.w + y.w;
		z.x = z.x * t.x + y.x;
		z.y = z.y * t.y + y.y;
		z.z = z.z * t.z + y.z;
		z.w = z.w * t.w + y.w;
	}

	// Prevent the compiler from optimizing the loop away
	out[tid] = x.x + x.y + x.z + x.w + z.x + z.y + z.z + z.w;
}

__global__ void int64_modadd(int* out, int iters) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	longlong4 x = { tid, tid << 1, tid << 2, tid << 3 };
	longlong4 y = { tid << 4, tid << 5, tid << 6, tid << 7 };
	longlong4 z = { tid << 5, tid << 6, tid << 7, tid << 8 };
	longlong4 t = { tid << 9, tid << 10, tid << 11, tid << 12 };

	// Unroll to reduce loop overhead and help reach peak throughput
#pragma unroll 16
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = FIDESlib::modadd(x.x, y.x, 0);
		x.y = FIDESlib::modadd(x.y, y.y, 0);
		x.z = FIDESlib::modadd(x.z, y.z, 0);
		x.w = FIDESlib::modadd(x.w, y.w, 0);
		z.x = FIDESlib::modadd(z.x, y.x, 0);
		z.y = FIDESlib::modadd(z.y, y.y, 0);
		z.z = FIDESlib::modadd(z.z, y.z, 0);
		z.w = FIDESlib::modadd(z.w, y.w, 0);
	}

	// Prevent the compiler from optimizing the loop away
	out[tid] = x.x + x.y + x.z + x.w + z.x + z.y + z.z + z.w;
}

__global__ void int64_modadd2(int* out, int iters) {
	uint tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	ulonglong4 x = { tid, tid << 1, tid << 2, tid << 3 };
	ulonglong4 y = { tid << 4, tid << 5, tid << 6, tid << 7 };
	ulonglong4 z = { tid << 5, tid << 6, tid << 7, tid << 8 };
	ulonglong4 t = { tid << 9, tid << 10, tid << 11, tid << 12 };

	// Unroll to reduce loop overhead and help reach peak throughput
	const uint64_t prime_p = tid << 32 ^ tid;
	/*
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = x.x + y.x;
		x.x = x.x >= prime_p ? x.x - prime_p : x.x;
		x.y = x.y + y.y;
		x.y = x.y >= prime_p ? x.y - prime_p : x.y;
		x.z = x.z + y.z;
		x.z = x.z >= prime_p ? x.z - prime_p : x.z;
		x.w = x.w + y.w;
		x.w = x.w >= prime_p ? x.w - prime_p : x.w;
		z.x = z.x + y.x;
		z.x = z.x >= prime_p ? z.x - prime_p : z.x;
		z.y = z.y + y.y;
		z.y = z.y >= prime_p ? z.y - prime_p : z.y;
		z.z = z.z + y.z;
		z.z = z.z >= prime_p ? z.z - prime_p : z.z;
		z.w = z.w + y.w;
		z.w = z.w >= prime_p ? z.w - prime_p : z.w;
	}*/
#pragma unroll 1
	for (int i = 0; i < iters; ++i) {
		asm volatile(
		  // x.x += y.x; if >= prime_p x.x -= prime_p  (repeat pattern for all 8)
		  ".reg .pred p0, p1, p2, p3, p4, p5, p6, p7; \n\t"
		  "add.u64 %0, %0, %4; \n\t"
		  "setp.ge.u64 p0, %0, %12; \n\t"
		  "@p0 sub.u64 %0, %0, %12; \n\t"

		  "add.u64 %1, %1, %5; \n\t"
		  "setp.ge.u64 p1, %1, %12; \n\t"
		  "@p1 sub.u64 %1, %1, %12; \n\t"

		  "add.u64 %2, %2, %6; \n\t"
		  "setp.ge.u64 p2, %2, %12; \n\t"
		  "@p2 sub.u64 %2, %2, %12; \n\t"

		  "add.u64 %3, %3, %7; \n\t"
		  "setp.ge.u64 p3, %3, %12; \n\t"
		  "@p3 sub.u64 %3, %3, %12; \n\t"

		  // z += y (parallel adds)
		  "add.u64 %8, %8, %4; \n\t"
		  "setp.ge.u64 p4, %8, %12; \n\t"
		  "@p4 sub.u64 %8, %8, %12; \n\t"

		  "add.u64 %9, %9, %5; \n\t"
		  "setp.ge.u64 p5, %9, %12; \n\t"
		  "@p5 sub.u64 %9, %9, %12; \n\t"

		  "add.u64 %10, %10, %6; \n\t"
		  "setp.ge.u64 p6, %10, %12; \n\t"
		  "@p6 sub.u64 %10, %10, %12; \n\t"

		  "add.u64 %11, %11, %7; \n\t"
		  "setp.ge.u64 p7, %11, %12; \n\t"
		  "@p7 sub.u64 %11, %11, %12; \n\t"

		  : "+l"(x.x), "+l"(x.y), "+l"(x.z), "+l"(x.w), "+l"(z.x), "+l"(z.y), "+l"(z.z), "+l"(z.w)
		  : "l"((long long)x.x),
		  "l"((long long)x.y),
		  "l"((long long)x.z),
		  "l"((long long)x.w),
		  "l"((long long)y.x),
		  "l"((long long)y.y),
		  "l"((long long)y.z),
		  "l"((long long)y.w),
		  "l"((long long)z.x),
		  "l"((long long)z.y),
		  "l"((long long)z.z),
		  "l"((long long)z.w),
		  "l"((long long)prime_p),
		  "l"((long long)prime_p)
		  : "memory");
	}
	// Prevent the compiler from optimizing the loop away
	out[tid] = x.x + x.y + x.z + x.w + z.x + z.y + z.z + z.w;
}

__global__ void int64_modadd3(int* out, int iters) {
	uint tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	uint32_t x[8];
	((ulonglong4*)x)[0] = { tid, tid << 1, tid << 2, tid << 3 };
	ulonglong4 y		= { tid << 4, tid << 5, tid << 6, tid << 7 };
	uint32_t z[8];
	((ulonglong4*)z)[8] = { tid << 5, tid << 6, tid << 7, tid << 8 };
	ulonglong4 t		= { tid << 9, tid << 10, tid << 11, tid << 12 };

	// Unroll to reduce loop overhead and help reach peak throughput
	const uint64_t prime_p = tid << 32 ^ tid;
	/*
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = x.x + y.x;
		x.x = x.x >= prime_p ? x.x - prime_p : x.x;
		x.y = x.y + y.y;
		x.y = x.y >= prime_p ? x.y - prime_p : x.y;
		x.z = x.z + y.z;
		x.z = x.z >= prime_p ? x.z - prime_p : x.z;
		x.w = x.w + y.w;
		x.w = x.w >= prime_p ? x.w - prime_p : x.w;
		z.x = z.x + y.x;
		z.x = z.x >= prime_p ? z.x - prime_p : z.x;
		z.y = z.y + y.y;
		z.y = z.y >= prime_p ? z.y - prime_p : z.y;
		z.z = z.z + y.z;
		z.z = z.z >= prime_p ? z.z - prime_p : z.z;
		z.w = z.w + y.w;
		z.w = z.w >= prime_p ? z.w - prime_p : z.w;
	}*/
#pragma unroll 16
	for (int i = 0; i < iters; ++i) {

		asm(

		  //  ".func (.reg .u32 lo, .reg .u32 hi) modadd (.reg .u32 a0, .reg .u32 a1, .reg .u32 b0, .reg .u32 b1, .reg .u32 p0, .reg .u32 p1)\n\t"
		  //  "{\n\t"
		  //  " mov.u32 lo a0;\n\t"
		  //  " mov.u32 a1;\n\t"
		  //  "	ret;\n\t"
		  //  "}\n\t"
		  // x.x += y.x; if >= prime_p x.x -= prime_p  (repeat pattern for all 8)
		  "{\n\t.reg .pred p0, p1, p2, p3, p4, p5, p6, p7; \n\t"
		  "add.cc.u32     %0, %0, %16; \n\t"   // lo add
		  "addc.u32    %1, %1, %17; \n\t"	   // hi add + carry-in 0
		  "add.cc.u32     %2, %2, %16; \n\t"   // lo add
		  "addc.u32    %3, %3, %17; \n\t"	   // hi add + carry-in 0
		  "add.cc.u32     %4, %4, %16; \n\t"   // lo add
		  "addc.u32    %5, %5, %17; \n\t"	   // hi add + carry-in 0
		  "add.cc.u32     %6, %6, %16; \n\t"   // lo add
		  "addc.u32    %7, %7, %17; \n\t"	   // hi add + carry-in 0
		  "add.cc.u32     %8, %8, %16; \n\t"   // lo add
		  "addc.u32    %9, %9, %17; \n\t"	   // hi add + carry-in 0
		  "add.cc.u32     %10, %10, %16; \n\t" // lo add
		  "addc.u32    %11, %11, %17; \n\t"	   // hi add + carry-in 0
		  "add.cc.u32     %12, %12, %16; \n\t" // lo add
		  "addc.u32    %13, %13, %17; \n\t"	   // hi add + carry-in 0
		  "add.cc.u32     %14, %14, %16; \n\t" // lo add
		  "addc.u32    %15, %15, %17; \n\t"	   // hi add + carry-in 0

		  "setp.ge.u32 p0, %1, %25; \n\t"  // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p1, %3, %25; \n\t"  // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p2, %5, %25; \n\t"  // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p3, %7, %25; \n\t"  // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p4, %9, %25; \n\t"  // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p5, %11, %25; \n\t" // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p6, %13, %25; \n\t" // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p7, %15, %25; \n\t" // hi >= prime_hi OR carry (p0)

		  "@p0 sub.cc.u32 %0, %0, %24; \n\t"	  // lo -= prime_lo if overflow
		  "@p0 subc.u32 %1, %1, %25; \n\t"		  // hi -= prime_hi if overflow
		  "@p1 sub.cc.u32 %2, %2, %24; \n\t"	  // lo -= prime_lo if overflow
		  "@p1 subc.u32 %3, %3, %25; \n\t"		  // hi -= prime_hi if overflow
		  "@p2 sub.cc.u32 %4, %4, %24; \n\t"	  // lo -= prime_lo if overflow
		  "@p2 subc.u32 %5, %5, %25; \n\t"		  // hi -= prime_hi if overflow
		  "@p3 sub.cc.u32 %6, %6, %24; \n\t"	  // lo -= prime_lo if overflow
		  "@p3 subc.u32 %7, %7, %25; \n\t"		  // hi -= prime_hi if overflow
		  "@p4 sub.cc.u32 %8, %8, %24; \n\t"	  // lo -= prime_lo if overflow
		  "@p4 subc.u32 %9, %9, %25; \n\t"		  // hi -= prime_hi if overflow
		  "@p5 sub.cc.u32 %10, %10, %24; \n\t"	  // lo -= prime_lo if overflow
		  "@p5 subc.u32 %11, %11, %25; \n\t"	  // hi -= prime_hi if overflow
		  "@p6 sub.cc.u32 %12, %12, %24; \n\t"	  // lo -= prime_lo if overflow
		  "@p6 subc.u32 %13, %13, %25; \n\t"	  // hi -= prime_hi if overflow
		  "@p7 sub.cc.u32 %14, %14, %24; \n\t"	  // lo -= prime_lo if overflow
		  "@p7 subc.u32 %15, %15, %25; \n\t}\n\t" // hi -= prime_hi if overflow

		  : "+r"(x[0]),
		  "+r"(x[1]),
		  "+r"(x[2]),
		  "+r"(x[3]),
		  "+r"(x[4]),
		  "+r"(x[5]),
		  "+r"(x[6]),
		  "+r"(x[7]),
		  "+r"(z[0]),
		  "+r"(z[1]),
		  "+r"(z[2]),
		  "+r"(z[3]),
		  "+r"(z[4]),
		  "+r"(z[5]),
		  "+r"(z[6]),
		  "+r"(z[7])
		  : "r"(((uint2)y.x).x),
		  "r"(((uint2)y.x).y),
		  "r"(((uint2)y.y).x),
		  "r"(((uint2)y.y).y),
		  "r"(((uint2)y.z).x),
		  "r"(((uint2)y.z).y),
		  "r"(((uint2)y.w).x),
		  "r"(((uint2)y.w).y),
		  "r"(((uint2)prime_p).x),
		  "r"(((uint2)prime_p).y));
	}
	if (tid == 5555)
		printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);

	// Prevent the compiler from optimizing the loop away
	out[tid] = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + z[0] + z[1] + z[2] + z[3] + z[4] + z[5] + z[6] + z[7];
}

__global__ void int64_modadd4(int* out, int iters) {
	uint tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	uint32_t x[8] = { tid, tid << 1, tid << 2, tid << 3, tid << 5, tid << 6, tid << 7, tid << 8 };
	uint32_t y[8] = { tid, tid << 1, tid << 2, tid << 3, tid << 5, tid << 6, tid << 7, tid << 8 };
	// ulonglong4 y  = { tid << 4, tid << 5, tid << 6, tid << 7 };
	uint32_t z[8] = { tid, tid << 11, tid << 12, tid << 13, tid << 15, tid << 16, tid << 17, tid << 18 };
	//((ulonglong4*)z)[8] = { tid << 5, tid << 6, tid << 7, tid << 8 };
	ulonglong4 t = { tid << 9, tid << 10, tid << 11, tid << 12 };

	if (tid == 111111)
		printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);

	// Unroll to reduce loop overhead and help reach peak throughput
	const uint64_t prime_p = tid << 32 ^ tid;
	/*
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = x.x + y.x;
		x.x = x.x >= prime_p ? x.x - prime_p : x.x;
		x.y = x.y + y.y;
		x.y = x.y >= prime_p ? x.y - prime_p : x.y;
		x.z = x.z + y.z;
		x.z = x.z >= prime_p ? x.z - prime_p : x.z;
		x.w = x.w + y.w;
		x.w = x.w >= prime_p ? x.w - prime_p : x.w;
		z.x = z.x + y.x;
		z.x = z.x >= prime_p ? z.x - prime_p : z.x;
		z.y = z.y + y.y;
		z.y = z.y >= prime_p ? z.y - prime_p : z.y;
		z.z = z.z + y.z;
		z.z = z.z >= prime_p ? z.z - prime_p : z.z;
		z.w = z.w + y.w;
		z.w = z.w >= prime_p ? z.w - prime_p : z.w;
	}*/
#pragma unroll 1
	for (int i = 0; i < iters; ++i) {

		asm volatile(

		  //  ".func (.reg .u32 lo, .reg .u32 hi) modadd (.reg .u32 a0, .reg .u32 a1, .reg .u32 b0, .reg .u32 b1, .reg .u32 p0, .reg .u32 p1)\n\t"
		  //  "{\n\t"
		  //  " mov.u32 lo a0;\n\t"
		  //  " mov.u32 a1;\n\t"
		  //  "	ret;\n\t"
		  //  "}\n\t"
		  // x.x += y.x; if >= prime_p x.x -= prime_p  (repeat pattern for all 8)
		  "{\n\t.reg .pred p0, p1, p2, p3, p4, p5, p6, p7; \n\t"
		  "add.u32     %0, %0, %16; \n\t" // lo add

		  "add.u32    %1, %1, %17; \n\t"	// hi add + carry-in 0
		  "add.u32     %2, %2, %18; \n\t"	// lo add
		  "add.u32    %3, %3, %19; \n\t"	// hi add + carry-in 0
		  "add.u32     %4, %4, %20; \n\t"	// lo add
		  "add.u32    %5, %5, %21; \n\t"	// hi add + carry-in 0
		  "add.u32     %6, %6, %22; \n\t"	// lo add
		  "add.u32    %7, %7, %23; \n\t"	// hi add + carry-in 0
		  "add.u32     %8, %8, %16; \n\t"	// lo add
		  "add.u32    %9, %9, %17; \n\t"	// hi add + carry-in 0
		  "add.u32     %10, %10, %18; \n\t" // lo add
		  "add.u32    %11, %11, %19; \n\t"	// hi add + carry-in 0
		  "add.u32     %12, %12, %20; \n\t" // lo add
		  "add.u32    %13, %13, %21; \n\t"	// hi add + carry-in 0
		  "add.u32     %14, %14, %22; \n\t" // lo add
		  "add.u32    %15, %15, %23; \n\t"	// hi add + carry-in 0

		  "setp.ge.u32 p0, %1, %25; \n\t"  // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p1, %3, %25; \n\t"  // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p2, %5, %25; \n\t"  // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p3, %7, %25; \n\t"  // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p4, %9, %25; \n\t"  // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p5, %11, %25; \n\t" // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p6, %13, %25; \n\t" // hi >= prime_hi OR carry (p0)
		  "setp.ge.u32 p7, %15, %25; \n\t" // hi >= prime_hi OR carry (p0)

		  "@p0 sub.u32 %0, %0, %24; \n\t"		 // lo -= prime_lo if overflow
		  "@p0 sub.u32 %1, %1, %25; \n\t"		 // hi -= prime_hi if overflow
		  "@p1 sub.u32 %2, %2, %24; \n\t"		 // lo -= prime_lo if overflow
		  "@p1 sub.u32 %3, %3, %25; \n\t"		 // hi -= prime_hi if overflow
		  "@p2 sub.u32 %4, %4, %24; \n\t"		 // lo -= prime_lo if overflow
		  "@p2 sub.u32 %5, %5, %25; \n\t"		 // hi -= prime_hi if overflow
		  "@p3 sub.u32 %6, %6, %24; \n\t"		 // lo -= prime_lo if overflow
		  "@p3 sub.u32 %7, %7, %25; \n\t"		 // hi -= prime_hi if overflow
		  "@p4 sub.u32 %8, %8, %24; \n\t"		 // lo -= prime_lo if overflow
		  "@p4 sub.u32 %9, %9, %25; \n\t"		 // hi -= prime_hi if overflow
		  "@p5 sub.u32 %10, %10, %24; \n\t"		 // lo -= prime_lo if overflow
		  "@p5 sub.u32 %11, %11, %25; \n\t"		 // hi -= prime_hi if overflow
		  "@p6 sub.u32 %12, %12, %24; \n\t"		 // lo -= prime_lo if overflow
		  "@p6 sub.u32 %13, %13, %25; \n\t"		 // hi -= prime_hi if overflow
		  "@p7 sub.u32 %14, %14, %24; \n\t"		 // lo -= prime_lo if overflow
		  "@p7 sub.u32 %15, %15, %25; \n\t}\n\t" // hi -= prime_hi if overflow

		  : "+r"(x[0]),
		  "+r"(x[1]),
		  "+r"(x[2]),
		  "+r"(x[3]),
		  "+r"(x[4]),
		  "+r"(x[5]),
		  "+r"(x[6]),
		  "+r"(x[7]),
		  "+r"(z[0]),
		  "+r"(z[1]),
		  "+r"(z[2]),
		  "+r"(z[3]),
		  "+r"(z[4]),
		  "+r"(z[5]),
		  "+r"(z[6]),
		  "+r"(z[7])
		  : "r"(y[0]), "r"(y[1]), "r"(y[2]), "r"(y[3]), "r"(y[4]), "r"(y[5]), "r"(y[6]), "r"(y[7]), "r"(((uint2)prime_p).x), "r"(((uint2)prime_p).y));
	}
	if (tid == 111111)
		printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
	// Prevent the compiler from optimizing the loop away
	out[tid] = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + z[0] + z[1] + z[2] + z[3] + z[4] + z[5] + z[6] + z[7];
}

__global__ void int64_modmult(int* out, int iters) {
	uint tid = blockDim.x * blockIdx.x + threadIdx.x;

	// Simple integer state
	ulonglong4 x = { tid, tid << 1, tid << 2, tid << 3 };
	ulonglong4 y = { tid << 4, tid << 5, tid << 6, tid << 7 };
	ulonglong4 z = { tid << 5, tid << 6, tid << 7, tid << 8 };
	ulonglong4 t = { tid << 9, tid << 10, tid << 11, tid << 12 };

	// Unroll to reduce loop overhead and help reach peak throughput
	for (int i = 0; i < iters; ++i) {
		// 1) add
		x.x = FIDESlib::modmult<FIDESlib::ALGO_BARRETT>((uint64_t)x.x, y.x, 0);
		x.y = FIDESlib::modmult<FIDESlib::ALGO_BARRETT>((uint64_t)x.y, y.y, 0);
		x.z = FIDESlib::modmult<FIDESlib::ALGO_BARRETT>((uint64_t)x.z, y.z, 0);
		x.w = FIDESlib::modmult<FIDESlib::ALGO_BARRETT>((uint64_t)x.w, y.w, 0);
		z.x = FIDESlib::modmult<FIDESlib::ALGO_BARRETT>((uint64_t)z.x, y.x, 0);
		z.y = FIDESlib::modmult<FIDESlib::ALGO_BARRETT>((uint64_t)z.y, y.y, 0);
		z.z = FIDESlib::modmult<FIDESlib::ALGO_BARRETT>((uint64_t)z.z, y.z, 0);
		z.w = FIDESlib::modmult<FIDESlib::ALGO_BARRETT>((uint64_t)z.w, y.w, 0);
	}

	// Prevent the compiler from optimizing the loop away
	out[tid] = x.x + x.y + x.z + x.w + z.x + z.y + z.z + z.w;
}

#define bench(func)                                                                                                       \
	{                                                                                                                     \
		int device = 0;                                                                                                   \
		CUDA_CHECK(cudaSetDevice(device));                                                                                \
                                                                                                                          \
		cudaDeviceProp prop;                                                                                              \
		CUDA_CHECK(cudaGetDeviceProperties(&prop, device));                                                               \
                                                                                                                          \
		int iters			= 100'000;                                                                                    \
		int threadsPerBlock = 128;                                                                                        \
		int blocks			= prop.multiProcessorCount * 48;                                                              \
                                                                                                                          \
		const long long numThreads = 1LL * threadsPerBlock * blocks;                                                      \
                                                                                                                          \
		printf("Device: %s\n", prop.name);                                                                                \
		printf("SMs: %d, clock: %.3f MHz\n", prop.multiProcessorCount, prop.clockRate / 1000.0);                          \
		printf("Launch config: blocks=%d, threadsPerBlock=%d, totalThreads=%lld\n", blocks, threadsPerBlock, numThreads); \
		printf("Loop iters per thread: %d, OPS_PER_ITER=%d\n", iters, OPS_PER_ITER);                                      \
                                                                                                                          \
		int* d_out = nullptr;                                                                                             \
		CUDA_CHECK(cudaMalloc(&d_out, numThreads * sizeof(int)));                                                         \
                                                                                                                          \
		func<<<blocks, threadsPerBlock>>>(d_out, iters);                                                                  \
		CUDA_CHECK(cudaGetLastError());                                                                                   \
		CUDA_CHECK(cudaDeviceSynchronize());                                                                              \
                                                                                                                          \
		cudaEvent_t start, stop;                                                                                          \
		CUDA_CHECK(cudaEventCreate(&start));                                                                              \
		CUDA_CHECK(cudaEventCreate(&stop));                                                                               \
                                                                                                                          \
		CUDA_CHECK(cudaEventRecord(start));                                                                               \
		func<<<blocks, threadsPerBlock>>>(d_out, iters);                                                                  \
		CUDA_CHECK(cudaGetLastError());                                                                                   \
		CUDA_CHECK(cudaEventRecord(stop));                                                                                \
		CUDA_CHECK(cudaEventSynchronize(stop));                                                                           \
                                                                                                                          \
		float ms = 0.0f;                                                                                                  \
		CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));                                                               \
                                                                                                                          \
		CUDA_CHECK(cudaEventDestroy(start));                                                                              \
		CUDA_CHECK(cudaEventDestroy(stop));                                                                               \
                                                                                                                          \
		double seconds	   = ms * 1e-3;                                                                                   \
		double total_iters = static_cast<double>(iters) * static_cast<double>(numThreads);                                \
		double total_ops   = total_iters * static_cast<double>(OPS_PER_ITER);                                             \
		double iops		   = total_ops / seconds;                                                                         \
		double gops		   = iops / 1e9;                                                                                  \
		double tops		   = iops / 1e12;                                                                                 \
		double iops_per_sm = iops / prop.multiProcessorCount;                                                             \
                                                                                                                          \
		printf("\nKernel time: %.3f ms\n", ms);                                                                           \
		printf("Total int32 ops: %.3e\n", total_ops);                                                                     \
		printf("Integer throughput: %.3e ops/s (%.3f GOPS, %.3f TOPS)\n", iops, gops, tops);                              \
		printf("Per-SM integer throughput: %.3e ops/s per SM\n", iops_per_sm);                                            \
                                                                                                                          \
		int sample = 0;                                                                                                   \
		CUDA_CHECK(cudaMemcpy(&sample, d_out + 1, sizeof(int), cudaMemcpyDeviceToHost));                                  \
		printf("Sample output[0] = %d\n", sample);                                                                        \
                                                                                                                          \
		CUDA_CHECK(cudaFree(d_out));                                                                                      \
		CUDA_CHECK(cudaDeviceReset());                                                                                    \
	}

TEST(Microbench, int32) {
	bench(int32_mult);
	bench(int32_add);
	bench(int32_mac);
	bench(int64_mult);
	bench(int64_add);
	bench(int64_mac);
	bench(int64_modadd);
	bench(int64_modadd2);
	bench(int64_modadd3);
	bench(int64_modadd4);
	bench(int64_modmult);
}

#endif