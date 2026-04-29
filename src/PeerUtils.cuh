//
// Created by carlosad on 18/12/25.
//

#ifndef FIDESLIB_PEERUTILS_CUH
#define FIDESLIB_PEERUTILS_CUH

#include "CudaUtils.cuh"
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <cuda.h>
#undef duration
#include <cuda/atomic>
#include <cuda_runtime.h>

namespace FIDESlib {

/**
 * Check if a CUDA stream is currently being captured
 * Returns true if capture is active, false otherwise
 */
bool is_stream_being_captured(cudaStream_t stream);

void verify_all_streams_joined(cudaStream_t main_stream);

const char* getNodeTypeName(cudaGraphNodeType type);

// Print detailed graph with all edges and dependencies
void printGraphDependencies(cudaGraph_t graph, const char* name = "Graph");

// Check if a path exists from source to sink
bool pathExists(cudaGraphNode_t source, cudaGraphNode_t sink, const std::vector<cudaGraphNode_t>& all_nodes, std::set<uintptr_t>& visited);

void printGraphDependencies2(cudaGraph_t graph, const char* name = "Graph");

// 1D P2P memcpy-style kernel: coalesced, vectorized, no warp specialization
__global__ void p2p_transfer_1d(const float* __restrict__ src, // pointer on source device (with P2P enabled)
  float* __restrict__ dst,									   // pointer on destination device
  size_t n													   // number of float elements
);

/** Initial profiling of peer transfers (PCIe 4.0x16) shows best performance with 4 thread blocks */
void transferKernel(float* src, float* dst, size_t elems, cudaStream_t s, int src_dev, int dst_dev, size_t involved_sm = 4);

__global__ void notify_kernel(volatile uint32_t* gpu_complete_flag, uint32_t value);

struct TimelineSemaphore {
	cuda::atomic<uint64_t, cuda::thread_scope_system> value{ 0 };
	char pad[120]; // Avoid false sharing
};

__global__ void notify_kernel_hostpin(TimelineSemaphore* gpu_complete_flag, uint64_t value);

void notifyKernel(TimelineSemaphore* gpu1_complete_flag, uint64_t value, cudaStream_t s);

__global__ void p2p_polling_kernel(volatile uint32_t* completion_flag, uint32_t value // Flag on destination GPU
);

__global__ void hostpin_polling_kernel(TimelineSemaphore* completion_flag,
  uint64_t value // Flag on destination GPU
);

void pollingKernel(TimelineSemaphore* gpu1_complete_flag, uint64_t value, cudaStream_t s);

[[maybe_unused]] __global__ static void dummy_kernel() {
}

} // namespace FIDESlib
#endif // FIDESLIB_PEERUTILS_CUH
