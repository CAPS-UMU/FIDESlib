//
// Created by carlosad on 22/12/25.
//
#include "PeerUtils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "CudaUtils.cuh"

std::vector<cudaGraphNode_t> get_current_capture_dependencies(cudaStream_t s) {
	cudaStreamCaptureStatus status;
	cudaGraphNode_t* depNodes = nullptr;
	size_t numDeps			  = 0;

	cudaError_t err = cudaStreamGetCaptureInfo(s, &status, nullptr, nullptr, ((const cudaGraphNode_t**)&depNodes), &numDeps);

	std::vector<cudaGraphNode_t> deps;
	if (err == cudaSuccess && numDeps > 0 && depNodes) {
		deps.assign(depNodes, depNodes + numDeps);
	}
	return deps;
}

static void launch(cudaFunction_t kernel, dim3 grid, dim3 block, void** args, uint32_t args_size, cudaStream_t s) {
	// === Step 1: Snapshot current graph dependencies BEFORE launch ===
	// These are the nodes that must complete before our kernel
	CudaCheckErrorModNoSync;
	std::vector<cudaGraphNode_t> deps_before = get_current_capture_dependencies(s);
	std::cout << "[P2P] Dependencies before launch: " << deps_before.size() << std::endl;

	CudaCheckErrorModNoSync;
	CUlaunchAttribute attr[] = { { .id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN, .value = { .memSyncDomain = CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE } },
		{ .id = CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE, .value = { .deviceUpdatableKernelNode = { .deviceUpdatable = 1, .devNode = nullptr } } } };

	// void* extra[] = { CU_LAUNCH_PARAM_BUFFER_POINTER, args, CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size, CU_LAUNCH_PARAM_END };
	/* Cooperative Group Array (CGA)
	 * On sm90 and later we have an extra level of hierarchy where we
	 * can group together several blocks within the Grid, called
	 * Thread Block Clusters.
	 * Clusters enable multiple thread blocks running concurrently
	 * across multiple SMs to synchronize and collaboratively fetch
	 * and exchange data. A cluster of blocks are guaranteed to be
	 * concurrently scheduled onto a group of SMs.
	 * The maximum value is 8 and it must be divisible into the grid dimensions
	 */
	/*
	CUlaunchConfig launchConfig = {0};
	CUlaunchAttribute launchAttrs[3];
	int attrs = 0;
		if (clusterSize) {
			// Grid dimension must be divisible by clusterSize
			if (grid.x % clusterSize) clusterSize = 1;
			launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
			launchAttrs[attrs++].value.clusterDim = {clusterSize, 1, 1};
			launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
			launchAttrs[attrs++].value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
		}
#if CUDART_VERSION >= 12000
		if (compCap >= 90 && driverVersion >= 12000) {
			// Set the NCCL Mem Sync domain on CUDA 12.0 and later (sm90)
			launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
			launchAttrs[attrs++].value.memSyncDomain = (CUlaunchMemSyncDomain) ncclParamMemSyncDomain();
		}
#endif
*/

	CUlaunchConfig config = {
		.gridDimX		= grid.x,
		.gridDimY		= grid.y,
		.gridDimZ		= grid.z,
		.blockDimX		= block.x,
		.blockDimY		= block.y,
		.blockDimZ		= block.z,
		.sharedMemBytes = 0, // grid, block, shared memory
		.hStream		= s, // stream
		.attrs			= attr,
		.numAttrs		= sizeof(attr) / sizeof(CUlaunchAttribute),
	};

	// This bypasses some capture restrictions
	CUresult launchResult = cuLaunchKernelEx(&config, kernel, args /*nullptr*/, nullptr /*extra*/);
	// CUresult launchResult = cuLaunchKernelExC(&config, &p2p_polling_kernel, args);
	// CUresult launchResult = cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, s, args, nullptr);

	if (launchResult != CUDA_SUCCESS) {
		std::cerr << "Failed to launch kernel: " << launchResult << "\n";
		CudaCheckErrorModNoSync;
		return;
	}

	// auto& kernelNode = attr[1].value.deviceUpdatableKernelNode.devNode;

	if (0) {
		// === Step 7: Retrieve snapshot of graph AFTER launch ===
		// NCCL explicitly updates capture dependencies after cuLaunchKernelEx
		cudaError_t err = cudaStreamUpdateCaptureDependencies(s, // The capturing stream
		  nullptr,												 // No specific nodes to depend on
		  0,													 // No additional nodes
		  cudaStreamAddCaptureDependencies						 // Add implicit dependencies
		);

		CudaCheckErrorModNoSync;

		if (err != cudaSuccess) {
			std::cerr << "[P2P] WARNING: cudaStreamUpdateCaptureDependencies failed: " << cudaGetErrorString(err) << "\n";
		} else {
			std::cout << "[P2P] Capture dependencies updated\n";
		}

		// === Step 8: Get the newly captured graph info ===
		// At this point, a new kernel node should be in the graph
		cudaStreamCaptureStatus status;
		cudaGraph_t captured_graph		  = nullptr;
		cudaGraphNode_t* deps_after_array = nullptr;
		size_t numDeps_after			  = 0;

		err = cudaStreamGetCaptureInfo(s,
		  &status,
		  nullptr, // captureID (optional)
		  &captured_graph,
		  (const cudaGraphNode_t**)&deps_after_array,
		  &numDeps_after);

		if (err != cudaSuccess) {
			std::cerr << "[P2P] ERROR: Failed to get capture info: " << cudaGetErrorString(err) << "\n";
			return;
		}

		std::cout << "[P2P] Capture status: " << (status == cudaStreamCaptureStatusActive ? "ACTIVE" : "INACTIVE") << "\n";
		std::cout << "[P2P] Current capture dependencies: " << numDeps_after << "\n";

		std::vector<cudaGraphNode_t> deps_after;
		if (numDeps_after > 0 && deps_after_array) {
			deps_after.assign(deps_after_array, deps_after_array + numDeps_after);
		}

		// === Step 9: Get all nodes in the graph to identify the new kernel node ===
		if (captured_graph) {
			size_t numNodes = 0;
			err				= cudaGraphGetNodes(captured_graph, nullptr, &numNodes);

			if (err == cudaSuccess && numNodes > 0) {
				std::vector<cudaGraphNode_t> all_nodes(numNodes);
				err = cudaGraphGetNodes(captured_graph, all_nodes.data(), &numNodes);

				if (err == cudaSuccess) {
					std::cout << "[P2P] Graph now contains " << numNodes << " total nodes\n";

					// The last node in the graph is likely our newly added kernel
					// But let's examine more carefully
					if (!all_nodes.empty()) {
						cudaGraphNode_t latest_node = all_nodes.back();
						cudaGraphNodeType nodeType;

						err = cudaGraphNodeGetType(latest_node, &nodeType);
						if (err == cudaSuccess && nodeType == cudaGraphNodeTypeKernel) {
							std::cout << "[P2P] Latest node is a KERNEL node\n";

							/*
								if (out_newly_added_nodes) {

									out_newly_added_nodes->push_back(latest_node);
								}
								*/

							// === Step 10: EXPLICITLY ADD DEPENDENCIES ===
							// This is the critical part NCCL does:
							// For peer kernels, explicitly establish dependencies
							// to ensure proper synchronization

							if (!deps_before.empty()) {
								std::cout << "[P2P] Adding explicit dependencies from " << deps_before.size() << " predecessor nodes\n";
								CudaCheckErrorModNoSync;
								err = cudaGraphAddDependencies(captured_graph,
								  deps_before.data(), // From these nodes
								  &latest_node,		  // To this kernel node
								  deps_before.size()  // This many dependencies
								);
								CudaCheckErrorModNoSync;
								if (err != cudaSuccess) {
									std::cerr << "[P2P] ERROR: Failed to add dependencies: " << cudaGetErrorString(err) << "\n";
								} else {
									std::cout << "[P2P] Dependencies added successfully\n";
								}
							}
						}
					}
				}
			}
		}
	}

	std::cout << "[P2P] P2P kernel launch in graph capture complete\n";
	CudaCheckErrorModNoSync;
}

namespace FIDESlib {

void pollingKernel(TimelineSemaphore* gpu1_complete_flag, uint64_t value, cudaStream_t s) {
	return;
	if (1 || !is_stream_being_captured(s)) {
		// p2p_polling_kernel<<<1, 32, 0, s>>>(gpu1_complete_flag, value);
		hostpin_polling_kernel<<<1, 32, 0, s>>>(gpu1_complete_flag, value);

	} else if (0) {

		if (0) {
			void* args[] = { &gpu1_complete_flag, &value };

			cudaFunction_t kernel;
			CudaCheckErrorModNoSync;
			cudaGetFuncBySymbol(&kernel, (const void*)&p2p_polling_kernel);
			CudaCheckErrorModNoSync;

			launch(kernel, 1, 32, args, 2, s);
		}

	} else {
		// Get the current capturing graph context
		cudaStreamCaptureStatus capture_status;
		cudaGraph_t capturing_graph;
		const cudaGraphNode_t* deps;
		size_t dep_count;

		cudaStreamGetCaptureInfo(s, &capture_status, nullptr, &capturing_graph, &deps, &dep_count);

		// ========================================================================
		// Create peer access kernel node with V2 parameters (supports attributes)
		// ========================================================================

		// Setup kernel parameters with launch attributes
		cudaKernelNodeParams kernel_params = { 0 };
		kernel_params.func				   = (void*)p2p_polling_kernel;
		kernel_params.gridDim			   = { 1, 1, 1 };
		kernel_params.blockDim			   = { 32, 1, 1 };
		kernel_params.sharedMemBytes	   = 0;

		// Kernel arguments
		void* args[]			   = { &gpu1_complete_flag, &value };
		kernel_params.kernelParams = args;
		kernel_params.extra		   = nullptr;

		// Add the kernel node to the capturing graph
		cudaGraphNode_t peer_kernel_node;
		cudaGraphAddKernelNode(&peer_kernel_node,
		  capturing_graph,
		  deps,
		  dep_count,
		  /*cudaGraphNodeTypeKernel,*/ &kernel_params);

		// ✅ CRITICAL: Set memory sync domain to REMOTE for peer access
		cudaLaunchAttributeValue attr_value;
		attr_value.memSyncDomain = cudaLaunchMemSyncDomainRemote;

		cudaGraphKernelNodeSetAttribute(peer_kernel_node, cudaLaunchAttributeMemSyncDomain, &attr_value);

		// Update stream dependencies so subsequent work depends on peer kernel
		cudaStreamUpdateCaptureDependencies(s, &peer_kernel_node, 1, 1);
	}
}

__global__ void notify_kernel_hostpin(TimelineSemaphore* gpu_complete_flag, uint64_t value) {
	if (threadIdx.x == 0) {
		__threadfence_system();
		gpu_complete_flag->value.fetch_max(value, cuda::memory_order_relaxed);
		//__nv_atomic_fetch_max_u64_system();
		//__nv_atomic_fetch_max_u64_system((unsigned long long*)hostSem, value, __NV_ATOMIC_RELAXED);
		// atomicStore_system((unsigned long long*)hostSyncFlag, 1ULL);  // Lightweight write to GPU1
	}
}

void notifyKernel(TimelineSemaphore* gpu1_complete_flag, uint64_t value, cudaStream_t s) {
	return;
	if (1 || !is_stream_being_captured(s)) {
		notify_kernel_hostpin<<<1, 32, 0, s>>>(gpu1_complete_flag, value);
		// notify_kernel<<<1, 32, 0, s>>>(gpu1_complete_flag, value);
	} else if (1) {

		void* args[] = { &gpu1_complete_flag, &value };

		cudaFunction_t kernel;
		CudaCheckErrorModNoSync;
		cudaGetFuncBySymbol(&kernel, (const void*)&notify_kernel);
		CudaCheckErrorModNoSync;

		launch(kernel, 1, 32, args, 2, s);

	} else {
		// Get the current capturing graph context
		cudaStreamCaptureStatus capture_status;
		cudaGraph_t capturing_graph;
		const cudaGraphNode_t* deps;
		size_t dep_count;

		cudaStreamGetCaptureInfo(s, &capture_status, nullptr, &capturing_graph, &deps, &dep_count);

		// ========================================================================
		// Create peer access kernel node with V2 parameters (supports attributes)
		// ========================================================================

		// Setup kernel parameters with launch attributes
		cudaKernelNodeParams kernel_params = { 0 };
		kernel_params.func				   = (void*)notify_kernel;
		kernel_params.gridDim			   = { 1, 1, 1 };
		kernel_params.blockDim			   = { 32, 1, 1 };
		kernel_params.sharedMemBytes	   = 0;

		// Kernel arguments
		void* args[]			   = { &gpu1_complete_flag, &value };
		kernel_params.kernelParams = args;
		kernel_params.extra		   = nullptr;

		// Add the kernel node to the capturing graph
		cudaGraphNode_t peer_kernel_node;
		cudaGraphAddKernelNode(&peer_kernel_node,
		  capturing_graph,
		  deps,
		  dep_count,
		  /*cudaGraphNodeTypeKernel,*/ &kernel_params);

		// ✅ CRITICAL: Set memory sync domain to REMOTE for peer access
		cudaLaunchAttributeValue attr_value;
		attr_value.memSyncDomain = cudaLaunchMemSyncDomainRemote;

		cudaGraphKernelNodeSetAttribute(peer_kernel_node, cudaLaunchAttributeMemSyncDomain, &attr_value);

		// Update stream dependencies so subsequent work depends on peer kernel
		cudaStreamUpdateCaptureDependencies(s, &peer_kernel_node, 1, 1);
	}
}

__global__ void p2p_polling_kernel(volatile uint32_t* completion_flag, uint32_t value) {
	if (threadIdx.x == 0) {
		while (*completion_flag < value) {
			// Busy-wait: poll until flag becomes non-zero
		}
	}
}

__global__ void hostpin_polling_kernel(TimelineSemaphore* completion_flag, uint64_t value) {
	if (threadIdx.x == 0) {
		// while (__nv_atomic_load_u64_system((unsigned long long*)completion_flag, __NV_ATOMIC_RELAXED) < value) {}
		//__threadfence_system();
		while (completion_flag->value.load(cuda::memory_order_relaxed) < value) {
		}
		__threadfence_system();
		// printf("Read signal %lu, expected %lu\n", completion_flag->value.load(cuda::memory_order_relaxed), value);
	}
}

bool is_stream_being_captured(cudaStream_t stream) {
	cudaStreamCaptureStatus capture_status;
	cudaStreamGetCaptureInfo(stream, &capture_status, nullptr, nullptr, nullptr, nullptr);

	return capture_status == cudaStreamCaptureStatusActive;
}

void verify_all_streams_joined(cudaStream_t main_stream) {
	// Check if we're in capture mode
	cudaStreamCaptureStatus status;
	cudaStreamIsCapturing(main_stream, &status);

	if (status == cudaStreamCaptureStatusActive) {
		printf("Still capturing - about to end capture\n");

		// Get all subsidiary streams involved
		cudaGraph_t capturing_graph;
		cudaStreamGetCaptureInfo(main_stream, &status, NULL, &capturing_graph, NULL, NULL);

		// Get all nodes to check for any outstanding work
		size_t num_nodes;
		cudaGraphGetNodes(capturing_graph, NULL, &num_nodes);

		printf("Current nodes in capturing graph: %zu\n", num_nodes);

		// This will help identify if any work is orphaned
		// If EndCapture fails with "unjoined work", you'll see it here
	}
}

const char* getNodeTypeName(cudaGraphNodeType type) {
	switch (type) {
	case cudaGraphNodeTypeKernel: return "Kernel";
	case cudaGraphNodeTypeMemcpy: return "Memcpy";
	case cudaGraphNodeTypeMemset: return "Memset";
	case cudaGraphNodeTypeHost: return "Host";
	case cudaGraphNodeTypeGraph: return "Graph";
	case cudaGraphNodeTypeEmpty: return "Empty";
	case cudaGraphNodeTypeWaitEvent: return "WaitEvent";
	case cudaGraphNodeTypeEventRecord: return "EventRecord";
	default: return "Unknown";
	}
}

void printGraphDependencies(cudaGraph_t graph, const char* name) {
	printf("\n================== GRAPH: %s ==================\n", name);

	// Get all nodes
	size_t num_nodes;
	cudaGraphGetNodes(graph, NULL, &num_nodes);

	if (num_nodes == 0) {
		printf("⚠ Graph is empty (0 nodes)\n\n");
		return;
	}

	std::vector<cudaGraphNode_t> nodes(num_nodes);
	cudaGraphGetNodes(graph, nodes.data(), &num_nodes);

	printf("Total nodes: %zu\n\n", num_nodes);

	// Create node index map for readable output
	std::map<uintptr_t, size_t> node_to_idx;
	for (size_t i = 0; i < num_nodes; i++) {
		node_to_idx[(uintptr_t)nodes[i]] = i;
	}

	// Print each node with full dependency info
	for (size_t i = 0; i < num_nodes; i++) {
		cudaGraphNodeType type;
		cudaGraphNodeGetType(nodes[i], &type);

		// Incoming edges (dependencies)
		size_t num_incoming;
		cudaGraphNodeGetDependencies(nodes[i], NULL, &num_incoming);

		// Outgoing edges (dependents)
		size_t num_outgoing;

		cudaGraphNodeGetDependentNodes(nodes[i], NULL, &num_outgoing);

		printf("Node[%zu]: %s | Incoming: %zu | Outgoing: %zu\n", i, getNodeTypeName(type), num_incoming, num_outgoing);

		// Print incoming edges
		if (num_incoming > 0) {
			std::vector<cudaGraphNode_t> deps(num_incoming);
			cudaGraphNodeGetDependencies(nodes[i], deps.data(), &num_incoming);

			printf("  ← Depends on: ");
			for (size_t j = 0; j < num_incoming; j++) {
				printf("[%zu]", node_to_idx[(uintptr_t)deps[j]]);
				if (j < num_incoming - 1)
					printf(", ");
			}
			printf("\n");
		}

		// Print outgoing edges
		if (num_outgoing > 0) {
			std::vector<cudaGraphNode_t> dependents(num_outgoing);
			cudaGraphNodeGetDependentNodes(nodes[i], dependents.data(), &num_outgoing);

			printf("  → Used by: ");
			for (size_t j = 0; j < num_outgoing; j++) {
				printf("[%zu]", node_to_idx[(uintptr_t)dependents[j]]);
				if (j < num_outgoing - 1)
					printf(", ");
			}
			printf("\n");
		}

		// Kernel-specific info
		if (type == cudaGraphNodeTypeKernel) {
			cudaKernelNodeParams params = {};
			cudaGraphKernelNodeGetParams(nodes[i], &params);
			printf("  Kernel: Grid(%u,%u,%u) Block(%u,%u,%u) Shared=%u\n",
			  params.gridDim.x,
			  params.gridDim.y,
			  params.gridDim.z,
			  params.blockDim.x,
			  params.blockDim.y,
			  params.blockDim.z,
			  params.sharedMemBytes);
		}
	}

	printf("\n--- ANALYSIS ---\n");

	// Find sources (no incoming)
	printf("Source nodes (no dependencies): ");
	int source_count = 0;
	for (size_t i = 0; i < num_nodes; i++) {
		size_t num_deps;
		cudaGraphNodeGetDependencies(nodes[i], NULL, &num_deps);
		if (num_deps == 0) {
			printf("[%zu] ", i);
			source_count++;
		}
	}
	printf("(count: %d)\n", source_count);

	// Find sinks (no outgoing)
	printf("Sink nodes (no dependents): ");
	int sink_count = 0;
	for (size_t i = 0; i < num_nodes; i++) {
		size_t num_dependents;
		cudaGraphNodeGetDependentNodes(nodes[i], NULL, &num_dependents);
		if (num_dependents == 0) {
			printf("[%zu] ", i);
			sink_count++;
		}
	}
	printf("(count: %d)\n", sink_count);

	// Check for unjoined work (orphaned nodes)
	printf("Unjoined work check: ");
	bool all_connected = true;
	for (size_t i = 0; i < num_nodes; i++) {
		size_t num_deps;
		size_t num_dependents = 0;
		cudaGraphNodeGetDependentNodes(nodes[i], NULL, &num_deps);

		// Orphaned if no incoming AND no outgoing (except sources/sinks)
		if (num_deps == 0 && num_dependents == 0 && num_nodes > 1) {
			printf("\n⚠ WARNING: Node[%zu] is ORPHANED (isolated)!", i);
			all_connected = false;
		}
	}
	if (all_connected) {
		printf("✓ All nodes connected properly");
	}

	printf("\nGraph ready for EndCapture: %s\n", (source_count > 0 && sink_count > 0) ? "✓ YES" : "✗ NO");
	printf("================================================\n\n");
}

bool pathExists(cudaGraphNode_t source, cudaGraphNode_t sink, const std::vector<cudaGraphNode_t>& all_nodes, std::set<uintptr_t>& visited) {
	if ((uintptr_t)source == (uintptr_t)sink)
		return true;

	visited.insert((uintptr_t)source);

	// Get dependents of source
	size_t num_dependents;
	cudaGraphNodeGetDependentNodes(source, NULL, &num_dependents);

	if (num_dependents > 0) {
		std::vector<cudaGraphNode_t> dependents(num_dependents);
		cudaGraphNodeGetDependentNodes(source, dependents.data(), &num_dependents);

		for (auto& dep : dependents) {
			if (visited.find((uintptr_t)dep) == visited.end()) {
				if (pathExists(dep, sink, all_nodes, visited)) {
					return true;
				}
			}
		}
	}

	return false;
}

void printGraphDependencies2(cudaGraph_t graph, const char* name) {
	printf("\n================== GRAPH: %s ==================\n", name);

	// Get all nodes
	size_t num_nodes;
	cudaGraphGetNodes(graph, NULL, &num_nodes);

	if (num_nodes == 0) {
		printf("⚠ Graph is empty (0 nodes)\n\n");
		return;
	}
	CudaCheckErrorModNoSync;

	std::vector<cudaGraphNode_t> nodes(num_nodes);
	cudaGraphGetNodes(graph, nodes.data(), &num_nodes);

	printf("Total nodes: %zu\n\n", num_nodes);

	// Create node index map
	std::map<uintptr_t, size_t> node_to_idx;
	for (size_t i = 0; i < num_nodes; i++) {
		node_to_idx[(uintptr_t)nodes[i]] = i;
	}

	CudaCheckErrorModNoSync;
	// Print each node
	for (size_t i = 0; i < num_nodes; i++) {
		cudaGraphNodeType type;
		cudaGraphNodeGetType(nodes[i], &type);
		CudaCheckErrorModNoSync;
		size_t num_incoming;
		cudaGraphNodeGetDependencies(nodes[i], NULL, &num_incoming);
		CudaCheckErrorModNoSync;
		size_t num_outgoing;
		cudaGraphNodeGetDependentNodes(nodes[i], NULL, &num_outgoing);
		CudaCheckErrorModNoSync;
		printf("Node[%zu]: %s | Incoming: %zu | Outgoing: %zu\n", i, getNodeTypeName(type), num_incoming, num_outgoing);

		// Print incoming edges
		if (num_incoming > 0) {
			std::vector<cudaGraphNode_t> deps(num_incoming);
			cudaGraphNodeGetDependencies(nodes[i], deps.data(), &num_incoming);
			CudaCheckErrorModNoSync;
			printf("  ← Depends on: ");
			for (size_t j = 0; j < num_incoming; j++) {
				printf("[%zu]", node_to_idx[(uintptr_t)deps[j]]);
				if (j < num_incoming - 1)
					printf(", ");
			}
			printf("\n");
		}

		// Print outgoing edges
		if (num_outgoing > 0) {
			std::vector<cudaGraphNode_t> dependents(num_outgoing);
			cudaGraphNodeGetDependentNodes(nodes[i], dependents.data(), &num_outgoing);
			CudaCheckErrorModNoSync;
			printf("  → Used by: ");
			for (size_t j = 0; j < num_outgoing; j++) {
				printf("[%zu]", node_to_idx[(uintptr_t)dependents[j]]);
				if (j < num_outgoing - 1)
					printf(", ");
			}
			printf("\n");
		}

		// Kernel info
		if (type == cudaGraphNodeTypeKernel) {
			// cudaKernelNodeParams params = {};
			//  cudaGraphKernelNodeGetParams(nodes[i], &params);
			//  CudaCheckErrorModNoSync;
			//     printf("  Grid(%u,%u,%u) Block(%u,%u,%u) Shared=%zu\n", params.gridDim.x, params.gridDim.y,
			//            params.gridDim.z, params.blockDim.x, params.blockDim.y, params.blockDim.z, params.sharedMemBytes);
		}
		CudaCheckErrorModNoSync;
	}

	printf("\n--- CONNECTIVITY ANALYSIS ---\n");

	// Find source nodes
	std::vector<size_t> source_nodes;
	printf("Source nodes: ");
	for (size_t i = 0; i < num_nodes; i++) {
		size_t num_deps;
		cudaGraphNodeGetDependencies(nodes[i], NULL, &num_deps);
		if (num_deps == 0) {
			printf("[%zu] ", i);
			source_nodes.push_back(i);
		}
	}
	printf("(count: %zu)\n", source_nodes.size());
	CudaCheckErrorModNoSync;
	// Find sink nodes
	std::vector<size_t> sink_nodes;
	printf("Sink nodes: ");
	for (size_t i = 0; i < num_nodes; i++) {
		size_t num_dependents;
		cudaGraphNodeGetDependentNodes(nodes[i], NULL, &num_dependents);
		if (num_dependents == 0) {
			printf("[%zu] ", i);
			sink_nodes.push_back(i);
		}
	}
	printf("(count: %zu)\n", sink_nodes.size());
	CudaCheckErrorModNoSync;
	// ===== CRITICAL: Check if multiple sinks are properly joined =====
	if (sink_nodes.size() > 1) {
		printf("\n⚠ MULTIPLE SINKS DETECTED (%zu sinks)\n", sink_nodes.size());
		printf("Checking if sinks are properly joined back to main stream...\n\n");

		bool all_properly_joined = true;

		for (size_t sink_idx : sink_nodes) {
			size_t num_incoming;
			cudaGraphNodeGetDependencies(nodes[sink_idx], NULL, &num_incoming);
			CudaCheckErrorModNoSync;
			printf("Sink[%zu]: Has %zu incoming dependencies\n", sink_idx, num_incoming);

			// Get dependencies for this sink
			std::vector<cudaGraphNode_t> deps(num_incoming);
			cudaGraphNodeGetDependencies(nodes[sink_idx], deps.data(), &num_incoming);
			CudaCheckErrorModNoSync;
			printf("  Depends on: ");
			for (size_t i = 0; i < num_incoming; i++) {
				printf("[%zu]", node_to_idx[(uintptr_t)deps[i]]);
				if (i < num_incoming - 1)
					printf(", ");
			}
			printf("\n");

			// Trace back to find which source this sink connects to
			std::set<uintptr_t> visited;
			for (auto& source_idx : source_nodes) {
				std::set<uintptr_t> temp_visited;
				if (pathExists(nodes[source_idx], nodes[sink_idx], nodes, temp_visited)) {
					printf("  ✓ Path exists from Source[%zu] to Sink[%zu]\n", source_idx, sink_idx);
				} else {
					printf("  ✗ NO path from Source[%zu] to Sink[%zu] - ORPHANED!\n", source_idx, sink_idx);
					all_properly_joined = false;
				}
			}
			CudaCheckErrorModNoSync;
		}

		printf("\n");
		if (all_properly_joined) {
			printf("✓ All sinks properly joined back to sources (stream fork/join is valid)\n");
		} else {
			printf("✗ Some sinks are orphaned (unjoined work detected)\n");
		}
	}

	// Check for cycles
	printf("\nCycle detection: Checking for circular dependencies...\n");
	bool has_cycle = false;

	// Simple cycle check: if any node can reach itself
	for (size_t i = 0; i < num_nodes; i++) {
		std::set<uintptr_t> visited;
		if (pathExists(nodes[i], nodes[i], nodes, visited) && visited.size() > 1) {
			printf("✗ CYCLE DETECTED at Node[%zu]\n", i);
			has_cycle = true;
		}
	}

	if (!has_cycle) {
		printf("✓ No cycles detected\n");
	}
	CudaCheckErrorModNoSync;
	printf("\n--- VERDICT ---\n");
	bool graph_valid = (source_nodes.size() > 0) && (!has_cycle) && (sink_nodes.size() > 0);

	printf("Graph ready for EndCapture: %s\n", graph_valid ? "✓ YES" : "✗ NO");
	printf("================================================\n\n");
}

__global__ void p2p_transfer_1d(const float* src, float* dst, size_t n) {
	// Global linear index
	size_t idx	  = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	// if (idx == 0)
	//     printf("Pointers: %p, %p, size: %d\n", src, dst, n);

	// Vectorized path: copy 4 floats (16B) at a time
	size_t n4 = n / 4; // number of float4 chunks
	for (size_t i = idx; i < n4; i += stride) {
		reinterpret_cast<float4*>(dst)[i] = reinterpret_cast<const float4*>(src)[i];
	}

	// Remainder path: handle last 0–3 elements
	size_t base = n4 * 4;
	for (size_t i = base + idx; i < n; i += stride) {
		dst[i] = src[i];
	}
}

void transferKernel(float* src, float* dst, size_t elems, cudaStream_t s, int src_dev, int dst_dev, size_t involved_sm) {
	// return;
	if (is_stream_being_captured(s)) {

		cudaGraph_t _capturing_graph;
		cudaStreamCaptureStatus _capture_status;
		const cudaGraphNode_t* _deps;
		size_t _dep_count;
		cudaStreamGetCaptureInfo(s, &_capture_status, nullptr, &_capturing_graph, &_deps, &_dep_count);

		cudaGraphNode_t copy_0to1;
		cudaMemcpy3DParms memcpyParams = { 0 };

		memset(&memcpyParams, 0, sizeof(memcpyParams));
		memcpyParams.srcArray = NULL;
		memcpyParams.srcPos	  = make_cudaPos(0, 0, 0);
		memcpyParams.srcPtr	  = make_cudaPitchedPtr(src, elems * sizeof(float), elems * sizeof(float), 1);

		memcpyParams.dstArray = NULL;
		memcpyParams.dstPos	  = make_cudaPos(0, 0, 0);
		memcpyParams.dstPtr	  = make_cudaPitchedPtr(dst, elems * sizeof(float), elems * sizeof(float), 1);
		memcpyParams.extent	  = make_cudaExtent(elems * sizeof(float), 1, 1);
		memcpyParams.kind	  = cudaMemcpyDefault;

		cudaGraphAddMemcpyNode(&copy_0to1, _capturing_graph, _deps, _dep_count, &memcpyParams);

		cudaStreamUpdateCaptureDependencies(s, &copy_0to1, 1, 1);

	} else if (!is_stream_being_captured(s)) {
		// p2p_transfer_1d<<<involved_sm, 128, 0, s>>>(src, dst, elems);

		// cudaPointerAttributes attr_src, attr_dst;
		// cudaPointerGetAttributes(&attr_src, src);
		// cudaPointerGetAttributes(&attr_dst, dst);
		// std::cout << "src: " << attr_src.device << " dst: " << attr_dst.device << std::endl;
		cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, elems * sizeof(float), s);
	} else if (1) {

		CudaCheckErrorModNoSync;
		void* args[] = { &src, &dst, &elems };

		cudaFunction_t kernel;
		CudaCheckErrorModNoSync;
		cudaGetFuncBySymbol(&kernel, (const void*)&p2p_transfer_1d);
		CudaCheckErrorModNoSync;

		launch(kernel, involved_sm, 128, args, 3, s);
		CudaCheckErrorModNoSync;
	} else {
		// Get the current capturing graph context
		cudaStreamCaptureStatus capture_status;
		cudaGraph_t capturing_graph;
		const cudaGraphNode_t* deps;
		size_t dep_count;

		cudaStreamGetCaptureInfo(s, &capture_status, nullptr, &capturing_graph, &deps, &dep_count);

		// ========================================================================
		// Create peer access kernel node with V2 parameters (supports attributes)
		// ========================================================================

		// Setup kernel parameters with launch attributes
		cudaKernelNodeParams kernel_params = { 0 };
		kernel_params.func				   = (void*)p2p_transfer_1d;
		kernel_params.gridDim			   = { (uint32_t)involved_sm, 1, 1 };
		kernel_params.blockDim			   = { 128, 1, 1 };
		kernel_params.sharedMemBytes	   = 0;

		// Kernel arguments
		void* args[]			   = { &src, &dst, &elems };
		kernel_params.kernelParams = args;
		kernel_params.extra		   = nullptr;

		// Add the kernel node to the capturing graph
		cudaGraphNode_t peer_kernel_node;
		cudaGraphAddKernelNode(&peer_kernel_node,
		  capturing_graph,
		  deps,
		  dep_count,
		  /*cudaGraphNodeTypeKernel,*/ &kernel_params);

		// ✅ CRITICAL: Set memory sync domain to REMOTE for peer access
		cudaLaunchAttributeValue attr_value;
		attr_value.memSyncDomain = cudaLaunchMemSyncDomainRemote;

		cudaGraphKernelNodeSetAttribute(peer_kernel_node, cudaLaunchAttributeMemSyncDomain, &attr_value);

		// Update stream dependencies so subsequent work depends on peer kernel
		cudaStreamUpdateCaptureDependencies(s, &peer_kernel_node, 1, 1);
	}
}

__global__ void notify_kernel(volatile uint32_t* gpu_complete_flag, uint32_t value) {
	if (threadIdx.x == 0) {
		*gpu_complete_flag = value; // Lightweight write to GPU1
		__threadfence_system();
	}
}
} // namespace FIDESlib