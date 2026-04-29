//
// Created by seyda on 9/14/24.
//

#include "Rotation.cuh"

namespace FIDESlib::CKKS {

// to implement automorph on a single limb
template <typename T> __global__ void automorph_(T* a, T* a_rot, const int index, const int br) {
	automorph__(a, a_rot, C_.N, C_.logN, index, br);
}

template __global__ void automorph_(uint64_t* a, uint64_t* a_rot, const int index, const int br);

template __global__ void automorph_(uint32_t* a, uint32_t* a_rot, const int index, const int br);

// to implement automorph on multiple limbs
__global__ void automorph_multi_(void** a, void** a_rot, const int k, const int br, const int primeid_init) {
	const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
	if (ISU64(primeid)) {
		automorph__((uint64_t*)a[blockIdx.y], (uint64_t*)a_rot[blockIdx.y], C_.N, C_.logN, k, br);
	} else {
		automorph__((uint32_t*)a[blockIdx.y], (uint32_t*)a_rot[blockIdx.y], C_.N, C_.logN, k, br);
	}
}

} // namespace FIDESlib::CKKS
