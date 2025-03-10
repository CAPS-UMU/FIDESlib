//
// Created by seyda on 9/14/24.
//


#include "Rotation.cuh"

namespace FIDESlib::CKKS {

    // to implement automorph on a single limb
    template<typename T>
    __global__ void automorph_(T *a, T *a_rot, const int index, const int br) {
        automorph__(a, a_rot, C_.N, C_.logN, index, br);
    }

    template __global__ void
    automorph_(uint64_t *a, uint64_t *a_rot, const int index, const int br);

    template __global__ void
    automorph_(uint32_t *a, uint32_t *a_rot, const int index, const int br);

    // to implement automorph on multiple limbs
    __global__ void
    automorph_multi_(void **a, void **a_rot, const int k, const int br) {
        if (ISU64(blockIdx.y)) {
            automorph__((uint64_t *) a[blockIdx.y], (uint64_t *) a_rot[blockIdx.y], C_.N, C_.logN, k, br);
        } else {
            automorph__((uint32_t *) a[blockIdx.y], (uint32_t *) a_rot[blockIdx.y], C_.N, C_.logN, k, br);
        }
    }

    // to implement automorph on 2 ciphertexts
    template<typename T>
    __global__ void
    automorph_multi_ct(T ****a, T ****a_rot, const int n, const int n_bits, const int k, const int limb_count,
                       const int ct_count, const int br) {
        int idx = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
        // const int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // 4n correspondsto 2 polynomial for 2 ciphertexts = 4n elements
        if (idx >= 4 * n) {
            return;
        } else if (idx >= 3 * n) {
            idx = idx - 3 * n;

            int jTmp = ((idx << 1) + 1);
            int rotIndex = ((jTmp * k) - (((jTmp * k) >> (n_bits + 1)) << (n_bits + 1))) >> 1;

            // Bit reversal:
            if (br == 1) {
                idx = __brev(idx) >> (32 - n_bits);
                rotIndex = __brev(rotIndex) >> (32 - n_bits);
            }

            for (int l = 0; l < limb_count; ++l) {
                a_rot[1][1][l][idx] = a[1][1][l][rotIndex];
            }
            return;

        } else if (idx >= 2 * n) {
            idx = idx - 2 * n;

            int jTmp = ((idx << 1) + 1);
            int rotIndex = ((jTmp * k) - (((jTmp * k) >> (n_bits + 1)) << (n_bits + 1))) >> 1;

            // Bit reversal:
            if (br == 1) {
                idx = __brev(idx) >> (32 - n_bits);
                rotIndex = __brev(rotIndex) >> (32 - n_bits);
            }

            for (int l = 0; l < limb_count; ++l) {
                a_rot[1][0][l][idx] = a[1][0][l][rotIndex];
            }
            return;
        } else if (idx >= n) {
            idx = idx - n;

            int jTmp = ((idx << 1) + 1);
            int rotIndex = ((jTmp * k) - (((jTmp * k) >> (n_bits + 1)) << (n_bits + 1))) >> 1;

            // Bit reversal:
            if (br == 1) {
                idx = __brev(idx) >> (32 - n_bits);
                rotIndex = __brev(rotIndex) >> (32 - n_bits);
            }

            for (int l = 0; l < limb_count; ++l) {
                a_rot[0][1][l][idx] = a[0][1][l][rotIndex];
            }
            return;
        } else {

            int jTmp = ((idx << 1) + 1);
            int rotIndex = ((jTmp * k) - (((jTmp * k) >> (n_bits + 1)) << (n_bits + 1))) >> 1;

            // Bit reversal:
            if (br == 1) {
                idx = __brev(idx) >> (32 - n_bits);
                rotIndex = __brev(rotIndex) >> (32 - n_bits);
            }

            for (int l = 0; l < limb_count; ++l) {
                a_rot[0][0][l][idx] = a[0][0][l][rotIndex];
            }
            return;
        }

    }

    template __global__ void
    automorph_multi_ct(uint64_t ****a, uint64_t ****a_rot, const int n, const int n_bits, const int index,
                       const int limb_count, const int ct_count, const int br);

    template __global__ void
    automorph_multi_ct(uint32_t ****a, uint32_t ****a_rot, const int n, const int n_bits, const int index,
                       const int limb_count, const int ct_count, const int br);

}
