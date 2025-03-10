//
// Created by carlosad on 16/03/24.
//
#ifndef FIDESLIB_CONSTANTSGPU_CUH
#define FIDESLIB_CONSTANTSGPU_CUH

#include <cinttypes>
#include <vector>
//#include "CKKS/openfhe-interface/rawciphertext.cuh"
#include "LimbUtils.cuh"
//#include "CKKS/Parameters.cuh"

namespace FIDESlib {
constexpr int MAXP = 64;
constexpr int MAXD = 8;

enum version { BARRET, DHEM, NEIL };

constexpr version VERSION = NEIL;

namespace CKKS {
struct Scheme {
    struct Constants;
    struct Global;
};

struct Scheme::Constants {};

struct Scheme::Global {};
}  // namespace CKKS

struct Constants {
    int N, L, K, logN;
    uint64_t type;
    uint64_t primes[MAXP];
    uint64_t prime_better_barret_mu[MAXP];
    uint32_t prime_bits[MAXP];
    uint8_t table[MAXP * MAXP * 8];

    uint64_t N_shoup[MAXP];
    uint64_t N_inv[MAXP];
    uint64_t N_inv_shoup[MAXP];
    uint64_t root[MAXP];
    uint64_t root_shoup[MAXP];
    uint64_t inv_root[MAXP];
    uint64_t inv_root_shoup[MAXP];

    int dnum;
    uint64_t P[MAXP];
    uint64_t P_shoup[MAXP];
    uint64_t P_inv[MAXP];
    uint64_t P_inv_shoup[MAXP];

    int num_primeid_digit_from[MAXD][MAXP];
    int num_primeid_digit_to[MAXD][MAXP];

    union {
        struct {
            int primeid_partition[MAXD][MAXP];    // Limb primeids
            int primeid_digit_from[MAXD][MAXP];   // Decomp primeids
            int primeid_digit_to[MAXD][MAXP];     // Digit primeids
            int primeid_special_partition[MAXP];  // Special limb primeids
        };
        int primeid_flattened[(MAXD * 3 + 1) * MAXP];
    };
    union {
        CKKS::Scheme::Constants ckks;
        void* none;
    };
};

constexpr int PARTITION(int id, int j) {
    return (offsetof(Constants, primeid_partition) - offsetof(Constants, primeid_partition)) / sizeof(int) + id * MAXP +
           j;
}
constexpr int SPECIAL(int d, int j) {
    return (offsetof(Constants, primeid_special_partition) - offsetof(Constants, primeid_partition)) / sizeof(int) + j;
}
constexpr int DECOMP(int d, int j) {
    return (offsetof(Constants, primeid_digit_from) - offsetof(Constants, primeid_partition)) / sizeof(int) + d * MAXP +
           j;
}
constexpr int DIGIT(int d, int j) {
    return (offsetof(Constants, primeid_digit_to) - offsetof(Constants, primeid_partition)) / sizeof(int) + d * MAXP +
           j;
}

struct Global {

    // NTT
    void* psi[MAXP];
    void* inv_psi[MAXP];
    void* psi_no[MAXP];
    void* inv_psi_no[MAXP];
    void* psi_middle_scale[MAXP];
    void* inv_psi_middle_scale[MAXP];
    void* psi_shoup[MAXP];
    void* inv_psi_shoup[MAXP];
    // GPU
    void* psi_ptr[MAXD][MAXP];
    void* inv_psi_ptr[MAXD][MAXP];
    void* psi_no_ptr[MAXD][MAXP];
    void* inv_psi_no_ptr[MAXD][MAXP];
    void* psi_middle_scale_ptr[MAXD][MAXP];
    void* inv_psi_middle_scale_ptr[MAXD][MAXP];
    void* psi_barrett_ptr[MAXD][MAXP];
    void* inv_psi_barrett_ptr[MAXD][MAXP];

    uint64_t root[MAXP];
    uint64_t inv_root[MAXP];

    // Conv
    //uint64_t q_[MAXP][MAXP];
    //uint64_t Q_[MAXP][MAXP];
    uint64_t q_inv[MAXP][MAXP];
    uint64_t QlQlInvModqlDivqlModq[MAXP][MAXP];
    uint64_t aux;

    uint64_t ModDown_pre_scale[MAXP];
    uint64_t ModDown_pre_scale_shoup[MAXP];
    uint64_t ModDown_matrix[MAXP][MAXP];
    uint64_t ModDown_matrix_shoup[MAXP][MAXP];

    uint64_t DecompAndModUp_pre_scale[MAXD][MAXP][MAXP];
    uint64_t DecompAndModUp_pre_scale_shoup[MAXD][MAXP][MAXP];
    uint64_t DecompAndModUp_matrix[MAXP][MAXD][MAXP][MAXP];
    uint64_t DecompAndModUp_matrix_shoup[MAXP][MAXD][MAXP][MAXP];

    union {
        CKKS::Scheme::Global ckks;
        void* none;
    };
};

extern __constant__ Constants constants;
namespace Globals {
extern __device__ void* psi[MAXP];
extern __device__ void* inv_psi[MAXP];
extern __device__ void* psi_middle_scale[MAXP];
extern __device__ void* inv_psi_middle_scale[MAXP];
extern __device__ void* psi_no[MAXP];
extern __device__ void* inv_psi_no[MAXP];
extern __device__ void* psi_shoup[MAXP];
extern __device__ void* inv_psi_shoup[MAXP];

//extern __device__ uint64_t q_[MAXP][MAXP];
//extern __device__ uint64_t Q_[MAXP][MAXP];
extern __device__ uint64_t q_inv[MAXP * MAXP];
extern __device__ uint64_t QlQlInvModqlDivqlModq[MAXP * MAXP];

extern __device__ uint64_t ModDown_pre_scale[MAXP];
extern __device__ uint64_t ModDown_pre_scale_shoup[MAXP];
extern __device__ uint64_t ModDown_matrix[MAXP * MAXP];
extern __device__ uint64_t ModDown_matrix_shoup[MAXP * MAXP];

extern __device__ uint64_t DecompAndModUp_pre_scale[MAXD * MAXP * MAXP];
extern __device__ uint64_t DecompAndModUp_pre_scale_shoup[MAXD * MAXP * MAXP];
extern __device__ uint64_t DecompAndModUp_matrix[MAXP * MAXD * MAXP * MAXP];
extern __device__ uint64_t DecompAndModUp_matrix_shoup[MAXP * MAXD * MAXP * MAXP];
}  // namespace Globals
extern Constants host_constants;
extern Global host_global;

#define C_ (constants)
#define G_ Globals
#define TABLE32(i, j) (constants.table[MAXP * 8 * (i) + 4 * (j)])
#define TABLE64(i, j) (constants.table[MAXP * 8 * (i) + 8 * (j)])

#define MODDOWN_MATRIX(i, j) (MAXP * (i) + (j))
#define MODUPIDX_SCALE(i, j, k) (MAXP * MAXP * (i) + MAXP * (j) + (k))
#define MODUPIDX_MATRIX(i, j, k, l) (MAXD * MAXP * MAXP * (i) + MAXP * MAXP * (j) + MAXP * (k) + (l))
#define hC_ (FIDESlib::host_constants)
#define hG_ (FIDESlib::host_global)

#define ISU64(x) (constants.type & (((uint64_t)1) << (x)))
#define HISU64(x) (host_constants.type & (((uint64_t)1) << (x)))

template <typename Scheme>
__global__ void printConstants();

template <typename Scheme>
void SetupConstants(const std::vector<PrimeRecord>& q, const std::vector<std::vector<LimbRecord>>& meta,
                    const std::vector<PrimeRecord>& p, const std::vector<LimbRecord>& smeta,
                    const std::vector<std::vector<std::vector<LimbRecord>>>& DECOMPmeta,
                    const std::vector<std::vector<std::vector<LimbRecord>>>& DIGITmeta,
                    const std::vector<std::vector<int>>& digitGPUid, const std::vector<int>& GPUid, const int N,
                    const Scheme& parameters);

}  // namespace FIDESlib
#endif  //FIDESLIB_CONSTANTSGPU_CUH
