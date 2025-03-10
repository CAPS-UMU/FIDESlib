//
// Created by carlosad on 24/03/24.
//
#include <algorithm>
#include <bit>
#include <cassert>
#include <execution>
#include "ConstantsGPU.cuh"
#include "CudaUtils.cuh"
#include "Math.cuh"

#include "CKKS/Parameters.cuh"

namespace FIDESlib {

__constant__ Constants constants;
namespace Globals {
__device__ void* psi[MAXP];
__device__ void* inv_psi[MAXP];
__device__ void* psi_middle_scale[MAXP];
__device__ void* inv_psi_middle_scale[MAXP];
__device__ void* psi_no[MAXP];
__device__ void* inv_psi_no[MAXP];
__device__ void* psi_shoup[MAXP];
__device__ void* inv_psi_shoup[MAXP];

//__device__ uint64_t q_[MAXP][MAXP];
//__device__ uint64_t Q_[MAXP][MAXP];
__device__ uint64_t q_inv[MAXP * MAXP];
__device__ uint64_t QlQlInvModqlDivqlModq[MAXP * MAXP];

__device__ uint64_t ModDown_pre_scale[MAXP];
__device__ uint64_t ModDown_pre_scale_shoup[MAXP];
__device__ uint64_t ModDown_matrix[MAXP * MAXP];
__device__ uint64_t ModDown_matrix_shoup[MAXP * MAXP];

__device__ uint64_t DecompAndModUp_pre_scale[MAXD * MAXP * MAXP];
__device__ uint64_t DecompAndModUp_pre_scale_shoup[MAXD * MAXP * MAXP];
__device__ uint64_t DecompAndModUp_matrix[MAXP * MAXD * MAXP * MAXP];
__device__ uint64_t DecompAndModUp_matrix_shoup[MAXP * MAXD * MAXP * MAXP];
}  // namespace Globals
Constants host_constants;
Global host_global;

template <typename Scheme>
__global__ void printConstants() {
    printf("L = %d, K = %d, type = %lx ", C_.L, C_.K, C_.type);
    for (int i = 0; i < 64; ++i) {
        printf("Prime %d: %lu ", i, C_.primes[i]);
    }
    printf("\n");
}

uint64_t mu_new(const uint64_t q, const uint32_t num_bits) {
    __uint128_t res =
        (((__uint128_t)1) << (2 * num_bits + (VERSION == DHEM ? 3 : (VERSION == NEIL ? 1 : 0)))) / ((__uint128_t)q);
    return res;
}

// Find a primitive root
int generator(uint64_t p) {
    std::vector<uint64_t> fact;
    uint64_t phi = p - 1, n = phi;
    for (uint64_t i = 2; i * i <= n; ++i)
        if (n % i == 0) {
            fact.push_back(i);
            while (n % i == 0)
                n /= i;
        }
    if (n > 1)
        fact.push_back(n);

    for (uint64_t res = 2; res <= p; ++res) {
        bool ok = true;
        for (size_t i = 0; i < fact.size() && ok; ++i)
            ok &= FIDESlib::modpow(res, phi / fact[i], p) != 1;
        if (ok)
            return res;
    }
    return 0;
}

uint64_t unity_root(const uint64_t q, const uint64_t order) {
    assert(q % order == 1);
    const uint64_t c = q / order;
    const uint64_t g = generator(q);
    assert(g != 0);
    const uint64_t res = FIDESlib::modpow(g, c, q);
    assert(FIDESlib::modpow(res, order, q) == 1);
    return res;
}

uint64_t shoup_precomp(uint64_t val, int primeid) {
    return (__uint128_t)((val) << 1) * (1ul << 63) / hC_.primes[primeid];
}

#define freegpu(name)                                \
    do {                                             \
        for (int j = 0; j < MAXD; ++j) {             \
            if (host_global.name[j][i] != nullptr) { \
                cudaFree(host_global.name[j][i]);    \
                host_global.name[j][i] = nullptr;    \
            }                                        \
        }                                            \
    } while (false)

#define free(name)                                     \
    do {                                               \
        if (host_global.name[i] != nullptr) {          \
            if (HISU64(i)) {                           \
                delete (uint64_t*)host_global.name[i]; \
            } else {                                   \
                delete (uint32_t*)host_global.name[i]; \
            }                                          \
            host_global.name[i] = nullptr;             \
        }                                              \
    } while (false)

void cleanUpPrevious() {
    for (int i = 0; i < MAXP; ++i) {
        free(psi);
        free(inv_psi);
        free(psi_middle_scale);
        free(inv_psi_middle_scale);
        free(psi_no);
        free(inv_psi_no);
        free(psi_shoup);
        free(inv_psi_shoup);

        freegpu(psi_ptr);
        freegpu(inv_psi_ptr);
        freegpu(psi_middle_scale_ptr);
        freegpu(inv_psi_middle_scale_ptr);
        freegpu(psi_no_ptr);
        freegpu(inv_psi_no_ptr);
        freegpu(psi_barrett_ptr);
        freegpu(inv_psi_barrett_ptr);
    }
}

template <typename Scheme>
void SetupConstants(const std::vector<PrimeRecord>& q, const std::vector<std::vector<LimbRecord>>& meta,
                    const std::vector<PrimeRecord>& p, const std::vector<LimbRecord>& smeta,
                    const std::vector<std::vector<std::vector<LimbRecord>>>& DECOMPmeta,
                    const std::vector<std::vector<std::vector<LimbRecord>>>& DIGITmeta,
                    const std::vector<std::vector<int>>& digitGPUid, const std::vector<int>& GPUid, const int N,
                    const Scheme& parameters) {
    CudaCheckErrorMod;

    cleanUpPrevious();

    for (int id : GPUid) {
        cudaSetDevice(id);
        hC_.N = N;
        hC_.logN = (int)std::bit_width((uint32_t)N) - 1;
        hC_.L = q.size();
        hC_.K = p.size();

        hC_.type = 0;
        for (auto& i : meta)
            for (auto& j : i) {
                if (j.type == U64) {
                    hC_.type |= (((uint64_t)1) << j.id);
                }
            }

        for (auto& j : smeta)
            if (j.type == U64) {
                hC_.type |= (((uint64_t)1) << j.id);
            }
    }

    {
        for (size_t i = 0; i < q.size(); ++i) {
            hC_.primes[i] = q[i].p;
            hC_.N_shoup[i] = shoup_precomp(hC_.N, i);
            hC_.N_inv[i] = modinv(hC_.N, q[i].p);
            hC_.N_inv_shoup[i] = shoup_precomp(hC_.N_inv[i], i);

            hC_.prime_better_barret_mu[i] = mu_new(q[i].p, q[i].bits);
            hC_.prime_bits[i] = q[i].bits;
        }

        for (size_t i = 0; i < p.size(); ++i) {
            hC_.primes[hC_.L + i] = p[i].p;

            hC_.N_shoup[hC_.L + i] = shoup_precomp(hC_.N, hC_.L + i);
            hC_.N_inv[hC_.L + i] = modinv(N, hC_.primes[hC_.L + i]);
            hC_.N_inv_shoup[hC_.L + i] = shoup_precomp(hC_.N_inv[hC_.L + i], hC_.L + i);
            ;

            hC_.prime_better_barret_mu[hC_.L + i] = mu_new(hC_.primes[hC_.L + i], p[i].bits);
            hC_.prime_bits[hC_.L + i] = p[i].bits;
        }
    }

    {

        std::vector<int> index(hC_.L + hC_.K);
        std::iota(index.begin(), index.end(), 0);
        std::for_each(/*std::execution::par_unseq,*/ index.begin(), index.end(), [&](int i) {
            if (std::is_same_v<CKKS::Parameters, Scheme>) {
                auto param = static_cast<CKKS::Parameters>(parameters);
                if (param.raw != nullptr) {
                    if (i < hC_.L) {
                        hG_.root[i] = param.raw->root_of_unity[i];
                        //hG_.root[i] = modpow(hG_.root[i], param.raw->cyclotomic_order[i] / (N), hC_.primes[i]);
                        //std::cout << "Ciclotomic order " << i <<  ": " << param.raw->cyclotomic_order[i] << std::endl;
                    } else if (i < hC_.L + hC_.K) {
                        hG_.root[i] = param.raw->SPECIALroot_of_unity.at(i - hC_.L);
                    }
                } else {
                    hG_.root[i] = unity_root(hC_.primes[i], 2 * hC_.N);
                }
            } else {
                hG_.root[i] = unity_root(hC_.primes[i], 2 * hC_.N);
            }

            hG_.inv_root[i] = modinv(hG_.root[i], hC_.primes[i]);

            hC_.root[i] = hG_.root[i];
            hC_.root_shoup[i] = shoup_precomp(hG_.root[i], i);
            hC_.inv_root[i] = hG_.inv_root[i];
            hC_.inv_root_shoup[i] = shoup_precomp(hG_.inv_root[i], i);

            //std::swap(hG_.root[i], hG_.inv_root[i]);
            assert(modpow(hG_.root[i], 2 * N, hC_.primes[i]) == 1);
            assert(modpow(hG_.root[i], N, hC_.primes[i]) == hC_.primes[i] - 1);
            assert(modpow(hG_.inv_root[i], 2 * N, hC_.primes[i]) == 1);
            assert(modpow(hG_.inv_root[i], N, hC_.primes[i]) == hC_.primes[i] - 1);
            assert(modprod(hG_.root[i], hG_.inv_root[i], hC_.primes[i]) == 1);

            int bytes = hC_.N * ((HISU64(i)) ? sizeof(uint64_t) : sizeof(uint32_t));

            hG_.psi[i] = malloc(bytes);
            hG_.inv_psi[i] = malloc(bytes);
            hG_.psi_no[i] = malloc(2 * bytes);
            hG_.inv_psi_no[i] = malloc(2 * bytes);
            hG_.psi_middle_scale[i] = malloc(bytes);
            hG_.inv_psi_middle_scale[i] = malloc(bytes);
            hG_.psi_shoup[i] = malloc(bytes);
            hG_.inv_psi_shoup[i] = malloc(bytes);

            if (!HISU64(i)) {
                ((uint32_t*)hG_.psi_no[i])[0] = 1;
                ((uint32_t*)hG_.inv_psi_no[i])[0] = 1;
            } else {
                ((uint64_t*)hG_.psi_no[i])[0] = 1;
                ((uint64_t*)hG_.inv_psi_no[i])[0] = 1;
            }
            for (int j = 1; j < 2 * N; ++j) {
                if (!HISU64(i)) {
                    ((uint32_t*)hG_.psi_no[i])[j] =
                        modprod(hG_.root[i], ((uint32_t*)hG_.psi_no[i])[j - 1], hC_.primes[i]);
                    ((uint32_t*)hG_.inv_psi_no[i])[j] =
                        modprod(hG_.inv_root[i], ((uint32_t*)hG_.inv_psi_no[i])[j - 1], hC_.primes[i]);
                } else {
                    ((uint64_t*)hG_.psi_no[i])[j] =
                        modprod(hG_.root[i], ((uint64_t*)hG_.psi_no[i])[j - 1], hC_.primes[i]);
                    ((uint64_t*)hG_.inv_psi_no[i])[j] =
                        modprod(hG_.inv_root[i], ((uint64_t*)hG_.inv_psi_no[i])[j - 1], hC_.primes[i]);
                }
            }

            for (int j = 0; j < N; ++j) {
                int pow = 1 << (std::bit_width((uint32_t)j));

                if (!HISU64(i)) {
                    ((uint32_t*)hG_.psi[i])[j] = ((uint32_t*)hG_.psi_no[i])[bit_reverse(j, hC_.logN)];
                    ((uint32_t*)hG_.inv_psi[i])[j] = ((uint32_t*)hG_.inv_psi_no[i])[bit_reverse(j, hC_.logN)];

                    assert(modpow(((uint32_t*)hG_.psi[i])[j], 2 * pow, hC_.primes[i]) == 1);
                    if (j > 0)
                        assert(modpow(((uint32_t*)hG_.psi[i])[j], pow, hC_.primes[i]) == (hC_.primes[i] - 1));
                    assert(modpow(((uint32_t*)hG_.inv_psi[i])[j], 2 * pow, hC_.primes[i]) == 1);
                    if (j > 0)
                        assert(modpow(((uint32_t*)hG_.inv_psi[i])[j], pow, hC_.primes[i]) == (hC_.primes[i] - 1));

                } else {
                    ((uint64_t*)hG_.psi[i])[j] = ((uint64_t*)hG_.psi_no[i])[bit_reverse(j, hC_.logN)];
                    ((uint64_t*)hG_.inv_psi[i])[j] = ((uint64_t*)hG_.inv_psi_no[i])[bit_reverse(j, hC_.logN)];

                    assert(modpow(((uint64_t*)hG_.psi[i])[j], 2 * pow, hC_.primes[i]) == 1);
                    if (j > 0)
                        assert(modpow(((uint64_t*)hG_.psi[i])[j], pow, hC_.primes[i]) == (hC_.primes[i] - 1ul));
                    assert(modpow(((uint64_t*)hG_.inv_psi[i])[j], 2 * pow, hC_.primes[i]) == 1);
                    if (j > 0)
                        assert(modpow(((uint64_t*)hG_.inv_psi[i])[j], pow, hC_.primes[i]) == (hC_.primes[i] - 1ul));
                }
            }

            int auxWidth = ((hC_.logN + 1) / 2);
            for (int j = 0; j < N / (1 << auxWidth); ++j) {
                for (int k = 0; k < (1 << auxWidth); ++k) {
                    if (!HISU64(i)) {
                        ((uint32_t*)hG_.psi_middle_scale[i])[j * (1 << auxWidth) + k] =
                            ((uint32_t*)hG_.psi_no[i])[j * bit_reverse(k, auxWidth)];
                        ((uint32_t*)hG_.inv_psi_middle_scale[i])[j * (1 << auxWidth) + k] = modprod(
                            ((uint32_t*)hG_.inv_psi_no[i])[j * bit_reverse(k, auxWidth)], hC_.N_inv[i], hC_.primes[i]);
                    } else {
                        ((uint64_t*)hG_.psi_middle_scale[i])[j * (1 << auxWidth) + k] =
                            ((uint64_t*)hG_.psi_no[i])[j * bit_reverse(k, auxWidth)];
                        ((uint64_t*)hG_.inv_psi_middle_scale[i])[j * (1 << auxWidth) + k] = modprod(
                            ((uint64_t*)hG_.inv_psi_no[i])[j * bit_reverse(k, auxWidth)], hC_.N_inv[i], hC_.primes[i]);
                    }
                }
            }

            for (int j = 0; j < 2 * N; ++j) {
                if (!HISU64(i)) {
                    ((uint32_t*)hG_.inv_psi_no[i])[j] =
                        modprod(((uint32_t*)hG_.inv_psi_no[i])[j], hC_.N_inv[i], hC_.primes[i]);
                } else {
                    ((uint64_t*)hG_.inv_psi_no[i])[j] =
                        modprod(((uint64_t*)hG_.inv_psi_no[i])[j], hC_.N_inv[i], hC_.primes[i]);
                }
            }

            for (int j = 0; j < N; ++j) {
                if (!HISU64(i)) {
                    ((uint32_t*)hG_.psi_shoup[i])[j] =
                        (uint64_t)(((uint32_t*)hG_.psi[i])[j] << 1) * (1ul << 31) / hC_.primes[i];
                    ((uint32_t*)hG_.inv_psi_shoup[i])[j] =
                        (uint64_t)(((uint32_t*)hG_.inv_psi[i])[j] << 1) * (1ul << 31) / hC_.primes[i];
                } else {
                    assert(hC_.primes[i] != 0);

                    ((uint64_t*)hG_.psi_shoup[i])[j] =
                        (__uint128_t)(((uint64_t*)hG_.psi[i])[j] << 1) * (1ul << 63) / hC_.primes[i];
                    ((uint64_t*)hG_.inv_psi_shoup[i])[j] =
                        (__uint128_t)(((uint64_t*)hG_.inv_psi[i])[j] << 1) * (1ul << 63) / hC_.primes[i];
                }
            }
        });
    }

    if (std::is_same_v<CKKS::Parameters, Scheme>) {
        auto param = static_cast<CKKS::Parameters>(parameters);
        if (param.raw != nullptr) {
            for (size_t i = 0; i < q.size(); ++i) {
                for (int k = 0; k < N; ++k) {
                    assert(param.raw->psi[i][k] == ((uint64_t*)hG_.psi[i])[k]);
                    assert(param.raw->psi_inv[i][k] == ((uint64_t*)hG_.inv_psi[i])[k]);
                }
                assert(hC_.N_inv[i] == param.raw->N_inv[i]);
            }
        }
    }

    for (size_t i = 0; i < GPUid.size(); ++i) {
        cudaSetDevice(0/*GPUid.at(i)*/);
CudaCheckErrorMod;
        for (int j = 0; j < hC_.L + hC_.K; ++j) {
            int bytes = hC_.N * ((HISU64(j)) ? sizeof(uint64_t) : sizeof(uint32_t));

            cudaMalloc(&(hG_.psi_ptr[i][j]), bytes);
            cudaMalloc(&(hG_.inv_psi_ptr[i][j]), bytes);
            cudaMalloc(&(hG_.psi_no_ptr[i][j]), 2 * bytes);
            cudaMalloc(&(hG_.inv_psi_no_ptr[i][j]), 2 * bytes);
            cudaMalloc(&(hG_.psi_middle_scale_ptr[i][j]), bytes);
            cudaMalloc(&(hG_.inv_psi_middle_scale_ptr[i][j]), bytes);
            cudaMalloc(&(hG_.psi_barrett_ptr[i][j]), bytes);
            cudaMalloc(&(hG_.inv_psi_barrett_ptr[i][j]), bytes);
            CudaCheckErrorMod;
            cudaMemcpy(hG_.psi_ptr[i][j], hG_.psi[j], bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(hG_.inv_psi_ptr[i][j], hG_.inv_psi[j], bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(hG_.psi_no_ptr[i][j], hG_.psi_no[j], 2 * bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(hG_.inv_psi_no_ptr[i][j], hG_.inv_psi_no[j], 2 * bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(hG_.psi_middle_scale_ptr[i][j], hG_.psi_middle_scale[j], bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(hG_.inv_psi_middle_scale_ptr[i][j], hG_.inv_psi_middle_scale[j], bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(hG_.psi_barrett_ptr[i][j], hG_.psi_shoup[j], bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(hG_.inv_psi_barrett_ptr[i][j], hG_.inv_psi_shoup[j], bytes, cudaMemcpyHostToDevice);
            CudaCheckErrorMod;
        }

        cudaMemcpyToSymbol(Globals::psi, hG_.psi_ptr[i], sizeof(Globals::psi), 0, cudaMemcpyHostToDevice);
CudaCheckErrorMod;
        cudaMemcpyToSymbol(G_::psi_no, hG_.psi_no_ptr[i], sizeof(hG_.psi_no_ptr[i]), 0, cudaMemcpyHostToDevice);
CudaCheckErrorMod;
        cudaMemcpyToSymbol(G_::psi_middle_scale, hG_.psi_middle_scale_ptr[i], sizeof(hG_.psi_middle_scale_ptr[i]), 0,
                           cudaMemcpyHostToDevice);
CudaCheckErrorMod;
        cudaMemcpyToSymbol(G_::inv_psi, hG_.inv_psi_ptr[i], sizeof(hG_.inv_psi_ptr[i]), 0, cudaMemcpyHostToDevice);
CudaCheckErrorMod;
        cudaMemcpyToSymbol(G_::inv_psi_no, hG_.inv_psi_no_ptr[i], sizeof(hG_.inv_psi_no_ptr[i]), 0,
                           cudaMemcpyHostToDevice);
CudaCheckErrorMod;
        cudaMemcpyToSymbol(G_::inv_psi_middle_scale, hG_.inv_psi_middle_scale_ptr[i],
                           sizeof(hG_.inv_psi_middle_scale_ptr[i]), 0, cudaMemcpyHostToDevice);
CudaCheckErrorMod;
        cudaMemcpyToSymbol(G_::psi_shoup, hG_.psi_barrett_ptr[i], sizeof(hG_.psi_barrett_ptr[i]), 0,
                           cudaMemcpyHostToDevice);
CudaCheckErrorMod;
        cudaMemcpyToSymbol(G_::inv_psi_shoup, hG_.inv_psi_barrett_ptr[i], sizeof(hG_.inv_psi_barrett_ptr[i]), 0,
                           cudaMemcpyHostToDevice);
CudaCheckErrorMod;
    }

    {
        for (int i = 0; i < MAXP && hC_.primes[i] != 0; ++i) {
            for (int j = 0; j < MAXP && hC_.primes[j] != 0; ++j) {
                hG_.q_inv[i][j] = modinv(hC_.primes[i], hC_.primes[j]);
            }
        }

        for (int i = 0; i < MAXP && hC_.primes[i] != 0; ++i) {
            for (int j = 0; j < MAXP && hC_.primes[j] != 0; ++j) {
                if (hC_.primes[i] != hC_.primes[j]) {
                    assert(modprod(hG_.q_inv[i][j], hC_.primes[i] % hC_.primes[j], hC_.primes[j]) == 1);
                }
            }
        }
        constexpr int bytes = sizeof(Global::q_inv);

        for (int id : GPUid) {
            cudaSetDevice(id);
            cudaMemcpyToSymbol(Globals::q_inv, hG_.q_inv, bytes, 0, cudaMemcpyHostToDevice);
            CudaCheckErrorMod;
        }
    }for (size_t j = 0; j < smeta.size(); ++j) {
                hC_.primeid_special_partition[j] = smeta[j].id;
             //   hC_.primeid_flattened[SPECIAL(0, j)] = smeta[j].id;
            }

            for (size_t i = 0; i < GPUid.size(); ++i) {
                for (size_t j = 0; j < meta[i].size(); ++j) {
                    hC_.primeid_partition[i][j] = meta[i][j].id;
                   // hC_.primeid_flattened[PARTITION(i, j)] = meta[i][j].id;
                }

                for (size_t j = 0; j < digitGPUid[i].size(); ++j) {
                    for (int k = 0; k < hC_.L; ++k) {
                        {
                            int num = 0;
                            for (auto& l : DECOMPmeta.at(i).at(j))
                                if (l.id <= k) {
                                    hC_.primeid_digit_from[digitGPUid.at(i).at(j)][num] = l.id;
                                  //  hC_.primeid_flattened[DECOMP(digitGPUid.at(i).at(j), num)] = l.id;
                                    num++;
                                }
                            hC_.num_primeid_digit_from[digitGPUid.at(i).at(j)][k] = num;
                        }

                        {
                            int num = 0;
                            for (auto& l : DIGITmeta.at(i).at(j))
                                if (l.id <= k || l.id >= hC_.L) {
                                    hC_.primeid_digit_to[digitGPUid.at(i).at(j)][num] = l.id;
                                 //   hC_.primeid_flattened[DIGIT(digitGPUid.at(i).at(j), num)] = l.id;
                                    num++;
                                }
                            hC_.num_primeid_digit_to[digitGPUid.at(i).at(j)][k] = num;
                        }
                    }
                }
            }




    if constexpr (std::is_same_v<Scheme, CKKS::Parameters>) {
        auto param = static_cast<CKKS::Parameters>(parameters);
        if (param.raw) {
            {
                for (size_t i = 0; i < q.size(); ++i) {
                    for (size_t j = 0; j < q.size(); ++j) {
                        if (i < j) {
                            hG_.QlQlInvModqlDivqlModq[j][i] = param.raw->m_QlQlInvModqlDivqlModq[q.size() - 1 - j][i];
                        }
                    }
                }

                constexpr int bytes = sizeof(Global::QlQlInvModqlDivqlModq);

                for (int id : GPUid) {
                    cudaSetDevice(id);
                    cudaMemcpyToSymbol(G_::QlQlInvModqlDivqlModq, hG_.QlQlInvModqlDivqlModq, bytes, 0,
                                       cudaMemcpyHostToDevice);
                    CudaCheckErrorMod;
                }
            }

            ////////////////// KEY SWITCH //////////////////

            hC_.dnum = param.raw->dnum;
            {
                auto& src = param.raw->PInvModq;
                for (size_t i = 0; i < src.size(); ++i) {
                    hC_.P_inv[i] = src[i];
                    hC_.P_inv_shoup[i] = shoup_precomp(hC_.P_inv[i], i);
                    hC_.P[i] = modinv(src[i], hC_.primes[i]);
                    hC_.P_shoup[i] = shoup_precomp(hC_.P[i], i);
                }
            }

            

            {
                auto& src = param.raw->PHatInvModp;

                for (size_t k = 0; k < src.size(); ++k) {
                    hG_.ModDown_pre_scale[hC_.L + k] = src[k];
                    hG_.ModDown_pre_scale_shoup[hC_.L + k] = shoup_precomp(hG_.ModDown_pre_scale[hC_.L + k], hC_.L + k);
                }

                constexpr int bytes = sizeof(Global::ModDown_pre_scale);
                for (int id : GPUid) {
                    cudaSetDevice(id);
                    cudaMemcpyToSymbol(G_::ModDown_pre_scale, hG_.ModDown_pre_scale, bytes, 0, cudaMemcpyHostToDevice);
                    cudaMemcpyToSymbol(G_::ModDown_pre_scale_shoup, hG_.ModDown_pre_scale_shoup, bytes, 0,
                                       cudaMemcpyHostToDevice);
                    CudaCheckErrorMod;
                }
            }

            {
                auto& src = param.raw->PHatModq;

                for (size_t k = 0; k < src.size(); ++k) {
                    for (size_t i = 0; i < src[k].size(); ++i) {
                        hG_.ModDown_matrix[k][i] = src[k][i];
                        hG_.ModDown_matrix_shoup[k][i] = shoup_precomp(hG_.ModDown_matrix[k][i], i);
                    }
                }

                constexpr int bytes = sizeof(Global::ModDown_matrix);
                for (int id : GPUid) {
                    cudaSetDevice(id);
                    cudaMemcpyToSymbol(G_::ModDown_matrix, hG_.ModDown_matrix, bytes, 0, cudaMemcpyHostToDevice);
                    cudaMemcpyToSymbol(G_::ModDown_matrix_shoup, hG_.ModDown_matrix_shoup, bytes, 0,
                                       cudaMemcpyHostToDevice);
                    CudaCheckErrorMod;
                }
            }

            {
                auto& src = param.raw->PartQlHatInvModq;
                int init_primeid = 0;
                for (size_t k = 0; k < src.size(); ++k) {
                    for (size_t i = 0; i < src[k].size(); ++i) {
                        for (size_t j = 0; j < src[k][i].size(); ++j) {
                            assert(src[k][i][j] != 0);
                            hG_.DecompAndModUp_pre_scale[k][i][init_primeid + j] = src[k][i][j];
                            hG_.DecompAndModUp_pre_scale_shoup[k][i][init_primeid + j] =
                                shoup_precomp(hG_.DecompAndModUp_pre_scale[k][i][init_primeid + j], init_primeid + j);
                        }
                    }
                    init_primeid += src[k].size();
                }

                constexpr int bytes = sizeof(Global::DecompAndModUp_pre_scale);
                assert(bytes == 8 * 64 * 64 * 8);
                for (int id : GPUid) {
                    cudaSetDevice(id);
                    cudaMemcpyToSymbol(G_::DecompAndModUp_pre_scale, hG_.DecompAndModUp_pre_scale, bytes, 0,
                                       cudaMemcpyHostToDevice);
                    cudaMemcpyToSymbol(G_::DecompAndModUp_pre_scale_shoup, hG_.DecompAndModUp_pre_scale_shoup, bytes, 0,
                                       cudaMemcpyHostToDevice);
                    CudaCheckErrorMod;
                }
            }

            {

                auto& src = param.raw->PartQlHatModp;

                for (size_t k = 0; k < src.size(); ++k) {
                    for (size_t i = 0; i < src[k].size(); ++i) {
                        size_t gpu = 0;
                        size_t gpu_d = 0;
                        for (; gpu < digitGPUid.size(); ++gpu) {
                            for (size_t j = 0; j < digitGPUid.at(gpu).size(); ++j) {
                                if (digitGPUid.at(gpu).at(j) == (int)i) {
                                    gpu_d = j;
                                    goto out;
                                }
                            }
                        }
                    out:
                        for (size_t j = 0; j < src[k][i].size(); ++j) {
                            for (size_t l = 0; l < src[k][i][j].size(); ++l) {
                                assert(src[k][i][j][l] != 0);
                                int DIGITmeta_primeid =
                                    l >= src[k][i][j].size() - hC_.K
                                        ? DIGITmeta.at(gpu).at(gpu_d).at(l - src[k][i][j].size() + hC_.K).id
                                        : DIGITmeta.at(gpu).at(gpu_d).at(l + hC_.K).id;
                                hG_.DecompAndModUp_matrix[k][i][j][DIGITmeta_primeid] = src[k][i][j][l];
                                hG_.DecompAndModUp_matrix_shoup[k][i][j][DIGITmeta_primeid] =
                                    shoup_precomp(src[k][i][j][l], DIGITmeta_primeid);
                            }
                        }
                    }
                }

                constexpr int bytes = sizeof(Global::DecompAndModUp_matrix);
                assert(bytes == 8 * 64 * 64 * 64 * 8);
                for (int id : GPUid) {
                    cudaSetDevice(id);
                    cudaMemcpyToSymbol(G_::DecompAndModUp_matrix, hG_.DecompAndModUp_matrix, bytes, 0,
                                       cudaMemcpyHostToDevice);
                    cudaMemcpyToSymbol(G_::DecompAndModUp_matrix_shoup, hG_.DecompAndModUp_matrix_shoup, bytes, 0,
                                       cudaMemcpyHostToDevice);
                    CudaCheckErrorMod;

                    cudaMemcpyFromSymbol(hG_.DecompAndModUp_matrix, G_::DecompAndModUp_matrix, bytes, 0,
                                         cudaMemcpyDeviceToHost);
                    cudaMemcpyFromSymbol(hG_.DecompAndModUp_matrix_shoup, G_::DecompAndModUp_matrix_shoup, bytes, 0,
                                         cudaMemcpyDeviceToHost);
                }
            }
        }
    }

    CudaCheckErrorMod;
    for (int id : GPUid) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(constants, &host_constants, sizeof(Constants), 0, cudaMemcpyHostToDevice);
        CudaCheckErrorMod;
    }
}

template void SetupConstants<CKKS::Parameters>(const std::vector<PrimeRecord>& q,
                                               const std::vector<std::vector<LimbRecord>>& meta,
                                               const std::vector<PrimeRecord>& p, const std::vector<LimbRecord>& smeta,
                                               const std::vector<std::vector<std::vector<LimbRecord>>>& DECOMPmeta,
                                               const std::vector<std::vector<std::vector<LimbRecord>>>& DIGITmeta,
                                               const std::vector<std::vector<int>>& digitGPUid,
                                               const std::vector<int>& GPUid, const int N,
                                               const CKKS::Parameters& parameters);

}  // namespace FIDESlib
