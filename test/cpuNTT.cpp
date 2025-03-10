//
// Created by carlosad on 4/05/24.
//

#include <cinttypes>
#include <iostream>

#include "ConstantsGPU.cuh"
#include "Math.cuh"
#include "cpuNTT.hpp"

void FIDESlib::Testing::fft(std::vector<uint64_t>& a, bool invert, uint64_t root, uint64_t root_1, uint64_t mod,
                            int its) {
    int n = a.size();

    int lg_n = 0;
    while ((1 << lg_n) < n)
        lg_n++;

    for (int i = 0; i < n; i++) {
        if (i < bit_reverse(i, lg_n))
            std::swap(a[i], a[bit_reverse(i, lg_n)]);
    }

    int count_its = 0;
    for (uint64_t len = 2; len <= (uint64_t)n && count_its < its; len <<= 1, ++count_its) {
        uint64_t wlen = invert ? root_1 : root;
        for (uint64_t i = len; i < (uint64_t)n; i <<= 1)
            wlen = ((__uint128_t)1 * wlen * wlen % mod);

        for (uint64_t i = 0; i < (uint64_t)n; i += len) {
            uint64_t w = 1;
            for (uint64_t j = 0; j < len / 2; j++) {
                uint64_t u = a[i + j], v = ((__uint128_t)1 * a[i + j + len / 2] * w % mod);
                a[i + j] = u + v < mod ? u + v : u + v - mod;
                a[i + j] %= mod;
                a[i + j + len / 2] = u - v < u ? u - v : u - v + mod;
                a[i + j + len / 2] %= mod;
                w = ((__uint128_t)1 * w * wlen % mod);
            }
        }
    }

    if (invert) {
        uint64_t n_1 = FIDESlib::modinv(n, mod);
        for (uint64_t& x : a)
            x = ((__uint128_t)x * n_1 % mod);
    }
}

void FIDESlib::Testing::fft_forPrime(std::vector<uint64_t>& a, bool invert, int primeid, int its) {
    std::cout << "N: " << a.size() << (invert ? "INTT" : "NTT")
              << " w: " << ((uint64_t*)hG_.psi[primeid])[a.size() >> 2]
              << " w^-1: " << ((uint64_t*)hG_.inv_psi[primeid])[a.size() >> 2] << " p: " << hC_.primes[primeid]
              << std::endl;
    fft(a, invert, /*hG_.root[primeid]*/ ((uint64_t*)hG_.psi[primeid])[a.size() >> 2],
        /*hG_.inv_root[primeid]*/ ((uint64_t*)hG_.inv_psi[primeid])[a.size() >> 2], hC_.primes[primeid], its);
}

#define M (4)

#define OFFSET_T(i) ((blockDim * 2 * M) * blockIdx + 2 * blockDim * (i) + 2 * tid)

#define OFFSET_2T(i) ((blockDim * M) * blockIdx + blockDim * (i) + tid)

void CT_butterfly(uint64_t& a, uint64_t& b, uint64_t psiaux, int primeid) {
    uint64_t c = FIDESlib::modprod(b, psiaux, FIDESlib::host_constants.primes[primeid]);
    uint64_t d = a;
    a = FIDESlib::modadd(c, d, FIDESlib::host_constants.primes[primeid]);
    b = FIDESlib::modsub(c, d, FIDESlib::host_constants.primes[primeid]);
}

template <typename T>
void FIDESlib::Testing::fft_2d(std::vector<T>& a, int sqrtN, int primeid) {
    T* dat = a.data();
    std::vector<T> res_aux(a.size());
    T* res = res_aux.data();
    const int N = a.size();
    int blockDim = sqrtN / 2;
    int gridDim = N / sqrtN / M;
    std::vector<T> psi_(sqrtN / 2);
    std::vector<T> A[M] = {std::vector<T>(sqrtN), std::vector<T>(sqrtN), std::vector<T>(sqrtN), std::vector<T>(sqrtN)};
    T* psi_dat = ((T*)hG_.psi[primeid]);
    T* full_psi = ((T*)hG_.psi_middle_scale[primeid]);

    for (int second = 0; second <= 1; ++second) {
        for (int blockIdx = 0; blockIdx < gridDim; ++blockIdx) {
            for (int tid = 0; tid < blockDim; ++tid) {
                const int j = tid << 1;

                psi_[tid] = psi_dat[tid];

                if constexpr (sizeof(T) == 8) {
                    const int col_init = j & ~2;

                    for (int i = 0; i < 4; ++i) {
                        T aux[2];
                        const int pos_transp = M * gridDim * (col_init + i) + M * blockIdx + (j & 2);
                        const int pos_res = (col_init + i);
                        //((int4 *) aux)[0] = ((int4 *) dat)[pos_transp >> 1];
                        aux[0] = dat[pos_transp];
                        aux[1] = dat[pos_transp + 1];
                        A[j & 2][pos_res] = aux[0];
                        A[(j & 2) + 1][pos_res] = aux[1];
                    }
                } else {
                    //  *(int2 *) aux = ((int2 *) dat)[gridDim.x * (j) | blockIdx.x];
                    //  ((int2 *) aux)[1] = ((int2 *) dat)[gridDim.x * (j + 1) | blockIdx.x];
                }
            }
        }
        int m = blockDim;
        int maskPsi = m;
        //printf("m: %d\n", m);

        // Iteración 0 optimizada.`
        for (int blockIdx = 0; blockIdx < gridDim; ++blockIdx) {
            for (int tid = 0; tid < blockDim; ++tid) {
                for (int i = 0; i < M; ++i) {
                    T aux[2];
                    aux[0] = A[i][tid];
                    aux[1] = A[i][tid + m];
                    A[i][tid] = modadd(aux[0], aux[1], FIDESlib::host_constants.primes[primeid]);
                    A[i][tid + m] = modsub(aux[0], aux[1], FIDESlib::host_constants.primes[primeid]);
                }
            }
        }

        m >>= 1;
        maskPsi |= (maskPsi >> 1);
        int log_psi = std::bit_width((uint32_t)blockDim) - 2;  // Ojo al logaritmo.
        //if(blockIdx.x == 0 && threadIdx.x == 0) printf("log_psi: %d", log_psi);

        for (; m >= 1; m >>= 1, log_psi--, maskPsi |= (maskPsi >> 1)) {
            for (int blockIdx = 0; blockIdx < gridDim; ++blockIdx) {
                for (int tid = 0; tid < blockDim; ++tid) {
                    //if (m >= WARP_SIZE)
                    //    __syncthreads();
                    const int mask = m - 1;
                    const int j1 = (mask & tid) | ((~mask & tid) << 1);
                    const int j2 = j1 + m;
                    const int psiid = (tid & maskPsi) >> log_psi;
                    const T& psiaux = psi_[psiid];

                    for (int i = 0; i < M; ++i) {
                        T& aux1 = A[i][j1];
                        T& aux2 = A[i][j2];
                        //  if(i == 0 && blockIdx.x == 0)printf("m: %d, j1: %d, j2: %d, psi: %lu, psi_id: %d, a1: %d, a2: %d\n", m, j1, j2, psiaux, psiid, (int) aux1, (int) aux2);
                        CT_butterfly(aux1, aux2, psiaux, primeid);
                        //  if(i == 0 && blockIdx.x == 0)printf("m: %d, j1: %d, j2: %d, psi: %lu, psi_id: %d, a1: %d, a2: %d\n", m, j1, j2, psiaux, psiid, (int) aux1, (int) aux2);
                    }
                }
            }
        }

        // Idea: calcular full_psi en función de ambos arrays psi
        for (int blockIdx = 0; blockIdx < gridDim; ++blockIdx) {
            for (int tid = 0; tid < blockDim; ++tid) {
                const int j = tid << 1;

                if (!second) {
                    for (int i = 0; i < M; ++i) {
                        T aux[2];

                        aux[0] = modprod(A[i][j], full_psi[OFFSET_T(i)], FIDESlib::host_constants.primes[primeid]);
                        aux[1] =
                            modprod(A[i][j + 1], full_psi[OFFSET_T(i) + 1], FIDESlib::host_constants.primes[primeid]);
                        if constexpr (sizeof(T) == 8) {
                            res[OFFSET_T(i)] = aux[0];
                            res[OFFSET_T(i) + 1] = aux[1];
                        } else {
                        }
                    }
                } else {
                    for (int i = 0; i < M; ++i) {
                        if constexpr (sizeof(T) == 8) {
                            res[OFFSET_T(i)] = A[i][j];
                            res[OFFSET_T(i) + 1] = A[i][j + 1];
                        }
                    }
                }
            }
        }

        blockDim = N / sqrtN / 2;
        gridDim = N / 2 / blockDim / M;
        dat = res_aux.data();
        res = a.data();
    }
}

template void FIDESlib::Testing::fft_2d<uint64_t>(std::vector<uint64_t>& a, int sqrtN, int primeid);
