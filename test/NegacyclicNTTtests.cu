//
// Created by carlosad on 12/09/24.
//
#include <cassert>
#include <execution>
#include <iomanip>

#include <gtest/gtest.h>

#include "CKKS/Context.cuh"
#include "CKKS/Limb.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"
#include "ConstantsGPU.cuh"
#include "Math.cuh"
#include "NTT.cuh"
#include "ParametrizedTest.cuh"
#include "cpuNTT.hpp"
#include "cpuNTT_nega.hpp"

namespace FIDESlib::Testing {
class NegacyclicNTTTest : public FIDESlibParametrizedTest {};

TEST_P(NegacyclicNTTTest, TestCpuNTT) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters custom = fideslibParams;
    custom.logN = 3;
    custom.primes[0].p = 17;
    custom.L = 0;
    custom.primes[0].type = FIDESlib::TYPE::U64;
    FIDESlib::CKKS::Context cc{custom, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);

        std::vector<uint64_t> v1{1, 2, 3, 7, 5, 4, 3, 9};
        std::vector<uint64_t> v2{1, 2, 3, 7, 5, 4, 1, 2};

        int N = 8;

        int primeid = 0;

        std::vector<uint64_t> res_cpu(v2);

        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, false, primeid, 1000);

        std::vector<uint64_t> expected{8, 11, 14, 2, 12, 16, 7, 6};
        // ASSERT_EQ(res_cpu, expected);

        nega_fft2_forPrime(res_cpu, true, primeid, 1000);

        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, true, primeid, 1000);

        nega_fft2_forPrime(res_cpu, false, primeid, 1000);

        ASSERT_EQ(res_cpu, v2);

        //nega_fft_forPrime(res_cpu, false, 0, 1000);

        //std::sort(res_gpu.begin(), res_gpu.end());
        //std::sort(res_cpu.begin(), res_cpu.end());

        std::cout << "Polynomial multiplication\n";

        std::vector<uint64_t> res(8, primeid);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i + j >= N) {
                    res[i + j - N] += 17 - (v1[i] * v2[j]) % 17;
                    res[i + j - N] %= 17;
                } else {
                    res[i + j] += (v1[i] * v2[j]) % 17;
                    res[i + j] %= 17;
                }
            }
        }

        std::vector<uint64_t> res2(16, primeid);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                {
                    res2[i + j] += (v1[i] * v2[j]) % 17;
                    res2[i + j] %= 17;
                }
            }
        }

        std::vector<uint64_t> res3(8, primeid);

        for (int i = 0; i < N; ++i) {
            res3[i] = (res2[i] + 17 - res2[i + N]) % 17;
        }

        ASSERT_EQ(res, res3);

        std::vector<uint64_t> res_fft(8, primeid);

        nega_fft2_forPrime(v1, false, primeid, 1000);
        nega_fft2_forPrime(v2, false, primeid, 1000);

        for (int i = 0; i < N; ++i) {
            res_fft[i] = (v1[i] * v2[i]) % 17;
        }

        nega_fft2_forPrime(res_fft, true, primeid, 1000);

        ASSERT_EQ(res, res_fft);
    }
    CudaCheckErrorMod;
}

TEST_P(NegacyclicNTTTest, TestCpuNTT_adapt_cyclic) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters custom = fideslibParams;
    custom.logN = 3;
    custom.primes[0].p = 17;
    custom.L = 0;
    custom.primes[0].type = FIDESlib::TYPE::U64;
    FIDESlib::CKKS::Context cc{custom, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);

        std::vector<uint64_t> v1{1, 2, 3, 7, 5, 4, 3, 9};
        std::vector<uint64_t> v2{1, 2, 3, 7, 5, 4, 1, 2};

        int N = 8;

        int primeid = 0;

        std::vector<uint64_t> res_cpu(v2);

        ASSERT_EQ(res_cpu, v2);

        for (int i = 0; i < N; ++i) {
            res_cpu[i] = FIDESlib::modprod(res_cpu[i], FIDESlib::modpow(hG_.root[primeid], i, hC_.primes[primeid]),
                                           hC_.primes[primeid]);
        }
        fft_forPrime(res_cpu, false, 0, 1000);

        std::vector<uint64_t> expected{8, 11, 14, 2, 12, 16, 7, 6};
        // ASSERT_EQ(res_cpu, expected);

        std::vector<uint64_t> res_cpu2(res_cpu);

        fft_forPrime(res_cpu2, true, 0, 1000);
        for (int i = 0; i < N; ++i) {
            res_cpu2[i] = FIDESlib::modprod(
                res_cpu2[i], FIDESlib::modpow(hG_.inv_root[primeid], i, hC_.primes[primeid]), hC_.primes[primeid]);
        }

        nega_fft2_forPrime(res_cpu, true, 0, 1000);

        ASSERT_EQ(res_cpu2, v2);
        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, true, 0, 1000);

        nega_fft_forPrime(res_cpu, false, 0, 1000);

        ASSERT_EQ(res_cpu, v2);

        //nega_fft_forPrime(res_cpu, false, 0, 1000);

        //std::sort(res_gpu.begin(), res_gpu.end());
        //std::sort(res_cpu.begin(), res_cpu.end());

        std::cout << "Polynomial multiplication\n";

        std::vector<uint64_t> res(8, 0);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i + j >= N) {
                    res[i + j - N] += 17 - (v1[i] * v2[j]) % 17;
                    res[i + j - N] %= 17;
                } else {
                    res[i + j] += (v1[i] * v2[j]) % 17;
                    res[i + j] %= 17;
                }
            }
        }

        std::vector<uint64_t> res2(16, 0);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                {
                    res2[i + j] += (v1[i] * v2[j]) % 17;
                    res2[i + j] %= 17;
                }
            }
        }

        std::vector<uint64_t> res3(8, 0);

        for (int i = 0; i < N; ++i) {
            res3[i] = (res2[i] + 17 - res2[i + N]) % 17;
        }

        ASSERT_EQ(res, res3);

        std::vector<uint64_t> res_fft(8, 0);

        nega_fft_forPrime(v1, false, 0, 1000);
        nega_fft_forPrime(v2, false, 0, 1000);

        for (int i = 0; i < N; ++i) {
            res_fft[i] = (v1[i] * v2[i]) % 17;
        }

        nega_fft2_forPrime(res_fft, true, 0, 1000);

        ASSERT_EQ(res, res_fft);
    }
    CudaCheckErrorMod;
}

TEST_P(NegacyclicNTTTest, TestCpuNTT2) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters custom = fideslibParams;
    custom.logN = 4;
    custom.primes[0].p = 97;
    custom.primes[0].type = FIDESlib::U64;
    custom.L = 0;
    FIDESlib::CKKS::Context cc{custom, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);

        std::vector<uint64_t> v1{1, 2, 3, 7, 5, 4, 3, 9, 1, 2, 3, 7, 5, 4, 3, 9};
        std::vector<uint64_t> v2{1, 2, 3, 7, 5, 4, 1, 2, 1, 2, 3, 7, 5, 4, 3, 9};

        int N = 1 << custom.logN;

        int primeid = 0;

        std::vector<uint64_t> res_cpu(v2);

        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, false, primeid, 1000);

        nega_fft2_forPrime(res_cpu, true, primeid, 1000);

        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, true, primeid, 1000);

        nega_fft2_forPrime(res_cpu, false, primeid, 1000);

        ASSERT_EQ(res_cpu, v2);

        //nega_fft_forPrime(res_cpu, false, 0, 1000);

        //std::sort(res_gpu.begin(), res_gpu.end());
        //std::sort(res_cpu.begin(), res_cpu.end());

        std::cout << "Polynomial multiplication\n";

        std::vector<uint64_t> res(N, 0);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i + j >= N) {
                    res[i + j - N] += cc.prime[primeid].p - ((__uint128_t)v1[i] * v2[j]) % cc.prime[primeid].p;
                    res[i + j - N] %= cc.prime[primeid].p;
                } else {
                    res[i + j] += ((__uint128_t)v1[i] * v2[j]) % cc.prime[primeid].p;
                    res[i + j] %= cc.prime[primeid].p;
                }
            }
        }

        std::vector<uint64_t> res2(2 * N, 0);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                {
                    res2[i + j] += ((__uint128_t)v1[i] * v2[j]) % cc.prime[primeid].p;
                    res2[i + j] %= cc.prime[primeid].p;
                }
            }
        }

        std::vector<uint64_t> res3(N, 0);

        for (int i = 0; i < N; ++i) {
            res3[i] = (res2[i] + cc.prime[primeid].p - res2[i + N]) % cc.prime[primeid].p;
        }

        ASSERT_EQ(res, res3);

        std::vector<uint64_t> res_fft(N, 0);

        nega_fft2_forPrime(v1, false, primeid, 1000);
        nega_fft2_forPrime(v2, false, primeid, 1000);

        for (int i = 0; i < N; ++i) {
            res_fft[i] = ((__uint128_t)v1[i] * v2[i]) % cc.prime[primeid].p;
        }

        nega_fft2_forPrime(res_fft, true, primeid, 1000);

        ASSERT_EQ(res, res_fft);
    }
    CudaCheckErrorMod;
}

TEST_P(NegacyclicNTTTest, TestCpuNTTBigMod) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters custom = fideslibParams;
    custom.logN = 3;
    custom.primes[0].p = params32_15.primes[0].p;
    custom.primes[0].type = FIDESlib::U64;
    custom.L = 0;
    FIDESlib::CKKS::Context cc{custom, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);

        std::vector<uint64_t> v1{1, 2, 3, 7, 5, 4, 3, 9};
        std::vector<uint64_t> v2{1, 2, 3, 7, 5, 4, 1, 2};

        int N = 1 << custom.logN;

        int primeid = 0;

        std::vector<uint64_t> res_cpu(v2);

        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, false, primeid, 1000);

        std::vector<uint64_t> expected{8, 11, 14, 2, 12, 16, 7, 6};
        // ASSERT_EQ(res_cpu, expected);

        nega_fft2_forPrime(res_cpu, true, primeid, 1000);

        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, true, primeid, 1000);

        nega_fft2_forPrime(res_cpu, false, primeid, 1000);

        ASSERT_EQ(res_cpu, v2);

        //nega_fft_forPrime(res_cpu, false, 0, 1000);

        //std::sort(res_gpu.begin(), res_gpu.end());
        //std::sort(res_cpu.begin(), res_cpu.end());

        std::cout << "Polynomial multiplication\n";

        std::vector<uint64_t> res(N, 0);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i + j >= N) {
                    res[i + j - N] += cc.prime[primeid].p - ((__uint128_t)v1[i] * v2[j]) % cc.prime[primeid].p;
                    res[i + j - N] %= cc.prime[primeid].p;
                } else {
                    res[i + j] += ((__uint128_t)v1[i] * v2[j]) % cc.prime[primeid].p;
                    res[i + j] %= cc.prime[primeid].p;
                }
            }
        }

        std::vector<uint64_t> res2(2 * N, 0);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                {
                    res2[i + j] += ((__uint128_t)v1[i] * v2[j]) % cc.prime[primeid].p;
                    res2[i + j] %= cc.prime[primeid].p;
                }
            }
        }

        std::vector<uint64_t> res3(N, 0);

        for (int i = 0; i < N; ++i) {
            res3[i] = (res2[i] + cc.prime[primeid].p - res2[i + N]) % cc.prime[primeid].p;
        }

        ASSERT_EQ(res, res3);

        std::vector<uint64_t> res_fft(N, 0);

        nega_fft2_forPrime(v1, false, primeid, 1000);
        nega_fft2_forPrime(v2, false, primeid, 1000);

        for (int i = 0; i < N; ++i) {
            res_fft[i] = ((__uint128_t)v1[i] * v2[i]) % cc.prime[primeid].p;
        }

        nega_fft2_forPrime(res_fft, true, primeid, 1000);

        ASSERT_EQ(res, res_fft);
    }
    CudaCheckErrorMod;
}

TEST_P(NegacyclicNTTTest, TestCpuNTTindependentOfN) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters custom = fideslibParams;
    custom.logN = 16;
    custom.L = 0;
    FIDESlib::CKKS::Context cc{custom, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        std::vector<uint64_t> v1(cc.N, 10);
        std::vector<uint64_t> v2(cc.N, 10);

        int N = v1.size();

        int primeid = 0;

        std::vector<uint64_t> res_cpu(v2);

        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, false, primeid, 1000);

        nega_fft2_forPrime(res_cpu, true, primeid, 1000);

        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, true, primeid, 1000);

        nega_fft2_forPrime(res_cpu, false, primeid, 1000);

        ASSERT_EQ(res_cpu, v2);

        //nega_fft_forPrime(res_cpu, false, 0, 1000);

        //std::sort(res_gpu.begin(), res_gpu.end());
        //std::sort(res_cpu.begin(), res_cpu.end());

        ///////////////////////////////////////////////////////
        /*
                std::cout << "Polynomial multiplication\n";


                std::vector<uint64_t> res(N, 0);

                for(int i = 0; i < N; ++i){
                    for(int j = 0; j < N; ++j){
                        if(i + j >= N){
                            res[i + j - N] += cc.prime[primeid].p - ((__uint128_t)v1[i] * v2[j]) % cc.prime[primeid].p;
                            res[i + j - N] %= cc.prime[primeid].p;
                        } else {
                            res[i + j] += ((__uint128_t) v1[i] * v2[j]) % cc.prime[primeid].p;
                            res[i + j] %= cc.prime[primeid].p;
                        }
                    }
                }

                std::vector<uint64_t> res2(2*N, 0);

                for(int i = 0; i < N; ++i){
                    for(int j = 0; j < N; ++j){
                        {
                            res2[i + j] += ((__uint128_t)v1[i] * v2[j]) % cc.prime[primeid].p;
                            res2[i + j] %= cc.prime[primeid].p;
                        }
                    }
                }

                std::vector<uint64_t> res3(N, 0);

                for(int i = 0; i < N; ++i){
                    res3[i] = (res2[i] + cc.prime[primeid].p - res2[i + N]) % cc.prime[primeid].p;
                }

                ASSERT_EQ(res, res3);

                std::vector<uint64_t> res_fft(N, 0);

                nega_fft2_forPrime(v1, false, primeid, 1000);
                nega_fft2_forPrime(v2, false, primeid, 1000);

                for(int i = 0; i < N; ++i){
                    res_fft[i] = ((__uint128_t) v1[i] * v2[i]) % cc.prime[primeid].p;
                }

                nega_fft2_forPrime(res_fft, true, primeid, 1000);

                ASSERT_EQ(res, res_fft);

        */
        ////////////////////////////////////////
        ASSERT_EQ(res_cpu, v2);
        std::cout << "Compare with GPU\n";
        FIDESlib::CKKS::Limb<uint64_t> limb(cc, GPUs[0], s, primeid);
        limb.load(v2);
        cudaDeviceSynchronize();
        limb.NTT();

        std::vector<uint64_t> res_gpu(v2);
        limb.store(res_gpu);
        cudaDeviceSynchronize();

        nega_fft2_forPrime(res_cpu, false, primeid, 1000);

        FIDESlib::bit_reverse_vector(res_cpu);
        ASSERT_EQ(res_cpu, res_gpu);
        FIDESlib::bit_reverse_vector(res_cpu);
        limb.INTT();
        limb.store(res_gpu);
        cudaDeviceSynchronize();
        nega_fft2_forPrime(res_cpu, true, primeid, 1000);
        ASSERT_EQ(res_cpu, v2);
        ASSERT_EQ(res_cpu, res_gpu);
    }
    CudaCheckErrorMod;
}

INSTANTIATE_TEST_SUITE_P(NegacyclicNTTTests, NegacyclicNTTTest,
                         testing::Values(params64_13, params64_14, params64_15, params64_16));
}  // namespace FIDESlib::Testing