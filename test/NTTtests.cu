//
// Created by carlosad on 25/03/24.
//
#include <iomanip>
#include "CKKS/Context.cuh"
#include "CKKS/Limb.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"
#include "ConstantsGPU.cuh"
#include "CudaUtils.cuh"
#include "Math.cuh"
#include "NTT.cuh"
#include "ParametrizedTest.cuh"
#include "cpuNTT.hpp"
#include "cpuNTT_nega.hpp"
#include "gtest/gtest.h"

namespace FIDESlib::Testing {
class NTTTest : public FIDESlibParametrizedTest {};

class NTTTest32 : public FIDESlibParametrizedTest {};

class FailNTTTest : public FIDESlibParametrizedTest {};

class FailNTTTest32 : public FIDESlibParametrizedTest {};

TEST_P(NTTTest, TestCpuNTT) {
    CudaCheckErrorMod;
    std::vector<uint64_t> v2{1, 2, 3, 7, 5, 4, 1, 2};

    std::vector<uint64_t> res_cpu(v2);

    ASSERT_EQ(res_cpu, v2);

    fft(res_cpu, false, 9, 2, 17);
    std::vector<uint64_t> expected{8, 11, 14, 2, 12, 16, 7, 6};

    ASSERT_EQ(res_cpu, expected);

    fft(res_cpu, true, 9, 2, 17);

    ASSERT_EQ(res_cpu, v2);
}

TEST_P(FailNTTTest, TestConstantsSmall) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters custom = fideslibParams;
    custom.logN = 3;
    custom.primes[0].p = 17;
    FIDESlib::CKKS::Context cc{custom, GPUs};

    CudaCheckErrorMod;

    for (int i = 0; i < 4; ++i) {
        int pow = 1 << (std::bit_width((uint32_t)i));
        if (pow > 1) {
            //   std::cout << ((uint64_t *) hpsi[0])[i] << std::endl;
            ASSERT_EQ(FIDESlib::modpow(((uint64_t*)hG_.psi[0])[i], pow, 17), 16);
        }
        ASSERT_TRUE(FIDESlib::modpow(((uint64_t*)hG_.psi[0])[i], 2 * pow, 17) == 1);
    }

    for (int i = 0; i < 4; ++i) {
        int pow = 1 << (std::bit_width((uint32_t)i));
        if (pow > 1) {
            //   std::cout << ((uint64_t *) hpsi[0])[i] << std::endl;
            ASSERT_EQ(FIDESlib::modpow(((uint64_t*)hG_.inv_psi[0])[i], pow, 17), 16);
        }
        ASSERT_TRUE(FIDESlib::modpow(((uint64_t*)hG_.inv_psi[0])[i], 2 * pow, 17) == 1);
    }

    for (int i = 0; i < 2 * cc.N; ++i) {
        int pow = cc.N;
        int aux_i = i;
        while ((aux_i & 1) == 0 && aux_i != 0) {
            pow >>= 1;
            aux_i >>= 1;
        }

        if (i > 0) {
            //std::cout << ((uint64_t *) hpsi_no[0])[i]<< " " << i << " " << pow << std::endl;
            ASSERT_EQ(FIDESlib::modpow(((uint64_t*)hG_.psi_no[0])[i], pow, 17), 16);
        }
        ASSERT_TRUE(FIDESlib::modpow(((uint64_t*)hG_.psi_no[0])[i], 2 * pow, 17) == 1);
    }

    for (int i = 0; i < 2 * cc.N; ++i) {
        int pow = cc.N;
        int aux_i = i;
        while ((aux_i & 1) == 0 && aux_i != 0) {
            pow >>= 1;
            aux_i >>= 1;
        }
        if (i > 0) {
            //   std::cout << ((uint64_t *) hpsi[0])[i] << std::endl;
            ASSERT_EQ(FIDESlib::modpow(((uint64_t*)hG_.inv_psi_no[0])[i], pow, 17), 16);
        }
        ASSERT_TRUE(FIDESlib::modpow(((uint64_t*)hG_.inv_psi_no[0])[i], 2 * pow, 17) == 1);
    }
}

TEST_P(NTTTest, TestConstants) {

    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    if (fideslibParams.logN == 17)
        return;
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    for (int j = 0; j <= cc.L; ++j) {
        ASSERT_EQ(1, hC_.primes[j] % (2 * cc.N));
        for (int i = 0; i < cc.N; ++i) {
            int pow = 1 << (std::bit_width((uint32_t)i));
            if (pow > 1) {
                //   std::cout << ((uint64_t *) hpsi[0])[i] << std::endl;
                //assert(modpow(((uint64_t *) hG_.psi[j])[i], pow, hC_.primes[j]) == (hC_.primes[j] - 1ul));
                ASSERT_EQ(FIDESlib::modpow(((uint64_t*)hG_.psi[j])[i], pow, hC_.primes[j]), (hC_.primes[j] - 1ul));
            }
            assert(FIDESlib::modpow(((uint64_t*)hG_.psi[j])[i], 2 * pow, hC_.primes[j]) == 1);
            ASSERT_EQ(FIDESlib::modpow(((uint64_t*)hG_.psi[j])[i], 2 * pow, hC_.primes[j]), 1);
        }
        std::cout << std::endl;

        for (int i = 0; i < cc.N; ++i) {
            int pow = cc.N;
            int aux_i = i;
            while ((aux_i & 1) == 0 && aux_i != 0) {
                pow >>= 1;
                aux_i >>= 1;
            }
            if (i > 0) {
                //   std::cout << ((uint64_t *) hpsi[0])[i] << std::endl;
                ASSERT_NE(FIDESlib::modpow(((uint64_t*)hG_.psi_no[j])[i], pow, hC_.primes[j]), 1);
            }
            assert(FIDESlib::modpow(((uint64_t*)hG_.psi_no[j])[i], 2 * pow, hC_.primes[j]) == 1);
            ASSERT_TRUE(FIDESlib::modpow(((uint64_t*)hG_.psi_no[j])[i], 2 * pow, hC_.primes[j]) == 1);
        }
    }

    for (int j = 0; j <= cc.L; ++j) {
        for (int i = 0; i < cc.N; ++i) {
            int pow = 1 << (std::bit_width((uint64_t)i));
            if (pow > 1) {
                //   std::cout << ((uint64_t *) hpsi[0])[i] << std::endl;
                ASSERT_EQ(FIDESlib::modpow(((uint64_t*)hG_.inv_psi[j])[i], pow, hC_.primes[j]), (hC_.primes[j] - 1ul));
            }
            ASSERT_TRUE(FIDESlib::modpow(((uint64_t*)hG_.inv_psi[j])[i], 2 * pow, hC_.primes[j]) == 1);
        }

        for (int i = 0; i < 2 * cc.N; ++i) {
            int pow = cc.N;
            int aux_i = i;
            while ((aux_i & 1) == 0 && aux_i != 0) {
                pow >>= 1;
                aux_i >>= 1;
            }
            if (i > 0) {
                //   std::cout << ((uint64_t *) hpsi[0])[i] << std::endl;
                ASSERT_NE(FIDESlib::modpow(((uint64_t*)hG_.inv_psi_no[j])[i], pow, hC_.primes[j]), 1);
            }
            ASSERT_EQ(FIDESlib::modpow(FIDESlib::modprod(((uint64_t*)hG_.inv_psi_no[j])[i], hC_.N, hC_.primes[j]),
                                       2 * pow, hC_.primes[j]),
                      1ul);
        }
    }

    for (int j = 0; j <= cc.L; ++j) {
        ASSERT_TRUE(FIDESlib::modprod(hG_.root[j], hG_.inv_root[j], hC_.primes[j]) == 1);
        for (int i = 0; i < 2 * cc.N; ++i) {
            //if(modprod(((uint64_t *) hG_.inv_psi_no[j])[i], ((uint64_t *) hG_.psi_no[j])[i] , hC_.primes[j]) != 1) std::cout << i << std::endl;
            ASSERT_TRUE(FIDESlib::modprod(FIDESlib::modprod(((uint64_t*)hG_.inv_psi_no[j])[i], hC_.N, hC_.primes[j]),
                                          ((uint64_t*)hG_.psi_no[j])[i], hC_.primes[j]) == 1);
        }
        for (int i = 0; i < cc.N / 2; ++i) {
            ASSERT_TRUE(FIDESlib::modprod(((uint64_t*)hG_.inv_psi[j])[i], ((uint64_t*)hG_.psi[j])[i], hC_.primes[j]) ==
                        1);
        }
    }
}

TEST_P(NTTTest32, TestConstants32) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    for (int j = 0; j <= cc.L; ++j) {
        for (int i = 0; i < cc.N; ++i) {
            int pow = 1 << (std::bit_width((uint32_t)i));
            if (pow > 1) {
                //   std::cout << ((uint32_t *) hpsi[0])[i] << std::endl;
                ASSERT_EQ(FIDESlib::modpow(((uint32_t*)hG_.psi[j])[i], pow, hC_.primes[j]), hC_.primes[j] - 1);
            }
            ASSERT_TRUE(FIDESlib::modpow(((uint32_t*)hG_.psi[j])[i], 2 * pow, hC_.primes[j]) == 1);
        }

        for (int i = 0; i < 2 * cc.N; ++i) {
            int pow = cc.N;
            int aux_i = i;
            while ((aux_i & 1) == 0 && aux_i != 0) {
                pow >>= 1;
                aux_i >>= 1;
            }
            if (i > 0) {
                //   std::cout << ((uint32_t *) hpsi[0])[i] << std::endl;
                ASSERT_EQ(FIDESlib::modpow(((uint32_t*)hG_.psi_no[j])[i], pow, hC_.primes[j]), hC_.primes[j] - 1);
            }
            ASSERT_TRUE(FIDESlib::modpow(((uint32_t*)hG_.psi_no[j])[i], 2 * pow, hC_.primes[j]) == 1);
        }
    }

    for (int j = 0; j <= cc.L; ++j) {
        for (int i = 0; i < cc.N / 2; ++i) {
            int pow = 1 << (std::bit_width((uint32_t)i));
            if (pow > 1) {
                //   std::cout << ((uint32_t *) hpsi[0])[i] << std::endl;
                ASSERT_EQ(FIDESlib::modpow(((uint32_t*)hG_.inv_psi[j])[i], pow, hC_.primes[j]), hC_.primes[j] - 1);
            }
            ASSERT_TRUE(FIDESlib::modpow(((uint32_t*)hG_.inv_psi[j])[i], 2 * pow, hC_.primes[j]) == 1);
        }

        for (int i = 0; i < 2 * cc.N; ++i) {
            int pow = cc.N;
            int aux_i = i;
            while ((aux_i & 1) == 0 && aux_i != 0) {
                pow >>= 1;
                aux_i >>= 1;
            }
            if (i > 0) {
                ASSERT_EQ(FIDESlib::modpow(((uint32_t*)hG_.inv_psi_no[j])[i], pow, hC_.primes[j]), hC_.primes[j] - 1);
            }
            ASSERT_TRUE(FIDESlib::modpow(FIDESlib::modprod(((uint32_t*)hG_.inv_psi_no[j])[i], hC_.N, hC_.primes[j]),
                                         2 * pow, hC_.primes[j]) == 1);
        }
    }

    for (int j = 0; j <= cc.L; ++j) {
        ASSERT_TRUE(FIDESlib::modprod(hG_.root[j], hG_.inv_root[j], hC_.primes[j]) == 1);
        for (int i = 0; i < cc.N; ++i) {
            //if(modprod(((uint32_t *) hG_.inv_psi_no[j])[i], ((uint32_t *) hG_.psi_no[j])[i] , hC_.primes[j]) != 1) std::cout << i << std::endl;
            ASSERT_TRUE(FIDESlib::modprod(FIDESlib::modprod(((uint32_t*)hG_.inv_psi_no[j])[i], hC_.N, hC_.primes[j]),
                                          ((uint32_t*)hG_.psi_no[j])[i], hC_.primes[j]) == 1);
        }
        for (int i = 0; i < cc.N / 2; ++i) {
            ASSERT_TRUE(FIDESlib::modprod(((uint32_t*)hG_.inv_psi[j])[i], ((uint32_t*)hG_.psi[j])[i], hC_.primes[j]) ==
                        1);
        }
    }
}

TEST_P(FailNTTTest, TestLimbNTT_1D_small) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters custom = fideslibParams;
    custom.logN = 3;
    custom.primes[0].p = 17;
    FIDESlib::CKKS::Context cc{custom, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        std::vector<uint64_t> v2{1, 2, 3, 7, 5, 4, 1, 2};

        FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);

        //std::vector<uint64_t> vRev(v2);

        limb2.load(v2);
        int N = 8;

        // GPU SETUP
        int primeid = 0;
        auto& v = limb2.v;
        //uint64_t *psi_arr = (uint64_t *) hG_.psi[primeid];

        dim3 blockDim = N / 2;

        dim3 gridDim = {1};
        int bytes = sizeof(uint64_t) * blockDim.x * 3;

        FIDESlib::NTT_1D<uint64_t>
            <<<gridDim, blockDim, bytes, limb2.stream.ptr>>>(v.data, nullptr, 2 * blockDim.x, primeid, 3);

        std::vector<uint64_t> res_gpu;
        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        res_gpu.resize(blockDim.x * 2);
        // END GPU SETUP

        std::vector<uint64_t> res_cpu(v2);
        res_cpu.resize(blockDim.x * 2);
        std::vector<uint64_t> v2_small(res_cpu);

        ASSERT_EQ(res_cpu, v2_small);

        fft_forPrime(res_cpu, false, 0, 1000);
        fft_forPrime(res_cpu, true, 0, 1000);

        ASSERT_EQ(res_cpu, v2_small);

        fft_forPrime(res_cpu, false, 0, 1000);

        //std::sort(res_gpu.begin(), res_gpu.end());
        //std::sort(res_cpu.begin(), res_cpu.end());
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] == 0)
                std::cout << i << std::endl;
        }
        FIDESlib::bit_reverse_vector(res_gpu);

        CudaCheckErrorMod;
        ASSERT_EQ(res_cpu, res_gpu);

        FIDESlib::INTT_1D<uint64_t><<<gridDim, blockDim, bytes, limb2.stream.ptr>>>(
            v.data, nullptr, 2 * blockDim.x, primeid, FIDESlib::modinv(N, hC_.primes[primeid]), 3);
        cudaDeviceSynchronize();

        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        res_gpu.resize(blockDim.x * 2);

        ASSERT_EQ(v2_small, res_gpu);
        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(NTTTest, TestLimbNTT_1D) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        std::vector<uint64_t> v2(cc.N, 10);
        for (auto& i : v2)
            i = rand();

        FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
        limb2.load(v2);
        int logN = 8;
        int N = 1 << logN;

        // GPU SETUP
        int primeid = 0;
        auto& v = limb2.v;
        // uint64_t *psi_arr = (uint64_t *) G_::psi[primeid];

        dim3 blockDim = N / 2;

        dim3 gridDim = {1};
        int bytes = sizeof(uint64_t) * blockDim.x * 3;

        FIDESlib::NTT_1D<uint64_t>
            <<<gridDim, blockDim, bytes, limb2.stream.ptr>>>(v.data, nullptr, blockDim.x, primeid, logN);

        std::vector<uint64_t> res_gpu;
        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        res_gpu.resize(blockDim.x * 2);
        // END GPU SETUP

        std::vector<uint64_t> res_cpu(v2);
        res_cpu.resize(blockDim.x * 2);
        std::vector<uint64_t> v2_small(res_cpu);

        ASSERT_EQ(res_cpu, v2_small);

        //fft_forPrime(res_cpu, false, 0);
        //fft_forPrime(res_cpu, true, 0

        fft(res_cpu, false, FIDESlib::modpow(hG_.root[primeid], cc.N / (blockDim.x), hC_.primes[primeid]),
            FIDESlib::modpow(hG_.inv_root[primeid], cc.N / (blockDim.x), hC_.primes[primeid]), hC_.primes[primeid]);
        fft(res_cpu, true, FIDESlib::modpow(hG_.root[primeid], cc.N / (blockDim.x), hC_.primes[primeid]),
            FIDESlib::modpow(hG_.inv_root[primeid], cc.N / (blockDim.x), hC_.primes[primeid]), hC_.primes[primeid]);

        ASSERT_EQ(res_cpu, v2_small);

        //fft_forPrime(res_cpu, false, 0);

        fft(res_cpu, false, FIDESlib::modpow(hG_.root[primeid], cc.N / (blockDim.x), hC_.primes[primeid]),
            FIDESlib::modpow(hG_.inv_root[primeid], cc.N / (blockDim.x), hC_.primes[primeid]), hC_.primes[primeid]);

        //std::sort(res_gpu.begin(), res_gpu.end());
        //std::sort(res_cpu.begin(), res_cpu.end());
        FIDESlib::bit_reverse_vector(res_gpu);
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] == 0)
                std::cout << i << std::endl;
        }
        ASSERT_EQ(res_cpu, res_gpu);

        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(NTTTest, TestLimbBitReverse) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        std::vector<uint64_t> v2(cc.N, 10);
        for (auto& i : v2)
            i = rand();

        FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
        limb2.load(v2);

        FIDESlib::Bit_Reverse<<<cc.N / 128, 128, 0, limb2.stream.ptr>>>(limb2.v.data, cc.N);

        std::vector<uint64_t> res_gpu;
        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        FIDESlib::bit_reverse_vector(v2);
        ASSERT_EQ(v2, res_gpu);

        FIDESlib::Bit_Reverse<<<cc.N / 128, 128, 0, limb2.stream.ptr>>>(limb2.v.data, cc.N);

        // END GPU SETUP

        //std::sort(res_gpu.begin(), res_gpu.end());
        //std::sort(v2.begin(), v2.end());

        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] == 0)
                std::cout << i << std::endl;
        }
        std::vector<uint64_t> res_gpu2;
        limb2.store(res_gpu2);
        cudaDeviceSynchronize();
        FIDESlib::bit_reverse_vector(v2);
        ASSERT_EQ(v2, res_gpu2);

        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(NTTTest, TestLimbNTTSecondHalf) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        std::vector<uint64_t> v2(cc.N, 10);
        int j = 0;
        for (auto& i : v2)
            i = j++;  //rand();
        for (size_t i = 0; i < v2.size(); ++i) {
            if (v2[i] == 0)
                std::cout << i << std::endl;
        }
        FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
        limb2.load(v2);

        // GPU SETUP
        int primeid = 0;
        auto& v = limb2.v;
        // uint64_t *psi_arr = (uint64_t *) G_::psi[primeid];

        dim3 blockDim = (1 << ((cc.logN) / 2 - 1));

        constexpr FIDESlib::ALGO algo = FIDESlib::ALGO_NATIVE;
        dim3 gridDim{v.size / blockDim.x / 2 / 4};
        int bytes = sizeof(uint64_t) * blockDim.x * (9 + (algo == 2 || algo == 3 ? 1 : 0));

        FIDESlib::VectorGPU<uint64_t> aux(limb2.stream, v.size, v.device);

        FIDESlib::NTT_<uint64_t, true, algo><<<gridDim, blockDim, bytes, limb2.stream.ptr>>>(v.data, primeid, aux.data);
        std::vector<uint64_t> res_gpu;
        limb2.load(aux);
        aux.free(limb2.stream);
        limb2.store(res_gpu);
        cudaDeviceSynchronize();

        std::vector<uint64_t> res_cpu(v2);
        for (size_t i = 0; i < blockDim.x * 2; ++i) {
            res_cpu[i] = v2.at(i * cc.N / (blockDim.x * 2));
        }
        res_cpu.resize(blockDim.x * 2);
        std::vector<uint64_t> v2_small(res_cpu);

        ASSERT_EQ(res_cpu, v2_small);

        fft_forPrime(res_cpu, false, 0);
        fft_forPrime(res_cpu, true, 0);
        ASSERT_EQ(res_cpu, v2_small);

        fft_forPrime(res_cpu, false, 0);

        res_gpu.resize(res_cpu.size());
        //        std::sort(res_gpu.begin(), res_gpu.end());
        //        std::sort(res_cpu.begin(), res_cpu.end());
        FIDESlib::bit_reverse_vector(res_gpu);
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] == 0)
                std::cout << i << std::endl;
        }
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] != res_cpu[i])
                std::cout << i << std::endl;
        }
        CudaCheckErrorMod;
        ASSERT_EQ(res_cpu, res_gpu);

        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(FailNTTTest32, TestLimbNTTSecondHalf32) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        std::vector<uint64_t> v2(cc.N, 10);
        int j = 0;
        for (auto& i : v2)
            i = j++;  //rand();
        for (size_t i = 0; i < v2.size(); ++i) {
            if (v2[i] == 0)
                std::cout << i << std::endl;
        }
        FIDESlib::CKKS::Limb<uint32_t> limb2(cc, GPUs[0], s, 0);
        limb2.load(v2);

        // GPU SETUP
        int primeid = 0;
        auto& v = limb2.v;

        dim3 blockDim = (1 << ((cc.logN) / 2 - 1));

        constexpr FIDESlib::ALGO algo = FIDESlib::ALGO_NATIVE;
        dim3 gridDim{v.size / blockDim.x / 2 / 4};
        int bytes = sizeof(uint64_t) * blockDim.x * (9 + (algo == 2 || algo == 3 ? 1 : 0));

        FIDESlib::VectorGPU<uint32_t> aux(limb2.stream, v.size, v.device);

        FIDESlib::NTT_<uint32_t, true, algo><<<gridDim, blockDim, bytes, limb2.stream.ptr>>>(v.data, primeid, aux.data);

        std::vector<uint32_t> res_gpu;
        limb2.load(aux);
        aux.free(limb2.stream);
        limb2.store(res_gpu);
        cudaDeviceSynchronize();

        std::vector<uint64_t> res_cpu(v2);
        for (size_t i = 0; i < blockDim.x * 2; ++i) {
            res_cpu[i] = v2.at(i * cc.N / (blockDim.x * 2));
        }
        res_cpu.resize(blockDim.x * 2);
        std::vector<uint64_t> v2_small(res_cpu);

        ASSERT_EQ(res_cpu, v2_small);

        fft_forPrime(res_cpu, false, 0);
        fft_forPrime(res_cpu, true, 0);
        ASSERT_EQ(res_cpu, v2_small);

        fft_forPrime(res_cpu, false, 0);

        res_gpu.resize(res_cpu.size());
        //std::sort(res_gpu.begin(), res_gpu.end());
        //std::sort(res_cpu.begin(), res_cpu.end());
        FIDESlib::bit_reverse_vector(res_gpu);
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] == 0)
                std::cout << i << std::endl;
        }
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] != res_cpu[i])
                std::cout << i << std::endl;
        }
        CudaCheckErrorMod;
        std::vector<uint64_t> res_gpu64(res_gpu.size());
        for (size_t i = 0; i < res_gpu.size(); ++i)
            res_gpu64[i] = res_gpu[i];
        ASSERT_EQ(res_cpu, res_gpu64);

        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(FailNTTTest, TestLimbINTTSecondHalfSmall) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters aux_params{fideslibParams};
    aux_params.logN = 4;
    FIDESlib::CKKS::Context cc{aux_params, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        std::vector<uint64_t> v2(cc.N, 10);
        for (auto& i : v2)
            i = rand();
        for (size_t i = 0; i < v2.size(); ++i) {
            if (v2[i] == 0)
                std::cout << i << std::endl;
        }
        FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
        limb2.load(v2);

        // GPU SETUP
        int primeid = 1;
        auto& v = limb2.v;
        dim3 blockDim = (1 << ((cc.logN) / 2 - 1));

        constexpr FIDESlib::ALGO algo = FIDESlib::ALGO_NATIVE;
        dim3 gridDim{v.size / blockDim.x / 2 / 4};
        int bytes = sizeof(uint64_t) * blockDim.x * (9 + (algo == 2 || algo == 3 ? 1 : 0));

        FIDESlib::VectorGPU<uint64_t> aux(limb2.stream, v.size, v.device);

        FIDESlib::INTT_<uint64_t, false, algo>
            <<<gridDim, blockDim, bytes, limb2.stream.ptr>>>(v.data, primeid, aux.data);

        std::vector<uint64_t> res_gpu;
        cudaDeviceSynchronize();
        limb2.load(aux);
        cudaDeviceSynchronize();
        aux.free(limb2.stream);
        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        //res_gpu.resize(blockDim.x * 2);
        //

        std::vector<uint64_t> res_cpu(v2);
        for (size_t i = 0; i < blockDim.x * 2; ++i) {
            //res_cpu[i] = res_cpu.at(i * 4);
        }
        res_cpu.resize(blockDim.x * 2);
        FIDESlib::bit_reverse_vector(res_cpu);

        std::vector<uint64_t> v2_small(res_cpu);

        ASSERT_EQ(res_cpu, v2_small);
        fft_forPrime(res_cpu, true, primeid);
        fft_forPrime(res_cpu, false, primeid);
        ASSERT_EQ(res_cpu, v2_small);

        fft_forPrime(res_cpu, true, primeid);

        uint64_t n_1 = FIDESlib::modinv(2 * blockDim.x, hC_.primes[primeid]);
        for (uint64_t& x : res_gpu)
            x = ((uint128_t)1 * x * n_1 % hC_.primes[primeid]);

        for (size_t i = 0; i < blockDim.x * 2; ++i) {
            //  res_gpu[i] = res_gpu.at(i * 4);
        }
        //res_gpu.resize(res_cpu.size());

        for (size_t i = 0; i < res_gpu.size(); ++i) {
            //      if(res_gpu[i] == 0) std::cout << i << std::endl;
        }
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] != res_cpu[i])
                std::cout << i << " " << res_gpu[i] << " " << res_cpu[i] << std::endl;
        }
        //  std::sort(res_gpu.begin(), res_gpu.end());
        //  std::sort(res_cpu.begin(), res_cpu.end());

        CudaCheckErrorMod;
        ASSERT_EQ(res_cpu, res_gpu);
        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(FailNTTTest, TestLimbINTTSecondHalf) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        std::vector<uint64_t> v2(cc.N, 10);
        for (auto& i : v2)
            i = rand();
        for (size_t i = 0; i < v2.size(); ++i) {
            if (v2[i] == 0)
                std::cout << i << std::endl;
        }
        FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
        limb2.load(v2);

        // GPU SETUP
        int primeid = 0;
        auto& v = limb2.v;

        dim3 blockDim = (1 << ((cc.logN) / 2 - 1));

        constexpr FIDESlib::ALGO algo = FIDESlib::ALGO_NATIVE;
        dim3 gridDim{v.size / blockDim.x / 2 / 4};
        int bytes = sizeof(uint64_t) * blockDim.x * (9 + (algo == 2 || algo == 3 ? 1 : 0));

        FIDESlib::VectorGPU<uint64_t> aux(limb2.stream, v.size, v.device);

        FIDESlib::INTT_<uint64_t, true, algo>
            <<<gridDim, blockDim, bytes, limb2.stream.ptr>>>(v.data, primeid, aux.data);

        std::vector<uint64_t> res_gpu;
        limb2.load(aux);
        aux.free(limb2.stream);
        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        //res_gpu.resize(blockDim.x * 2);
        //

        std::vector<uint64_t> res_cpu(v2);
        for (size_t i = 0; i < blockDim.x * 2; ++i) {
            // res_cpu[i] = res_cpu.at(i * 256);
        }
        res_cpu.resize(blockDim.x * 2);
        FIDESlib::bit_reverse_vector(res_cpu);
        std::vector<uint64_t> v2_small(res_cpu);

        ASSERT_EQ(res_cpu, v2_small);
        fft_forPrime(res_cpu, true, 0);
        fft_forPrime(res_cpu, false, 0);
        ASSERT_EQ(res_cpu, v2_small);

        fft_forPrime(res_cpu, true, 0);

        for (size_t i = 0; i < blockDim.x * 2; ++i) {
            res_gpu[i] = res_gpu.at(i * 256);
        }
        res_gpu.resize(res_cpu.size());

        uint64_t n_1 = FIDESlib::modinv(2 * blockDim.x, hC_.primes[primeid]);
        for (uint64_t& x : res_gpu) {
            x = ((uint128_t)1 * x * n_1 % hC_.primes[primeid]);
        }
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            //      if(res_gpu[i] == 0) std::cout << i << std::endl;
        }
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] != res_cpu[i])
                std::cout << i << " " << res_gpu[i] << " " << res_cpu[i] << std::endl;
        }
        //  std::sort(res_gpu.begin(), res_gpu.end());
        //  std::sort(res_cpu.begin(), res_cpu.end());

        CudaCheckErrorMod;
        ASSERT_EQ(res_cpu, res_gpu);
        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(NTTTest, TestLimbNTTsmall) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters aux_params{fideslibParams};
    aux_params.logN = 4;
    FIDESlib::CKKS::Context cc{aux_params, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        std::vector<uint64_t> v2{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        //for(auto & i : v2) i = rand();

        // GPU
        FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
        limb2.load(v2);

        int primeid = 0;

        limb2.NTT<FIDESlib::ALGO_NATIVE>();
        CudaCheckErrorMod;
        std::vector<uint64_t> res_gpu;
        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        CudaCheckErrorMod;
        //std::sort(res_cpu.begin(), res_cpu.end());
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] == 0)
                std::cout << i << std::endl;
        }
        sleep(1);
        //std::sort(res_gpu.begin(), res_gpu.end());
        FIDESlib::bit_reverse_vector(res_gpu);
        // CPU
        std::vector<uint64_t> res_cpu(v2);
        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, false, primeid);
        nega_fft2_forPrime(res_cpu, true, primeid);

        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, false, primeid);

        ASSERT_EQ(res_cpu.size(), res_gpu.size());
        ASSERT_EQ(res_cpu, res_gpu);

        std::cout << "Better Barrett" << std::endl;
        limb2.load(v2);

        limb2.NTT<FIDESlib::ALGO_BARRETT>();
        CudaCheckErrorMod;

        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        FIDESlib::bit_reverse_vector(res_gpu);
        CudaCheckErrorMod;
        //std::sort(res_cpu.begin(), res_cpu.end());
        //std::sort(res_gpu.begin(), res_gpu.end());
        ASSERT_EQ(res_cpu, res_gpu);

        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(NTTTest, TestLimbNTT) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        std::vector<uint64_t> v2(cc.N, 10);
        for (auto& i : v2)
            i = rand();

        // GPU
        FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
        limb2.load(v2);

        int primeid = 0;

        limb2.NTT<FIDESlib::ALGO_NATIVE>();
        CudaCheckErrorMod;
        std::vector<uint64_t> res_gpu;
        limb2.store(res_gpu);
        cudaDeviceSynchronize();

        //std::sort(res_cpu.begin(), res_cpu.end());
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] == 0)
                std::cout << i << std::endl;
        }
        //std::sort(res_gpu.begin(), res_gpu.end());
        FIDESlib::bit_reverse_vector(res_gpu);
        // CPU
        std::vector<uint64_t> res_cpu(v2);
        CudaCheckErrorMod;
        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, false, primeid);
        nega_fft2_forPrime(res_cpu, true, primeid);

        ASSERT_EQ(res_cpu, v2);

        nega_fft2_forPrime(res_cpu, false, primeid);

        ASSERT_EQ(res_cpu.size(), res_gpu.size());
        ASSERT_EQ(res_cpu, res_gpu);

        std::cout << "Better Barrett" << std::endl;
        limb2.load(v2);

        limb2.NTT<FIDESlib::ALGO_BARRETT>();

        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        FIDESlib::bit_reverse_vector(res_gpu);
        //std::sort(res_cpu.begin(), res_cpu.end());
        //std::sort(res_gpu.begin(), res_gpu.end());
        ASSERT_EQ(res_cpu, res_gpu);

        std::cout << "Shoup" << std::endl;
        limb2.load(v2);

        limb2.NTT<FIDESlib::ALGO_SHOUP>();

        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        FIDESlib::bit_reverse_vector(res_gpu);
        //std::sort(res_cpu.begin(), res_cpu.end());
        //std::sort(res_gpu.begin(), res_gpu.end());
        ASSERT_EQ(res_cpu, res_gpu);

        std::cout << "Barrett fp64" << std::endl;
        limb2.load(v2);

        if (cc.prime.at(limb2.primeid).bits <= 51) {
            limb2.NTT<FIDESlib::ALGO_BARRETT_FP64>();
        } else {
            limb2.NTT<FIDESlib::ALGO_NATIVE>();
        }

        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        FIDESlib::bit_reverse_vector(res_gpu);
        //std::sort(res_cpu.begin(), res_cpu.end());
        //std::sort(res_gpu.begin(), res_gpu.end());
        ASSERT_EQ(res_cpu, res_gpu);

        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(FailNTTTest32, TestLimbNTT32) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        std::vector<uint64_t> v2(cc.N, 10);
        for (auto& i : v2)
            i = rand();

        // GPU
        FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
        limb2.load(v2);

        int primeid = 0;

        limb2.NTT<FIDESlib::ALGO_NATIVE>();
        CudaCheckErrorMod;
        std::vector<uint64_t> res_gpu;
        limb2.store(res_gpu);
        cudaDeviceSynchronize();

        //std::sort(res_cpu.begin(), res_cpu.end());
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] == 0)
                std::cout << i << std::endl;
        }
        sleep(1);
        //std::sort(res_gpu.begin(), res_gpu.end());
        FIDESlib::bit_reverse_vector(res_gpu);
        // CPU
        std::vector<uint64_t> res_cpu(v2);
        ASSERT_EQ(res_cpu, v2);

        fft_forPrime(res_cpu, false, primeid);
        fft_forPrime(res_cpu, true, primeid);

        ASSERT_EQ(res_cpu, v2);

        fft_forPrime(res_cpu, false, primeid);

        ASSERT_EQ(res_cpu.size(), res_gpu.size());
        ASSERT_EQ(res_cpu, res_gpu);

        std::cout << "Better Barrett" << std::endl;
        limb2.load(v2);

        limb2.NTT<FIDESlib::ALGO_BARRETT>();

        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        FIDESlib::bit_reverse_vector(res_gpu);
        //std::sort(res_cpu.begin(), res_cpu.end());
        //std::sort(res_gpu.begin(), res_gpu.end());
        ASSERT_EQ(res_cpu, res_gpu);

        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(FailNTTTest, TestLimbNTTtoINTT) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        std::vector<uint64_t> v2(cc.N, 10);
        for (auto& i : v2)
            i = rand();

        // GPU
        int primeid = 0;
        FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, primeid);
        limb2.load(v2);

        /// limb2.NTT<0>();
        const FIDESlib::ALGO algo = FIDESlib::ALGO_NATIVE;

        assert(primeid >= 0);
        dim3 blockDim{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
        dim3 gridDim{limb2.v.size / blockDim.x / 2 / 4};
        int bytes = sizeof(uint64_t) * blockDim.x * (9 + (algo == 2 || algo == 3 ? 1 : 0));

        FIDESlib::NTT_<uint64_t, false, algo>
            <<<gridDim, blockDim, bytes, limb2.stream.ptr>>>(limb2.v.data, primeid, limb2.aux.data);
        CudaCheckErrorMod;
        blockDim = (1 << ((cc.logN) / 2 - 1));
        gridDim = {limb2.v.size / blockDim.x / 2 / 4};
        bytes = sizeof(uint64_t) * blockDim.x * (9 + (algo == 2 || algo == 3 ? 1 : 0));

        FIDESlib::NTT_<uint64_t, true, algo>
            <<<gridDim, blockDim, bytes, limb2.stream.ptr>>>(limb2.aux.data, primeid, limb2.v.data);
        ///
        CudaCheckErrorMod;

        std::vector<uint64_t> res_gpu;
        limb2.store(res_gpu);
        cudaDeviceSynchronize();

        //std::sort(res_cpu.begin(), res_cpu.end());
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] == 0)
                std::cout << i << std::endl;
        }
        sleep(1);
        //std::sort(res_gpu.begin(), res_gpu.end());
        FIDESlib::bit_reverse_vector(res_gpu);
        // CPU
        std::vector<uint64_t> res_cpu(v2);
        ASSERT_EQ(res_cpu, v2);

        fft_forPrime(res_cpu, false, primeid);
        fft_forPrime(res_cpu, true, primeid);

        ASSERT_EQ(res_cpu, v2);

        fft_forPrime(res_cpu, true, primeid);

        ASSERT_EQ(res_cpu.size(), res_gpu.size());
        ASSERT_EQ(res_cpu, res_gpu);
        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(FailNTTTest, TestCPUntt_2d) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    for (int l : {0}) {
        cudaSetDevice(GPUs[0]);

        std::vector<uint64_t> v2(cc.N, 10);
        for (auto& i : v2)
            i = rand();

        // GPU
        int primeid = 0;

        std::vector<uint64_t> res_gpu(v2);
        fft_2d(res_gpu, (1 << ((cc.logN + 1) / 2)), primeid);

        //std::sort(res_cpu.begin(), res_cpu.end());
        for (size_t i = 0; i < res_gpu.size(); ++i) {
            if (res_gpu[i] == 0)
                std::cout << i << std::endl;
        }
        sleep(1);
        //std::sort(res_gpu.begin(), res_gpu.end());
        FIDESlib::bit_reverse_vector(res_gpu);
        // CPU
        std::vector<uint64_t> res_cpu(v2);
        ASSERT_EQ(res_cpu, v2);

        fft_forPrime(res_cpu, false, primeid);
        fft_forPrime(res_cpu, true, primeid);

        ASSERT_EQ(res_cpu, v2);

        fft_forPrime(res_cpu, false, primeid);

        ASSERT_EQ(res_cpu.size(), res_gpu.size());
        ASSERT_EQ(res_cpu, res_gpu);

        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(NTTTest, TestLimbInverseNTTsmall) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters aux_params{fideslibParams};
    aux_params.logN = 4;
    FIDESlib::CKKS::Context cc{aux_params, GPUs};
    CudaCheckErrorMod;

    for (int l : {0}) {
        FIDESlib::Stream s;
        s.init();
        cudaSetDevice(GPUs[0]);

        std::vector<uint64_t> v2(cc.N, 10);
        for (auto& i : v2)
            i = rand();

        FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
        limb2.load(v2);

        limb2.NTT<FIDESlib::ALGO_NATIVE>();

        std::vector<uint64_t> res_gpu;
        limb2.store(res_gpu);
        cudaDeviceSynchronize();

        std::vector<uint64_t> res_cpu(v2);
        nega_fft2_forPrime(res_cpu, false, 0);
        FIDESlib::bit_reverse_vector(res_cpu);
        ASSERT_EQ(res_cpu, res_gpu);

        limb2.INTT<FIDESlib::ALGO_NATIVE>();

        limb2.store(res_gpu);
        cudaDeviceSynchronize();
        FIDESlib::bit_reverse_vector(res_cpu);
        nega_fft2_forPrime(res_cpu, true, 0);

        //std::sort(res_cpu.begin(), res_cpu.end());
        //std::sort(res_gpu.begin(), res_gpu.end());
        ASSERT_EQ(res_cpu, res_gpu);
        ASSERT_EQ(v2, res_gpu);

        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(NTTTest, TestLimbInverseNTT) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    CudaCheckErrorMod;

    for (int l : {0}) {
        for (int primeid = 0; primeid < cc.L + cc.K + 1; ++primeid) {
            std::cout << primeid << std::endl;
            FIDESlib::Stream s;
            s.init();
            cudaSetDevice(GPUs[0]);

            std::vector<uint64_t> v2(cc.N, 10);
            for (auto& i : v2)
                i = rand();

            FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, primeid);
            {
                limb2.load(v2);

                limb2.NTT<FIDESlib::ALGO_NATIVE>();

                std::vector<uint64_t> res_gpu;
                limb2.store(res_gpu);
                cudaDeviceSynchronize();

                std::vector<uint64_t> res_cpu(v2);
                nega_fft2_forPrime(res_cpu, false, primeid);
                FIDESlib::bit_reverse_vector(res_cpu);
                if ((cc.logN & 1) == 0)
                    ASSERT_EQ(res_cpu, res_gpu);
                FIDESlib::bit_reverse_vector(res_cpu);

                limb2.INTT<FIDESlib::ALGO_NATIVE>();

                limb2.store(res_gpu);
                cudaDeviceSynchronize();

                nega_fft2_forPrime(res_cpu, true, primeid);

                //std::sort(res_cpu.begin(), res_cpu.end());
                //std::sort(res_gpu.begin(), res_gpu.end());
                ASSERT_EQ(res_cpu, res_gpu);
                ASSERT_EQ(v2, res_gpu);
            }
            {
                limb2.load(v2);

                limb2.NTT<FIDESlib::ALGO_NATIVE>();

                std::vector<uint64_t> res_gpu;
                limb2.store(res_gpu);
                cudaDeviceSynchronize();

                std::vector<uint64_t> res_cpu(v2);
                nega_fft2_forPrime(res_cpu, false, primeid);
                FIDESlib::bit_reverse_vector(res_cpu);
                if ((cc.logN & 1) == 0)
                    ASSERT_EQ(res_cpu, res_gpu);
                FIDESlib::bit_reverse_vector(res_cpu);

                limb2.INTT<FIDESlib::ALGO_BARRETT>();

                limb2.store(res_gpu);
                cudaDeviceSynchronize();

                nega_fft2_forPrime(res_cpu, true, primeid);

                //std::sort(res_cpu.begin(), res_cpu.end());
                //std::sort(res_gpu.begin(), res_gpu.end());
                ASSERT_EQ(res_cpu, res_gpu);
                ASSERT_EQ(v2, res_gpu);
            }
            {
                limb2.load(v2);

                limb2.NTT<FIDESlib::ALGO_NATIVE>();

                std::vector<uint64_t> res_gpu;
                limb2.store(res_gpu);
                cudaDeviceSynchronize();

                std::vector<uint64_t> res_cpu(v2);
                nega_fft2_forPrime(res_cpu, false, primeid);
                FIDESlib::bit_reverse_vector(res_cpu);
                if ((cc.logN & 1) == 0)
                    ASSERT_EQ(res_cpu, res_gpu);
                FIDESlib::bit_reverse_vector(res_cpu);

                limb2.INTT<FIDESlib::ALGO_SHOUP>();

                limb2.store(res_gpu);
                cudaDeviceSynchronize();

                nega_fft2_forPrime(res_cpu, true, primeid);

                //std::sort(res_cpu.begin(), res_cpu.end());
                //std::sort(res_gpu.begin(), res_gpu.end());
                ASSERT_EQ(res_cpu, res_gpu);
                ASSERT_EQ(v2, res_gpu);
            }
            {
                limb2.load(v2);

                limb2.NTT<FIDESlib::ALGO_NATIVE>();

                std::vector<uint64_t> res_gpu;
                limb2.store(res_gpu);
                cudaDeviceSynchronize();

                std::vector<uint64_t> res_cpu(v2);
                nega_fft2_forPrime(res_cpu, false, primeid);
                FIDESlib::bit_reverse_vector(res_cpu);
                if ((cc.logN & 1) == 0)
                    ASSERT_EQ(res_cpu, res_gpu);
                FIDESlib::bit_reverse_vector(res_cpu);

                if ((primeid <= cc.L && cc.prime.at(limb2.primeid).bits <= 51) ||
                    (primeid > cc.L && cc.specialPrime.at(limb2.primeid - cc.L - 1).bits <= 51)) {
                    limb2.INTT<FIDESlib::ALGO_BARRETT_FP64>();
                } else {
                    limb2.INTT<FIDESlib::ALGO_NATIVE>();
                }

                limb2.store(res_gpu);
                cudaDeviceSynchronize();

                nega_fft2_forPrime(res_cpu, true, primeid);

                //std::sort(res_cpu.begin(), res_cpu.end());
                //std::sort(res_gpu.begin(), res_gpu.end());
                ASSERT_EQ(res_cpu, res_gpu);
                ASSERT_EQ(v2, res_gpu);
            }
            // destructor implícito
        }
    }
    CudaCheckErrorMod;
}

TEST_P(NTTTest, LimbBatchTestINTT) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    CudaCheckErrorMod;

    FIDESlib::CKKS::RNSPoly poly(cc, cc.L);
    FIDESlib::CKKS::RNSPoly poly2(cc, cc.L);
    for (int batch : {1, -1, 2, 3, 4, 10, 100}) {
        std::cout << "Batch arg: " << batch << std::endl;
        std::vector<uint64_t> v2(cc.N, 10);
        for (auto& i : v2)
            i = rand();

        for (auto& part : poly.GPU) {
            for (auto& limb : part.limb)
                SWITCH(limb, load(v2));
        }

        std::vector<std::vector<uint64_t>> res_gpu;
        cudaDeviceSynchronize();
        poly.NTT<FIDESlib::ALGO_SHOUP>(batch);
        cudaDeviceSynchronize();
        poly.store(res_gpu);
        std::vector<std::vector<uint64_t>> res_cpu(cc.L + 1, std::vector<uint64_t>(v2));
        for (int i = 0; i <= cc.L; ++i) {
            nega_fft2_forPrime(res_cpu[i], false, i);
            FIDESlib::bit_reverse_vector(res_cpu[i]);
        }
        if ((cc.logN & 1) == 0)
            ASSERT_EQ(res_cpu, res_gpu);

        cudaDeviceSynchronize();
        poly.INTT<FIDESlib::ALGO_SHOUP>(batch);
        cudaDeviceSynchronize();
        poly.store(res_gpu);
        for (int i = 0; i <= cc.L; ++i) {
            FIDESlib::bit_reverse_vector(res_cpu[i]);
            nega_fft2_forPrime(res_cpu[i], true, i);
        }

        ASSERT_EQ(res_cpu, res_gpu);
    }
    CudaCheckErrorMod;
    for (int batch : {1, -1, 2, 3, 4, 10, 100}) {
        std::cout << "Batch arg: " << batch << std::endl;
        std::vector<uint64_t> v2(cc.N, 10);
        for (auto& i : v2)
            i = rand();

        for (auto& part : poly.GPU) {
            for (auto& limb : part.limb)
                SWITCH(limb, load(v2));
        }

        std::vector<std::vector<uint64_t>> res_gpu;
        cudaDeviceSynchronize();
        poly.NTT<FIDESlib::ALGO_BARRETT>(batch);
        cudaDeviceSynchronize();
        poly.store(res_gpu);
        std::vector<std::vector<uint64_t>> res_cpu(cc.L + 1, std::vector<uint64_t>(v2));
        for (int i = 0; i <= cc.L; ++i) {
            nega_fft2_forPrime(res_cpu[i], false, i);
            FIDESlib::bit_reverse_vector(res_cpu[i]);
        }
        if ((cc.logN & 1) == 0)
            ASSERT_EQ(res_cpu, res_gpu);

        cudaDeviceSynchronize();
        poly.INTT<FIDESlib::ALGO_BARRETT>(batch);
        cudaDeviceSynchronize();
        poly.store(res_gpu);
        for (int i = 0; i <= cc.L; ++i) {
            FIDESlib::bit_reverse_vector(res_cpu[i]);
            nega_fft2_forPrime(res_cpu[i], true, i);
        }

        ASSERT_EQ(res_cpu, res_gpu);
    }
    CudaCheckErrorMod;
}

INSTANTIATE_TEST_SUITE_P(NTTTests, NTTTest, testing::Values(params64_13, params64_14, params64_15, params64_16));

// INSTANTIATE_TEST_SUITE_P(NTTTests32, NTTTest32, testing::Values(params32_15));

//INSTANTIATE_TEST_SUITE_P(FailNTTTests, FailNTTTest, testing::Values(params64_13, params64_14, params64_15, params64_16));

// INSTANTIATE_TEST_SUITE_P(FailNTTTests32, FailNTTTest32, testing::Values(params32_15));
}  // namespace FIDESlib::Testing