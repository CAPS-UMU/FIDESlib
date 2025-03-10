//
// Created by carlosad on 25/03/24.
//
#include <algorithm>
#include <execution>
#include <iomanip>

#include <gtest/gtest.h>

#include "CKKS/Context.cuh"
#include "CKKS/Limb.cuh"
#include "ConstantsGPU.cuh"
#include "CudaUtils.cuh"
#include "ModMult.cuh"
#include "ParametrizedTest.cuh"

namespace FIDESlib::Testing {
class LimbKernelTest : public FIDESlibParametrizedTest {};

class LimbKernelTest32 : public FIDESlibParametrizedTest {};

TEST_P(LimbKernelTest, AllLimbKernel32) {
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

        FIDESlib::CKKS::Limb<uint32_t> limb(cc, 0, s, 0);
        FIDESlib::CKKS::LimbImpl limb2(FIDESlib::CKKS::Limb<uint32_t>(cc, 0, s, 0));

        std::vector<uint32_t> v(cc.N, 10);
        std::vector<uint32_t> v2(cc.N, 0);

        limb.load(v);
        limb.store(v2);
        FIDESlib::CudaHostSync();

        ASSERT_EQ(v, v2);
        assert(v2[0] == 10);

        std::get<FIDESlib::U32>(limb2).load(v);

        std::vector<uint32_t> v3(cc.N, 20);

        limb.add(limb2);

        limb.store(v2);
        FIDESlib::CudaHostSync();
        for (int i = 0; i < cc.N; ++i)
            if (v2[i] != v3[i])
                std::cout << i << std::endl;
        ASSERT_EQ(v2, v3);
        ASSERT_NE(v, v2);

        limb.sub(limb2);

        limb.store(v2);
        FIDESlib::CudaHostSync();
        ASSERT_NE(v2, v3);
        ASSERT_EQ(v, v2);

        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(LimbKernelTest, AllLimbKernel64) {
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

        FIDESlib::CKKS::Limb<uint64_t> limb(cc, 0, s, 0);
        FIDESlib::CKKS::LimbImpl limb2(FIDESlib::CKKS::Limb<uint64_t>(cc, 0, s, 0));

        std::vector<uint64_t> v(cc.N, 10);
        std::vector<uint64_t> v2(cc.N, 0);

        limb.load(v);
        limb.store(v2);
        FIDESlib::CudaHostSync();

        ASSERT_EQ(v, v2);
        assert(v2[0] == 10);

        std::get<FIDESlib::U64>(limb2).load(v);

        std::vector<uint64_t> v3(cc.N, 20);

        limb.add(limb2);

        limb.store(v2);
        FIDESlib::CudaHostSync();
        for (int i = 0; i < cc.N; ++i)
            if (v2[i] != v3[i])
                std::cout << i << std::endl;
        ASSERT_EQ(v2, v3);
        ASSERT_NE(v, v2);

        limb.sub(limb2);

        limb.store(v2);
        FIDESlib::CudaHostSync();
        ASSERT_NE(v2, v3);
        ASSERT_EQ(v, v2);
        // destructor implícito
    }
    CudaCheckErrorMod;
}

TEST_P(LimbKernelTest, TestMultKernel64) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    CudaCheckErrorMod;

    for (int i = 0; i <= cc.L + cc.K; ++i)
        for (int l : {0}) {
            cudaSetDevice(GPUs[0]);
            FIDESlib::Stream s;
            s.init();

            FIDESlib::CKKS::Limb<uint64_t> limb(cc, GPUs[0], s, i);
            FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, i);

            std::vector<uint64_t> v(cc.N, 10);
            std::vector<uint64_t> v2(cc.N, 0);
            for (auto& e : v2)
                e = (uint64_t)rand() * rand() % hC_.primes[i];

            limb.load(v);
            limb2.load(v2);

            std::vector<uint64_t> v3(v);
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < cc.N; ++k)
                    v3[k] = ((__uint128_t)v3[k]) * ((__uint128_t)v2[k]) % ((__uint128_t)hC_.primes[i]);

                FIDESlib::mult_<uint64_t, FIDESlib::ALGO_NATIVE>
                    <<<cc.N / 256, 256, 0, limb.stream.ptr>>>(limb.v.data, limb2.v.data, i);

                limb.store(v);

                cudaDeviceSynchronize();
                CudaCheckErrorMod;
                for (int i = 0; i < cc.N; ++i)
                    if (v[i] != v3[i])
                        std::cout << i << std::endl;
                ASSERT_EQ(v, v3);
            }

            // destructor implícito
        }
    CudaCheckErrorMod;
}

TEST_P(LimbKernelTest, TestBetterBarretMultKernel64) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    CudaCheckErrorMod;

    for (int its = 0; its < 10; ++its) {
        for (int i = 0; i <= cc.L + cc.K; ++i) {
            cudaSetDevice(GPUs[0]);
            FIDESlib::Stream s;
            s.init();

            FIDESlib::CKKS::Limb<uint64_t> limb(cc, GPUs[0], s, i);
            FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, i);

            std::vector<uint64_t> v(cc.N, rand() * rand() % hC_.primes[i]);
            std::vector<uint64_t> v2(cc.N, 0);
            for (auto& e : v2)
                e = (uint64_t)rand() * rand() % hC_.primes[i];

            limb.load(v);
            limb2.load(v2);

            std::vector<uint64_t> v3(v);
            for (int j = 0; j < 10; ++j) {
                for (int k = 0; k < cc.N; ++k)
                    v3[k] = ((__uint128_t)v3[k]) * ((__uint128_t)v2[k]) % ((__uint128_t)hC_.primes[i]);

                FIDESlib::mult_<uint64_t, FIDESlib::ALGO_BARRETT>
                    <<<cc.N / 256, 256, 0, limb.stream.ptr>>>(limb.v.data, limb2.v.data, i);

                limb.store(v);
                FIDESlib::CudaHostSync();
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
                for (int k = 0; k < cc.N; ++k)
                    if (v[k] != v3[k]) {
                        std::cout << i << ":" << k << " " << v[k] << " " << v3[k] << std::endl;
                    }
                ASSERT_EQ(v, v3);
            }
            // destructor implícito
        }
    }
}

TEST_P(LimbKernelTest, Test53bitFp64debMultKernel64) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    CudaCheckErrorMod;

    for (int its = 0; its < 10; ++its) {
        for (int i = 0; i <= cc.L + cc.K; ++i) {
            if (hC_.prime_bits[i] >= 53 - 1)
                continue;

            cudaSetDevice(GPUs[0]);
            FIDESlib::Stream s;
            s.init();

            FIDESlib::CKKS::Limb<uint64_t> limb(cc, GPUs[0], s, i);
            FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, i);

            std::vector<uint64_t> v(cc.N, rand() * rand() % hC_.primes[i]);
            std::vector<uint64_t> v2(cc.N, 0);
            for (auto& e : v2)
                e = (uint64_t)rand() * rand() % hC_.primes[i];

            limb.load(v);
            limb2.load(v2);

            std::vector<uint64_t> v3(v);
            for (int j = 0; j < 10; ++j) {
                for (int k = 0; k < cc.N; ++k)
                    v3[k] = ((__uint128_t)v3[k]) * ((__uint128_t)v2[k]) % ((__uint128_t)hC_.primes[i]);

                FIDESlib::mult_<uint64_t, FIDESlib::ALGO_BARRETT_FP64>
                    <<<cc.N / 256, 256, 0, limb.stream.ptr>>>(limb.v.data, limb2.v.data, i);

                limb.store(v);
                FIDESlib::CudaHostSync();
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
                for (int k = 0; k < cc.N; ++k)
                    if (v[k] != v3[k]) {
                        std::cout << std::hex << i << ":" << k << " " << v[k] << " " << v3[k] << std::endl;
                    }
                ASSERT_EQ(v, v3);
            }
            // destructor implícito
        }
    }
}

TEST_P(LimbKernelTest, TestBarretPsiKernel64) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    CudaCheckErrorMod;

    for (int its = 0; its < 10; ++its) {
        for (int i = 0; i <= cc.L + cc.K; ++i) {
            if (hC_.prime_bits[i] >= 53 - 1)
                continue;

            cudaSetDevice(GPUs[0]);
            FIDESlib::Stream s;
            s.init();

            FIDESlib::CKKS::Limb<uint64_t> limb(cc, GPUs[0], s, i);
            FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, i);

            std::vector<uint64_t> v(cc.N, rand() * rand() % hC_.primes[i]);
            std::vector<uint64_t> v2(cc.N, rand() * rand() % hC_.primes[i]);
            //for (auto &e: v2) e = (uint64_t) rand() * rand() % hC_.primes[i];
            uint64_t aux_psi = (__uint128_t)(v2[0] << 1) * (1ul << 63) / hC_.primes[i];
            limb.load(v);
            limb2.load(v2);

            std::vector<uint64_t> v3(v);
            for (int j = 0; j < 10; ++j) {
                for (int k = 0; k < cc.N; ++k)
                    v3[k] = ((__uint128_t)v3[k]) * ((__uint128_t)v2[k]) % ((__uint128_t)hC_.primes[i]);

                FIDESlib::scalar_mult_<uint64_t, FIDESlib::ALGO_SHOUP>
                    <<<cc.N / 256, 256, 0, limb.stream.ptr>>>(limb.v.data, v2[0], i, aux_psi);

                limb.store(v);
                FIDESlib::CudaHostSync();
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
                for (int k = 0; k < cc.N; ++k)
                    if (v[k] != v3[k]) {
                        std::cout << std::hex << i << ":" << k << " " << v[k] << " " << v3[k] << std::endl;
                    }
                ASSERT_EQ(v, v3);
            }
            // destructor implícito
        }
    }
}

TEST_P(LimbKernelTest32, TestMultKernel32) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    CudaCheckErrorMod;

    std::vector<int> limbs(cc.L + cc.K);
    std::iota(limbs.begin(), limbs.end(), 0);
    std::for_each(std::execution::par_unseq, limbs.begin(), limbs.end(), [&](int i) {
        cudaSetDevice(GPUs[0]);
        FIDESlib::Stream s;
        s.init();

        FIDESlib::CKKS::Limb<uint32_t> limb(cc, GPUs[0], s, i);
        FIDESlib::CKKS::Limb<uint32_t> limb2(cc, GPUs[0], s, i);

        std::vector<uint32_t> v(cc.N, 1);
        std::vector<uint32_t> v2(cc.N, 0);
        int errores = 0;
        for (uint64_t j = 0; j < hC_.primes[i] && j < 1000000; j += cc.N) {
            for (uint64_t k = 0; k < (uint64_t)cc.N; ++k) {
                v2[k] = k + j;
            }

            limb.load(v);
            limb2.load(v2);

            std::vector<uint32_t> v3(v);
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < cc.N; ++k)
                    v3[k] = ((__uint128_t)v3[k]) * ((__uint128_t)v2[k]) % ((__uint128_t)hC_.primes[i]);

                FIDESlib::mult_<uint32_t, FIDESlib::ALGO_NATIVE>
                    <<<cc.N / 256, 256, 0, limb.stream.ptr>>>(limb.v.data, limb2.v.data, i);

                limb.store(v);
                FIDESlib::CudaHostSync();
                CudaCheckErrorMod;
                for (int i = 0; i < cc.N; ++i)
                    if (v[i] != v3[i])
                        ++errores;

                ASSERT_EQ(v, v3);
            }
        }
        std::cout << "errores: " << errores << std::endl;
    });

    CudaCheckErrorMod;
}

TEST_P(LimbKernelTest32, TestBetterBarretMultKernel32) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{params32_15, GPUs};
    CudaCheckErrorMod;

    for (int i = 0; i <= cc.L + cc.K; ++i)
        for (int l : {0}) {
            cudaSetDevice(GPUs[0]);
            FIDESlib::Stream s;
            s.init();

            FIDESlib::CKKS::Limb<uint32_t> limb(cc, GPUs[0], s, i);
            FIDESlib::CKKS::Limb<uint32_t> limb2(cc, GPUs[0], s, i);

            std::vector<uint32_t> v(cc.N, 10);
            std::vector<uint32_t> v2(cc.N, 0);
            for (auto& i : v2)
                i = rand() % hC_.primes[i];

            limb.load(v);
            limb2.load(v2);

            std::vector<uint32_t> v3(v);
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < cc.N; ++k)
                    v3[k] = ((__uint128_t)v3[k]) * ((__uint128_t)v2[k]) % ((__uint128_t)hC_.primes[i]);

                FIDESlib::mult_<uint32_t, FIDESlib::ALGO_BARRETT><<<cc.N / 256, 256, 0>>>(limb.v.data, limb2.v.data, i);

                limb.store(v);
                FIDESlib::CudaHostSync();
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
                for (int k = 0; k < cc.N; ++k)
                    if (v[k] != v3[k]) {
                        std::cout << i << ":" << k << " " << v[k] << " " << v3[k] << std::endl;
                    }
                ASSERT_EQ(v, v3);
            }

            // destructor implícito
        }
    CudaCheckErrorMod;
}

/*
TEST(LimbKernelTests, TestShoupMultKernel64) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i) GPUs.push_back(i);

    Context cc{params, GPUs};
    CudaCheckErrorMod;

    for (int l: {0}) {
        cudaSetDevice(GPUs[0]);
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

        Limb<uint64_t> limb(cc, 0, 0);
        Limb<uint64_t> limb2(cc, 0, 0);

        uint64_t val2 = 420000000;
        std::vector<uint64_t> v(cc.N, 10);
        std::vector<uint64_t> v2(cc.N, val2);

        uint64_t shoup_mu = shoup_const(val2, 0);


        limb.load(v);
        limb2.load(v2);


        std::vector<uint64_t> v3(v);
        for(int j = 0; j < 4; ++j) {
            for (int i = 0; i < cc.N; ++i) v3[i] = ((__uint128_t)v3[i]) * ((__uint128_t)v2[i]) % ((__uint128_t)cc.prime[0].p);

            mult_<uint64_t, 2><<<cc.N / 256, 256, 0>>>(limb.v.data, limb2.v.data, 0,
                                                          shoup_mu );


            limb.store(v);
            CudaHostSync();
            cudaDeviceSynchronize();
            CudaCheckErrorMod;
            for(int i = 0; i < 1; ++i) if(v[i] != v3[i]) std::cout << i << std::endl;
            ASSERT_EQ(v, v3);
        }

        limb.free();
        limb2.free();
// destructor implícito
    }
    CudaCheckErrorMod;
}
*/

INSTANTIATE_TEST_SUITE_P(LimbKernelTests, LimbKernelTest,
                         testing::Values(params64_13, params64_14, params64_15, params64_16));

INSTANTIATE_TEST_SUITE_P(LimbKernelTests, LimbKernelTest32, testing::Values(params32_15));
}  // namespace FIDESlib::Testing