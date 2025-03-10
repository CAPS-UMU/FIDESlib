//
// Created by carlosad on 14/03/24.
//
#include <iomanip>

#include <gtest/gtest.h>

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Limb.cuh"
#include "CKKS/Parameters.cuh"
#include "CKKS/RNSPoly.cuh"
#include "ParametrizedTest.cuh"
#include "VectorGPU.cuh"

namespace FIDESlib::Testing {
class BaseTest : public FIDESlibParametrizedTest {};

TEST_P(BaseTest, ConstructVectorGPU) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);
    for (int i = 0; i < devcount; ++i) {
        cudaSetDevice(i);
        FIDESlib::Stream s;
        s.init();
        CudaCheckErrorMod;
        FIDESlib::VectorGPU<int> gpuvec(s, 1, i);

        gpuvec.free(s);

        // implicit destruction expected now
    }
    CudaCheckErrorMod;

    // Freeing in a different Stream.
    for (int i = 0; i < devcount; ++i) {
        cudaSetDevice(i);
        FIDESlib::Stream s;
        s.init();
        FIDESlib::Stream s2;
        s2.init();
        FIDESlib::VectorGPU<int> gpuvec(s, 1, i);

        cudaStreamSynchronize(s.ptr);

        gpuvec.free(s2);
        // implicit destruction expected now
    }
    CudaCheckErrorMod;

    for (int i = 0; i < devcount; ++i) {
        cudaSetDevice(i);
        int* ptr;
        cudaMalloc(&ptr, 1);

        FIDESlib::VectorGPU<int> gpuvec(ptr, 1, i);

        cudaFree(ptr);
        // implicit destruction expected now
    }
    CudaCheckErrorMod;

    for (int i = 0; i < devcount; ++i) {
        cudaSetDevice(i);
        int* ptr;
        cudaMalloc(&ptr, 10);

        FIDESlib::VectorGPU<int> gpuvec(ptr, 5, i, 5);

        cudaFree(ptr);
        // implicit destruction expected now
    }
    CudaCheckErrorMod;

    GTEST_ASSERT_TRUE(true);
}

TEST_P(BaseTest, ConstructLimb) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc(fideslibParams, GPUs);

    // Constructor memoria automática:
    for (int i = 0; i < devcount; ++i) {
        cudaSetDevice(i);
        FIDESlib::Stream s;
        s.init();

        FIDESlib::CKKS::Limb<uint32_t> limb(cc, i, s);
        FIDESlib::CKKS::Limb<uint64_t> limb2(cc, i, s);
        FIDESlib::CKKS::Limb<uint32_t> limb3(cc, i, s, 0);
        FIDESlib::CKKS::Limb<uint64_t> limb4(cc, i, s, 1);

        // implicit destruction expected now
    }
    CudaCheckErrorMod;

    // Constructor memoria manual:
    /*
        for (int i = 0; i < devcount; ++i) {
            cudaSetDevice(i);
            uint32_t *ptr;
            cudaMalloc(&ptr, cc.N * 6);

            Limb<uint32_t> limb(cc, ptr, i, 0);
            Limb<uint32_t> limb3(cc, ptr, i, cc.N, 0);
            Limb<uint64_t> limb2(cc, reinterpret_cast<uint64_t *>(ptr), i, cc.N);
            Limb<uint64_t> limb4(cc, reinterpret_cast<uint64_t *>(ptr), i, cc.N, 1);

            Stream s;
            s.init();

            limb.free();
            limb4.free();

            cudaFree(ptr);
            // implicit destruction expected now
        }
         */
    CudaCheckErrorMod;
}

TEST_P(BaseTest, ConstructRNSPoly) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs{0};
    //for (int i = 0; i < devcount; ++i) GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    FIDESlib::Stream s;
    FIDESlib::Stream s2;
    FIDESlib::Stream s3;
    s2.init();
    s3.init();
    FIDESlib::Stream s4;

    {
        FIDESlib::CKKS::RNSPoly poly(cc);
        // destructor implícito
    }

    {
        FIDESlib::CKKS::RNSPoly poly(cc, 2);
        // destructor implícito
    }

    {
        FIDESlib::CKKS::RNSPoly poly(cc, cc.L);
        // destructor implícito
    }

    CudaCheckErrorMod;
}

TEST_P(BaseTest, ConstructRNSPolySimulateMultiGPU) {

    std::vector<int> GPUs{0, 0};

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    {
        FIDESlib::CKKS::RNSPoly poly(cc);
        // destructor implícito
    }

    {
        FIDESlib::CKKS::RNSPoly poly(cc, 2);
        // destructor implícito
    }

    CudaCheckErrorMod;
}

TEST_P(BaseTest, ConstructCiphertext) {

    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;

    {
        FIDESlib::CKKS::Ciphertext c(cc);
        // Implicit destructor
    }

    CudaCheckErrorMod;
}

INSTANTIATE_TEST_SUITE_P(BaseTests, BaseTest,
                         testing::Values(params64_13, params64_14, params64_15, params64_16, params32_15));
}  // namespace FIDESlib::Testing
