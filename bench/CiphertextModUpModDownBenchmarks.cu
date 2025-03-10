//
// Created by oscar on 21/10/24.
//

#include <benchmark/benchmark.h>

#include "Benchmark.cuh"
namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(GeneralFixture, CiphertextModUp)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs = generalTestParams.GPUs;

    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::LEVELEDSHE);
    auto keys = cc->KeyGen();

    fideslibParams.batch = state.range(2);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), GPUs};

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    state.counters["p_batch"] = state.range(2);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        GPUct1.modUp();
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed.count());
        GPUct1.c0.generateSpecialLimbs();
        GPUct1.c1.generateSpecialLimbs();
        GPUct1.modDown();
    }

    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, CiphertextModDown)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs = generalTestParams.GPUs;

    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::LEVELEDSHE);
    auto keys = cc->KeyGen();

    fideslibParams.batch = state.range(2);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), GPUs};

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    state.counters["p_batch"] = state.range(2);
    for (auto _ : state) {
        GPUct1.modUp();
        GPUct1.c0.generateSpecialLimbs();
        GPUct1.c1.generateSpecialLimbs();
        auto start = std::chrono::high_resolution_clock::now();
        GPUct1.modDown();
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }

    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, CiphertextModUpModDown)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs = generalTestParams.GPUs;

    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::LEVELEDSHE);
    auto keys = cc->KeyGen();

    fideslibParams.batch = state.range(2);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), GPUs};

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    for (auto _ : state) {
        GPUct1.modUp();
        GPUct1.c0.generateSpecialLimbs();
        GPUct1.c1.generateSpecialLimbs();
        GPUct1.modDown();
        CudaCheckErrorMod;
    }

    CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(GeneralFixture, CiphertextModUp)->ArgsProduct({{0, 1, 2, 3}, {0}, BATCH_CONFIG})->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextModDown)->ArgsProduct({{0, 1, 2, 3}, {0}, BATCH_CONFIG})->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextModUpModDown)->ArgsProduct({{0, 1, 2, 3}, {0}, BATCH_CONFIG});

}  // namespace FIDESlib::Benchmarks