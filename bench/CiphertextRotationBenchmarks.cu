//
// Created by carlosad on 5/11/24.
//
#include <benchmark/benchmark.h>

#include "Benchmark.cuh"
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(GeneralFixture, CiphertextRotation)(benchmark::State& state) {
    if (this->generalTestParams.multDepth < static_cast<uint64_t>(state.range(3))) {
        state.SkipWithMessage("cc.L < level");
        return;
    }

    int devcount = -1;
    cudaGetDeviceCount(&devcount);
    std::vector<int> GPUs = generalTestParams.GPUs;

    fideslibParams.batch = state.range(2);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), GPUs};

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
    ptxt1->SetLevel(state.range(3));
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    FIDESlib::CKKS::KeySwitchingKey kskRot(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);
    kskRot.Initialize(GPUcc, rawKskEval);

    state.counters["p_batch"] = state.range(2);
    state.counters["p_limbs"] = state.range(3);
    for (auto _ : state) {
        GPUct1.rotate(2 * GPUcc.N - 1, kskRot, true);
        CudaCheckErrorMod;
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, CiphertextHoistedRotation)(benchmark::State& state) {
    if (this->generalTestParams.multDepth < static_cast<uint64_t>(state.range(3))) {
        state.SkipWithMessage("cc.L < level");
        return;
    }

    int devcount = -1;
    cudaGetDeviceCount(&devcount);
    std::vector<int> GPUs = generalTestParams.GPUs;

    fideslibParams.batch = state.range(2);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), GPUs};

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
    ptxt1->SetLevel(state.range(3));
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
    FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw1);
    FIDESlib::CKKS::Ciphertext GPUct3(GPUcc, raw1);
    FIDESlib::CKKS::Ciphertext GPUct4(GPUcc, raw1);

    FIDESlib::CKKS::KeySwitchingKey kskRot1(GPUcc);
    FIDESlib::CKKS::KeySwitchingKey kskRot2(GPUcc);
    FIDESlib::CKKS::KeySwitchingKey kskRot3(GPUcc);
    FIDESlib::CKKS::KeySwitchingKey kskRot4(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval1 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);
    kskRot1.Initialize(GPUcc, rawKskEval1);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval2 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 2, cc);
    kskRot2.Initialize(GPUcc, rawKskEval2);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval3 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 3, cc);
    kskRot3.Initialize(GPUcc, rawKskEval3);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval4 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 4, cc);
    kskRot4.Initialize(GPUcc, rawKskEval4);

    state.counters["p_batch"] = state.range(2);
    state.counters["p_limbs"] = state.range(3);
    for (auto _ : state) {
        GPUct1.rotate_hoisted({&kskRot1, &kskRot2, &kskRot3, &kskRot4}, {1, 2, 3, 4},
                              {&GPUct2, &GPUct3, &GPUct4, &GPUct1}, false);
        CudaCheckErrorMod;
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, CiphertextRotateAndAccumulate)(benchmark::State& state) {
    if (this->generalTestParams.multDepth < static_cast<uint64_t>(state.range(3))) {
        state.SkipWithMessage("cc.L < level");
        return;
    }

    int devcount = -1;
    cudaGetDeviceCount(&devcount);
    std::vector<int> GPUs = generalTestParams.GPUs;

    fideslibParams.batch = state.range(2);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), GPUs};

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
    ptxt1->SetLevel(state.range(3));
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

    int bstep = state.range(4);
    std::vector<int> indexes = FIDESlib::CKKS::GetAccumulateRotationIndices(bstep, 1, GPUcc.N / 2);
    FIDESlib::CKKS::GenAndAddRotationKeys(cc, keys, GPUcc, indexes);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
    state.counters["p_batch"] = state.range(2);
    state.counters["p_limbs"] = state.range(3);
    for (auto _ : state) {

        FIDESlib::CKKS::Accumulate(GPUct1, bstep, 1, GPUcc.N / 2);
        CudaCheckErrorMod;
    }
    CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(GeneralFixture, CiphertextRotation)->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG});
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextHoistedRotation)
    ->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG});

BENCHMARK_REGISTER_F(GeneralFixture, CiphertextRotateAndAccumulate)
    ->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, LEVEL_CONFIG, {2, 4, 8}});
}  // namespace FIDESlib::Benchmarks