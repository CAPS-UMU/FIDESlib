//
// Created by carlosad on 19/11/24.
//
#include "Benchmark.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/KeySwitchingKey.cuh"

namespace FIDESlib::Benchmarks {

BENCHMARK_DEFINE_F(GeneralFixture, ApproxModReduction)(benchmark::State& state) {
    if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
        state.SkipWithMessage("cc.L <= level");
        return;
    }

    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs = generalTestParams.GPUs;

    state.counters["p_batch"] = state.range(2);
    state.counters["p_limbs"] = state.range(3);

    fideslibParams.batch = state.range(2);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), GPUs};

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2, 1, state.range(3));

    ptxt1->SetLevel(state.range(3));
    ptxt2->SetLevel(state.range(3));
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
    FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    kskEval.Initialize(GPUcc, rawKskEval);

    for (auto _ : state) {
        CudaCheckErrorMod;
        auto start = std::chrono::high_resolution_clock::now();
        FIDESlib::CKKS::approxModReduction(GPUct1, GPUct2, kskEval, 1.0);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());

        GPUct1.c0.grow(GPUcc.L - state.range(3));
        GPUct1.c1.grow(GPUcc.L - state.range(3));
        GPUct2.c0.grow(GPUcc.L - state.range(3));
        GPUct2.c1.grow(GPUcc.L - state.range(3));
    }
    CudaCheckErrorMod;
    cc->GetEvalAutomorphismKeyMap(this->keys.publicKey->GetKeyTag()).clear();
}

BENCHMARK_DEFINE_F(GeneralFixture, ApproxModReductionSparse)(benchmark::State& state) {
    if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
        state.SkipWithMessage("cc.L <= level");
        return;
    }

    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs = generalTestParams.GPUs;

    state.counters["p_batch"] = state.range(2);
    state.counters["p_limbs"] = state.range(3);
    fideslibParams.batch = state.range(2);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), GPUs};

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));

    ptxt1->SetLevel(state.range(3));
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    kskEval.Initialize(GPUcc, rawKskEval);

    for (auto _ : state) {
        CudaCheckErrorMod;
        auto start = std::chrono::high_resolution_clock::now();
        FIDESlib::CKKS::approxModReductionSparse(GPUct1, kskEval, 1.0);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());

        GPUct1.c0.grow(GPUcc.L - state.range(3));
        GPUct1.c1.grow(GPUcc.L - state.range(3));
    }
    CudaCheckErrorMod;
    cc->GetEvalAutomorphismKeyMap(this->keys.publicKey->GetKeyTag()).clear();
}

BENCHMARK_DEFINE_F(GeneralFixture, CoeffsToSlots)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs = generalTestParams.GPUs;

    state.counters["p_batch"] = state.range(2);
    state.counters["p_slots"] = state.range(3);
    fideslibParams.batch = state.range(2);
    const int slots = state.range(3);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), GPUs};

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    cc->EvalBootstrapSetup({2, 2}, {0, 0}, slots);
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);
    const int start_level = GPUcc.GetBootPrecomputation(slots).CtS.at(0).A.at(0).c0.getLevel();
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - start_level);

    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    {
        FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
        FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
        kskEval.Initialize(GPUcc, rawKskEval);
        GPUcc.AddEvalKey(std::move(kskEval));
    }

    for (auto _ : state) {
        CudaCheckErrorMod;
        auto start = std::chrono::high_resolution_clock::now();

        FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, false);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());

        GPUct1.c0.grow(start_level);
        GPUct1.c1.grow(start_level);
    }
    CudaCheckErrorMod;
    cc->GetEvalAutomorphismKeyMap(this->keys.publicKey->GetKeyTag()).clear();
}

BENCHMARK_DEFINE_F(GeneralFixture, SlotsToCoeffs)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs = generalTestParams.GPUs;

    state.counters["p_batch"] = state.range(2);
    state.counters["p_slots"] = state.range(3);

    fideslibParams.batch = state.range(2);
    const int slots = state.range(3);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), GPUs};

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    cc->EvalBootstrapSetup({2, 2}, {0, 0}, slots);
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

    const int init_level = GPUcc.GetBootPrecomputation(slots).StC.at(0).A.at(0).c0.getLevel();
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - init_level);
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    {
        FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
        FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
        kskEval.Initialize(GPUcc, rawKskEval);
        GPUcc.AddEvalKey(std::move(kskEval));
    }

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();

        FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, true);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());

        GPUct1.c0.grow(init_level);
        GPUct1.c1.grow(init_level);
    }
    CudaCheckErrorMod;
    cc->GetEvalAutomorphismKeyMap(this->keys.publicKey->GetKeyTag()).clear();
}

struct BootConfig {
    uint32_t slots, a, b, dim1, dim2;
};

BootConfig conf[] = {BootConfig{1 << 6, 1, 1, 16, 16},  BootConfig{1 << 9, 2, 2, 16, 16},
                     BootConfig{1 << 14, 5, 5, 8, 8},   BootConfig{1 << 15, 5, 5, 8, 8},
                     BootConfig{1 << 14, 4, 4, 8, 8},   BootConfig{1 << 15, 4, 4, 8, 8},
                     BootConfig{1 << 14, 3, 3, 16, 16}, BootConfig{1 << 15, 3, 3, 16, 16},
                     BootConfig{1 << 16, 4, 4, 16, 16}, BootConfig{1 << 16, 3, 3, 16, 16}};

#include <openfhe/pke/openfhe.h>

BENCHMARK_DEFINE_F(GeneralFixture, BootstrapGPU)(benchmark::State& state) {
    CudaCheckErrorMod;
    std::vector<int> GPUs = generalTestParams.GPUs;

    state.counters["p_batch"] = state.range(2);
    state.counters["p_slots"] = conf[state.range(3)].slots;

    fideslibParams.batch = state.range(2);
    const int slots = conf[state.range(3)].slots;
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    cudaSetDevice(GPUs[0]);
    CudaCheckErrorMod;
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), GPUs};

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

    cc->EvalBootstrapSetup({conf[state.range(3)].a, conf[state.range(3)].b},
                           {conf[state.range(3)].dim1, conf[state.range(3)].dim2}, slots);
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

    const int init_level = 1;
    ptxt1->SetLevel(GPUcc.L - init_level);
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    std::cout << "LoadEvalKey" << std::endl;
    {
        FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
        FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
        kskEval.Initialize(GPUcc, rawKskEval);
        GPUcc.AddEvalKey(std::move(kskEval));
    }

    int endlevel = 0;

    for (auto _ : state) {
        // std::cout << "Boot" << std::endl;
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        FIDESlib::CKKS::Bootstrap(GPUct1, slots, false);
        CudaCheckErrorMod;
        // std::cout << "end1" << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
        endlevel = GPUct1.getLevel();
        GPUct1.c0.grow(init_level + 1);
        GPUct1.c1.grow(init_level + 1);
        //  std::cout << "end" << std::endl;
    }
    std::cout << "Remaining levels: " << endlevel << std::endl;
    CudaCheckErrorMod;
    cc->GetEvalAutomorphismKeyMap(this->keys.publicKey->GetKeyTag()).clear();
}

BENCHMARK_REGISTER_F(GeneralFixture, ApproxModReduction)
    ->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, {0, 1, 2, 3, 4, 5}})
    ->Iterations(50);
BENCHMARK_REGISTER_F(GeneralFixture, ApproxModReductionSparse)
    ->ArgsProduct({PARAMETERS, {0}, BATCH_CONFIG, {0, 1, 2, 3, 4, 5}})
    ->Iterations(50);
BENCHMARK_REGISTER_F(GeneralFixture, CoeffsToSlots)->ArgsProduct({{3}, {0}, BATCH_CONFIG, {64}})->Iterations(50);
BENCHMARK_REGISTER_F(GeneralFixture, SlotsToCoeffs)->ArgsProduct({{3}, {0}, BATCH_CONFIG, {64}})->Iterations(50);

BENCHMARK_REGISTER_F(GeneralFixture, BootstrapGPU)
    ->ArgsProduct({{3, 4}, {0}, BATCH_CONFIG, {0, 1, 2, 3, 4, 5, 6, 7}})
    ->Iterations(50);

BENCHMARK_REGISTER_F(GeneralFixture, BootstrapGPU)
    ->ArgsProduct({{6}, {0}, BATCH_CONFIG, {0, 1, 3, 5, 7, 8, 9}})
    ->Iterations(50);

BENCHMARK_DEFINE_F(GeneralFixture, BootstrapCPU)(benchmark::State& state) {

    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs = generalTestParams.GPUs;

    state.counters["p_batch"] = state.range(2);
    state.counters["p_slots"] = conf[state.range(3)].slots;

    fideslibParams.batch = state.range(2);
    const int slots = conf[state.range(3)].slots;
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), GPUs};

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, slots);

    cc->EvalBootstrapSetup({conf[state.range(3)].a, conf[state.range(3)].b}, {0, 0}, slots);
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    //FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

    //const int init_level = GPUcc.GetBootPrecomputation(slots).StC.at(0).A.at(0).c0.getLevel();
    ////ptxt1->SetLevel(GPUcc.L - init_level);
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

    //FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    //FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    {
        //FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
        //FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
        //kskEval.Initialize(GPUcc, rawKskEval);
        // GPUcc.AddEvalKey(std::move(kskEval));
    }

    int endlevel = 0;

    for (auto _ : state) {
        auto ct = c1->Clone();

        auto start = std::chrono::high_resolution_clock::now();
        ct = cc->EvalBootstrap(ct);
        //FIDESlib::CKKS::Bootstrap(GPUct1, slots);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    std::cout << "Remaining levels: " << endlevel << std::endl;
    CudaCheckErrorMod;
    cc->GetEvalAutomorphismKeyMap(this->keys.publicKey->GetKeyTag()).clear();
}
BENCHMARK_REGISTER_F(GeneralFixture, BootstrapCPU)->ArgsProduct({{3}, {0}, {1}, {0, 1, 6, 7}})->Iterations(3);

//BENCHMARK_REGISTER_F(GeneralFixture, BootstrapCPU)->ArgsProduct({{4, 3}, {0}, {2, 6, 12}, {4, 5}});

}  // namespace FIDESlib::Benchmarks
