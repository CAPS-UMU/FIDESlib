//
// Created by oscar on 22/10/24.
//

#include "benchmark/benchmark.h"

#include "Benchmark.cuh"
namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(FIDESlibFixture, LimbBatchAdd)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    int n = state.range(1);
    state.counters["p_limbs"] = n;

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    std::vector<std::pair<FIDESlib::CKKS::Limb<uint32_t>, FIDESlib::CKKS::Limb<uint32_t>>> limb32;

    for (int i = 0; i < n; ++i) {
        limb32.emplace_back(FIDESlib::CKKS::Limb<uint32_t>(cc, GPUs[0], s, 0),
                            FIDESlib::CKKS::Limb<uint32_t>(cc, GPUs[0], s, 0));
        limb32.back().first.load(v);
        limb32.back().second.load(v);
    }

    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        for (auto& i : limb32)
            i.first.add(i.second);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbBatchAdd64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    int n = state.range(1);
    state.counters["p_limbs"] = n;

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    std::vector<std::pair<FIDESlib::CKKS::Limb<uint64_t>, FIDESlib::CKKS::Limb<uint64_t>>> limb64;

    for (int i = 0; i < n; ++i) {
        limb64.emplace_back(FIDESlib::CKKS::Limb<uint64_t>(cc, GPUs[0], s, 0),
                            FIDESlib::CKKS::Limb<uint64_t>(cc, GPUs[0], s, 0));
        limb64.back().first.load(v2);
        limb64.back().second.load(v2);
    }

    for (auto _ : state) {

        auto start = std::chrono::high_resolution_clock::now();
        for (auto& i : limb64)
            i.first.add(i.second);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbBatchSub)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    int n = state.range(1);
    state.counters["p_limbs"] = n;

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    std::vector<std::pair<FIDESlib::CKKS::Limb<uint32_t>, FIDESlib::CKKS::Limb<uint32_t>>> limb32;

    for (int i = 0; i < n; ++i) {
        limb32.emplace_back(FIDESlib::CKKS::Limb<uint32_t>(cc, GPUs[0], s, 0),
                            FIDESlib::CKKS::Limb<uint32_t>(cc, GPUs[0], s, 0));
        limb32.back().first.load(v);
        limb32.back().second.load(v);
    }

    for (auto _ : state) {

        auto start = std::chrono::high_resolution_clock::now();
        for (auto& i : limb32)
            i.first.sub(i.second);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbBatchSub64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    int n = state.range(1);
    state.counters["p_limbs"] = n;

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    std::vector<std::pair<FIDESlib::CKKS::Limb<uint64_t>, FIDESlib::CKKS::Limb<uint64_t>>> limb64;

    for (int i = 0; i < n; ++i) {
        limb64.emplace_back(FIDESlib::CKKS::Limb<uint64_t>(cc, GPUs[0], s, 0),
                            FIDESlib::CKKS::Limb<uint64_t>(cc, GPUs[0], s, 0));
        limb64.back().first.load(v2);
        limb64.back().second.load(v2);
    }

    CudaCheckErrorMod;

    for (auto _ : state) {

        auto start = std::chrono::high_resolution_clock::now();
        for (auto& i : limb64)
            i.first.sub(i.second);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(FIDESlibFixture, LimbBatchAdd)->ArgsProduct({{1}, {1, 8, 16, 32, 64, 128}})->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbBatchAdd64)->ArgsProduct({{1}, {1, 8, 16, 32, 64, 128}})->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbBatchSub)->ArgsProduct({{1}, {1, 8, 16, 32, 64, 128}})->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbBatchSub64)->ArgsProduct({{1}, {1, 8, 16, 32, 64, 128}})->UseManualTime();
}  // namespace FIDESlib::Benchmarks