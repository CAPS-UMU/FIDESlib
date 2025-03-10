//
// Created by oscar on 22/10/24.
//

#include <benchmark/benchmark.h>

#include "Benchmark.cuh"
#include "CKKS/RNSPoly.cuh"
namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyAdd)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_limbs"] = state.range(1);
    for (auto _ : state) {
        if (cc.L <= state.range(1)) {
            state.SkipWithMessage("cc.L <= initial levels");
            break;
        }
        FIDESlib::CKKS::RNSPoly a(cc, state.range(1));
        FIDESlib::CKKS::RNSPoly b(cc, state.range(1));

        auto start = std::chrono::high_resolution_clock::now();
        a.add(b);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyAddContextLimbCount)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    CudaCheckErrorMod;
    for (auto _ : state) {
        FIDESlib::CKKS::RNSPoly a(cc, cc.L);
        FIDESlib::CKKS::RNSPoly b(cc, cc.L);

        auto start = std::chrono::high_resolution_clock::now();
        a.add(b);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyMultiAdd)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_limbs"] = state.range(1);
    for (auto _ : state) {
        if (cc.L <= state.range(1)) {
            state.SkipWithMessage("cc.L <= initial levels");
            break;
        }
        FIDESlib::CKKS::RNSPoly a(cc, state.range(1));
        FIDESlib::CKKS::RNSPoly b(cc, state.range(1));

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i)
            a.add(b);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyMultiAddContextLimbCount)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    for (auto _ : state) {
        FIDESlib::CKKS::RNSPoly a(cc, cc.L);
        FIDESlib::CKKS::RNSPoly b(cc, cc.L);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i)
            a.add(b);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolySub)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    CudaCheckErrorMod;
    state.counters["p_limbs"] = state.range(1);
    for (auto _ : state) {
        if (cc.L <= state.range(1)) {
            state.SkipWithMessage("cc.L <= initial levels");
            break;
        }
        FIDESlib::CKKS::RNSPoly a(cc, state.range(1));
        FIDESlib::CKKS::RNSPoly b(cc, state.range(1));

        auto start = std::chrono::high_resolution_clock::now();
        a.sub(b);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolySubContextLimbCount)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    CudaCheckErrorMod;
    for (auto _ : state) {
        FIDESlib::CKKS::RNSPoly a(cc, cc.L);
        FIDESlib::CKKS::RNSPoly b(cc, cc.L);

        auto start = std::chrono::high_resolution_clock::now();
        a.sub(b);
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
    CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyAdd)->ArgsProduct({{1}, {0, 1, 8, 16}})->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyAddContextLimbCount)->ArgsProduct({{1}})->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyMultiAdd)->ArgsProduct({{1}, {0, 1, 8, 16}})->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyMultiAddContextLimbCount)->ArgsProduct({{1}})->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolySub)->ArgsProduct({{1}, {0, 1, 8, 16}})->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolySubContextLimbCount)->ArgsProduct({{1}})->UseManualTime();

}  // namespace FIDESlib::Benchmarks