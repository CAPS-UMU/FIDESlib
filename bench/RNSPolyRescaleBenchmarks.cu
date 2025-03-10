//
// Created by oscar on 22/10/24.
//

#include "benchmark/benchmark.h"

#include "Benchmark.cuh"
namespace FIDESlib::Benchmarks {

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyRescale)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);
    fideslibParams.batch = state.range(2);
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    CudaCheckErrorMod;
    state.counters["p_limbs"] = state.range(1);
    state.counters["p_batch"] = state.range(2);
    for (auto _ : state) {
        if (cc.L <= state.range(1)) {
            state.SkipWithMessage("cc.L <= initial levels");
            break;
        }
        FIDESlib::CKKS::RNSPoly a(cc, state.range(1));
        auto start = std::chrono::high_resolution_clock::now();
        a.rescale();
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
        CudaCheckErrorMod;
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyRescaleContextLimbCount)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    fideslibParams.batch = state.range(1);
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    state.counters["p_batch"] = state.range(1);
    CudaCheckErrorMod;
    FIDESlib::CKKS::RNSPoly a(cc, cc.L);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        a.rescale();
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
        a.grow(cc.L);
        CudaCheckErrorMod;
    }
    CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyRescale)
    ->ArgsProduct({{2, 3, 4, 5}, {1, 8, 16}, BATCH_CONFIG})
    ->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyRescaleContextLimbCount)
    ->ArgsProduct({{2, 3, 4, 5}, BATCH_CONFIG})
    ->UseManualTime();

}  // namespace FIDESlib::Benchmarks