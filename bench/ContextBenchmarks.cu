//
// Created by oscar on 22/10/24.
//

#include <benchmark/benchmark.h>

#include "Benchmark.cuh"
namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(FIDESlibFixture, ContextCreation)(benchmark::State& state) {
    for (auto _ : state) {
        FIDESlib::CKKS::Context c(fideslibParams, {0});
    }
}

BENCHMARK_REGISTER_F(FIDESlibFixture, ContextCreation)->ArgsProduct({{2, 3, 4, 5}});

}  // namespace FIDESlib::Benchmarks