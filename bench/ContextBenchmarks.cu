//
// Created by oscar on 22/10/24.
//

#include <benchmark/benchmark.h>

#include "Benchmark.cuh"
#include "CKKS/Context.cuh"

namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(FIDESlibFixture, ContextCreation)(benchmark::State& state) {
	for (auto _ : state) {
		FIDESlib::CKKS::Context c = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams, { 0 });
	}
}

BENCHMARK_REGISTER_F(FIDESlibFixture, ContextCreation)->ArgsProduct({ PARAMETERS });

} // namespace FIDESlib::Benchmarks