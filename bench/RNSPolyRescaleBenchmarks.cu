//
// Created by oscar on 22/10/24.
//

#include "benchmark/benchmark.h"

#include "Benchmark.cuh"
#include "CKKS/Context.cuh"

namespace FIDESlib::Benchmarks {

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyRescale)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs;
	for (int i = 0; i < devcount; ++i)
		GPUs.push_back(i);
	fideslibParams.batch	   = state.range(2);
	FIDESlib::CKKS::Context cc = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams, GPUs);
	CudaCheckErrorMod;
	state.counters["p_limbs"] = state.range(1);
	state.counters["p_batch"] = state.range(2);
	for (auto _ : state) {
		if (cc->L <= state.range(1)) {
			state.SkipWithMessage("cc.L <= initial levels");
			break;
		}
		FIDESlib::CKKS::RNSPoly a(*cc, state.range(1));
		auto start = std::chrono::high_resolution_clock::now();
		a.rescale();
		if constexpr (SYNC)
			CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		if constexpr (SYNC)
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

	fideslibParams.batch	   = state.range(2);
	FIDESlib::CKKS::Context cc = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams, GPUs);
	state.counters["p_batch"]  = state.range(2);
	CudaCheckErrorMod;
	FIDESlib::CKKS::RNSPoly a(*cc, cc->L - state.range(3));
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		a.rescale();
		if constexpr (SYNC)
			CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		a.grow(cc->L);
		if constexpr (SYNC)
			CudaCheckErrorMod;
	}
	CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyRescale)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyRescaleContextLimbCount)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG })->UseManualTime();

} // namespace FIDESlib::Benchmarks