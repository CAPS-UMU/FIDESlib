//
// Created by oscar on 21/10/24.
//

#include <benchmark/benchmark.h>

#include "Benchmark.cuh"
#include "CKKS/KeySwitchingKey.cuh"

namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(GeneralFixture, CiphertextAdd)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc //{fideslibParams.adaptTo(raw_param), GPUs};
	  = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2, 1, state.range(3));

	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["limbs"]	  = GPUcc->param.L - state.range(3);
	CudaCheckErrorMod;
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++) {
			GPUct1.add(GPUct2);
		}
		auto end_cpu = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count() / 100);
		state.counters["cpu"] = std::chrono::duration_cast<std::chrono::duration<double>>(end_cpu - start).count() / 100;
		state.counters["gpu"] = elapsed.count() / 100;
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, AddPlaintext)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
	auto c1					  = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawPlainText raw2  = FIDESlib::CKKS::GetRawPlainText(cc, ptxt1);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Plaintext GPUpt2(GPUcc, raw2);
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["limbs"]	  = GPUcc->param.L - state.range(3);

	CudaCheckErrorMod;
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		for (auto i = 0; i < 100; i++) {
			GPUct1.addPt(GPUpt2);
		}
		auto end_cpu = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count() / 100);
		state.counters["cpu"] = std::chrono::duration_cast<std::chrono::duration<double>>(end_cpu - start).count() / 100;
		state.counters["gpu"] = elapsed.count() / 100;
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, AddScalar)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));

	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["limbs"]	  = GPUcc->param.L - state.range(3);

	CudaCheckErrorMod;
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		for (auto i = 0; i < 100; i++) {
			GPUct1.addScalar(1.00123123);
		}
		auto end_cpu = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count() / 100);
		state.counters["cpu"] = std::chrono::duration_cast<std::chrono::duration<double>>(end_cpu - start).count() / 100;
		state.counters["gpu"] = elapsed.count() / 100;
	}
	CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(GeneralFixture, CiphertextAdd)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG });
BENCHMARK_REGISTER_F(GeneralFixture, AddPlaintext)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG });
BENCHMARK_REGISTER_F(GeneralFixture, AddScalar)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG });

/// TDPS Experiments

BENCHMARK_REGISTER_F(GeneralFixture, CiphertextAdd)->ArgsProduct({ { 30, 31 }, { 0 }, BATCH_CONFIG, { 0, 12 } })->UseManualTime()->Iterations(10);
BENCHMARK_REGISTER_F(GeneralFixture, AddPlaintext)->ArgsProduct({ { 30, 31 }, { 0 }, BATCH_CONFIG, { 0, 12 } })->UseManualTime()->Iterations(10);
BENCHMARK_REGISTER_F(GeneralFixture, AddScalar)->ArgsProduct({ { 30, 31 }, { 0 }, BATCH_CONFIG, { 0, 12 } })->UseManualTime()->Iterations(10);

} // namespace FIDESlib::Benchmarks