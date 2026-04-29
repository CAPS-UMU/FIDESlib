//
// Created by oscar on 21/10/24.
//

#include <benchmark/benchmark.h>

#include "Benchmark.cuh"

namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(GeneralFixture, MultPlaintext)(benchmark::State& state) {
	if (this->generalTestParams.multDepth < static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L < level");
		return;
	}
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
	FIDESlib::CKKS::RawPlainText raw2  = FIDESlib::CKKS::GetRawPlainText(cc, ptxt1);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Plaintext GPUpt2(GPUcc, raw2);
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["limbs"]	  = GPUcc->param.L - state.range(3);
	CudaCheckErrorMod;
	{
		for (auto _ : state) {
			auto start = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < 200; i++) {
				GPUct1.multPt(GPUpt2, false);
			}
			auto end_cpu = std::chrono::high_resolution_clock::now();
			CudaCheckErrorMod;
			auto end	 = std::chrono::high_resolution_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
			state.SetIterationTime(elapsed.count() / 200);
			state.counters["cpu"] = std::chrono::duration_cast<std::chrono::duration<double>>(end_cpu - start).count() / 200;
			state.counters["gpu"] = elapsed.count() / 200;
		}
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, Rescale)(benchmark::State& state) {
	if (this->generalTestParams.multDepth < static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L < level");
		return;
	}
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 2, state.range(3));

	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["limbs"]	  = GPUcc->param.L - state.range(3);

	CudaCheckErrorMod;
	{
		for (auto _ : state) {
			auto start = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < 200; i++) {
				GPUct1.rescale();
				GPUct1.c0.grow(GPUct1.c0.getLevel() + 1);
				GPUct1.c1.grow(GPUct1.c1.getLevel() + 1);
				GPUct1.NoiseLevel = 2;
			}
			auto end_cpu = std::chrono::high_resolution_clock::now();
			CudaCheckErrorMod;
			auto end	 = std::chrono::high_resolution_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
			state.SetIterationTime(elapsed.count() / 200);
			state.counters["cpu"] = std::chrono::duration_cast<std::chrono::duration<double>>(end_cpu - start).count() / 200;
			state.counters["gpu"] = elapsed.count() / 200;
		}
	}

	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, AdjustAddSub)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, state.range(4), state.range(3));
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, state.range(6), state.range(5));

	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);

	CudaCheckErrorMod;
	int its = 0;
	for (auto _ : state) {
		its++;

		auto start = std::chrono::high_resolution_clock::now();
		if (!GPUct1.adjustForAddOrSub(GPUct2)) {
			CKKS::Ciphertext b_(GPUcc);
			b_.copy(GPUct2);
			if (!b_.adjustForAddOrSub(GPUct1)) {
				state.SkipWithMessage("Can't adjust");
				return;
			}
		}
		if constexpr (SYNC)
			CudaCheckErrorMod;
		else if (its % 100 == 99)
			CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		GPUct1.load(raw1);
		GPUct2.load(raw2);
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, AdjustMult)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, state.range(4), state.range(3));
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, state.range(6), state.range(5));

	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;
	int its = 0;
	for (auto _ : state) {
		its++;

		auto start = std::chrono::high_resolution_clock::now();
		if (!GPUct1.adjustForMult(GPUct2)) {
			CKKS::Ciphertext b_(GPUcc);
			b_.copy(GPUct2);
			if (!b_.adjustForMult(GPUct1)) {
				state.SkipWithMessage("Can't adjust");
				return;
			}
		}
		if constexpr (SYNC)
			CudaCheckErrorMod;
		else if (its % 100 == 99)
			CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		GPUct1.load(raw1);
		GPUct2.load(raw2);
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, AdjustPlaintext)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, state.range(4), state.range(3));
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, state.range(6), state.range(5));

	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	// auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawPlainText raw2  = FIDESlib::CKKS::GetRawPlainText(cc, ptxt2);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Plaintext b(GPUcc, raw2);

	state.counters["p_batch"] = state.range(2);
	state.counters["p_limbs"] = state.range(3);
	CudaCheckErrorMod;
	int its = 0;
	for (auto _ : state) {
		its++;

		auto start = std::chrono::high_resolution_clock::now();
		if (GPUcc->rescaleTechnique == CKKS::FLEXIBLEAUTO || GPUcc->rescaleTechnique == CKKS::FLEXIBLEAUTOEXT || GPUcc->rescaleTechnique == CKKS::FIXEDAUTO) {
			if (b.c0.getLevel() != GPUct1.getLevel() || b.NoiseLevel == 2 || (b.NoiseLevel == 1 && GPUct1.NoiseLevel == 2) /*!hasSameScalingFactor(b)*/) {
				CKKS::Plaintext b_(GPUcc);
				if (!b_.adjustPlaintextToCiphertext(b, GPUct1)) {
					state.SkipWithMessage("Can't adjust");
					return;
				}
			}
		}
		if constexpr (SYNC)
			CudaCheckErrorMod;
		else if (its % 100 == 99)
			CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		GPUct1.load(raw1);
		b.load(raw2);
	}
	CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(GeneralFixture, MultPlaintext)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, Rescale)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG })->UseManualTime();

/// TDPS Experiments

BENCHMARK_REGISTER_F(GeneralFixture, MultPlaintext)->ArgsProduct({ { 30, 31 }, { 0 }, BATCH_CONFIG, { 0, 12 } })->UseManualTime()->Iterations(10);
BENCHMARK_REGISTER_F(GeneralFixture, Rescale)->ArgsProduct({ { 30, 31 }, { 0 }, BATCH_CONFIG, { 0, 12 } })->UseManualTime()->Iterations(10);

} // namespace FIDESlib::Benchmarks