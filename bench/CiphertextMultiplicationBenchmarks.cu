//
// Created by oscar on 21/10/24.
//

#include <benchmark/benchmark.h>

#include "Benchmark.cuh"
#include "CKKS/KeySwitchingKey.cuh"

namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(GeneralFixture, CiphertextMultiplication)(benchmark::State& state) {

	if (this->generalTestParams.multDepth < static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L < level");
		return;
	}

	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	{
		FIDESlib::CKKS::Context GPUcc = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

		std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
		std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

		lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, state.range(3));
		lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2, 1, state.range(3));

		ptxt1->SetLevel(state.range(3));
		ptxt2->SetLevel(state.range(3));
		auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
		auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

		FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
		FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);
		{

			FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
			FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
			kskEval.Initialize(rawKskEval);
			GPUcc->AddEvalKey(std::move(kskEval));
			{
				FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
				{
					FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);
					{
						state.counters["config"] = state.range(0);

						state.counters["p_batch"] = state.range(2);
						state.counters["limbs"]	  = GPUcc->param.L - state.range(3);
						CudaCheckErrorMod;
						for (auto _ : state) {
							auto start = std::chrono::high_resolution_clock::now();
							for (auto i = 0; i < 100; i++) {
								GPUct1.mult(GPUct2, false);
								GPUct1.NoiseFactor = GPUct2.NoiseFactor;
								GPUct1.NoiseLevel  = 1;
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
					CudaCheckErrorMod;
				}
				CudaCheckErrorMod;
			}
			CudaCheckErrorMod;
		}
		CudaCheckErrorMod;
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, CiphertextSquaring)(benchmark::State& state) {
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

	ptxt1->SetLevel(state.range(3));
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	kskEval.Initialize(rawKskEval);
	GPUcc->AddEvalKey(std::move(kskEval));
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["limbs"]	  = GPUcc->param.L - state.range(3);
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		for (auto i = 0; i < 100; i++) {
			GPUct1.square(false);
			GPUct1.NoiseLevel = 1;
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

BENCHMARK_DEFINE_F(GeneralFixture, MultScalar)(benchmark::State& state) {
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

	ptxt1->SetLevel(state.range(3));
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
			GPUct1.multScalar(1.01231331, false);
			GPUct1.NoiseLevel = 1;
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

BENCHMARK_REGISTER_F(GeneralFixture, CiphertextMultiplication)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG });
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextSquaring)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG });
BENCHMARK_REGISTER_F(GeneralFixture, MultScalar)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG });

/// TDPS Experiments

BENCHMARK_REGISTER_F(GeneralFixture, CiphertextMultiplication)->ArgsProduct({ { 30, 31 }, { 0 }, BATCH_CONFIG, { 0, 12 } })->UseManualTime()->Iterations(10);
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextSquaring)->ArgsProduct({ { 30, 31 }, { 0 }, BATCH_CONFIG, { 0, 12 } })->UseManualTime()->Iterations(10);
BENCHMARK_REGISTER_F(GeneralFixture, MultScalar)->ArgsProduct({ { 30, 31 }, { 0 }, BATCH_CONFIG, { 0, 12 } })->UseManualTime()->Iterations(10);

} // namespace FIDESlib::Benchmarks