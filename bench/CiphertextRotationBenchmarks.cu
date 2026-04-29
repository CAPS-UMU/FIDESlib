//
// Created by carlosad on 5/11/24.
//
#include <benchmark/benchmark.h>

#include "Benchmark.cuh"
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(GeneralFixture, CiphertextRotation)(benchmark::State& state) {
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
	// Ensure ciphertext uses the requested level (drop extra DCRT towers so
	// GetRawCipherText picks the correct number of limbs)
	if (c1) {
		size_t currentTowers = c1->GetElements()[0].GetNumOfElements();
		size_t currentLevel	 = c1->GetLevel();
		size_t totalPrimes	 = currentTowers + currentLevel;
		size_t targetTowers	 = totalPrimes - static_cast<size_t>(state.range(3));
		if (currentTowers > targetTowers) {
			size_t towersToDrop = currentTowers - targetTowers;
			for (auto& elem : c1->GetElements()) {
				elem.DropLastElements(towersToDrop);
			}
		}
		c1->SetLevel(static_cast<size_t>(state.range(3)));
	}

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	CKKS::GenAndAddRotationKeys(cc, keys, GPUcc, { 1 });
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["limbs"]	  = GPUcc->param.L - state.range(3);
	CudaCheckErrorMod;
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++) {
			GPUct1.rotate(1, true);
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

BENCHMARK_DEFINE_F(GeneralFixture, CiphertextHoistedRotation)(benchmark::State& state) {
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
	// Ensure ciphertext uses the requested level (drop extra DCRT towers so
	// GetRawCipherText picks the correct number of limbs)
	if (c1) {
		size_t currentTowers = c1->GetElements()[0].GetNumOfElements();
		size_t currentLevel	 = c1->GetLevel();
		size_t totalPrimes	 = currentTowers + currentLevel;
		size_t targetTowers	 = totalPrimes - static_cast<size_t>(state.range(3));
		if (currentTowers > targetTowers) {
			size_t towersToDrop = currentTowers - targetTowers;
			for (auto& elem : c1->GetElements()) {
				elem.DropLastElements(towersToDrop);
			}
		}
		c1->SetLevel(static_cast<size_t>(state.range(3)));
	}

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct3(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct4(GPUcc, raw1);

	CKKS::GenAndAddRotationKeys(cc, keys, GPUcc, { 1, 2, 3, 4 });
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["limbs"]	  = GPUcc->param.L - state.range(3);
	CudaCheckErrorMod;
	// hoisted rotation runs multiple rotations per call; normalize timings per rotation
	std::vector<int> rot_indexes					  = { 1, 2, 3, 4 };
	std::vector<FIDESlib::CKKS::Ciphertext*> rot_outs = { &GPUct2, &GPUct3, &GPUct4, &GPUct1 };
	int nrot										  = static_cast<int>(rot_indexes.size());
	state.counters["rotations"]						  = nrot;
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++) {
			GPUct1.rotate_hoisted(rot_indexes, rot_outs, false);
		}
		auto end_cpu = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		// report per-rotation timings (divide by inner loop count and number of rotations)
		state.SetIterationTime(elapsed.count() / (100 * nrot));
		state.counters["cpu"] = std::chrono::duration_cast<std::chrono::duration<double>>(end_cpu - start).count() / (100 * nrot);
		state.counters["gpu"] = elapsed.count() / (100 * nrot);
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, CiphertextRotateAndAccumulate)(benchmark::State& state) {
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

	int bstep				 = state.range(4);
	std::vector<int> indexes = FIDESlib::CKKS::GetAccumulateRotationIndices(bstep, 1, GPUcc->N / 2);
	FIDESlib::CKKS::GenAndAddRotationKeys(cc, keys, GPUcc, indexes);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["limbs"]	  = GPUcc->param.L - state.range(3);
	CudaCheckErrorMod;
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < 100; i++) {
			FIDESlib::CKKS::Accumulate(GPUct1, bstep, 1, GPUcc->N / 2);
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

BENCHMARK_REGISTER_F(GeneralFixture, CiphertextRotation)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG });
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextHoistedRotation)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG });
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextRotateAndAccumulate)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, LEVEL_CONFIG, { 2, 4, 8 } });

// TDPS Experiments

BENCHMARK_REGISTER_F(GeneralFixture, CiphertextRotation)->ArgsProduct({ { 30, 31 }, { 0 }, BATCH_CONFIG, { 0, 12 } })->UseManualTime()->Iterations(10);
BENCHMARK_REGISTER_F(GeneralFixture, CiphertextHoistedRotation)->ArgsProduct({ { 30, 31 }, { 0 }, BATCH_CONFIG, { 0, 12 } })->UseManualTime()->Iterations(10);

} // namespace FIDESlib::Benchmarks