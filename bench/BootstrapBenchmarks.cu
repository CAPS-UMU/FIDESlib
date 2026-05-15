//
// Created by carlosad on 19/11/24.
//

#include "Benchmark.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/KeySwitchingKey.cuh"

namespace FIDESlib::Benchmarks {

BENCHMARK_DEFINE_F(GeneralFixture, ApproxModReduction)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L <= level");
		return;
	}

	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	state.counters["p_batch"] = state.range(2);

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

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

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);

	FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	kskEval.Initialize(rawKskEval);
	GPUcc->AddEvalKey(std::move(kskEval));

	for (auto _ : state) {
		CudaCheckErrorMod;
		auto start = std::chrono::high_resolution_clock::now();
		FIDESlib::CKKS::approxModReduction(GPUct1, GPUct2, kskEval, 1.0);
		auto cpu_end = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end								  = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed	  = end - start;
		std::chrono::duration<double> cpu_elapsed = cpu_end - start;
		state.counters["cpu"]					  = cpu_elapsed.count() * 1e9;
		state.counters["gpu"]					  = elapsed.count() * 1e9;
		state.SetIterationTime(elapsed.count());
		GPUct1.c0.grow(GPUcc->L - state.range(3));
		GPUct1.c1.grow(GPUcc->L - state.range(3));
		GPUct2.c0.grow(GPUcc->L - state.range(3));
		GPUct2.c1.grow(GPUcc->L - state.range(3));
	}
	CudaCheckErrorMod;
	// cc->GetEvalAutomorphismKeyMap(this->keys.publicKey->GetKeyTag()).clear();
}

BENCHMARK_DEFINE_F(GeneralFixture, ApproxModReductionSparse)(benchmark::State& state) {
	if (this->generalTestParams.multDepth <= static_cast<uint64_t>(state.range(3))) {
		state.SkipWithMessage("cc.L <= level");
		return;
	}

	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	state.counters["p_batch"]			= state.range(2);
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

	for (auto _ : state) {
		CudaCheckErrorMod;
		auto start = std::chrono::high_resolution_clock::now();
		FIDESlib::CKKS::approxModReductionSparse(GPUct1, 1.0);
		auto cpu_end = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end								  = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed	  = end - start;
		std::chrono::duration<double> cpu_elapsed = cpu_end - start;
		state.counters["cpu"]					  = cpu_elapsed.count() * 1e9;
		state.counters["gpu"]					  = elapsed.count() * 1e9;
		state.SetIterationTime(elapsed.count());

		GPUct1.c0.grow(GPUcc->L - state.range(3));
		GPUct1.c1.grow(GPUcc->L - state.range(3));
	}
	CudaCheckErrorMod;
	// cc->GetEvalAutomorphismKeyMap(this->keys.publicKey->GetKeyTag()).clear();
}

BENCHMARK_DEFINE_F(GeneralFixture, CoeffsToSlots)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	state.counters["p_batch"]			= state.range(2);
	state.counters["p_slots"]			= state.range(3);
	fideslibParams.batch				= state.range(2);
	const int slots						= state.range(3);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	cc->EvalBootstrapSetup({ 2, 2 }, { 0, 0 }, slots);
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);
	const int start_level	  = GPUcc->GetBootPrecomputation(slots).CtS.at(0).A.at(0).c0.getLevel();
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc->L - start_level);

	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	{
		FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
		FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
		kskEval.Initialize(rawKskEval);
		GPUcc->AddEvalKey(std::move(kskEval));
	}

	for (auto _ : state) {
		CudaCheckErrorMod;
		auto start = std::chrono::high_resolution_clock::now();
		FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, false);
		auto cpu_end = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end								  = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed	  = end - start;
		std::chrono::duration<double> cpu_elapsed = cpu_end - start;
		state.counters["CPU time (ns)"]			  = cpu_elapsed.count() * 1e9;
		state.counters["GPU time (ns)"]			  = elapsed.count() * 1e9;
		state.SetIterationTime(elapsed.count());

		GPUct1.c0.grow(start_level);
		GPUct1.c1.grow(start_level);
	}
	CudaCheckErrorMod;
	cc->GetEvalAutomorphismKeyMap(this->keys.publicKey->GetKeyTag()).clear();
}

BENCHMARK_DEFINE_F(GeneralFixture, SlotsToCoeffs)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	state.counters["p_batch"] = state.range(2);
	state.counters["p_slots"] = state.range(3);

	fideslibParams.batch				= state.range(2);
	const int slots						= state.range(3);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	cc->EvalBootstrapSetup({ 2, 2 }, { 0, 0 }, slots);
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

	const int init_level	  = GPUcc->GetBootPrecomputation(slots).StC.at(0).A.at(0).c0.getLevel();
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc->L - init_level);
	auto c1					  = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	{
		FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
		FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
		kskEval.Initialize(rawKskEval);
		GPUcc->AddEvalKey(std::move(kskEval));
	}

	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, true);
		auto cpu_end = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end								  = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed	  = end - start;
		std::chrono::duration<double> cpu_elapsed = cpu_end - start;
		state.counters["CPU time (ns)"]			  = cpu_elapsed.count() * 1e9;
		state.counters["GPU time (ns)"]			  = elapsed.count() * 1e9;
		state.SetIterationTime(elapsed.count());

		GPUct1.c0.grow(init_level);
		GPUct1.c1.grow(init_level);
	}
	CudaCheckErrorMod;
	cc->GetEvalAutomorphismKeyMap(this->keys.publicKey->GetKeyTag()).clear();
}

struct BootConfig {
	uint32_t slots, a, b, dim1, dim2;
};

BootConfig conf[] = {
	BootConfig{ 1 << 6, 1, 1, 16, 16 },
	BootConfig{ 1 << 9, 2, 2, 16, 16 },
	BootConfig{ 1 << 14, 5, 5, 8, 8 },
	BootConfig{ 1 << 15, 5, 5, 8, 8 },
	BootConfig{ 1 << 14, 4, 4, 8, 8 },
	BootConfig{ 1 << 15, 4, 4, 8, 8 },
	BootConfig{ 1 << 14, 3, 3, 16, 16 },
	BootConfig{ 1 << 15, 3, 3, 16, 16 },
	BootConfig{ 1 << 16, 4, 4, 16, 16 },
	BootConfig{ 1 << 16, 3, 3, 16, 16 }, // 9

	BootConfig{ 1 << 15, 3, 3, 1, 1 },
	BootConfig{ 1 << 15, 3, 3, 2, 2 },
	BootConfig{ 1 << 15, 3, 3, 4, 4 },
	BootConfig{ 1 << 15, 3, 3, 8, 8 },
	BootConfig{ 1 << 15, 3, 3, 16, 16 },
	BootConfig{ 1 << 15, 3, 3, 32, 32 },
	BootConfig{ 1 << 15, 3, 3, 63, 63 }, // 16

	BootConfig{ 1 << 14, 3, 3, 1, 1 },
	BootConfig{ 1 << 14, 3, 3, 2, 2 },
	BootConfig{ 1 << 15, 3, 3, 4, 4 },
	BootConfig{ 1 << 14, 3, 3, 8, 8 },
	BootConfig{ 1 << 14, 3, 3, 16, 16 },
	BootConfig{ 1 << 15, 3, 3, 32, 32 },
	BootConfig{ 1 << 14, 3, 3, 63, 63 }, // 23

	BootConfig{ 1 << 8, 2, 2, 16, 16 },
	BootConfig{ 1 << 14, 3, 3, 16, 16 },
	BootConfig{ 1 << 15, 3, 3, 16, 16 },
	BootConfig{ 1 << 8, 2, 2, 16, 16 },
	BootConfig{ 1 << 15, 3, 3, 16, 16 },
	BootConfig{ 1 << 16, 3, 3, 16, 16 }, // 29

};

/** MinKS */
/*
BootConfig conf[] = {BootConfig{1 << 6, 1, 1, 1, 1},  BootConfig{1 << 9, 2, 2, 1, 1},  BootConfig{1 << 14, 5, 5, 1, 1},
					 BootConfig{1 << 15, 5, 5, 1, 1}, BootConfig{1 << 14, 4, 4, 1, 1}, BootConfig{1 << 15, 4, 4, 1, 1},
					 BootConfig{1 << 14, 3, 3, 1, 1}, BootConfig{1 << 15, 3, 3, 1, 1}, BootConfig{1 << 16, 4, 4, 1, 1},
					 BootConfig{1 << 16, 3, 3, 1, 1}};
*/
#include <openfhe.h>

BENCHMARK_DEFINE_F(GeneralFixture, HyperParamBootstrapGPU)(benchmark::State& state) {
	CKKS::DeregisterAllContexts();
	for (auto& i : context_map) {
		i.second->ClearEvalAutomorphismKeys();
		// i.second->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// bool verbose = true;
	CudaCheckErrorMod;
	std::vector<int> GPUs = generalTestParams.GPUs;

	state.counters["p_batch"] = state.range(2);
	state.counters["p_slots"] = conf[state.range(3)].slots;

	fideslibParams.batch				= state.range(2);
	const int slots						= conf[state.range(3)].slots;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, UNIFORM);
	cudaSetDevice(GPUs[0]);
	CudaCheckErrorMod;
	FIDESlib::CKKS::Context GPUcc = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

	size_t min_global_openfhe = 1e9;
	size_t min_global_fides	  = 1e9;
	for (int fftiter = 1; (1 << fftiter) <= slots; ++fftiter) {
		size_t min_openfhe = 1e9;
		size_t min_fides   = 1e9;
		for (int b = 1; b <= (1 << ((int)ceil(log2(pow((double)slots, ((double)1 / (double)fftiter))))) + (fftiter > 1)); b <<= 1) {
			if (b == (1 << ((int)ceil(log2(pow((double)slots, ((double)1 / (double)fftiter))))) + (fftiter > 1)))
				b--;
			cc->EvalBootstrapSetup({ (uint32_t)fftiter, (uint32_t)fftiter },
			  { (uint32_t)b, (uint32_t)b },
			  slots,
			  0,
			  false,
			  false,
			  lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc->GetCoeffsChebyshev(), false) + GPUcc->GetDoubleAngleIts());

			std::cout << "slots=" << slots << ", fftIter=" << fftiter << ", b=" << b << ":" << std::endl;
			auto rots_cpu = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->FindBootstrapRotationIndices(slots, cc->GetCyclotomicOrder());
			std::set<int32_t> rots_cpu_set(rots_cpu.begin(), rots_cpu.end());
			min_openfhe = std::min(min_openfhe, rots_cpu_set.size());
			std::cout << "OpenFHE requests " << rots_cpu_set.size() << " distinct rotation keys." << std::endl;

			auto rots_gpu = FIDESlib::CKKS::GetBootstrapIndexes(cc, slots, nullptr);
			std::set<int> rots_gpu_set(rots_gpu.begin(), rots_gpu.end());
			min_fides = std::min(min_fides, rots_gpu_set.size());
			std::cout << "FIDESlib requests " << rots_gpu_set.size() << " distinct rotation keys." << std::endl;
		}
		std::cout << "For fftIter=" << fftiter << ": min openfhe=" << min_openfhe << ", min fides=" << min_fides << std::endl;
		min_global_openfhe = std::min(min_global_openfhe, min_openfhe);
		min_global_fides   = std::min(min_global_fides, min_fides);
	}
	std::cout << "For slots=" << slots << ": min openfhe=" << min_global_openfhe << ", min fides=" << min_global_fides << std::endl;

	state.SkipWithMessage("");
	CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(GeneralFixture, HyperParamBootstrapGPU)->ArgsProduct({ { 5 }, { 0 }, BATCH_CONFIG, { 0, 1, 2, 3 } })->Iterations(50);

BENCHMARK_REGISTER_F(GeneralFixture, HyperParamBootstrapGPU)->ArgsProduct({ { 6 }, { 0 }, BATCH_CONFIG, { 0, 1, 2, 3, 8 } })->Iterations(50);

BENCHMARK_DEFINE_F(GeneralFixture, PrintRotationsBootstrapGPU)(benchmark::State& state) {
	CKKS::DeregisterAllContexts();
	for (auto& i : context_map) {
		i.second->ClearEvalAutomorphismKeys();
		// i.second->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// bool verbose = true;
	CudaCheckErrorMod;
	std::vector<int> GPUs = generalTestParams.GPUs;

	state.counters["p_batch"] = state.range(2);
	state.counters["p_slots"] = conf[state.range(3)].slots;

	fideslibParams.batch				= state.range(2);
	const int slots						= conf[state.range(3)].slots;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, UNIFORM);
	cudaSetDevice(GPUs[0]);
	CudaCheckErrorMod;
	FIDESlib::CKKS::Context GPUcc = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	{
		cc->EvalBootstrapSetup({ conf[state.range(3)].a, conf[state.range(3)].b },
		  { conf[state.range(3)].dim1, conf[state.range(3)].dim2 },
		  slots,
		  0,
		  false,
		  false,
		  lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc->GetCoeffsChebyshev(), false) + GPUcc->GetDoubleAngleIts());

		auto rots_cpu = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->FindBootstrapRotationIndices(slots, cc->GetCyclotomicOrder());
		std::set<int32_t> rots_cpu_set(rots_cpu.begin(), rots_cpu.end());
		std::cout << "OpenFHE requests " << rots_cpu_set.size() << " distinct rotation keys." << std::endl;

		auto rots_gpu = FIDESlib::CKKS::GetBootstrapIndexes(cc, slots, nullptr);
		std::set<int> rots_gpu_set(rots_gpu.begin(), rots_gpu.end());
		std::cout << "FIDESlib requests " << rots_gpu_set.size() << " distinct rotation keys." << std::endl;

		state.counters["OpenFHE rots: "]  = rots_cpu_set.size();
		state.counters["FIDESlib rots: "] = rots_gpu_set.size();
	}
	state.SkipWithMessage("");
	CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(GeneralFixture, PrintRotationsBootstrapGPU)
  ->ArgsProduct({ { 5 }, { 0 }, BATCH_CONFIG, { 0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16 } })
  ->Iterations(50);

BENCHMARK_REGISTER_F(GeneralFixture, PrintRotationsBootstrapGPU)->ArgsProduct({ { 6 }, { 0 }, BATCH_CONFIG, { 0, 1, 2, 3, 6, 7, 8, 9 } })->Iterations(50);

BENCHMARK_DEFINE_F(GeneralFixture, BootstrapGPU)(benchmark::State& state) {
	CKKS::DeregisterAllContexts();
	for (auto& i : context_map) {
		i.second->ClearEvalAutomorphismKeys();
		// i.second->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// bool verbose = true;
	CudaCheckErrorMod;
	std::vector<int> GPUs = generalTestParams.GPUs;

	state.counters["p_batch"] = state.range(2);
	state.counters["slots"]	  = conf[state.range(3)].slots;
	state.counters["config"]  = state.range(0);

	fideslibParams.batch				= state.range(2);
	const int slots						= conf[state.range(3)].slots;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, UNIFORM);
	cudaSetDevice(GPUs[0]);
	CudaCheckErrorMod;
	FIDESlib::CKKS::Context GPUcc = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	{
		cc->EvalBootstrapSetup({ conf[state.range(3)].a, conf[state.range(3)].b },
		  { conf[state.range(3)].dim1, conf[state.range(3)].dim2 },
		  slots,
		  0,
		  false,
		  lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc->GetCoeffsChebyshev(), false) + GPUcc->GetDoubleAngleIts());

		auto rots_cpu = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->FindBootstrapRotationIndices(slots, cc->GetCyclotomicOrder());
		std::set<int32_t> rots_cpu_set(rots_cpu.begin(), rots_cpu.end());

		auto rots_gpu = FIDESlib::CKKS::GetBootstrapIndexes(cc, slots, nullptr);
		std::set<int> rots_gpu_set(rots_gpu.begin(), rots_gpu.end());

		state.counters["OpenFHE rots"]	= rots_cpu_set.size();
		state.counters["FIDESlib rots"] = rots_gpu_set.size();
	}

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

	cc->EvalBootstrapSetup({ conf[state.range(3)].a, conf[state.range(3)].b },
	  { conf[state.range(3)].dim1, conf[state.range(3)].dim2 },
	  slots,
	  0,
	  true,
	  false,
	  lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc->GetCoeffsChebyshev(), false) + GPUcc->GetDoubleAngleIts());

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

	const int init_level = 1;
	ptxt1->SetLevel(GPUcc->L - init_level);
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	int endlevel = 0;

	for (auto _ : state) {
		cudaDeviceSynchronize();
		auto start = std::chrono::high_resolution_clock::now();
		FIDESlib::CKKS::Bootstrap(GPUct1, slots, false);
		auto cpu_end = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end								  = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed	  = end - start;
		std::chrono::duration<double> cpu_elapsed = cpu_end - start;
		state.counters["cpu"]					  = cpu_elapsed.count() * 1e9;
		state.counters["gpu"]					  = elapsed.count() * 1e9;
		state.SetIterationTime(elapsed.count());
		endlevel = GPUct1.getLevel();
		if (endlevel < init_level) {
			GPUct1.c0.grow(init_level + 1);
			GPUct1.c1.grow(init_level + 1);
		}
	}

	{
		state.counters["Leff"]	 = endlevel;
		state.counters["Lrecov"] = endlevel - init_level;

		if (endlevel > 0) {
			FIDESlib::CKKS::RawCipherText raw_res;
			GPUct1.store(raw_res);
			auto result(c1);
			GetOpenFHECipherText(result, raw_res);

			lbcrypto::Plaintext result_pt;
			cc->Decrypt(keys.secretKey, result, &result_pt);

			state.counters["bits"] = result_pt->GetLogPrecision();
		} else {
			state.counters["bits"] = -1;
		}
	}
	/*
	FIDESlib::CKKS::RawCipherText raw_res1;
	GPUct1.store(raw_res1);
	auto cResGPU(c1);

	GetOpenFHECipherText(cResGPU, raw_res1);
	lbcrypto::Plaintext resultGPU;
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

	resultGPU->SetLength(8);
	std::cout << "Result GPU " << resultGPU;
	*/
	CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(GeneralFixture, ApproxModReduction)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, { 0, 1, 2, 3, 4, 5 } })->Iterations(50);
BENCHMARK_REGISTER_F(GeneralFixture, ApproxModReductionSparse)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, { 0, 1, 2, 3, 4, 5 } })->Iterations(50);
BENCHMARK_REGISTER_F(GeneralFixture, CoeffsToSlots)->ArgsProduct({ { 5 }, { 0 }, BATCH_CONFIG, { 64 } })->Iterations(50);
BENCHMARK_REGISTER_F(GeneralFixture, SlotsToCoeffs)->ArgsProduct({ { 5 }, { 0 }, BATCH_CONFIG, { 64 } })->Iterations(50);

BENCHMARK_DEFINE_F(GeneralFixture, SSEBootstrapGPU)(benchmark::State& state) {
	CKKS::DeregisterAllContexts();
	for (auto& i : context_map) {
		i.second->ClearEvalAutomorphismKeys();
		// i.second->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// bool verbose = true;
	CudaCheckErrorMod;
	std::vector<int> GPUs	 = generalTestParams.GPUs;
	state.counters["config"] = state.range(0);

	state.counters["p_batch"] = state.range(2);
	state.counters["slots"]	  = conf[state.range(3)].slots;

	fideslibParams.batch				= state.range(2);
	const int slots						= conf[state.range(3)].slots;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, ENCAPS);
	cudaSetDevice(GPUs[0]);
	CudaCheckErrorMod;
	FIDESlib::CKKS::Context GPUcc = FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	{
		cc->EvalBootstrapSetup({ conf[state.range(3)].a, conf[state.range(3)].b },
		  { conf[state.range(3)].dim1, conf[state.range(3)].dim2 },
		  slots,
		  0,
		  false,
		  lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc->GetCoeffsChebyshev(), false) + GPUcc->GetDoubleAngleIts());

		auto rots_cpu = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->FindBootstrapRotationIndices(slots, cc->GetCyclotomicOrder());
		std::set<int32_t> rots_cpu_set(rots_cpu.begin(), rots_cpu.end());

		auto rots_gpu = FIDESlib::CKKS::GetBootstrapIndexes(cc, slots, nullptr);
		std::set<int> rots_gpu_set(rots_gpu.begin(), rots_gpu.end());

		state.counters["OpenFHE rots"]	= rots_cpu_set.size();
		state.counters["FIDESlib rots"] = rots_gpu_set.size();
	}

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

	cc->EvalBootstrapSetup({ conf[state.range(3)].a, conf[state.range(3)].b },
	  { conf[state.range(3)].dim1, conf[state.range(3)].dim2 },
	  slots,
	  0,
	  true,
	  false,
	  lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc->GetCoeffsChebyshev(), false) + GPUcc->GetDoubleAngleIts());

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

	const int init_level = 1;
	ptxt1->SetLevel(GPUcc->L - init_level);
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	int endlevel = 0;

	for (auto _ : state) {
		cudaDeviceSynchronize();
		auto start = std::chrono::high_resolution_clock::now();
		FIDESlib::CKKS::Bootstrap(GPUct1, slots, false);
		auto end_cpu = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end								  = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed	  = end - start;
		std::chrono::duration<double> cpu_elapsed = end_cpu - start;
		state.counters["cpu"]					  = cpu_elapsed.count() * 1e9;
		state.counters["gpu"]					  = elapsed.count() * 1e9;
		state.SetIterationTime(elapsed.count());
		endlevel = GPUct1.getLevel();
	}

	{
		state.counters["Leff"]	 = endlevel;
		state.counters["Lrecov"] = endlevel - init_level;

		FIDESlib::CKKS::RawCipherText raw_res;
		GPUct1.store(raw_res);
		auto result(c1);
		GetOpenFHECipherText(result, raw_res);

		lbcrypto::Plaintext result_pt;
		cc->Decrypt(keys.secretKey, result, &result_pt);
		state.counters["bits"] = result_pt->GetLogPrecision();
	}
	/*
	FIDESlib::CKKS::RawCipherText raw_res1;
	GPUct1.store(raw_res1);
	auto cResGPU(c1);

	GetOpenFHECipherText(cResGPU, raw_res1);
	lbcrypto::Plaintext resultGPU;
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

	resultGPU->SetLength(8);
	std::cout << "Result GPU " << resultGPU;
	*/
	CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(GeneralFixture, SSEBootstrapGPU)->ArgsProduct({ { 18 }, { 0 }, BATCH_CONFIG, { 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23 } })->Iterations(50);

BENCHMARK_REGISTER_F(GeneralFixture, BootstrapGPU)->ArgsProduct({ { 3, 4, 5, 18 }, { 0 }, BATCH_CONFIG, { 0, 1, 2, 3, 4, 5, 6, 7 } })->Iterations(50);

BENCHMARK_REGISTER_F(GeneralFixture, SSEBootstrapGPU)->ArgsProduct({ { 3, 4, 5, 18 }, { 0 }, BATCH_CONFIG, { 0, 1, 2, 3, 4, 5, 6, 7 } })->Iterations(50);

// BENCHMARK_REGISTER_F(GeneralFixture, BootstrapGPU)->ArgsProduct({ { 7 }, { 0 }, BATCH_CONFIG, {  0, 1, 3, 5, 7, 8, 9} })->Iterations(50);
BENCHMARK_REGISTER_F(GeneralFixture, BootstrapGPU)->ArgsProduct({ { 20 }, { 0 }, BATCH_CONFIG, { 0, 1, 3, 5, 7, 8, 9 } })->Iterations(50);

// BENCHMARK_REGISTER_F(GeneralFixture, SSEBootstrapGPU)->ArgsProduct({ { 7 }, { 0 }, BATCH_CONFIG, { 0, 1, 3, 5, 7, 8, 9 } })->Iterations(50);
BENCHMARK_REGISTER_F(GeneralFixture, SSEBootstrapGPU)->ArgsProduct({ { 20 }, { 0 }, BATCH_CONFIG, { 0, 1, 3, 5, 7, 8, 9 } })->Iterations(50);

BENCHMARK_DEFINE_F(GeneralFixture, BootstrapCPU)(benchmark::State& state) {

	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	state.counters["config"]  = state.range(0);
	state.counters["p_batch"] = state.range(2);
	state.counters["slots"]	  = conf[state.range(3)].slots;

	fideslibParams.batch				= state.range(2);
	const int slots						= conf[state.range(3)].slots;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc->L - 1, nullptr, slots);

	cc->EvalBootstrapSetup({ conf[state.range(3)].a, conf[state.range(3)].b }, { 0, 0 }, slots);
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	// FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

	// const int init_level = GPUcc.GetBootPrecomputation(slots).StC.at(0).A.at(0).c0.getLevel();
	////ptxt1->SetLevel(GPUcc.L - init_level);
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	// FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	// FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

	{
		// FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
		// FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
		// kskEval.Initialize(GPUcc, rawKskEval);
		//  GPUcc.AddEvalKey(std::move(kskEval));
	}

	int endlevel = 0;

	for (auto _ : state) {
		auto ct = c1->Clone();

		auto start	 = std::chrono::high_resolution_clock::now();
		ct			 = cc->EvalBootstrap(ct);
		auto cpu_end = std::chrono::high_resolution_clock::now();
		CudaCheckErrorMod;
		auto end								  = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed	  = end - start;
		std::chrono::duration<double> cpu_elapsed = cpu_end - start;
		state.counters["cpu"]					  = cpu_elapsed.count() * 1e9;
		state.counters["gpu"]					  = elapsed.count() * 1e9;
		state.SetIterationTime(elapsed.count());
		endlevel = ct->GetLevel();
	}
	state.counters["Leff"]	 = endlevel;
	state.counters["Lrecov"] = static_cast<int>(GPUcc->L) - 1 - endlevel;
	CudaCheckErrorMod;
	cc->GetEvalAutomorphismKeyMap(this->keys.publicKey->GetKeyTag()).clear();
}

BENCHMARK_REGISTER_F(GeneralFixture, BootstrapCPU)->ArgsProduct({ PARAMETERS, { 0 }, { 1 }, { 0, 1, 6, 7 } })->Iterations(3);

// BENCHMARK_REGISTER_F(GeneralFixture, BootstrapCPU)->ArgsProduct({{4, 3}, {0}, {2, 6, 12}, {4, 5}});

/// TDPS Boot benchmarks
BENCHMARK_REGISTER_F(GeneralFixture, BootstrapGPU)->ArgsProduct({ { 28 }, { 0 }, BATCH_CONFIG, { 24, 25, 26 } })->Iterations(10)->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, SSEBootstrapGPU)->ArgsProduct({ { 28 }, { 0 }, BATCH_CONFIG, { 24, 25, 26 } })->Iterations(10)->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, BootstrapGPU)->ArgsProduct({ { 29 }, { 0 }, BATCH_CONFIG, { 27, 28, 29 } })->Iterations(10)->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, SSEBootstrapGPU)->ArgsProduct({ { 29 }, { 0 }, BATCH_CONFIG, { 27, 28, 29 } })->Iterations(10)->UseManualTime();

} // namespace FIDESlib::Benchmarks