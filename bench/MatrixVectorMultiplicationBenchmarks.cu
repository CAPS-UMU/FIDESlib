//
// Created by oscar on 21/10/24.
//

#include <benchmark/benchmark.h>

#include "Benchmark.cuh"

namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(GeneralFixture, GPUMatVecMult)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::LEVELEDSHE);
	auto keys = cc->KeyGen();

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	constexpr int n = 8;

	std::vector<double> x1	 = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
	std::vector<double> x[n] = {
		{ 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 },
		//                            {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}
	};

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

	std::vector<lbcrypto::Plaintext> ptxt;
	for (int i = 0; i < n; ++i) {
		ptxt.emplace_back(cc->MakeCKKSPackedPlaintext(x[i]));
	}
	using Cipher = lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>;
	std::vector<Cipher> ct;
	for (int i = 0; i < n; ++i) {
		ct.emplace_back(cc->Encrypt(keys.publicKey, ptxt1));
	}

	std::vector<FIDESlib::CKKS::Ciphertext> GPUct;
	for (int i = 0; i < n; ++i) {
		FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ct[i]);
		GPUct.emplace_back(GPUcc, raw1);
	}

	std::vector<FIDESlib::CKKS::Plaintext> GPUpt;
	for (int i = 0; i < n; ++i) {
		FIDESlib::CKKS::RawPlainText raw2 = FIDESlib::CKKS::GetRawPlainText(cc, ptxt[i]);
		GPUpt.emplace_back(GPUcc, raw2);
	}

	for (auto _ : state) {
		cudaDeviceSynchronize();
		auto start = std::chrono::high_resolution_clock::now();
		GPUct[0].multPt(GPUpt[0], false);
		for (int i = 1; i < n; ++i) {
			// GPUct[i].multPt(GPUpt[i], false);
			// GPUct[0].add(GPUct[i]);
			GPUct[0].addMultPt(GPUct[i], GPUpt[i], false);
		}
		GPUct[0].rescale();
		cudaDeviceSynchronize();
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		GPUct[0].c0.grow(GPUcc->L);
		GPUct[0].c1.grow(GPUcc->L);
		if constexpr (SYNC)
			CudaCheckErrorMod;
	}

	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, GPUMatVecMultScalar)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::LEVELEDSHE);
	auto keys = cc->KeyGen();

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1	 = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
	std::vector<double> x[8] = { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 } };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

	using Cipher = lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>;
	std::vector<Cipher> ct;
	for (int i = 0; i < 8; ++i) {
		ct.emplace_back(cc->Encrypt(keys.publicKey, ptxt1));
	}

	std::vector<FIDESlib::CKKS::Ciphertext> GPUct;
	for (int i = 0; i < 8; ++i) {
		FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ct[i]);
		GPUct.emplace_back(GPUcc, raw1);
	}
	CudaCheckErrorMod;
	for (auto _ : state) {
		cudaDeviceSynchronize();
		auto start = std::chrono::high_resolution_clock::now();
		GPUct[0].multScalar(x[0][0], false);
		for (int i = 1; i < 8; ++i) {
			GPUct[i].multScalar(x[i][0], false);
			GPUct[0].add(GPUct[i]);
		}
		GPUct[0].rescale();
		cudaDeviceSynchronize();
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		GPUct[0].c0.grow(GPUcc->L);
		GPUct[0].c1.grow(GPUcc->L);
		if constexpr (SYNC)
			CudaCheckErrorMod;
	}

	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, GPUMatVecMultWSum)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::LEVELEDSHE);
	auto keys = cc->KeyGen();

	fideslibParams.batch				= state.range(2);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1 = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
	std::vector<double> x  = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

	using Cipher = lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>;
	std::vector<Cipher> ct;
	for (int i = 0; i < 8; ++i) {
		ct.emplace_back(cc->Encrypt(keys.publicKey, ptxt1));
	}

	std::vector<FIDESlib::CKKS::Ciphertext*> GPUct;
	std::vector<FIDESlib::CKKS::Ciphertext> GPUct_;
	for (int i = 0; i < 8; ++i) {
		FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ct[i]);
		GPUct_.emplace_back(GPUcc, raw1);
	}
	for (int i = 0; i < 8; ++i) {
		GPUct.emplace_back(&GPUct_[i]);
	}

	CudaCheckErrorMod;
	for (auto _ : state) {
		cudaDeviceSynchronize();
		auto start = std::chrono::high_resolution_clock::now();

		GPUct[0]->evalLinearWSumMutable(8, GPUct, x);
		GPUct[0]->rescale();
		cudaDeviceSynchronize();
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		GPUct[0]->c0.grow(GPUcc->L);
		GPUct[0]->c1.grow(GPUcc->L);
		if constexpr (SYNC)
			CudaCheckErrorMod;
	}

	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, CPUMatVecMult)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::LEVELEDSHE);
	auto keys = cc->KeyGen();

	fideslibParams.batch				= 12;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1	 = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
	std::vector<double> x[8] = { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 } };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

	std::vector<lbcrypto::Plaintext> ptxt;
	for (int i = 0; i < 8; ++i) {
		ptxt.emplace_back(cc->MakeCKKSPackedPlaintext(x[i]));
	}

	using Cipher = lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>;
	std::vector<Cipher> ct;
	for (int i = 0; i < 8; ++i) {
		ct.emplace_back(cc->Encrypt(keys.publicKey, ptxt1));
	}

	CudaCheckErrorMod;
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		cc->GetScheme()->EvalMultInPlace(ct[0], ptxt[0]);
		for (int i = 1; i < 8; ++i) {
			cc->GetScheme()->EvalMultInPlace(ct[i], ptxt[i]);
			cc->EvalAddInPlace(ct[0], ct[i]);
		}
		cc->RescaleInPlace(ct[0]);
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		ct[0] = cc->Encrypt(keys.publicKey, ptxt1);

		if constexpr (SYNC)
			CudaCheckErrorMod;
	}
	CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(GeneralFixture, CPUMatVecMultScalar)(benchmark::State& state) {
	int devcount = -1;
	cudaGetDeviceCount(&devcount);

	std::vector<int> GPUs = generalTestParams.GPUs;

	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::LEVELEDSHE);
	auto keys = cc->KeyGen();

	fideslibParams.batch				= 12;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	FIDESlib::CKKS::Context GPUcc		= FIDESlib::CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), GPUs);

	std::vector<double> x1	 = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
	std::vector<double> x[8] = { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 } };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

	using Cipher = lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>;
	std::vector<Cipher> ct;
	for (int i = 0; i < 8; ++i) {
		ct.emplace_back(cc->Encrypt(keys.publicKey, ptxt1));
	}
	CudaCheckErrorMod;
	for (auto _ : state) {
		auto start = std::chrono::high_resolution_clock::now();
		cc->GetScheme()->EvalMultInPlace(ct[0], x[0][0]);
		for (int i = 1; i < 8; ++i) {
			cc->GetScheme()->EvalMultInPlace(ct[i], x[i][0]);
			cc->EvalAddInPlace(ct[0], ct[i]);
		}
		cc->RescaleInPlace(ct[0]);
		auto end	 = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		state.SetIterationTime(elapsed.count());
		ct[0] = cc->Encrypt(keys.publicKey, ptxt1);

		if constexpr (SYNC)
			CudaCheckErrorMod;
	}
	CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(GeneralFixture, GPUMatVecMult)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, { 0 } })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, CPUMatVecMult)->ArgsProduct({ PARAMETERS, { 0 } })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, GPUMatVecMultScalar)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, { 0 } })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, CPUMatVecMultScalar)->ArgsProduct({ PARAMETERS, { 0 } })->UseManualTime();
BENCHMARK_REGISTER_F(GeneralFixture, GPUMatVecMultWSum)->ArgsProduct({ PARAMETERS, { 0 }, BATCH_CONFIG, { 0 } })->UseManualTime();
} // namespace FIDESlib::Benchmarks