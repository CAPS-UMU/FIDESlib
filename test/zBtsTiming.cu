
#include "CKKS/Bootstrap.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"

using namespace FIDESlib::CKKS;
using namespace std::chrono;

namespace FIDESlib::Testing {

class BtsTimingTests : public GeneralParametrizedTest {};

TEST_P(BtsTimingTests, Regular) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// bool verbose = true;

	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);

	// const bool sparse_encaps = false;

	std::cout << "Create context" << std::endl;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, UNIFORM);
	FIDESlib::CKKS::Context GPUcc_      = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *GPUcc_;
	std::cout << "Num large digits" << GPUcc.dnum << std::endl;
	// Parameters
	GPUcc.batch  = 128;
	int numSlots = cc->GetRingDimension() / 2;

	// Keys
	keys = cc->KeyGen();

	// Bootstrapping Precomputation

	cc->EvalBootstrapSetup({ 3, 3 }, { 16, 16 }, numSlots, 0, true, false);

	cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);
	std::cout << lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) << std::endl;
	std::cout << GPUcc.GetDoubleAngleIts() << std::endl;

	std::cout << "Add bootstrap precomputation" << std::endl;
	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

	std::vector<double> x1            = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1         = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, numSlots);
	auto c1                           = cc->Encrypt(keys.publicKey, ptxt1);
	FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	std::cout << "Create ciphertext" << std::endl;
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc_, raw);

	int N = 10;

	std::cout << "Begin boot" << std::endl;
	auto start_gpu = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++) {
		Bootstrap(GPUct1, numSlots, false);
		cudaDeviceSynchronize();
	}
	auto end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) / N << " ms." << std::endl;

	std::cout << GPUct1.getLevel() << std::endl;

	cudaDeviceSynchronize();

	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct1.store(raw_res);
	auto result(c1);
	GetOpenFHECipherText(result, raw_res);

	lbcrypto::Plaintext result_pt;
	cc->Decrypt(keys.secretKey, result, &result_pt);
	std::cout << result_pt->GetLogPrecision() << std::endl;
	for (int i = 0; i < 8; ++i) {
		std::cout << result_pt->GetRealPackedValue().at(i) << " ";
	}
	std::cout << std::endl;

	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
}

TEST_P(BtsTimingTests, SSE) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// bool verbose = true;

	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);

	// const bool sparse_encaps = true;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, ENCAPS);
	FIDESlib::CKKS::Context GPUcc_      = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *GPUcc_;

	// Parameters
	GPUcc.batch  = 128;
	int numSlots = cc->GetRingDimension() / 2;
	// int numSlots = 64;
	//  Keys
	keys = cc->KeyGen();

	// Bootstrapping Precomputation
	cc->EvalBootstrapSetup(
		{ 3, 3 },
		{ 16, 16 },
		numSlots,
		0,
		true,
		false,
		lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) + GPUcc.GetDoubleAngleIts());
	std::cout << lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) << std::endl;
	std::cout << GPUcc.GetDoubleAngleIts() << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

	std::vector<double> x1    = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, numSlots);
	auto c1                   = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc_, raw);

	GPUct1.dropToLevel(2);
	{
		FIDESlib::CKKS::RawCipherText raw_res;
		GPUct1.store(raw_res);
		auto result(c1);
		GetOpenFHECipherText(result, raw_res);

		lbcrypto::Plaintext result_pt;
		cc->Decrypt(keys.secretKey, result, &result_pt);
		std::cout << result_pt->GetLogPrecision() << std::endl;
	}

	int N = 10;

	auto start_gpu = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++) {
		cudaDeviceSynchronize();
		FIDESlib::CKKS::Ciphertext GPUct2(GPUcc_);
		cudaDeviceSynchronize();
		GPUct2.copy(GPUct1);
		cudaDeviceSynchronize();
		Bootstrap(GPUct2, numSlots, false);
		cudaDeviceSynchronize();
		GPUct1.copy(GPUct2);
		cudaDeviceSynchronize();
	}
	auto end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) / N << " ms." << std::endl;

	std::cout << GPUct1.getLevel() << std::endl;

	cudaDeviceSynchronize();

	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct1.store(raw_res);
	auto result(c1);
	GetOpenFHECipherText(result, raw_res);

	lbcrypto::Plaintext result_pt;
	cc->Decrypt(keys.secretKey, result, &result_pt);
	std::cout << result_pt->GetLogPrecision() << std::endl;
	for (int i = 0; i < 8; ++i) {
		std::cout << result_pt->GetRealPackedValue().at(i) << " ";
	}
	std::cout << std::endl;

	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
}

TEST_P(BtsTimingTests, REGULAR2) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// bool verbose = true;

	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);

	// const bool sparse_encaps = true;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, UNIFORM_2);
	FIDESlib::CKKS::Context GPUcc_      = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *GPUcc_;

	// Parameters
	GPUcc.batch  = 128;
	int numSlots = cc->GetRingDimension() / 2;

	// Keys
	keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(GPUcc_);
	eval_key_gpu.Initialize(eval_key);
	GPUcc.AddEvalKey(std::move(eval_key_gpu));

	// Bootstrapping Precomputation
	cc->EvalBootstrapSetup(
		{ 3, 3 },
		{ 16, 16 },
		numSlots,
		0,
		true,
		false,
		lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) + GPUcc.GetDoubleAngleIts());
	cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

	std::vector<double> x1    = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, numSlots);
	auto c1                   = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc_, raw);

	GPUct1.dropToLevel(2);

	int N = 2;

	auto start_gpu = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++) {
		Bootstrap(GPUct1, numSlots, false);
		cudaDeviceSynchronize();
	}
	auto end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) / N << " ms." << std::endl;

	std::cout << GPUct1.getLevel() << std::endl;

	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct1.store(raw_res);
	auto result(c1);
	GetOpenFHECipherText(result, raw_res);

	lbcrypto::Plaintext result_pt;
	cc->Decrypt(keys.secretKey, result, &result_pt);
	std::cout << result_pt->GetLogPrecision() << std::endl;
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
}

TEST_P(BtsTimingTests, SPARSE) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// bool verbose = true;

	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);

	// const bool sparse_encaps = true;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, SPARSE);
	FIDESlib::CKKS::Context GPUcc_      = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *GPUcc_;

	// Parameters
	GPUcc.batch  = 128;
	int numSlots = cc->GetRingDimension() / 2;

	// Keys
	keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(GPUcc_);
	eval_key_gpu.Initialize(eval_key);
	GPUcc.AddEvalKey(std::move(eval_key_gpu));

	// Bootstrapping Precomputation
	cc->EvalBootstrapSetup(
		{ 3, 3 },
		{ 0, 0 },
		numSlots,
		0,
		true,
		false,
		lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) + GPUcc.GetDoubleAngleIts());
	cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

	std::vector<double> x1    = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, numSlots);
	auto c1                   = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(GPUcc_, raw);

	GPUct1.dropToLevel(2);

	int N = 10;

	auto start_gpu = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++) {
		Bootstrap(GPUct1, numSlots, false);
		cudaDeviceSynchronize();
	}
	auto end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) / N << " ms." << std::endl;

	std::cout << GPUct1.getLevel() << std::endl;

	cudaDeviceSynchronize();

	FIDESlib::CKKS::RawCipherText raw_res;
	GPUct1.store(raw_res);
	auto result(c1);
	GetOpenFHECipherText(result, raw_res);

	lbcrypto::Plaintext result_pt;
	cc->Decrypt(keys.secretKey, result, &result_pt);
	std::cout << result_pt->GetLogPrecision() << std::endl;
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
}

INSTANTIATE_TEST_SUITE_P(LLMTests, BtsTimingTests, testing::Values(TTALL64BOOT));
} // namespace FIDESlib::Testing