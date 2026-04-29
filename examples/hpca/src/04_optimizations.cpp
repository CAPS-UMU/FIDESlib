#include <fideslib.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace fideslib;

// =====================================================
// Accumulation optimized.
// =====================================================
void accumulateOptimized(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly>& ct, uint32_t batchSize) {
	// TODO: Implement optimized accumulation.
	// Hint: Think in terms of logarithms and use hoisted rotations.
}

// =====================================================
// Accumulation original.
// =====================================================
void accumulateOriginal(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly>& ct, uint32_t batchSize) {
	auto ctRotated = ct;
	for (uint32_t i = 1; i < batchSize; ++i) {
		ctRotated = cc->EvalRotate(ctRotated, 1);
		ct		  = cc->EvalAdd(ct, ctRotated);
	}
}

// =====================================================
// Accumulation comparison.
// =====================================================
void accumulation_comparison(CryptoContext<DCRTPoly>& cc, const KeyPair<DCRTPoly>& keys, uint32_t batchSize) {
	std::cout << "\n=== Demo 1: Accumulation Performance ===" << std::endl;

	std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	Plaintext ptxtData		 = cc->MakeCKKSPackedPlaintext(data);

	// Original accumulation.
	auto ct1 = cc->Encrypt(keys.publicKey, ptxtData);
	cc->Synchronize();
	auto start = std::chrono::high_resolution_clock::now();
	accumulateOriginal(cc, ct1, batchSize);
	cc->Synchronize();
	auto end							 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> original = end - start;

	// Optimized accumulation.
	auto ct2 = cc->Encrypt(keys.publicKey, ptxtData);
	cc->Synchronize();
	start = std::chrono::high_resolution_clock::now();
	accumulateOptimized(cc, ct2, batchSize);
	cc->Synchronize();
	end									   = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> optimized = end - start;

	std::cout << std::endl << "==== Accumulation (sum reduction) ====" << std::endl;
	std::cout << "\tOriginal:   " << std::fixed << std::setprecision(4) << original.count() * 1000 << " ms" << std::endl;
	std::cout << "\tOptimized:  " << std::fixed << std::setprecision(4) << optimized.count() * 1000 << " ms" << std::endl;
	std::cout << "\tSpeedup:    " << std::fixed << std::setprecision(2) << original.count() / optimized.count() << "x" << std::endl;

	// Verify results.
	Plaintext result;
	cc->Decrypt(keys.secretKey, ct1, &result);
	result->SetLength(batchSize);
	std::cout << "\tOriginal result:        " << result;
	cc->Decrypt(keys.secretKey, ct2, &result);
	result->SetLength(batchSize);
	std::cout << "\tOptimized result:       " << result;
}

// =====================================================
// Hoisted rotations vs normal rotations.
// =====================================================
void hoisted_rotations(CryptoContext<DCRTPoly>& cc, const KeyPair<DCRTPoly>& keys, uint32_t batchSize) {
	std::cout << std::endl << "==== Hoisted vs Normal Rotations ====" << std::endl;

	std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	Plaintext ptxtData		 = cc->MakeCKKSPackedPlaintext(data);
	auto ct					 = cc->Encrypt(keys.publicKey, ptxtData);

	std::vector<int32_t> rotIndices = { 1, 2, 3 };
	uint32_t m						= cc->GetCyclotomicOrder();

	// Normal rotations.
	cc->Synchronize();
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10; ++i) {
		for (auto idx : rotIndices) {
			cc->EvalRotate(ct, idx);
		}
	}
	cc->Synchronize();
	auto end								 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> normalTime = end - start;

	// Hoisted rotations.
	cc->Synchronize();
	start		 = std::chrono::high_resolution_clock::now();
	auto precomp = cc->EvalFastRotationPrecompute(ct);
	for (int i = 0; i < 10; ++i) {
		auto rotations = cc->EvalFastRotation(ct, rotIndices, m, precomp);
	}
	cc->Synchronize();
	end										  = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> hoistedTime = end - start;

	std::cout << "\tNormal rotations:  " << std::fixed << std::setprecision(4) << normalTime.count() * 1000 / 10 << " ms" << std::endl;
	std::cout << "\tHoisted rotations: " << std::fixed << std::setprecision(4) << hoistedTime.count() * 1000 / 10 << " ms" << std::endl;
	std::cout << "\tSpeedup:           " << std::fixed << std::setprecision(2) << normalTime.count() / hoistedTime.count() << "x" << std::endl;
}

// =====================================================
// SPARSE_TERNARY Bootstrapping.
// =====================================================
void sparse_bootstrap() {
	std::cout << std::endl << "==== SPARSE_TERNARY Bootstrapping ====" << std::endl;

	uint32_t multDepth = 22;
	uint32_t ring_dim  = 1 << 16;
	uint32_t batchSize = 1 << 14;
	uint32_t dcrtBits  = 52;
	uint32_t firstMod  = 56;
	uint32_t dnum	   = 3;

	std::vector<uint32_t> levelBudget = { 3, 3 };
	std::vector<uint32_t> bsgs		  = { 16, 16 };

	CCParams<CryptoContextCKKSRNS> params;
	params.SetDevices({ 0 });
	params.SetSecurityLevel(SecurityLevel::HEStd_NotSet);
	params.SetRingDim(ring_dim);
	params.SetMultiplicativeDepth(multDepth);
	params.SetScalingModSize(dcrtBits);
	params.SetScalingTechnique(FLEXIBLEAUTO);
	params.SetFirstModSize(firstMod);
	params.SetKeySwitchTechnique(HYBRID);
	params.SetNumLargeDigits(dnum);
	params.SetBatchSize(batchSize);
	params.SetCiphertextAutoload(true);

	// ========
	// Secret key distribution.
	// ========
	params.SetSecretKeyDist(SPARSE_ENCAPSULATED);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);

	std::cout << "\tRing dimension: " << cc->GetRingDimension() << std::endl;

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalBootstrapSetup(levelBudget, bsgs, batchSize, 0);
	cc->EvalBootstrapKeyGen(keys.secretKey, batchSize);

	cc->LoadContext(keys.publicKey);
	
	std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	Plaintext ptxt			 = cc->MakeCKKSPackedPlaintext(data, 1, multDepth - 1);
	auto ct					 = cc->Encrypt(keys.publicKey, ptxt);

	uint32_t levelBefore = ct->GetLevel();
	cc->Synchronize();
	auto start = std::chrono::high_resolution_clock::now();
	auto ctBS  = cc->EvalBootstrap(ct);
	cc->Synchronize();
	auto end						   = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	uint32_t levelAfter				   = ctBS->GetLevel();

	std::cout << "\tLevel before bootstrap: " << levelBefore << std::endl;
	std::cout << "\tLevel after bootstrap:  " << levelAfter << std::endl;
	std::cout << "\tLevels remaining:       " << (multDepth - levelAfter) << std::endl;
	std::cout << "\tBootstrap time:         " << std::fixed << std::setprecision(4) << time.count() << " s" << std::endl;

	Plaintext result;
	cc->Decrypt(keys.secretKey, ctBS, &result);
	result->SetLength(batchSize);
}

void uniform_bootstrap() {
	std::cout << std::endl << "==== UNIFORM_TERNARY Bootstrapping ====" << std::endl;

	uint32_t multDepth = 22;
	uint32_t ring_dim  = 1 << 16;
	uint32_t batchSize = 1 << 14;
	uint32_t dcrtBits  = 52;
	uint32_t firstMod  = 56;
	uint32_t dnum	   = 3;

	std::vector<uint32_t> levelBudget = { 3, 3 };
	std::vector<uint32_t> bsgs		  = { 16, 16 };

	CCParams<CryptoContextCKKSRNS> params;
	params.SetDevices({ 0 });
	params.SetSecurityLevel(SecurityLevel::HEStd_NotSet);
	params.SetRingDim(ring_dim);
	params.SetMultiplicativeDepth(multDepth);
	params.SetScalingModSize(dcrtBits);
	params.SetScalingTechnique(FLEXIBLEAUTO);
	params.SetFirstModSize(firstMod);
	params.SetKeySwitchTechnique(HYBRID);
	params.SetNumLargeDigits(dnum);
	params.SetBatchSize(batchSize);
	params.SetCiphertextAutoload(true);

	// ========
	// Secret key distribution.
	// ========
	params.SetSecretKeyDist(UNIFORM_TERNARY);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);

	std::cout << "\tRing dimension: " << cc->GetRingDimension() << std::endl;

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalBootstrapSetup(levelBudget, bsgs, batchSize, 0);
	cc->EvalBootstrapKeyGen(keys.secretKey, batchSize);
	cc->LoadContext(keys.publicKey);

	std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	Plaintext ptxt			 = cc->MakeCKKSPackedPlaintext(data, 1, multDepth - 1);
	auto ct					 = cc->Encrypt(keys.publicKey, ptxt);

	uint32_t levelBefore = ct->GetLevel();
	cc->Synchronize();
	auto start = std::chrono::high_resolution_clock::now();
	auto ctBS  = cc->EvalBootstrap(ct);
	cc->Synchronize();
	auto end						   = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	uint32_t levelAfter				   = ctBS->GetLevel();

	std::cout << "\tLevel before bootstrap: " << levelBefore << std::endl;
	std::cout << "\tLevel after bootstrap:  " << levelAfter << std::endl;
	std::cout << "\tLevels remaining:       " << (multDepth - levelAfter) << std::endl;
	std::cout << "\tBootstrap time:         " << std::fixed << std::setprecision(4) << time.count() << " s" << std::endl;

	Plaintext result;
	cc->Decrypt(keys.secretKey, ctBS, &result);
	result->SetLength(batchSize);
}

int main() {
	// =====================================================
	// Part 1: Accumulation and Hoisting Demos.
	// =====================================================
	{
		uint32_t multDepth			 = 10;
		uint32_t batchSize			 = 8;
		uint32_t ring_dim 			 = 1 << 12;
		ScalingTechnique rescaleTech = FLEXIBLEAUTO;

		uint32_t dcrtBits = 59;
		uint32_t firstMod = 60;
		uint32_t dnum	  = 3;

		CCParams<CryptoContextCKKSRNS> parameters;
		parameters.SetSecurityLevel(SecurityLevel::HEStd_NotSet);
		parameters.SetRingDim(ring_dim);
		parameters.SetMultiplicativeDepth(multDepth);
		parameters.SetScalingModSize(dcrtBits);
		parameters.SetScalingTechnique(rescaleTech);
		parameters.SetFirstModSize(firstMod);
		parameters.SetKeySwitchTechnique(HYBRID);
		parameters.SetNumLargeDigits(dnum);
		parameters.SetBatchSize(batchSize);
		parameters.SetDevices({ 0 });
		parameters.SetPlaintextAutoload(false);
		parameters.SetCiphertextAutoload(true);

		CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		cc->Enable(FHE);

		std::cout << "CKKS scheme using ring dimension: " << cc->GetRingDimension() << std::endl << std::endl;

		auto keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);

		std::vector<int32_t> rotationIndices = { 1, -1, 2, -2, 3, 4, -4, 8, -8 };
		cc->EvalRotateKeyGen(keys.secretKey, rotationIndices);

		cc->LoadContext(keys.publicKey);

		accumulation_comparison(cc, keys, batchSize);
		hoisted_rotations(cc, keys, batchSize);
	}

	// =====================================================
	// Part 2: Bootstrapping Demos.
	// =====================================================

	uniform_bootstrap();
	sparse_bootstrap();

	return 0;
}
