//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

#include <cstdint>
#include <random>

#include <fideslib.hpp>
#include <vector>

using namespace fideslib;

void SimpleBootstrapExample();
void BootstrapExample(uint32_t numSlots);
void BootstrapExampleSSE(uint32_t numSlots);

int main(int argc, char* argv[]) {
	SimpleBootstrapExample();
	BootstrapExample(1 << 11);
	BootstrapExampleSSE(1 << 11);
}

std::vector<int> devices = { 0 };

void SimpleBootstrapExample() {
	CCParams<CryptoContextCKKSRNS> parameters;

	SecretKeyDist secretKeyDist = UNIFORM_TERNARY;

	parameters.SetSecretKeyDist(secretKeyDist);
	parameters.SetSecurityLevel(HEStd_NotSet);
	parameters.SetRingDim(1 << 12);

#if NATIVEINT == 128
	ScalingTechnique rescaleTech = FIXEDAUTO;
	uint32_t dcrtBits			 = 78;
	uint32_t firstMod			 = 89;
#else
	ScalingTechnique rescaleTech = FLEXIBLEAUTO;
	uint32_t dcrtBits			 = 59;
	uint32_t firstMod			 = 60;
#endif

	parameters.SetScalingModSize(dcrtBits);
	parameters.SetScalingTechnique(rescaleTech);
	parameters.SetFirstModSize(firstMod);
	parameters.SetKeySwitchTechnique(HYBRID);
	parameters.SetBackend(Backend::CUDA);

	std::vector<uint32_t> levelBudget	   = { 3, 3 };
	uint32_t levelsAvailableAfterBootstrap = 10;
	uint32_t depth						   = 25;
	parameters.SetMultiplicativeDepth(depth);

	CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);

	cryptoContext->Enable(PKE);
	cryptoContext->Enable(KEYSWITCH);
	cryptoContext->Enable(LEVELEDSHE);
	cryptoContext->Enable(ADVANCEDSHE);
	cryptoContext->Enable(FHE);

	uint32_t ringDim  = cryptoContext->GetRingDimension();
	uint32_t numSlots = ringDim >> 1;
	std::cout << "CKKS scheme is using ring dimension " << ringDim << std::endl;

	auto keyPair = cryptoContext->KeyGen();
	cryptoContext->EvalMultKeyGen(keyPair.secretKey);
	
	cryptoContext->LoadContext(keyPair.publicKey);

	cryptoContext->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
	cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

	std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	size_t encodedLength  = x.size();

	Plaintext ptxt = cryptoContext->MakeCKKSPackedPlaintext(x, 1, depth - 1);

	ptxt->SetLength(encodedLength);
	//std::cout << "Input: " << ptxt;

	Ciphertext<DCRTPoly> ciph = cryptoContext->Encrypt(keyPair.publicKey, ptxt);

	std::cout << "Initial number of levels remaining: " << depth - ciph->GetLevel() << std::endl;

	cryptoContext->EvalBootstrapInPlace(ciph);

	std::cout << "Number of levels remaining after bootstrapping: " << depth - ciph->GetLevel() - (ciph->GetNoiseScaleDeg() - 1) << std::endl;

	Plaintext result;
	cryptoContext->Decrypt(keyPair.secretKey, ciph, &result);
	result->SetLength(encodedLength);
	//std::cout << "Output after bootstrapping: " << result;
}

void BootstrapExample(uint32_t numSlots) {
	CCParams<CryptoContextCKKSRNS> parameters;

	SecretKeyDist secretKeyDist = SPARSE_TERNARY;
	parameters.SetSecretKeyDist(secretKeyDist);
	parameters.SetSecurityLevel(HEStd_NotSet);
	parameters.SetRingDim(1 << 12);
	parameters.SetNumLargeDigits(3);
	parameters.SetKeySwitchTechnique(HYBRID);
	parameters.SetBackend(Backend::CUDA);

#if NATIVEINT == 128 && !defined(__EMSCRIPTEN__)
	ScalingTechnique rescaleTech = FIXEDAUTO;
	uint32_t dcrtBits			 = 78;
	uint32_t firstMod			 = 89;
#else
	// All modes are supported for 64-bit CKKS bootstrapping.
	ScalingTechnique rescaleTech = FLEXIBLEAUTO;
	uint32_t dcrtBits			 = 59;
	uint32_t firstMod			 = 60;
#endif

	parameters.SetScalingModSize(dcrtBits);
	parameters.SetScalingTechnique(rescaleTech);
	parameters.SetFirstModSize(firstMod);

	std::vector<uint32_t> levelBudget	   = { 3, 3 };
	std::vector<uint32_t> bsgsDim		   = { 0, 0 };
	uint32_t levelsAvailableAfterBootstrap = 10;
	uint32_t depth						   = 25;
	parameters.SetMultiplicativeDepth(depth);

	CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);

	cryptoContext->Enable(PKE);
	cryptoContext->Enable(KEYSWITCH);
	cryptoContext->Enable(LEVELEDSHE);
	cryptoContext->Enable(ADVANCEDSHE);
	cryptoContext->Enable(FHE);

	uint32_t ringDim = cryptoContext->GetRingDimension();
	std::cout << "CKKS scheme is using ring dimension " << ringDim << std::endl;

	auto keyPair = cryptoContext->KeyGen();
	cryptoContext->EvalMultKeyGen(keyPair.secretKey);
	
	cryptoContext->LoadContext(keyPair.publicKey);

	cryptoContext->EvalBootstrapSetup(levelBudget, bsgsDim, numSlots, 0);
	cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

	std::vector<double> x;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);
	for (size_t i = 0; i < numSlots; i++) {
		x.push_back(dis(gen));
	}

	Plaintext ptxt = cryptoContext->MakeCKKSPackedPlaintext(x, 1, depth - 1, nullptr, numSlots);
	ptxt->SetLength(numSlots);
	//std::cout << "Input: " << ptxt;

	Ciphertext<DCRTPoly> ciph = cryptoContext->Encrypt(keyPair.publicKey, ptxt);

	std::cout << "Initial number of levels remaining: " << depth - ciph->GetLevel() << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	auto ciphertextAfter = cryptoContext->EvalBootstrap(ciph);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << "Bootstrapping time: " << diff.count() << " s" << std::endl;

	std::cout << "Number of levels remaining after bootstrapping: " << depth - ciphertextAfter->GetLevel() << std::endl;

	Plaintext result;
	cryptoContext->Decrypt(keyPair.secretKey, ciphertextAfter, &result);
	result->SetLength(numSlots);
	//std::cout << "Output after bootstrapping: " << result;
}


void BootstrapExampleSSE(uint32_t numSlots) {
	CCParams<CryptoContextCKKSRNS> parameters;

	SecretKeyDist secretKeyDist = fideslib::SPARSE_ENCAPSULATED;
	parameters.SetSecretKeyDist(secretKeyDist);
	parameters.SetSecurityLevel(HEStd_NotSet);
	parameters.SetRingDim(1 << 12);
	parameters.SetNumLargeDigits(3);
	parameters.SetKeySwitchTechnique(HYBRID);
	parameters.SetBackend(Backend::CUDA);

#if NATIVEINT == 128 && !defined(__EMSCRIPTEN__)
	ScalingTechnique rescaleTech = FIXEDAUTO;
	uint32_t dcrtBits			 = 78;
	uint32_t firstMod			 = 89;
#else
	// All modes are supported for 64-bit CKKS bootstrapping.
	ScalingTechnique rescaleTech = FLEXIBLEAUTO;
	uint32_t dcrtBits			 = 59;
	uint32_t firstMod			 = 60;
#endif

	parameters.SetScalingModSize(dcrtBits);
	parameters.SetScalingTechnique(rescaleTech);
	parameters.SetFirstModSize(firstMod);

	std::vector<uint32_t> levelBudget	   = { 3, 3 };
	std::vector<uint32_t> bsgsDim		   = { 0, 0 };
	uint32_t levelsAvailableAfterBootstrap = 10;
	uint32_t depth						   = 25;
	parameters.SetMultiplicativeDepth(depth);

	CryptoContext<DCRTPoly> cryptoContext = GenCryptoContext(parameters);

	cryptoContext->Enable(PKE);
	cryptoContext->Enable(KEYSWITCH);
	cryptoContext->Enable(LEVELEDSHE);
	cryptoContext->Enable(ADVANCEDSHE);
	cryptoContext->Enable(FHE);

	uint32_t ringDim = cryptoContext->GetRingDimension();
	std::cout << "CKKS scheme is using ring dimension " << ringDim << std::endl;

	auto keyPair = cryptoContext->KeyGen();
	cryptoContext->EvalMultKeyGen(keyPair.secretKey);
	
	cryptoContext->LoadContext(keyPair.publicKey);
	
	cryptoContext->EvalBootstrapSetup(levelBudget, bsgsDim, numSlots, 0);
	cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

	std::vector<double> x;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);
	for (size_t i = 0; i < numSlots; i++) {
		x.push_back(dis(gen));
	}

	Plaintext ptxt = cryptoContext->MakeCKKSPackedPlaintext(x, 1, depth - 1, nullptr, numSlots);
	ptxt->SetLength(numSlots);
	//std::cout << "Input: " << ptxt;

	Ciphertext<DCRTPoly> ciph = cryptoContext->Encrypt(keyPair.publicKey, ptxt);

	std::cout << "Initial number of levels remaining: " << depth - ciph->GetLevel() << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	auto ciphertextAfter = cryptoContext->EvalBootstrap(ciph);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << "Bootstrapping time: " << diff.count() << " s" << std::endl;

	std::cout << "Number of levels remaining after bootstrapping: " << depth - ciphertextAfter->GetLevel() << std::endl;

	Plaintext result;
	cryptoContext->Decrypt(keyPair.secretKey, ciphertextAfter, &result);
	result->SetLength(numSlots);
	//std::cout << "Output after bootstrapping: " << result;
}