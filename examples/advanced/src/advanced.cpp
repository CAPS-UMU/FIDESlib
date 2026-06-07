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

#include "fideslib.hpp"

#include <chrono>

using namespace fideslib;

void AutomaticRescaleDemo(ScalingTechnique scalTech);
void ManualRescaleDemo(ScalingTechnique scalTech);
void HybridKeySwitchingDemo1();
void HybridKeySwitchingDemo2();
void FastRotationsDemo1();
void FastRotationsDemo2();

int main(int argc, char* argv[]) {

	AutomaticRescaleDemo(FLEXIBLEAUTO);
	AutomaticRescaleDemo(FLEXIBLEAUTOEXT);
	AutomaticRescaleDemo(FIXEDAUTO);
	ManualRescaleDemo(FIXEDMANUAL);

	HybridKeySwitchingDemo1();
	HybridKeySwitchingDemo2();

	FastRotationsDemo1();
	FastRotationsDemo2();

	return 0;
}

std::vector<int> devices = { 0 };

void AutomaticRescaleDemo(ScalingTechnique scalTech) {

	if (scalTech == FLEXIBLEAUTO) {
		std::cout << " ===== FlexibleAutoDemo ============= " << std::endl;
	} else if (scalTech == FLEXIBLEAUTOEXT) {
		std::cout << " ===== FlexibleAutoExtDemo ============= " << std::endl;
	} else {
		std::cout << " ===== FixedAutoDemo ============= " << std::endl;
	}

	uint32_t batchSize = 8;
	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(5);
	parameters.SetScalingModSize(50);
	parameters.SetScalingTechnique(scalTech);
	parameters.SetBatchSize(batchSize);
	parameters.SetBackend(Backend::CUDA);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl;

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);

	cc->LoadContext(keys.publicKey);

	// Input
	std::vector<double> x = { 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07 };
	Plaintext ptxt		  = cc->MakeCKKSPackedPlaintext(x);

	std::cout << "Input x: " << ptxt;

	auto c = cc->Encrypt(ptxt, keys.publicKey);

	auto c2	  = cc->EvalMult(c, c);						// x^2.
	auto c4	  = cc->EvalMult(c2, c2);					// x^4.
	auto c8	  = cc->EvalMult(c4, c4);					// x^8.
	auto c16  = cc->EvalMult(c8, c8);					// x^16.
	auto c9	  = cc->EvalMult(c8, c);					// x^9.
	auto c18  = cc->EvalMult(c16, c2);					// x^18.
	auto cRes = cc->EvalAdd(cc->EvalAdd(c18, c9), 1.0); // Final result.

	Plaintext result;
	std::cout.precision(8);

	cc->Decrypt(cRes, keys.secretKey, &result);
	result->SetLength(batchSize);
	std::cout << "x^18 + x^9 + 1 = " << result;
}

void ManualRescaleDemo(ScalingTechnique scalTech) {

	std::cout << " ===== FixedManualDemo ============= " << std::endl;

	uint32_t batchSize = 8;
	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(5);
	parameters.SetScalingModSize(50);
	parameters.SetBatchSize(batchSize);
	parameters.SetScalingTechnique(scalTech);
	parameters.SetBackend(Backend::CUDA);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl;

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);

	cc->LoadContext(keys.publicKey);

	// Input.
	std::vector<double> x = { 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07 };
	Plaintext ptxt		  = cc->MakeCKKSPackedPlaintext(x);

	std::cout << "Input x: " << ptxt;

	auto c = cc->Encrypt(keys.publicKey, ptxt);

	// x^2.
	auto c2_depth2 = cc->EvalMult(c, c);
	auto c2_depth1 = cc->Rescale(c2_depth2);
	// x^4.
	auto c4_depth2 = cc->EvalMult(c2_depth1, c2_depth1);
	auto c4_depth1 = cc->Rescale(c4_depth2);
	// x^8.
	auto c8_depth2 = cc->EvalMult(c4_depth1, c4_depth1);
	auto c8_depth1 = cc->Rescale(c8_depth2);
	// x^16.
	auto c16_depth2 = cc->EvalMult(c8_depth1, c8_depth1);
	auto c16_depth1 = cc->Rescale(c16_depth2);
	// x^9.
	auto c9_depth2 = cc->EvalMult(c8_depth1, c);
	// x^18.
	auto c18_depth2 = cc->EvalMult(c16_depth1, c2_depth1);
	// Final result.
	auto cRes_depth2 = cc->EvalAdd(cc->EvalAdd(c18_depth2, c9_depth2), 1.0);
	auto cRes_depth1 = cc->Rescale(cRes_depth2);

	Plaintext result;
	std::cout.precision(8);

	cc->Decrypt(keys.secretKey, cRes_depth1, &result);
	result->SetLength(batchSize);
	std::cout << "x^18 + x^9 + 1 = " << result;
}

void HybridKeySwitchingDemo1() {

	std::cout << " ===== HybridKeySwitchingDemo1 ============= " << std::endl;

	uint32_t dnum	   = 2;
	uint32_t batchSize = 8;

	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(5);
	parameters.SetScalingModSize(50);
	parameters.SetBatchSize(batchSize);
	parameters.SetScalingTechnique(FLEXIBLEAUTO);
	parameters.SetNumLargeDigits(dnum);
	parameters.SetBackend(Backend::CUDA);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl;

	std::cout << "- Using HYBRID key switching with " << dnum << " digits" << std::endl;

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);

	auto keys = cc->KeyGen();
	cc->EvalRotateKeyGen(keys.secretKey, { 1, -2 });

	cc->LoadContext(keys.publicKey);

	// Input.
	std::vector<double> x = { 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 };
	Plaintext ptxt		  = cc->MakeCKKSPackedPlaintext(x);

	std::cout << "Input x: " << ptxt;

	auto c = cc->Encrypt(keys.publicKey, ptxt);

	auto start = std::chrono::high_resolution_clock::now();

	auto cRot1 = cc->EvalRotate(c, 1);
	auto cRot2 = cc->EvalRotate(cRot1, -2);

	auto end		 = std::chrono::high_resolution_clock::now();
	auto time2digits = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	Plaintext result;
	std::cout.precision(8);

	cc->Decrypt(keys.secretKey, cRot2, &result);
	result->SetLength(batchSize);
	std::cout << "x rotate by -1 = " << result;
	std::cout << " - 2 rotations with HYBRID (2 digits) took " << time2digits << "ms" << std::endl;
}

void HybridKeySwitchingDemo2() {

	std::cout << " ===== HybridKeySwitchingDemo2 ============= " << std::endl;

	uint32_t dnum	   = 3;
	uint32_t batchSize = 8;

	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(5);
	parameters.SetScalingModSize(50);
	parameters.SetBatchSize(batchSize);
	parameters.SetScalingTechnique(FLEXIBLEAUTO);
	parameters.SetNumLargeDigits(dnum);
	parameters.SetBackend(Backend::CUDA);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl;
	std::cout << "- Using HYBRID key switching with " << dnum << " digits" << std::endl;

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);

	auto keys = cc->KeyGen();
	cc->EvalRotateKeyGen(keys.secretKey, { 1, -2 });

	cc->LoadContext(keys.publicKey);

	// Input.
	std::vector<double> x = { 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 };
	Plaintext ptxt		  = cc->MakeCKKSPackedPlaintext(x);

	std::cout << "Input x: " << ptxt;

	auto c = cc->Encrypt(keys.publicKey, ptxt);

	auto start = std::chrono::high_resolution_clock::now();

	auto cRot1 = cc->EvalRotate(c, 1);
	auto cRot2 = cc->EvalRotate(cRot1, -2);

	auto end		   = std::chrono::high_resolution_clock::now();
	double time3digits = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	Plaintext result;
	std::cout.precision(8);

	cc->Decrypt(keys.secretKey, cRot2, &result);
	result->SetLength(batchSize);
	std::cout << "x rotate by -1 = " << result;
	std::cout << " - 2 rotations with HYBRID (3 digits) took " << time3digits << "ms" << std::endl;
}

void FastRotationsDemo1() {

	std::cout << " ===== FastRotationsDemo1 ============= " << std::endl;

	uint32_t dnum	   = 3;
	uint32_t batchSize = 8;

	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(5);
	parameters.SetScalingModSize(59);
	parameters.SetFirstModSize(60);
	parameters.SetBatchSize(batchSize);
	parameters.SetScalingTechnique(FLEXIBLEAUTO);
	parameters.SetNumLargeDigits(dnum);
	parameters.SetBackend(Backend::CUDA);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	uint32_t N = cc->GetRingDimension();
	std::cout << "CKKS scheme is using ring dimension " << N << std::endl;

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);

	auto keys = cc->KeyGen();
	cc->EvalRotateKeyGen(keys.secretKey, { 1, 2, 3, 4, 5, 6, 7 });

	cc->LoadContext(keys.publicKey);

	// Input.
	std::vector<double> x = { 0, 0, 0, 0, 0, 0, 0, 1 };
	Plaintext ptxt		  = cc->MakeCKKSPackedPlaintext(x);

	std::cout << "Input x: " << ptxt;

	auto c = cc->Encrypt(keys.publicKey, ptxt);

	Ciphertext<DCRTPoly> cRot1, cRot2, cRot3, cRot4, cRot5, cRot6, cRot7;

	auto startNoHoisting = std::chrono::high_resolution_clock::now();

	cRot1 = cc->EvalRotate(c, 1);
	cRot2 = cc->EvalRotate(c, 2);
	cRot3 = cc->EvalRotate(c, 3);
	cRot4 = cc->EvalRotate(c, 4);
	cRot5 = cc->EvalRotate(c, 5);
	cRot6 = cc->EvalRotate(c, 6);
	cRot7 = cc->EvalRotate(c, 7);

	double timeNoHoisting = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startNoHoisting).count();

	auto cResNoHoist = c; // + cRot1 + cRot2 + cRot3 + cRot4 + cRot5 + cRot6 + cRot7;

	// M is the cyclotomic order and we need it to call EvalFastRotation.
	uint32_t M = 2 * N;

	// Then, we perform 7 rotations with hoisting.

	auto startHoisting = std::chrono::high_resolution_clock::now();

	auto cPrecomp = cc->EvalFastRotationPrecompute(c);
	cRot1		  = cc->EvalFastRotation(c, 1, M, cPrecomp);
	cRot2		  = cc->EvalFastRotation(c, 2, M, cPrecomp);
	cRot3		  = cc->EvalFastRotation(c, 3, M, cPrecomp);
	cRot4		  = cc->EvalFastRotation(c, 4, M, cPrecomp);
	cRot5		  = cc->EvalFastRotation(c, 5, M, cPrecomp);
	cRot6		  = cc->EvalFastRotation(c, 6, M, cPrecomp);
	cRot7		  = cc->EvalFastRotation(c, 7, M, cPrecomp);

	double timeHoisting = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startHoisting).count();

	auto cResHoist = c; // + cRot1 + cRot2 + cRot3 + cRot4 + cRot5 + cRot6 + cRot7;

	auto startMultiIndexHoisting = std::chrono::high_resolution_clock::now();
	auto rots					 = cc->EvalFastRotation(c, { 1, 2, 3, 4, 5, 6, 7 }, M, cPrecomp);
	double timeMultiIndexHoisting = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startMultiIndexHoisting).count();

	auto cResMultiIndexHoist = c; // + rots[0] + rots[1] + rots[2] + rots[3] + rots[4] + rots[5] + rots[6];

	Plaintext result;
	std::cout.precision(8);

	cc->Decrypt(keys.secretKey, cResNoHoist, &result);
	result->SetLength(batchSize);
	std::cout << "Result without hoisting = " << result;
	std::cout << " - 7 rotations on x without hoisting took " << timeNoHoisting << "ms" << std::endl;

	cc->Decrypt(keys.secretKey, cResHoist, &result);
	result->SetLength(batchSize);
	std::cout << "Result with hoisting = " << result;
	std::cout << " - 7 rotations on x with hoisting took " << timeHoisting << "ms" << std::endl;

	cc->Decrypt(keys.secretKey, cResMultiIndexHoist, &result);
	result->SetLength(batchSize);
	std::cout << "Result with multi-index hoisting = " << result;
	std::cout << " - 7 rotations on x with multi-index hoisting took " << timeMultiIndexHoisting << "ms" << std::endl;
}

void FastRotationsDemo2() {

	std::cout << " ===== FastRotationsDemo2 ============= " << std::endl;

	uint32_t digitSize = 10;
	uint32_t batchSize = 8;

	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(1);
	parameters.SetScalingModSize(50);
	parameters.SetBatchSize(batchSize);
	parameters.SetScalingTechnique(FLEXIBLEAUTO);
	// parameters.SetKeySwitchTechnique(BV);
	parameters.SetFirstModSize(60);
	parameters.SetDigitSize(digitSize);
	parameters.SetBackend(Backend::CUDA);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	uint32_t N = cc->GetRingDimension();
	std::cout << "CKKS scheme is using ring dimension " << N << std::endl;

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);

	auto keys = cc->KeyGen();
	cc->EvalRotateKeyGen(keys.secretKey, { 1, 2, 3, 4, 5, 6, 7 });

	cc->LoadContext(keys.publicKey);

	// Input.
	std::vector<double> x = { 0, 0, 0, 0, 0, 0, 0, 1 };
	Plaintext ptxt		  = cc->MakeCKKSPackedPlaintext(x);

	std::cout << "Input x: " << ptxt;

	auto c = cc->Encrypt(keys.publicKey, ptxt);

	Ciphertext<DCRTPoly> cRot1, cRot2, cRot3, cRot4, cRot5, cRot6, cRot7;

	// First, we perform 7 regular (non-hoisted) rotations.
	auto startNoHoisting  = std::chrono::high_resolution_clock::now();
	cRot1				  = cc->EvalRotate(c, 1);
	cRot2				  = cc->EvalRotate(c, 2);
	cRot3				  = cc->EvalRotate(c, 3);
	cRot4				  = cc->EvalRotate(c, 4);
	cRot5				  = cc->EvalRotate(c, 5);
	cRot6				  = cc->EvalRotate(c, 6);
	cRot7				  = cc->EvalRotate(c, 7);
	double timeNoHoisting = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startNoHoisting).count();

	auto cResNoHoist = c; // + cRot1 + cRot2 + cRot3 + cRot4 + cRot5 + cRot6 + cRot7;

	// M is the cyclotomic order and we need it to call EvalFastRotation.
	uint32_t M = 2 * N;

	// Then, we perform 7 rotations with hoisting.
	auto startHoisting	= std::chrono::high_resolution_clock::now();
	auto cPrecomp		= cc->EvalFastRotationPrecompute(c);
	cRot1				= cc->EvalFastRotation(c, 1, M, cPrecomp);
	cRot2				= cc->EvalFastRotation(c, 2, M, cPrecomp);
	cRot3				= cc->EvalFastRotation(c, 3, M, cPrecomp);
	cRot4				= cc->EvalFastRotation(c, 4, M, cPrecomp);
	cRot5				= cc->EvalFastRotation(c, 5, M, cPrecomp);
	cRot6				= cc->EvalFastRotation(c, 6, M, cPrecomp);
	cRot7				= cc->EvalFastRotation(c, 7, M, cPrecomp);
	double timeHoisting = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startHoisting).count();

	auto cResHoist = c; // + cRot1 + cRot2 + cRot3 + cRot4 + cRot5 + cRot6 + cRot7;

	Plaintext result;
	std::cout.precision(8);

	cc->Decrypt(keys.secretKey, cResNoHoist, &result);
	result->SetLength(batchSize);
	std::cout << "Result without hoisting = " << result;
	std::cout << " - 7 rotations on x without hoisting took " << timeNoHoisting << "ms" << std::endl;

	cc->Decrypt(keys.secretKey, cResHoist, &result);
	result->SetLength(batchSize);
	std::cout << "Result with hoisting = " << result;
	std::cout << " - 7 rotations on x with hoisting took " << timeHoisting << "ms" << std::endl;
}