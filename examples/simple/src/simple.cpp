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

#include <fideslib.hpp>

#include <iostream>

using namespace fideslib;

int main() {

	// Step 1: Setup CryptoContext.

	uint32_t multDepth	  = 1;
	uint32_t scaleModSize = 50;
	uint32_t batchSize	  = 8;

	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(multDepth);
	parameters.SetScalingModSize(scaleModSize);
	parameters.SetBatchSize(batchSize);
	parameters.SetBackend(Backend::CUDA);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	// Step 1.1: Enable the features that you wish to use.
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);

	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl;

	// Step 2: Key Generation.

	auto keys = cc->KeyGen();

	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, { 1, -2 });

	// Load on the GPU.
	cc->LoadContext(keys.publicKey);

	// Step 3: Encoding and encryption of inputs.

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

	Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

	std::cout << "Input x1: " << ptxt1;
	std::cout << "Input x2: " << ptxt2;
	
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2); 

	// Step 4: Evaluation.

	// Homomorphic addition.
	auto cAdd = cc->EvalAdd(c1, c2);

	// Homomorphic subtraction.
	auto cSub = cc->EvalSub(c1, c2);

	// Homomorphic scalar multiplication..
	auto cScalar = cc->EvalMult(c1, 4.0);

	// Homomorphic multiplication.
	auto cMul = cc->EvalMult(c1, c2);

	// Homomorphic rotations.
	auto cRot1 = cc->EvalRotate(c1, 1);
	auto cRot2 = cc->EvalRotate(c1, -2);

	// Step 5: Decryption and output.

	Plaintext result;

	std::cout.precision(8);
	std::cout << "Results of homomorphic computations: " << std::endl;

	cc->Decrypt(keys.secretKey, c1, &result);
	result->SetLength(batchSize);
	std::cout << "x1 = " << result;
	std::cout << "Estimated precision in bits: " << result->GetLogPrecision() << std::endl;

	// Decrypt the result of addition.
	cc->Decrypt(keys.secretKey, cAdd, &result);
	result->SetLength(batchSize);
	std::cout << "x1 + x2 = " << result;
	std::cout << "Estimated precision in bits: " << result->GetLogPrecision() << std::endl;

	// Decrypt the result of subtraction.
	cc->Decrypt(keys.secretKey, cSub, &result);
	result->SetLength(batchSize);
	std::cout << "x1 - x2 = " << result;

	// Decrypt the result of scalar multiplication.
	cc->Decrypt(keys.secretKey, cScalar, &result);
	result->SetLength(batchSize);
	std::cout << "4 * x1 = " << result;

	// Decrypt the result of multiplication.
	cc->Decrypt(keys.secretKey, cMul, &result);
	result->SetLength(batchSize);
	std::cout << "x1 * x2 = " << result;

	// Decrypt the result of rotations.
	cc->Decrypt(keys.secretKey, cRot1, &result);
	result->SetLength(batchSize);
	std::cout << "In rotations, very small outputs (~10^-10 here) correspond to 0's:" << std::endl;
	std::cout << "x1 rotate by 1 = " << result;
	cc->Decrypt(keys.secretKey, cRot2, &result);
	result->SetLength(batchSize);
	std::cout << "x1 rotate by -2 = " << result;

	return 0;
}