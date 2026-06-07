#include <fideslib.hpp>

#include <iostream>

using namespace fideslib;

int main() {

	// =====================================================
	// Step 1: Define our parameters.
	// =====================================================

	// How many multiplications can be done.
	uint32_t multDepth = 10;
	// Ring dimension of the scheme.
	uint32_t ring_dim = 1 << 12;
	// How many slots per ciphertext (SIMD). Maximum is ring_dim/2 (fully packed).
	uint32_t batchSize = 8;
	// Internal rescaling technique.
	ScalingTechnique rescaleTech = FLEXIBLEAUTO;

	// Other parameters. Influences the security level.
	uint32_t dcrtBits = 59;
	uint32_t firstMod = 60;
	uint32_t dnum	  = 2;

	// Create the parameter struct.
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
	// GPU Settings. Devices and autoload configuration.
	parameters.SetBackend(Backend::CUDA);
	parameters.SetPlaintextAutoload(false);
	parameters.SetCiphertextAutoload(true);

	// =====================================================
	// Step 2: Generate the CryptoContext.
	// =====================================================

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	// Enable the features that you wish to use.
	// NOTE: This is done for compatibility with OpenFHE API.
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);

	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	// =====================================================
	// Step 3: Key Generation.
	// =====================================================

	// Generate the keypair.
	auto keys = cc->KeyGen();

	// Derive the evaluation keys.
	cc->EvalMultKeyGen(keys.secretKey);
	// Derive the rotation keys for the specified rotations.
	cc->EvalRotateKeyGen(keys.secretKey, { 1, -2 });

	// =====================================================
	// Step 4: Load the context on the GPU.
	// =====================================================

	cc->LoadContext(keys.publicKey);

	// =====================================================
	// Step 5: Encode and encrypt your data.
	// =====================================================

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };
	uint32_t noiseScaleDeg = 1;
	Plaintext ptxt1		   = cc->MakeCKKSPackedPlaintext(x1, noiseScaleDeg, 4);
	Plaintext ptxt2		   = cc->MakeCKKSPackedPlaintext(x2, noiseScaleDeg, 4);
	std::cout << std::endl << "Input x1: " << ptxt1;
	std::cout << "Input x2: " << ptxt2 << std::endl;
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	// =====================================================
	// Step 6: Evaluation and decryption.
	// =====================================================

	// Homomorphic ciphertext-ciphertext addition.
	{
		std::cout << "==== Ciphertext-Ciphertext addition ====" << std::endl;
		std::cout << "\tc1 levels: " << c1->GetLevel() << std::endl;
		std::cout << "\tc2 levels: " << c2->GetLevel() << std::endl;
		auto res = cc->EvalAdd(c1, c2);
		std::cout << "\tcAdd levels: " << res->GetLevel() << std::endl;

		// Decrypt the result.
		Plaintext result;
		cc->Decrypt(keys.secretKey, res, &result);
		result->SetLength(batchSize);
		std::cout << "\tc1 + c2 = " << result;
		std::cout << "\tEstimated precision in bits: " << result->GetLogPrecision() << std::endl << std::endl;
	}

	// Homomorphic ciphertext-plaintext addition.
	{
		std::cout << "==== Ciphertext-Plaintext addition ====" << std::endl;
		std::cout << "\tc1 levels: " << c1->GetLevel() << std::endl;
		std::cout << "\tptxt2 levels: " << ptxt2->GetLevel() << std::endl;
		auto res = cc->EvalAdd(c1, ptxt2);
		std::cout << "\tcAdd levels: " << res->GetLevel() << std::endl;

		// Decrypt the result.
		Plaintext result;
		cc->Decrypt(keys.secretKey, res, &result);
		result->SetLength(batchSize);
		std::cout << "\tc1 + ptxt2 = " << result;
		std::cout << "\tEstimated precision in bits: " << result->GetLogPrecision() << std::endl << std::endl;
	}

	// Homomorphic ciphertext-ciphertext subtraction.
	{
		std::cout << "==== Ciphertext-Ciphertext subtraction ====" << std::endl;
		std::cout << "\tc1 levels: " << c1->GetLevel() << std::endl;
		std::cout << "\tc2 levels: " << c2->GetLevel() << std::endl;
		auto res = cc->EvalSub(c1, c2);
		std::cout << "\tcSub levels: " << res->GetLevel() << std::endl;

		// Decrypt the result.
		Plaintext result;
		cc->Decrypt(keys.secretKey, res, &result);
		result->SetLength(batchSize);
		std::cout << "\tc1 - c2 = " << result;
		std::cout << "\tEstimated precision in bits: " << result->GetLogPrecision() << std::endl << std::endl;
	}

	// Homomorphic ciphertext-plaintext subtraction.
	{
		std::cout << "==== Ciphertext-Plaintext subtraction ====" << std::endl;
		std::cout << "\tc1 levels: " << c1->GetLevel() << std::endl;
		std::cout << "\tptxt2 levels: " << ptxt2->GetLevel() << std::endl;
		auto res = cc->EvalSub(c1, ptxt2);
		std::cout << "\tcSub levels: " << res->GetLevel() << std::endl;

		// Decrypt the result.
		Plaintext result;
		cc->Decrypt(keys.secretKey, res, &result);
		result->SetLength(batchSize);
		std::cout << "\tc1 - ptxt2 = " << result;
		std::cout << "\tEstimated precision in bits: " << result->GetLogPrecision() << std::endl << std::endl;
	}

	// Homomorphic ciphertext-ciphertext multiplication.
	{
		std::cout << "==== Ciphertext-Ciphertext multiplication ====" << std::endl;
		std::cout << "\tc1 levels: " << c1->GetLevel() << std::endl;
		std::cout << "\tc2 levels: " << c2->GetLevel() << std::endl;
		auto res = cc->EvalMult(c1, c2);
		std::cout << "\tcMul levels: " << res->GetLevel() << std::endl;

		// Decrypt the result.
		Plaintext result;
		cc->Decrypt(keys.secretKey, res, &result);
		result->SetLength(batchSize);
		std::cout << "\tc1 * c2 = " << result;
		std::cout << "\tEstimated precision in bits: " << result->GetLogPrecision() << std::endl << std::endl;
	}

	// Homomorphic ciphertext-plaintext multiplication.
	{
		std::cout << "==== Ciphertext-Plaintext multiplication ====" << std::endl;
		std::cout << "\tc1 levels: " << c1->GetLevel() << std::endl;
		std::cout << "\tptxt2 levels: " << ptxt2->GetLevel() << std::endl;
		auto res = cc->EvalMult(c1, ptxt2);
		std::cout << "\tcMul levels: " << res->GetLevel() << std::endl;

		// Decrypt the result.
		Plaintext result;
		cc->Decrypt(keys.secretKey, res, &result);
		result->SetLength(batchSize);
		std::cout << "\tc1 * ptxt2 = " << result;
		std::cout << "\tEstimated precision in bits: " << result->GetLogPrecision() << std::endl << std::endl;
	}

	// Homomorphic ciphertext-scalar multiplication.
	{
		std::cout << "==== Ciphertext-Scalar multiplication ====" << std::endl;
		std::cout << "\tc1 levels: " << c1->GetLevel() << std::endl;
		auto res = cc->EvalMult(c1, 4.0);
		std::cout << "\tcMul levels: " << res->GetLevel() << std::endl;

		// Decrypt the result.
		Plaintext result;
		cc->Decrypt(keys.secretKey, res, &result);
		result->SetLength(batchSize);
		std::cout << "\tc1 * 4 = " << result;
		std::cout << "\tEstimated precision in bits: " << result->GetLogPrecision() << std::endl << std::endl;
	}

	// Homomorphic rotations.
	{
		std::cout << "==== Rotations ====" << std::endl;
		std::cout << "\tc1 levels: " << c1->GetLevel() << std::endl;
		auto cRot1 = cc->EvalRotate(c1, 1);
		auto cRot2 = cc->EvalRotate(c1, -2);
		std::cout << "\tcRot1 levels: " << cRot1->GetLevel() << std::endl;
		std::cout << "\tcRot2 levels: " << cRot2->GetLevel() << std::endl;

		// Decrypt the result.
		Plaintext result;
		cc->Decrypt(keys.secretKey, cRot1, &result);
		result->SetLength(batchSize);
		std::cout << "\tcRot1 = " << result;
		std::cout << "\tEstimated precision in bits: " << result->GetLogPrecision() << std::endl;

		cc->Decrypt(keys.secretKey, cRot2, &result);
		result->SetLength(batchSize);
		std::cout << "\tcRot2 = " << result;
		std::cout << "\tEstimated precision in bits: " << result->GetLogPrecision() << std::endl << std::endl;
	}

	return 0;
}