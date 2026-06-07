#include <fideslib.hpp>

#include <iostream>

using namespace fideslib;

int main() {

	// =====================================================
	// Step 1: Define our parameters.
	// =====================================================

	// How many multiplications can be done.
	uint32_t multDepth = 25;
	// Ring dimension of the scheme.
	uint32_t ring_dim = 1 << 12;
	// How many slots per ciphertext (SIMD). Maximum is ring_dim/2 (fully packed).
	uint32_t batchSize = 8;
	// Internal rescaling technique.
	ScalingTechnique rescaleTech = FLEXIBLEAUTO;

	// Other parameters. Influences the security level.
	uint32_t dcrtBits = 59;
	uint32_t firstMod = 60;
	uint32_t dnum	  = 3;

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
	// This is done for compatibility with OpenFHE API.
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

	std::vector<uint32_t> level_budget = { 2, 2 };
	std::vector<uint32_t> bsgs		   = { 0, 0 };

	cc->EvalBootstrapSetup(level_budget, bsgs, batchSize, 0);
	cc->EvalBootstrapKeyGen(keys.secretKey, batchSize);

	// =====================================================
	// Step 4: Load the context on the GPU.
	// =====================================================

	cc->LoadContext(keys.publicKey);

	// =====================================================
	// Step 5: Encode and encrypt your data.
	// =====================================================

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	uint32_t noise_scale   = 1;
	Plaintext ptxt1		   = cc->MakeCKKSPackedPlaintext(x1, noise_scale, multDepth - 1);
	std::cout << std::endl << "==== Bootstrap ====" << std::endl;
	std::cout << "Number of levels before bootstrapping: " << ptxt1->GetLevel() << std::endl;
	std::cout << "Original input: " << ptxt1;
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	// =====================================================
	// Step 6: Bootstrap.
	// =====================================================

	cc->Synchronize();
	auto start						   = std::chrono::high_resolution_clock::now();
	auto ciphertextAfter			   = cc->EvalBootstrap(c1);
	cc->Synchronize();
	auto end						   = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << "Bootstrapping time: " << diff.count() << " s" << std::endl;

	std::cout << "Number of levels after bootstrapping: " << ciphertextAfter->GetLevel() << std::endl;

	Plaintext result;
	cc->Decrypt(keys.secretKey, ciphertextAfter, &result);
	result->SetLength(batchSize);

	std::cout << "Bootstrapped result: " << result << std::endl;

	return 0;
}