#include <fideslib.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

using namespace fideslib;

// =====================================================
// Selecting specific slots.
// =====================================================
void masking(CryptoContext<DCRTPoly>& cc, const KeyPair<DCRTPoly>& keys, uint32_t batchSize) {
	std::cout << std::endl << "==== Masking specific slots ====" << std::endl;

	// Input vector.
	std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	std::cout << "\tOriginal data: ";
	for (const auto& v : data)
		std::cout << v << " ";
	std::cout << std::endl;

	// Define masks.
	std::vector<double> evenMask	   = { 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 };
	std::vector<double> oddMask		   = { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 };
	std::vector<double> firstHalfMask  = { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
	std::vector<double> singleSlotMask = { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

	// Encrypt data.
	Plaintext ptxtData = cc->MakeCKKSPackedPlaintext(data);
	auto ctData		   = cc->Encrypt(keys.publicKey, ptxtData);

	// Apply masks.
	Plaintext ptxtEvenMask		 = cc->MakeCKKSPackedPlaintext(evenMask);
	Plaintext ptxtOddMask		 = cc->MakeCKKSPackedPlaintext(oddMask);
	Plaintext ptxtFirstHalfMask	 = cc->MakeCKKSPackedPlaintext(firstHalfMask);
	Plaintext ptxtSingleSlotMask = cc->MakeCKKSPackedPlaintext(singleSlotMask);
	auto ctEvenSlots	 = cc->EvalMult(ctData, ptxtEvenMask);
	auto ctOddSlots		 = cc->EvalMult(ctData, ptxtOddMask);
	auto ctFirstHalfOnly = cc->EvalMult(ctData, ptxtFirstHalfMask);
	auto ctSingleSlot	 = cc->EvalMult(ctData, ptxtSingleSlotMask);

	// Decrypt and display.
	Plaintext result;

	cc->Decrypt(keys.secretKey, ctEvenSlots, &result);
	result->SetLength(batchSize);
	std::cout << "\tEven slots masked:   " << result;

	cc->Decrypt(keys.secretKey, ctOddSlots, &result);
	result->SetLength(batchSize);
	std::cout << "\tOdd slots masked:    " << result;

	cc->Decrypt(keys.secretKey, ctFirstHalfOnly, &result);
	result->SetLength(batchSize);
	std::cout << "\tFirst half masked:   " << result;

	cc->Decrypt(keys.secretKey, ctSingleSlot, &result);
	result->SetLength(batchSize);
	std::cout << "\tSingle slot (2):     " << result;
}

// =====================================================
// Accumulation. Sum all slots using step-by-1 rotations.
// =====================================================
void accumulation(CryptoContext<DCRTPoly>& cc, const KeyPair<DCRTPoly>& keys, uint32_t batchSize) {
	std::cout << std::endl << "==== Accumulation (Sum Reduction case) ====" << std::endl;

	// Input vector.
	std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	std::cout << "\tOriginal data: ";
	for (const auto& v : data)
		std::cout << v << " ";
	std::cout << std::endl;

	// Encrypt data.
	Plaintext ptxtData = cc->MakeCKKSPackedPlaintext(data);
	auto ctSum		   = cc->Encrypt(keys.publicKey, ptxtData);

	// Iterative rotate and add.
	auto ctRotated = ctSum;
	for (uint32_t i = 1; i < batchSize; ++i) {
		ctRotated = cc->EvalRotate(ctRotated, 1);
		ctSum	  = cc->EvalAdd(ctSum, ctRotated);
	}

	// Decrypt and display.
	Plaintext result;
	cc->Decrypt(keys.secretKey, ctSum, &result);
	result->SetLength(batchSize);
	std::cout << "\tAfter accumulation:  " << result;

	double expectedSum = 0.0;
	for (const auto& v : data)
		expectedSum += v;
	std::cout << "\tExpected sum: " << expectedSum << " (replicated in all slots)" << std::endl;
}

// =====================================================
// Distribution. Replicate slot 0 to all slots.
// =====================================================
void distribution(CryptoContext<DCRTPoly>& cc, const KeyPair<DCRTPoly>& keys, uint32_t batchSize) {
	std::cout << std::endl << "==== Distribution (Replicate Value) ====" << std::endl;

	// Create a ciphertext with only slot 0 having a value.
	std::vector<double> singleValue = { 42.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	Plaintext ptxtSingle			= cc->MakeCKKSPackedPlaintext(singleValue);
	auto ctDistributed				= cc->Encrypt(keys.publicKey, ptxtSingle);

	Plaintext result;
	std::cout << "\tInitial (slot 0 only): ";
	cc->Decrypt(keys.secretKey, ctDistributed, &result);
	result->SetLength(batchSize);
	std::cout << result;

	// Distribute using rotate-by-(-1) and add iteratively.
	auto ctRotated = ctDistributed;
	for (uint32_t i = 1; i < batchSize; ++i) {
		ctRotated	  = cc->EvalRotate(ctRotated, -1);
		ctDistributed = cc->EvalAdd(ctDistributed, ctRotated);
	}

	// Decrypt and display.
	cc->Decrypt(keys.secretKey, ctDistributed, &result);
	result->SetLength(batchSize);
	std::cout << "\tAfter distribution:  " << result;
}

// =====================================================
// This function must do the following.
// 1. Accumulate all slot values.
// 2. Divide the even slots by 2.
// 3. Multiply odd slots by 2.
// 4. Accumulate only the first half of the slots.
// 5. Return the result.
// =====================================================
Ciphertext<DCRTPoly> task(CryptoContext<DCRTPoly>& cc, const KeyPair<DCRTPoly>& keys, uint32_t batchSize, Ciphertext<DCRTPoly>& ct) {
	
	// TODO: 1. Accumulate all slots.

	// TODO: 2. Divide the even slots by 2.
	
	// TODO: 3. Multiply the odd slots by 2.
	
	// TODO: 4. Accumulate only the first half of the slots.
	
	// TODO: 5. Return the result.

	return ct->Clone();
}

int main() {
	// =====================================================
	// Step 1: Define our parameters.
	// =====================================================

	uint32_t multDepth			 = 10;
	uint32_t batchSize			 = 8;
	uint32_t ring_dim			 = 1 << 12;
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
	parameters.SetBackend(Backend::CUDA);
	parameters.SetPlaintextAutoload(false);
	parameters.SetCiphertextAutoload(true);

	// =====================================================
	// Step 2: Generate the CryptoContext.
	// =====================================================

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);

	std::cout << "CKKS scheme using ring dimension: " << cc->GetRingDimension() << std::endl;
	std::cout << "Batch size (slots): " << batchSize << std::endl << std::endl;

	// =====================================================
	// Step 3: Key Generation.
	// =====================================================

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);

	// Generate rotation keys for step-by-1 rotations.
	cc->EvalRotateKeyGen(keys.secretKey, { 1, -1 });

	// =====================================================
	// Step 4: Load the context on the GPU.
	// =====================================================

	cc->LoadContext(keys.publicKey);

	// =====================================================
	// Run demos.
	// =====================================================

	masking(cc, keys, batchSize);
	accumulation(cc, keys, batchSize);
	distribution(cc, keys, batchSize);

	// =====================================================
	// Step 5: Task.
	// =====================================================

	std::cout << std::endl << "==== Task ====" << std::endl;

	std::vector<double> values = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
	auto input = cc->MakeCKKSPackedPlaintext(values);
	auto ct = cc->Encrypt(keys.publicKey, input);

	auto ct_task = task(cc, keys, batchSize, ct);

	Plaintext result;
	cc->Decrypt(keys.secretKey, ct_task, &result);
	result->SetLength(batchSize);
	std::cout << "\tTask result: " << result;

	std::vector<double> expected = { 40.0, 40.0, 40.0, 40.0, 4.0, 16.0, 4.0, 16.0 };
	Plaintext expected_plaintext = cc->MakeCKKSPackedPlaintext(expected);
	std::cout << "\tExpected result: " << expected_plaintext << std::endl;

	return 0;
}
