//
// Created by carlosad on 13/7/26.
//

//
// Symptom: the process's FIRST key-switch (here a single EvalRotate) on a
// ciphertext at level >= 2 returns all-NaN. The identical op run a second time
// is correct. Root cause is a cross-stream race in the caching GPU pool: the
// per-context key-switch scratch is allocated once, lazily, recycling chunks
// whose previous tenant (the rescale scratch freed by the preceding EvalMults)
// still has GPU writes in flight -> corruption on first use only.
//
// Build against either tree (public api/ layer is byte-identical):
//   cmake -S . -B build && cmake --build build
//   ./build/keyswitch_nan_repro          # level defaults to 3
//   REPRO_LEVEL=2 ./build/keyswitch_nan_repro
//
// Fixed tree -> exits 0 ("NO NaN"). Upstream (~/UpstreamFIDESlib) -> exits 1
// ("BUG REPRODUCED"). Params are the MWE-exact params: depth 30, ring 2^17,
// scaleMod 59, firstMod 60, dnum 3, 1024 slots, FLEXIBLEAUTO.
#include <gtest/gtest.h>
#include <fideslib.hpp>
#include <cmath>
#include <cstdlib>
#include <cuda.h>
using namespace fideslib;

TEST(Bug, CatchBugTest) {
	const char* env = std::getenv("REPRO_LEVEL");
	const int level = env ? std::atoi(env) : 2; // sink target; bug needs level >= 2

	uint32_t batchSize = 1024;

	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetMultiplicativeDepth(30);
	parameters.SetFirstModSize(60);
	parameters.SetScalingModSize(59);
	parameters.SetBatchSize(batchSize);
	parameters.SetRingDim(1u << 17);
	parameters.SetNumLargeDigits(3); // dnum
	parameters.SetScalingTechnique(FLEXIBLEAUTO);
	parameters.SetDevices({ 0 });
	parameters.SetPlaintextAutoload(false);
	parameters.SetCiphertextAutoload(true);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, { 1 });
	cc->LoadContext(keys.publicKey);

	std::vector<double> x = { 1, 2, 3, 4, 5, 6, 7, 8 };
	Plaintext ones_pt     = cc->MakeCKKSPackedPlaintext(std::vector<double>(8, 1.0));
	Plaintext x_pt        = cc->MakeCKKSPackedPlaintext(x);
	auto ct               = cc->Encrypt(keys.publicKey, x_pt);

	// Sink to `level` with plaintext mults on the GPU (NOT key-switches). This
	// leaves the rescale scratch with in-flight writes right before the rotate.
	for (int i = 0; i < level; ++i) {
		ct = cc->EvalMult(ct, ones_pt);
	}

	//cudaDeviceSynchronize();

	auto cRot = cc->EvalRotate(ct, 1); // <-- the process's first key-switch
	//auto cRot = ct;
	Plaintext result;
	cc->Decrypt(keys.secretKey, cRot, &result);
	result->SetLength(batchSize);

	// rotate-left-by-1 of [1..8] -> slots 0..6 should read 2..8.
	std::cout << "level=" << level << "  first 8 slots: [";
	bool has_nan = false;
	for (int i = 0; i < 8; ++i) {
		double v = result->GetRealPackedValue().at(i);
		has_nan  = has_nan || std::isnan(v);
		std::cout << v << (i < 7 ? " " : "");
	}
	std::cout << "]" << std::endl;

	if (has_nan) {
		std::cout << "BUG REPRODUCED: first-use key-switch returned NaN" << std::endl;
		ASSERT_FALSE(true);
	}
	std::cout << "NO NaN: first-use key-switch is correct" << std::endl;
	ASSERT_FALSE(false);
}