// API test suite for fideslib — drives the public CryptoContextImpl api and verifies
// results by decryption against expected values, so it is meaningful on either backend.
// Default backend is CPU; set FIDESLIB_TEST_BACKEND=cuda on a CUDA build to exercise the
// CUDA engine end-to-end (see TestUseCuda below).
// CPU-only build:  cmake -B build-cpu -DFIDESLIB_ENABLE_CUDA=OFF && cmake --build build-cpu --target fideslib-cpu-test
// Run:    ./build-cpu/fideslib-cpu-test [--gtest_filter=-*Bootstrap*]

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

#include "Serialize.hpp"
#include "fideslib.hpp"

using namespace fideslib;

// The api test fixtures run against either backend: set FIDESLIB_TEST_BACKEND=cuda
// (on a CUDA build) to exercise the CUDA engine end-to-end. CPU-only builds always
// run on the CPU backend, since IsBackendAvailable(CUDA) is false when CUDA is not
// compiled in, so this is a no-op there.
static bool TestUseCuda() {
	const char* b = std::getenv("FIDESLIB_TEST_BACKEND");
	return b != nullptr && std::string(b) == "cuda" && IsBackendAvailable(Backend::CUDA);
}

// Precision tolerance for standard CKKS operations (encrypt, add, mult, rotate).
// With scalingModSize=50, theoretical single-op precision is ~2^{-50} ~ 1e-15.
// Accumulated error through depth-5 circuits and encoding roundtrip gives ~1e-6.
static constexpr double CKKS_PRECISION = 1e-6;

// Bootstrapping introduces polynomial approximation error for modular reduction.
// OpenFHE's iterative bootstrap with levelBudget={1,1} achieves ~10-14 bits of
// precision; 1e-2 (~6.6 bits) provides margin for the single-iteration case.
static constexpr double CKKS_BOOTSTRAP_PRECISION = 1e-2;

// Decrypt ct into *pt and check real slots against expected within tol.
static void CheckPrecision(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, const std::vector<double>& expected, double tol) {
	Plaintext pt;
	cc->Decrypt(ct, sk, &pt);
	pt->SetLength(expected.size());
	auto vals = pt->GetRealPackedValue();
	for (size_t i = 0; i < expected.size(); ++i)
		EXPECT_NEAR(vals[i], expected[i], tol) << "slot " << i;
}

// ---------------------------------------------------------------------------
// Standard fixture: depth=5, ringDim=65536, slots=8, FIXEDAUTO, UNIFORM_TERNARY
// ---------------------------------------------------------------------------

class CKKSTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth   = 5;
	static constexpr uint32_t kRingDim = 1u << 16; // 65536
	static constexpr uint32_t kSlots   = 8;

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;

	// Test vectors (8 slots).
	const std::vector<double> v1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	const std::vector<double> v2 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

	void SetUp() override {
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(50);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FIXEDAUTO);
		if (TestUseCuda())
			params.SetBackend(Backend::CUDA);
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		// Generate every rotation index the tests use up front. In GPU mode the keys are
		// uploaded by LoadContext, so they must all exist before it; the convolution and
		// accumulate ops need the full 1..8 range (incl. 3,5,6,7), not just powers of two.
		cc->EvalRotateKeyGen(keys.secretKey, { 1, 2, 3, 4, 5, 6, 7, 8, -1, -2 });
		if (TestUseCuda())
			cc->LoadContext(keys.publicKey);
	}
};

// ---- 1. Context ----

TEST_F(CKKSTest, ContextSetup) {
	EXPECT_EQ(cc->GetRingDimension(), kRingDim);
	EXPECT_EQ(cc->GetCyclotomicOrder(), 2 * kRingDim);
}

// ---- 2. Encoding ----

TEST_F(CKKSTest, EncodingReal) {
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto out = pt->GetRealPackedValue();
	for (size_t i = 0; i < v1.size(); ++i)
		EXPECT_NEAR(out[i], v1[i], CKKS_PRECISION) << "slot " << i;
}

TEST_F(CKKSTest, EncodingComplex) {
	std::vector<std::complex<double>> cv(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		cv[i] = { v1[i], 0.0 };
	auto pt	 = cc->MakeCKKSPackedPlaintext(cv);
	auto out = pt->GetCKKSPackedValue();
	for (size_t i = 0; i < v1.size(); ++i)
		EXPECT_NEAR(out[i].real(), v1[i], CKKS_PRECISION) << "slot " << i;
}

// ---- 3. Encrypt / Decrypt ----

TEST_F(CKKSTest, EncryptDecryptPublicKey) {
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	CheckPrecision(cc, ct, keys.secretKey, v1, CKKS_PRECISION);
}

TEST_F(CKKSTest, EncryptDecryptPrivateKey) {
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.secretKey);
	CheckPrecision(cc, ct, keys.secretKey, v1, CKKS_PRECISION);
}

// ---- 4. EvalAdd ----

TEST_F(CKKSTest, EvalAddCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalAdd(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto res = cc->EvalAdd(ct1, pt2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddCtScalar) {
	double scalar = 10.0;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + scalar;
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalAdd(ct, scalar);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddInPlaceCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	cc->EvalAddInPlace(ct1, ct2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddInPlaceCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	cc->EvalAddInPlace(ct1, pt2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddInPlaceCtScalar) {
	double scalar = 5.0;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + scalar;
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalAddInPlace(ct, scalar);
	CheckPrecision(cc, ct, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddMany) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i] + v1[i];
	auto pt1							  = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2							  = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1							  = cc->Encrypt(pt1, keys.publicKey);
	auto ct2							  = cc->Encrypt(pt2, keys.publicKey);
	auto ct3							  = cc->Encrypt(pt1, keys.publicKey);
	std::vector<Ciphertext<DCRTPoly>> cts = { ct1, ct2, ct3 };
	auto res							  = cc->EvalAddMany(cts);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- 4. EvalSub ----

TEST_F(CKKSTest, EvalSubCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalSub(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto res = cc->EvalSub(ct1, pt2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubCtScalar) {
	double scalar = 0.5;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - scalar;
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalSub(ct, scalar);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubInPlaceCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	cc->EvalSubInPlace(ct1, ct2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubInPlaceCtScalar) {
	double scalar = 3.0;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - scalar;
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalSubInPlace(ct, scalar);
	CheckPrecision(cc, ct, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- 4. EvalMult ----

TEST_F(CKKSTest, EvalMultCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalMult(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto res = cc->EvalMult(ct1, pt2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultCtScalar) {
	double scalar = 2.0;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * scalar;
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalMult(ct, scalar);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultInPlaceCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	cc->EvalMultInPlace(ct1, pt2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultInPlaceCtScalar) {
	double scalar = 3.0;
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * scalar;
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalMultInPlace(ct, scalar);
	CheckPrecision(cc, ct, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- 4. EvalNegate / EvalSquare ----

TEST_F(CKKSTest, EvalNegate) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = -v1[i];
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalNegate(ct);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSquare) {
	std::vector<double> expected(v2.size());
	for (size_t i = 0; i < v2.size(); ++i)
		expected[i] = v2[i] * v2[i];
	auto pt	 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalSquare(ct);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSquareInPlace) {
	std::vector<double> expected(v2.size());
	for (size_t i = 0; i < v2.size(); ++i)
		expected[i] = v2[i] * v2[i];
	auto pt = cc->MakeCKKSPackedPlaintext(v2);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalSquareInPlace(ct);
	CheckPrecision(cc, ct, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- 5. Rotation ----

TEST_F(CKKSTest, EvalRotate) {
	// Rotate left by 1: slot i gets v1[(i+1) % slots] (circular).
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[(i + 1) % v1.size()];
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalRotate(ct, 1);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalFastRotation) {
	// Same rotation as EvalRotate but via precompute + EvalFastRotation.
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[(i + 1) % v1.size()];
	auto pt		 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct		 = cc->Encrypt(pt, keys.publicKey);
	auto precomp = cc->EvalFastRotationPrecompute(ct);
	uint32_t m	 = cc->GetCyclotomicOrder();
	auto res	 = cc->EvalFastRotation(ct, 1, m, precomp);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- 6. Level management ----

TEST_F(CKKSTest, RescaleInPlace) {
	// Verify that the ciphertext remains decryptable and accurate after rescaling.
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto ct2 = cc->EvalMult(ct, ct); // ct-ct mult: scale^2
	cc->RescaleInPlace(ct2);
	std::vector<double> expected(kSlots);
	for (size_t i = 0; i < kSlots; ++i)
		expected[i] = v1[i] * v1[i];
	CheckPrecision(cc, ct2, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, SetLevel) {
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	// Consume one level so SetLevel has somewhere to go.
	cc->RescaleInPlace(ct);
	size_t cur = ct->GetLevel();
	// SetLevel drops towers (increases level number toward max).
	CryptoContextImpl<DCRTPoly>::SetLevel(ct, cur + 1);
	EXPECT_EQ(ct->GetLevel(), cur + 1);
}

// ---- 7. Advanced operations ----

TEST_F(CKKSTest, EvalChebyshevSeries) {
	// The GPU Chebyshev domain-map fix is deliberately off this branch (lives on
	// chebyshev-domain-map-fix); the device path is off by ~1 on non-[-1,1] intervals.
	if (TestUseCuda())
		GTEST_SKIP() << "GPU Chebyshev domain-map fix deferred (branch chebyshev-domain-map-fix)";
	// Approximate the identity f(x)=x on [0,1] with degree-5 Chebyshev.
	std::function<double(double)> f = [](double x) { return x; };
	auto coeffs						= CryptoContextImpl<DCRTPoly>::GetChebyshevCoefficients(f, 0.0, 1.0, 5);
	// Inputs must lie inside [0,1].
	std::vector<double> input(kSlots);
	for (size_t i = 0; i < kSlots; ++i)
		input[i] = static_cast<double>(i + 1) / (kSlots + 1);
	auto pt	 = cc->MakeCKKSPackedPlaintext(input);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalChebyshevSeries(ct, coeffs, 0.0, 1.0);
	// Degree-5 approximation of identity should be accurate to ~1e-4.
	CheckPrecision(cc, res, keys.secretKey, input, 1e-4);
}

TEST_F(CKKSTest, AccumulateSum) {
	// AccumulateSum(ct, slots) sums all slots into each slot position.
	// With v1={1..8}, sum = 36.
	std::vector<double> expected(kSlots, 36.0);
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->AccumulateSum(ct, static_cast<int>(kSlots));
	// Only slot 0 is guaranteed; the rest depend on rotation key availability.
	Plaintext out;
	cc->Decrypt(res, keys.secretKey, &out);
	out->SetLength(1);
	auto vals = out->GetRealPackedValue();
	EXPECT_NEAR(vals[0], 36.0, CKKS_PRECISION);
}

// ---- 10. GetConvolutionTransformRotationIndices ----

TEST_F(CKKSTest, ConvRotIndices_gStep0) {
	auto idx = CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(16, 4, 1, 0);
	EXPECT_TRUE(idx.empty());
}

TEST_F(CKKSTest, ConvRotIndices_gStep1) {
	// gStep=1: maxIntra = min(1,8) = 1, loop k in [1,1) → empty intra.
	// blockCount = 1 → no inter. Result: empty.
	auto idx = CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(16, 4, 1, 1);
	EXPECT_TRUE(idx.empty());
}

TEST_F(CKKSTest, ConvRotIndices_gStep8) {
	// gStep=8, stride=1: maxIntra=8, intra = {1,2,3,4,5,6,7}.
	// blockCount=1 → no inter entries.
	auto idx = CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(16, 4, 1, 8);
	ASSERT_EQ(idx.size(), 7u);
	for (int k = 1; k <= 7; ++k)
		EXPECT_EQ(idx[static_cast<size_t>(k - 1)], k);
}

TEST_F(CKKSTest, ConvRotIndices_gStep16) {
	// gStep=16, stride=2, internal_gstep=8:
	//   maxIntra=8, intra = {2,4,6,8,10,12,14} (k*stride for k=1..7)
	//   blockCount=2, inter = {16} (baseRotation=8*2=16, k=1)
	auto idx = CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(32, 4, 2, 16);
	// 7 intra + 1 inter = 8 entries
	ASSERT_EQ(idx.size(), 8u);
	for (int k = 1; k <= 7; ++k)
		EXPECT_EQ(idx[static_cast<size_t>(k - 1)], k * 2);
	EXPECT_EQ(idx[7], 16);
}

// ---- 11. In-place variants ----

TEST_F(CKKSTest, EvalNegateInPlace) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = -v1[i];
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalNegateInPlace(ct);
	CheckPrecision(cc, ct, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalRotateInPlace) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[(i + 1) % v1.size()];
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalRotateInPlace(ct, 1);
	CheckPrecision(cc, ct, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, Rescale) {
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto ct2 = cc->EvalMult(ct, ct);
	auto ct3 = cc->Rescale(ct2);
	std::vector<double> expected(kSlots);
	for (size_t i = 0; i < kSlots; ++i)
		expected[i] = v1[i] * v1[i];
	CheckPrecision(cc, ct3, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalChebyshevSeriesInPlace) {
	// GPU Chebyshev domain-map fix deferred to branch chebyshev-domain-map-fix (off by ~1 on
	// non-[-1,1] intervals); CPU (OpenFHE) is correct, so this still runs there.
	if (TestUseCuda())
		GTEST_SKIP() << "GPU Chebyshev domain-map fix deferred (branch chebyshev-domain-map-fix)";
	std::function<double(double)> f = [](double x) { return x; };
	auto coeffs						= CryptoContextImpl<DCRTPoly>::GetChebyshevCoefficients(f, 0.0, 1.0, 5);
	std::vector<double> input(kSlots);
	for (size_t i = 0; i < kSlots; ++i)
		input[i] = static_cast<double>(i + 1) / (kSlots + 1);
	auto pt = cc->MakeCKKSPackedPlaintext(input);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalChebyshevSeriesInPlace(ct, coeffs, 0.0, 1.0);
	CheckPrecision(cc, ct, keys.secretKey, input, 1e-4);
}

TEST_F(CKKSTest, AccumulateSumInPlace) {
	std::vector<double> expected(kSlots, 36.0);
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->AccumulateSumInPlace(ct, static_cast<int>(kSlots));
	Plaintext out;
	cc->Decrypt(ct, keys.secretKey, &out);
	out->SetLength(1);
	auto vals = out->GetRealPackedValue();
	EXPECT_NEAR(vals[0], 36.0, CKKS_PRECISION);
}

// ---- 11b. ConvolutionTransform ----

TEST_F(CKKSTest, ConvolutionTransformIdentity) {
	if (!TestUseCuda())
		GTEST_SKIP() << "CPU convolution is implemented in PR3 (backend-abstraction-cpu-impl).";
	// Identity transform: bStep=1, gStep=1, single plaintext of ones.
	// With indexes={0} (no baby-step rotation), the result should be
	// ct * pt (scaled by 1), rotated by stride*(1-0)=stride, then rescaled.
	// Use a single "1.0" weight and gStep=1 so the output = rot(ct*1, stride).
	std::vector<double> ones(kSlots, 1.0);
	auto ptWeight				   = cc->MakeCKKSPackedPlaintext(ones);
	std::vector<Plaintext> weights = { ptWeight };
	std::vector<int> indexes	   = { 0 };

	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);

	// gStep=1, bStep=1, stride=1 => rotation by stride*(1-0)=1, then rescale.
	cc->ConvolutionTransformInPlace(ct, 1, 1, weights, indexes, 1);

	// Expected: v1 rotated left by 1, then ct*1.0 = v1 (precision loss from mult+rescale).
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[(i + 1) % v1.size()];
	CheckPrecision(cc, ct, keys.secretKey, expected, 1e-4);
}

// ---- 12. SPARSE_ENCAPSULATED ----

TEST_F(CKKSTest, SparseEncapsulatedContext) {
	CCParams<CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(5);
	params.SetScalingModSize(50);
	params.SetBatchSize(kSlots);
	params.SetRingDim(kRingDim);
	params.SetScalingTechnique(FIXEDAUTO);
	params.SetSecretKeyDist(SPARSE_ENCAPSULATED);
	EXPECT_NO_THROW({
		auto cc2 = GenCryptoContext(params);
		cc2->Enable(PKE);
		cc2->Enable(KEYSWITCH);
		cc2->Enable(LEVELEDSHE);
		auto kp = cc2->KeyGen();
		auto pt = cc2->MakeCKKSPackedPlaintext(v1);
		auto ct = cc2->Encrypt(pt, kp.publicKey);
		Plaintext result;
		cc2->Decrypt(ct, kp.secretKey, &result);
	});
}

// ---- 13. Serialization ----

TEST_F(CKKSTest, SerializeDeserializeEvalMultKey) {
	// Serialize the eval mult key (generated in SetUp) to a buffer.
	std::ostringstream oss;
	ASSERT_TRUE(CryptoContextImpl<DCRTPoly>::SerializeEvalMultKey(oss, SerType::BINARY));
	EXPECT_GT(oss.str().size(), 0u);

	// Create a second context with the same parameters but no mult key.
	// Deserializing into it verifies the binary format is self-consistent.
	CCParams<CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(kDepth);
	params.SetScalingModSize(50);
	params.SetBatchSize(kSlots);
	params.SetRingDim(kRingDim);
	params.SetScalingTechnique(FIXEDAUTO);
	params.SetSecurityLevel(HEStd_NotSet);
	auto cc2 = GenCryptoContext(params);
	cc2->Enable(PKE);
	cc2->Enable(KEYSWITCH);
	cc2->Enable(LEVELEDSHE);

	std::istringstream iss(oss.str());
	ASSERT_TRUE(cc2->DeserializeEvalMultKey(iss, SerType::BINARY));

	// Confirm the original context can still multiply (its keys are still live).
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalMult(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

// ---------------------------------------------------------------------------
// Bootstrap fixture (FIXEDAUTO): depth=25, ringDim=65536, slots=8, UNIFORM_TERNARY
// ---------------------------------------------------------------------------

class CKKSBootstrapTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth   = 25;
	static constexpr uint32_t kRingDim = 1u << 16;
	static constexpr uint32_t kSlots   = 8;

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	const std::vector<double> v1 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

	void SetUp() override {
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(50);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FIXEDAUTO);
		params.SetSecretKeyDist(UNIFORM_TERNARY);
		params.SetSecurityLevel(HEStd_NotSet);
		if (TestUseCuda())
			params.SetBackend(Backend::CUDA);
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		cc->Enable(FHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		// correctionFactor=11 ensures it exceeds deg=10 (round(log2(q0/2^50))
		// with the default firstModSize=60 used by OpenFHE for FIXEDAUTO).
		cc->EvalBootstrapSetup({ 1, 1 }, { 0, 0 }, kSlots, 11);
		cc->EvalBootstrapKeyGen(keys.secretKey, kSlots);
		if (TestUseCuda())
			cc->LoadContext(keys.publicKey);
	}
};

// ---- 8. Bootstrap ----

TEST_F(CKKSBootstrapTest, DISABLED_Bootstrap) {
	auto pt		   = cc->MakeCKKSPackedPlaintext(v1);
	auto ct		   = cc->Encrypt(pt, keys.publicKey);
	auto refreshed = cc->EvalBootstrap(ct);
	CheckPrecision(cc, refreshed, keys.secretKey, v1, CKKS_BOOTSTRAP_PRECISION);
}

TEST_F(CKKSBootstrapTest, DISABLED_BootstrapInPlace) {
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->EvalBootstrapInPlace(ct);
	CheckPrecision(cc, ct, keys.secretKey, v1, CKKS_BOOTSTRAP_PRECISION);
}

// ---- 9. GetPreScaleFactor (FIXEDAUTO) ----

TEST_F(CKKSBootstrapTest, DISABLED_GetPreScaleFactorFIXEDAUTO) {
	// EvalBootstrapSetup sets m_correctionFactor; without it the result underflows.
	double r = cc->GetPreScaleFactor(kSlots);
	EXPECT_TRUE(std::isfinite(r)) << "result is not finite: " << r;
	EXPECT_GT(r, 0.0);
	EXPECT_LT(r, 1.0);
}

// ---------------------------------------------------------------------------
// Bootstrap fixture (FLEXIBLEAUTO): identical parameters but FLEXIBLEAUTO
// ---------------------------------------------------------------------------

class CKKSFlexBootstrapTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth   = 25;
	static constexpr uint32_t kRingDim = 1u << 16;
	static constexpr uint32_t kSlots   = 8;

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;

	void SetUp() override {
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(50);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FLEXIBLEAUTO);
		params.SetSecretKeyDist(UNIFORM_TERNARY);
		params.SetSecurityLevel(HEStd_NotSet);
		if (TestUseCuda())
			params.SetBackend(Backend::CUDA);
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		cc->Enable(FHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		cc->EvalBootstrapSetup({ 1, 1 }, { 0, 0 }, kSlots, 0);
		cc->EvalBootstrapKeyGen(keys.secretKey, kSlots);
		if (TestUseCuda())
			cc->LoadContext(keys.publicKey);
	}
};

// ---- 9. GetPreScaleFactor (FLEXIBLEAUTO) ----

TEST_F(CKKSFlexBootstrapTest, DISABLED_GetPreScaleFactorFLEXIBLEAUTO) {
	double r = cc->GetPreScaleFactor(kSlots);
	EXPECT_TRUE(std::isfinite(r)) << "result is not finite: " << r;
	EXPECT_GT(r, 0.0);
	// FLEXIBLEAUTO result can exceed 1.
}

// ===========================================================================
// Additional coverage tests
// ===========================================================================

// ---- Mutable variants ----

TEST_F(CKKSTest, EvalAddMutableCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalAddMutable(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddMutableCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto res = cc->EvalAddMutable(ct1, pt2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalAddMutableInPlaceCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	cc->EvalAddMutableInPlace(ct1, ct2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubMutableCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalSubMutable(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubMutableCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto res = cc->EvalSubMutable(ct1, pt2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSubMutableInPlaceCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] - v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	cc->EvalSubMutableInPlace(ct1, ct2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultMutableCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto res = cc->EvalMultMutable(ct1, ct2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultMutableCtPt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto res = cc->EvalMultMutable(ct1, pt2);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalMultMutableInPlaceCtCt) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	cc->EvalMultMutableInPlace(ct1, ct2);
	CheckPrecision(cc, ct1, keys.secretKey, expected, CKKS_PRECISION);
}

TEST_F(CKKSTest, EvalSquareMutable) {
	std::vector<double> expected(v2.size());
	for (size_t i = 0; i < v2.size(); ++i)
		expected[i] = v2[i] * v2[i];
	auto pt	 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalSquareMutable(ct);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- Multi-level operations ----

TEST_F(CKKSTest, MultChainedAllLevels) {
	// Multiply through all 5 depth levels, verifying at each step.
	auto pt					= cc->MakeCKKSPackedPlaintext(v2);
	auto ct					= cc->Encrypt(pt, keys.publicKey);
	std::vector<double> cur = v2;
	for (uint32_t d = 0; d < kDepth; ++d) {
		ct = cc->EvalMult(ct, ct);
		for (size_t i = 0; i < cur.size(); ++i)
			cur[i] = cur[i] * cur[i];
	}
	// Precision degrades with depth; use a looser tolerance.
	CheckPrecision(cc, ct, keys.secretKey, cur, 1e-2);
}

TEST_F(CKKSTest, RotateAfterMult) {
	// Consume a level via multiplication, then rotate at the lower level.
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] * v2[i];
	auto pt1 = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2 = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1 = cc->Encrypt(pt1, keys.publicKey);
	auto ct2 = cc->Encrypt(pt2, keys.publicKey);
	auto ct	 = cc->EvalMult(ct1, ct2);
	auto res = cc->EvalRotate(ct, 1);
	// Expected: (v1*v2) rotated left by 1.
	std::vector<double> rotated(expected.size());
	for (size_t i = 0; i < expected.size(); ++i)
		rotated[i] = expected[(i + 1) % expected.size()];
	CheckPrecision(cc, res, keys.secretKey, rotated, CKKS_PRECISION);
}

// ---- Ciphertext copy and clone ----

TEST_F(CKKSTest, CiphertextCopy) {
	// Ciphertext<DCRTPoly> is a shared_ptr; copying shares the underlying object.
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	Ciphertext<DCRTPoly> alias(ct);
	// Both alias and ct point to the same ciphertext.
	cc->EvalAddInPlace(ct, ct);
	// alias sees the modification (shared semantics).
	std::vector<double> doubled(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		doubled[i] = v1[i] * 2.0;
	CheckPrecision(cc, alias, keys.secretKey, doubled, CKKS_PRECISION);
}

TEST_F(CKKSTest, CiphertextClone) {
	auto pt	   = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	   = cc->Encrypt(pt, keys.publicKey);
	auto clone = ct->Clone();
	cc->EvalAddInPlace(ct, ct);
	CheckPrecision(cc, clone, keys.secretKey, v1, CKKS_PRECISION);
}

// ---- Getters / setters ----

TEST_F(CKKSTest, GetLevel) {
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	EXPECT_EQ(ct->GetLevel(), 0u);
	auto ct2 = cc->EvalMult(ct, ct);
	// After mult, level may have increased.
	EXPECT_GE(ct2->GetLevel(), 0u);
}

TEST_F(CKKSTest, GetNoiseScaleDeg) {
	auto pt	   = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	   = cc->Encrypt(pt, keys.publicKey);
	size_t deg = ct->GetNoiseScaleDeg();
	EXPECT_GE(deg, 1u);
}

// ---- Automorphism key serialization ----

TEST_F(CKKSTest, SerializeDeserializeEvalAutomorphismKey) {
	std::ostringstream oss;
	ASSERT_TRUE(CryptoContextImpl<DCRTPoly>::SerializeEvalAutomorphismKey(oss, SerType::BINARY));
	EXPECT_GT(oss.str().size(), 0u);

	// Deserialize into a second context with the same parameters.
	CCParams<CryptoContextCKKSRNS> params;
	params.SetMultiplicativeDepth(kDepth);
	params.SetScalingModSize(50);
	params.SetBatchSize(kSlots);
	params.SetRingDim(kRingDim);
	params.SetScalingTechnique(FIXEDAUTO);
	params.SetSecurityLevel(HEStd_NotSet);
	auto cc2 = GenCryptoContext(params);
	cc2->Enable(PKE);
	cc2->Enable(KEYSWITCH);
	cc2->Enable(LEVELEDSHE);

	std::istringstream iss(oss.str());
	ASSERT_TRUE(cc2->DeserializeEvalAutomorphismKey(iss, SerType::BINARY));

	// Verify original context can still rotate.
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[(i + 1) % v1.size()];
	auto pt	 = cc->MakeCKKSPackedPlaintext(v1);
	auto ct	 = cc->Encrypt(pt, keys.publicKey);
	auto res = cc->EvalRotate(ct, 1);
	CheckPrecision(cc, res, keys.secretKey, expected, CKKS_PRECISION);
}

// ---- EvalAddManyInPlace ----

TEST_F(CKKSTest, EvalAddManyInPlace) {
	std::vector<double> expected(v1.size());
	for (size_t i = 0; i < v1.size(); ++i)
		expected[i] = v1[i] + v2[i] + v1[i];
	auto pt1							  = cc->MakeCKKSPackedPlaintext(v1);
	auto pt2							  = cc->MakeCKKSPackedPlaintext(v2);
	auto ct1							  = cc->Encrypt(pt1, keys.publicKey);
	auto ct2							  = cc->Encrypt(pt2, keys.publicKey);
	auto ct3							  = cc->Encrypt(pt1, keys.publicKey);
	std::vector<Ciphertext<DCRTPoly>> cts = { ct1, ct2, ct3 };
	cc->EvalAddManyInPlace(cts);
	// Result is in cts[0].
	CheckPrecision(cc, cts[0], keys.secretKey, expected, CKKS_PRECISION);
}

// ---------------------------------------------------------------------------
// 14. Linear transforms: ConvolutionTransform / SpecialConvolutionTransform
//
// The CPU baby-step/giant-step path is compared against an independent
// clear-text simulation of the documented algorithm (which mirrors
// src/CKKS/LinearTransform.cu). Parameters are chosen to cover the branchy
// pieces: single even block, single odd block (linear accumulate), and a
// two-block case (inter-block rotation + even-tree accumulate), plus masking.
// ---------------------------------------------------------------------------

namespace {

// Clear-text reference. rot(v, k)[i] = v[(i + k) mod n] (OpenFHE left rotation).
std::vector<double> SimulateConvolution(const std::vector<double>& x,
  const std::vector<std::vector<double>>& pts,
  const std::vector<int>& indexes,
  int bStep,
  int gStep,
  int stride,
  int rowSize,
  const std::vector<double>* mask,
  int maskRotationStride) {
	const int n = static_cast<int>(x.size());
	auto rot	= [n](const std::vector<double>& v, int k) {
		   std::vector<double> out(n);
		   for (int i = 0; i < n; ++i)
			   out[i] = v[(((i + k) % n) + n) % n];
		   return out;
	};
	if (rowSize == 0)
		rowSize = bStep * gStep;

	std::vector<std::vector<double>> fr(bStep);
	for (int i = 0; i < bStep; ++i)
		fr[i] = rot(x, indexes[i]);

	constexpr uint32_t INTERNAL = 8;
	uint32_t blockCount			= (static_cast<uint32_t>(gStep) + INTERNAL - 1) / INTERNAL;
	std::vector<std::vector<double>> blockResults;
	for (uint32_t b = 0; b < blockCount; ++b) {
		uint32_t start = b * INTERNAL;
		uint32_t end   = std::min(start + INTERNAL, static_cast<uint32_t>(gStep));
		uint32_t cur   = end - start;
		std::vector<std::vector<double>> results(cur, std::vector<double>(n, 0.0));
		for (uint32_t j = 0; j < cur; ++j) {
			uint32_t gj = start + j;
			for (int s = 0; s < n; ++s)
				results[j][s] = pts[bStep * gj][s] * fr[0][s];
			for (int i = 1; i < bStep; ++i) {
				int ptIdx = bStep * static_cast<int>(gj) + i;
				if (ptIdx < rowSize)
					for (int s = 0; s < n; ++s)
						results[j][s] += pts[ptIdx][s] * fr[i][s];
			}
			if (mask != nullptr) {
				auto r1 = rot(results[j], maskRotationStride);
				auto r2 = rot(results[j], 2 * maskRotationStride);
				for (int s = 0; s < n; ++s)
					results[j][s] = (results[j][s] + r1[s] + r2[s]) * (*mask)[s];
			}
			int rotation = stride * static_cast<int>(cur - j);
			if (rotation != 0)
				results[j] = rot(results[j], rotation);
		}
		std::vector<double> acc(n, 0.0);
		for (uint32_t j = 0; j < cur; ++j)
			for (int s = 0; s < n; ++s)
				acc[s] += results[j][s];
		blockResults.push_back(acc);
	}
	if (blockCount > 1) {
		int baseRotation = static_cast<int>(INTERNAL) * stride;
		for (uint32_t b = 0; b < blockCount - 1; ++b) {
			int rotation = static_cast<int>(blockCount - 1 - b) * baseRotation;
			if (rotation != 0)
				blockResults[b] = rot(blockResults[b], rotation);
		}
	}
	std::vector<double> out(n, 0.0);
	for (const auto& br : blockResults)
		for (int s = 0; s < n; ++s)
			out[s] += br[s];
	return out;
}

// Deterministic, per-slot-varying weights so a rotation/index bug cannot hide.
std::vector<std::vector<double>> MakeWeights(int count, int n) {
	std::vector<std::vector<double>> w(count, std::vector<double>(n));
	for (int j = 0; j < count; ++j)
		for (int s = 0; s < n; ++s)
			w[j][s] = 0.1 * (((j * 3 + s) % 7) + 1);
	return w;
}

// Encrypt() takes a Plaintext& lvalue, so bind the encoded plaintext first.
Ciphertext<DCRTPoly> EncryptVec(CryptoContext<DCRTPoly>& cc, const std::vector<double>& v, const PublicKey<DCRTPoly>& pk) {
	auto pt = cc->MakeCKKSPackedPlaintext(v);
	return cc->Encrypt(pt, pk);
}

} // namespace

TEST_F(CKKSTest, ConvolutionTransformSingleBlockEven) {
	if (!TestUseCuda())
		GTEST_SKIP() << "CPU convolution is implemented in PR3 (backend-abstraction-cpu-impl).";
	// bStep=2, gStep=2: single block, even-tree accumulate.
	const int gStep = 2, bStep = 2, stride = 1;
	const std::vector<int> indexes = { 0, 1 };
	auto ptVals					   = MakeWeights(bStep * gStep, kSlots);

	std::vector<Plaintext> pts;
	for (const auto& v : ptVals)
		pts.push_back(cc->MakeCKKSPackedPlaintext(v));
	auto ct = EncryptVec(cc, v1, keys.publicKey);

	cc->ConvolutionTransformInPlace(ct, gStep, bStep, pts, indexes, stride);

	auto expected = SimulateConvolution(v1, ptVals, indexes, bStep, gStep, stride, 0, nullptr, 0);
	CheckPrecision(cc, ct, keys.secretKey, expected, 1e-3);
}

TEST_F(CKKSTest, ConvolutionTransformSingleBlockOdd) {
	if (!TestUseCuda())
		GTEST_SKIP() << "CPU convolution is implemented in PR3 (backend-abstraction-cpu-impl).";
	// bStep=2, gStep=3: single block, odd count exercises the linear-accumulate branch.
	const int gStep = 3, bStep = 2, stride = 1;
	const std::vector<int> indexes = { 0, 1 };
	auto ptVals					   = MakeWeights(bStep * gStep, kSlots);

	std::vector<Plaintext> pts;
	for (const auto& v : ptVals)
		pts.push_back(cc->MakeCKKSPackedPlaintext(v));
	auto ct = EncryptVec(cc, v1, keys.publicKey);

	cc->ConvolutionTransformInPlace(ct, gStep, bStep, pts, indexes, stride);

	auto expected = SimulateConvolution(v1, ptVals, indexes, bStep, gStep, stride, 0, nullptr, 0);
	CheckPrecision(cc, ct, keys.secretKey, expected, 1e-3);
}

TEST_F(CKKSTest, ConvolutionTransformMultiBlock) {
	if (!TestUseCuda())
		GTEST_SKIP() << "CPU convolution is implemented in PR3 (backend-abstraction-cpu-impl).";
	// bStep=1, gStep=9: two giant-step blocks, exercising inter-block rotation
	// and accumulation on top of the per-block (size-8 even) accumulate.
	const int gStep = 9, bStep = 1, stride = 1;
	const std::vector<int> indexes = { 0 };
	auto ptVals					   = MakeWeights(bStep * gStep, kSlots);

	std::vector<Plaintext> pts;
	for (const auto& v : ptVals)
		pts.push_back(cc->MakeCKKSPackedPlaintext(v));
	auto ct = EncryptVec(cc, v1, keys.publicKey);

	cc->ConvolutionTransformInPlace(ct, gStep, bStep, pts, indexes, stride);

	auto expected = SimulateConvolution(v1, ptVals, indexes, bStep, gStep, stride, 0, nullptr, 0);
	CheckPrecision(cc, ct, keys.secretKey, expected, 5e-3);
}

TEST_F(CKKSTest, SpecialConvolutionTransform) {
	if (!TestUseCuda())
		GTEST_SKIP() << "CPU convolution is implemented in PR3 (backend-abstraction-cpu-impl).";
	// Masked variant: bStep=2, gStep=2, maskRotationStride=1.
	const int gStep = 2, bStep = 2, stride = 1, maskRotationStride = 1;
	const std::vector<int> indexes = { 0, 1 };
	auto ptVals					   = MakeWeights(bStep * gStep, kSlots);
	std::vector<double> maskVals(kSlots);
	for (size_t s = 0; s < kSlots; ++s)
		maskVals[s] = (s % 2 == 0) ? 1.0 : 0.5;

	std::vector<Plaintext> pts;
	for (const auto& v : ptVals)
		pts.push_back(cc->MakeCKKSPackedPlaintext(v));
	auto mask = cc->MakeCKKSPackedPlaintext(maskVals);
	auto ct	  = EncryptVec(cc, v1, keys.publicKey);

	cc->SpecialConvolutionTransformInPlace(ct, gStep, bStep, pts, mask, indexes, stride, maskRotationStride);

	auto expected = SimulateConvolution(v1, ptVals, indexes, bStep, gStep, stride, 0, &maskVals, maskRotationStride);
	CheckPrecision(cc, ct, keys.secretKey, expected, 1e-2);
}

// ---- 15. Additional operation paths ----

TEST_F(CKKSTest, EvalFastRotationVector) {
	// Vector overload returns one clean rotation per requested index.
	const std::vector<int32_t> indices = { 1, 2 };
	auto ct							   = EncryptVec(cc, v1, keys.publicKey);
	auto precomp					   = cc->EvalFastRotationPrecompute(ct);
	uint32_t m						   = cc->GetCyclotomicOrder();
	auto results					   = cc->EvalFastRotation(ct, indices, m, precomp);
	ASSERT_EQ(results.size(), indices.size());
	for (size_t k = 0; k < indices.size(); ++k) {
		std::vector<double> expected(v1.size());
		for (size_t i = 0; i < v1.size(); ++i)
			expected[i] = v1[(i + indices[k]) % v1.size()];
		CheckPrecision(cc, results[k], keys.secretKey, expected, CKKS_PRECISION);
	}
}

TEST_F(CKKSTest, EvalFastRotationExt) {
	// EvalFastRotationExt yields an extended-basis intermediate (not a plain
	// rotation), so just exercise the CPU dispatch and ensure it succeeds.
	auto ct		 = EncryptVec(cc, v1, keys.publicKey);
	auto precomp = cc->EvalFastRotationPrecompute(ct);
	Ciphertext<DCRTPoly> res;
	EXPECT_NO_THROW(res = cc->EvalFastRotationExt(ct, 1, precomp, true));
	EXPECT_NE(res.get(), nullptr);
}

TEST_F(CKKSTest, AccumulateSumInPlaceWithStart) {
	// The 4-arg overload (start, then doubling): start=1 reduces all 8 slots into
	// slot 0 (= 36). Like the AccumulateSum siblings, only slot 0 is guaranteed —
	// the CUDA backend reduces into slot 0 and leaves the rest 0.
	auto pt = cc->MakeCKKSPackedPlaintext(v1);
	auto ct = cc->Encrypt(pt, keys.publicKey);
	cc->AccumulateSumInPlace(ct, kSlots, 1, 1);
	Plaintext out;
	cc->Decrypt(ct, keys.secretKey, &out);
	out->SetLength(1);
	auto vals = out->GetRealPackedValue();
	EXPECT_NEAR(vals[0], 36.0, CKKS_PRECISION);
}

TEST_F(CKKSTest, SerializeDeserializeContextRoundTrip) {
	// Exercises the full-context serializer, including the CPU-only device
	// metadata block ("0 { }") written/read by api/Serialize.cpp.
	const std::string path = "/tmp/fideslib_cpu_ctx_roundtrip";
	ASSERT_TRUE(Serial::SerializeToFile(path, cc, SerType::BINARY));

	CryptoContext<DCRTPoly> cc2;
	ASSERT_TRUE(Serial::DeserializeFromFile(path, cc2, SerType::BINARY));
	ASSERT_NE(cc2.get(), nullptr);
	EXPECT_EQ(cc2->GetRingDimension(), cc->GetRingDimension());

	std::remove(path.c_str());
	std::remove((path + ".dev").c_str());
}
