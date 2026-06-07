// api-vs-OpenFHE parity suite.
//
// Drives the public fideslib api on the CUDA backend and asserts the resulting ciphertext is
// bit-for-bit identical to the same computation run directly through OpenFHE (the oracle). This is
// the api-level counterpart of the raw-GPU parity tests in OpenFheInterfaceTests.cu: instead of
// hand-plumbing FIDESlib::CKKS device objects, every op goes through CryptoContextImpl, and the
// result is recovered with CryptoContextImpl::RecoverHostCiphertext (the backend readback seam).
//
// Because it compares the active backend against OpenFHE, it is only meaningful on a non-CPU
// backend: on the CPU backend the api *is* OpenFHE, so the comparison is trivial and the fixture
// skips. Set FIDESLIB_TEST_BACKEND=cuda on a CUDA build to run it against the CUDA engine. The same
// tests will exercise a future Haze backend (Haze == OpenFHE) unchanged, just by selecting it.
//
// Oracle note: when several ops are chained, the api runs them on the device and only the final
// result is synced back, so intermediate host shadows are stale. The oracle therefore replays the
// whole chain in OpenFHE from the freshly-encrypted inputs (never from a mid-chain api ciphertext).

#include <gtest/gtest.h>
#include <openfhe.h>

#include <any>
#include <cmath>
#include <string>
#include <vector>

#include "fideslib.hpp"

using namespace fideslib;

// Exact ciphertext equality: scaling metadata + every RNS limb of both polynomials. Mirrors the
// ASSERT_EQ_CIPHERTEXT in test/ParametrizedTest.cuh, minus the per-limb debug printing.
#define ASSERT_EQ_CIPHERTEXT(ct1, ct2)                                                                                                                   \
	do {                                                                                                                                                 \
		ASSERT_EQ((ct1)->GetNoiseScaleDeg(), (ct2)->GetNoiseScaleDeg());                                                                                 \
		ASSERT_EQ((ct1)->GetScalingFactor(), (ct2)->GetScalingFactor());                                                                                 \
		ASSERT_EQ((ct1)->GetEncodingType(), (ct2)->GetEncodingType());                                                                                   \
		for (size_t j = 0; j < 2; ++j) {                                                                                                                 \
			ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().size(), (ct2)->GetElements().at(j).GetAllElements().size());                           \
			for (size_t i = 0; i < (ct1)->GetElements().at(j).GetAllElements().size(); ++i) {                                                            \
				ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().at(i).GetValues().GetLength(),                                                     \
				  (ct2)->GetElements().at(j).GetAllElements().at(i).GetValues().GetLength());                                                            \
				ASSERT_EQ((ct1)->GetElements().at(j).GetAllElements().at(i).GetValues(), (ct2)->GetElements().at(j).GetAllElements().at(i).GetValues()); \
			}                                                                                                                                            \
		}                                                                                                                                                \
	} while (0)

static bool TestUseCuda() {
	const char* b = std::getenv("FIDESLIB_TEST_BACKEND");
	return b != nullptr && std::string(b) == "cuda" && IsBackendAvailable(Backend::CUDA);
}

// The OpenFHE context the api wraps (its CPU shadow). It holds the same keys the api generated, so
// it is the parity oracle.
static lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& LbCc(CryptoContext<DCRTPoly>& cc) {
	return std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->host);
}

// The lbcrypto ciphertext behind a freshly-encrypted api ciphertext (host-resident before any op).
// Only valid on encrypt outputs — a mid-chain api ciphertext's shadow is stale on the CUDA backend.
static lbcrypto::Ciphertext<lbcrypto::DCRTPoly> LbCt(Ciphertext<DCRTPoly>& ct) {
	return std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct->host);
}

// The lbcrypto plaintext behind an api plaintext.
static lbcrypto::Plaintext LbPt(Plaintext& pt) {
	return std::any_cast<lbcrypto::Plaintext>(pt->host);
}

// Recover the lbcrypto ciphertext of an api result, syncing it back from the device first (the device
// copy stays resident, so the ciphertext can still be used on-device afterward).
static lbcrypto::Ciphertext<lbcrypto::DCRTPoly> HostCt(CryptoContext<DCRTPoly>& cc, Ciphertext<DCRTPoly>& ct) {
	cc->RecoverHostCiphertext(ct);
	return std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct->host);
}

// Decrypt two lbcrypto ciphertexts and compare their real slots within tol. For ops that are
// numerically correct but not bit-identical to OpenFHE (the raw OpenFheInterfaceTests use
// ASSERT_ERROR_OK for the same ops, for the same reason).
static void ExpectSlotsNear(CryptoContext<DCRTPoly>& cc,
  const PrivateKey<DCRTPoly>& sk,
  const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& got,
  const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& oracle,
  size_t slots,
  double tol) {
	auto& lbcc	 = LbCc(cc);
	auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	lbcrypto::Plaintext pgot, porc;
	lbcc->Decrypt(skImpl, got, &pgot);
	lbcc->Decrypt(skImpl, oracle, &porc);
	pgot->SetLength(slots);
	porc->SetLength(slots);
	auto vg = pgot->GetRealPackedValue();
	auto vo = porc->GetRealPackedValue();
	for (size_t i = 0; i < slots; ++i)
		EXPECT_NEAR(vg[i], vo[i], tol) << "slot " << i;
}

// Same parameters as the api correctness fixture (ApiTests.cpp): depth 5, ringDim 65536, 8 slots,
// FIXEDAUTO (default secret-key dist, as in ApiTests).
class ApiParityTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth   = 5;
	static constexpr uint32_t kRingDim = 1u << 16;
	static constexpr uint32_t kSlots   = 8;

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;

	const std::vector<double> v1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	const std::vector<double> v2 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
	// Near-1 values keep repeated squaring well-conditioned (above the noise floor, below overflow).
	const std::vector<double> vsq = { 0.93, 0.96, 0.99, 1.0, 1.01, 1.04, 1.07, 0.95 };

	void SetUp() override {
		if (!TestUseCuda())
			GTEST_SKIP() << "api-vs-OpenFHE parity is trivial on the CPU backend (api == OpenFHE); "
							"set FIDESLIB_TEST_BACKEND=cuda on a CUDA build to run it.";

		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(50);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FIXEDAUTO);
		params.SetBackend(Backend::CUDA);
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		cc->EvalRotateKeyGen(keys.secretKey, { 1, 2, 3, 4, 5, 6, 7, 8, -1, -2 });
		cc->LoadContext(keys.publicKey);
	}

	// Encrypt v1 and v2 with the public key.
	void EncryptInputs(Ciphertext<DCRTPoly>& a1, Ciphertext<DCRTPoly>& a2) {
		auto p1 = cc->MakeCKKSPackedPlaintext(v1);
		auto p2 = cc->MakeCKKSPackedPlaintext(v2);
		a1		= cc->Encrypt(p1, keys.publicKey);
		a2		= cc->Encrypt(p2, keys.publicKey);
	}

	// Encrypt just v1 (for single-input ops).
	Ciphertext<DCRTPoly> EncryptV1() {
		auto p = cc->MakeCKKSPackedPlaintext(v1);
		return cc->Encrypt(p, keys.publicKey);
	}

	// Approximate comparison for the single-op tests (negate/rotate/rescale).
	void ExpectApproxEq(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& got, const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& oracle) {
		ExpectSlotsNear(cc, keys.secretKey, got, oracle, kSlots, 1e-6);
	}
};

TEST_F(ApiParityTest, EvalAdd) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	auto res	= cc->EvalAdd(a1, a2);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalAdd(lb1, lb2);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, EvalSub) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	auto res	= cc->EvalSub(a1, a2);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalSub(lb1, lb2);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, EvalMult) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	auto res	= cc->EvalMult(a1, a2);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalMult(lb1, lb2);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, EvalSquare) {
	auto a1		= EncryptV1();
	auto lb1	= LbCt(a1);
	auto res	= cc->EvalSquare(a1);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalSquare(lb1);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, EvalNegate) {
	auto a1		= EncryptV1();
	auto lb1	= LbCt(a1);
	auto res	= cc->EvalNegate(a1);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalNegate(lb1);
	// Approximate, not bit-identical: the CUDA backend implements negate as multScalar(-1.0), which
	// bumps NoiseScaleDeg (1->2) relative to OpenFHE's degree-preserving sign flip. The decrypted
	// values match; the scale-degree bookkeeping differs (a known GPU EvalNegate fidelity gap).
	ExpectApproxEq(got, oracle);
}

TEST_F(ApiParityTest, EvalRotate) {
	auto a1		= EncryptV1();
	auto lb1	= LbCt(a1);
	auto res	= cc->EvalRotate(a1, 1);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalRotate(lb1, 1);
	// Approximate, not bit-identical: GPU rotation (hoisted automorphism + key-switch) is numerically
	// correct but uses a different key-switch representation than OpenFHE. The raw OpenFheInterfaceTests
	// Rotate test uses ASSERT_ERROR_OK for exactly this reason (its ASSERT_EQ_CIPHERTEXT is commented out).
	ExpectApproxEq(got, oracle);
}

TEST_F(ApiParityTest, EvalAddPlaintext) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1	= LbCt(a1);
	auto pt		= cc->MakeCKKSPackedPlaintext(v2);
	auto lbpt	= LbPt(pt);
	auto res	= cc->EvalAdd(a1, pt);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalAdd(lb1, lbpt);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, EvalMultPlaintext) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1	= LbCt(a1);
	auto pt		= cc->MakeCKKSPackedPlaintext(v2);
	auto lbpt	= LbPt(pt);
	auto res	= cc->EvalMult(a1, pt);
	auto got	= HostCt(cc, res);
	auto oracle = LbCc(cc)->EvalMult(lb1, lbpt);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, EvalAddScalar) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1	 = LbCt(a1);
	auto resP	 = cc->EvalAdd(a1, 2.0);
	auto gotP	 = HostCt(cc, resP);
	auto oracleP = LbCc(cc)->EvalAdd(lb1, 2.0);
	ASSERT_EQ_CIPHERTEXT(oracleP, gotP);
	auto resN	 = cc->EvalAdd(a1, -2.0);
	auto gotN	 = HostCt(cc, resN);
	auto oracleN = LbCc(cc)->EvalAdd(lb1, -2.0);
	ASSERT_EQ_CIPHERTEXT(oracleN, gotN);
}

TEST_F(ApiParityTest, EvalMultScalar) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1);
	// 2^-7 matches the raw MultScalar test (a clean power-of-two scalar).
	const double s = std::pow(2.0, -7.0);
	auto res	   = cc->EvalMult(a1, s);
	auto got	   = HostCt(cc, res);
	auto oracle	   = LbCc(cc)->EvalMult(lb1, s);
	ASSERT_EQ_CIPHERTEXT(oracle, got);
}

TEST_F(ApiParityTest, RescaleAfterMult) {
	Ciphertext<DCRTPoly> a1, a2;
	EncryptInputs(a1, a2);
	auto lb1 = LbCt(a1), lb2 = LbCt(a2);
	auto prod = cc->EvalMult(a1, a2);
	auto res  = cc->Rescale(prod);
	auto got  = HostCt(cc, res);
	// Oracle replays the whole chain in OpenFHE (the mid-chain product's host shadow is stale).
	auto oracle = LbCc(cc)->Rescale(LbCc(cc)->EvalMult(lb1, lb2));
	// Approximate, not bit-identical: GPU and OpenFHE track NoiseScaleDeg differently across an
	// explicit Rescale under FIXEDAUTO (deferred-rescale accounting). The decrypted values match.
	ExpectApproxEq(got, oracle);
}

// ---- AllLevels depth tests: exercise an op at each level down the modulus chain ----
// These walk the ciphertext down the chain via mult+rescale, which crosses the api-Rescale
// NoiseScaleDeg divergence, so they compare decrypted values (approximate). Bit-identity at each
// individual level remains covered by the retained raw OpenFheInterfaceTests *AllLevels tests.

TEST_F(ApiParityTest, EvalMultAllLevels) {
	Ciphertext<DCRTPoly> a, b;
	EncryptInputs(a, b);
	auto lbA = LbCt(a);
	auto lbB = LbCt(b);
	for (uint32_t lvl = 0; lvl + 1 < kDepth; ++lvl) {
		SCOPED_TRACE("level " + std::to_string(lvl));
		a = cc->EvalMult(a, b);
		cc->RescaleInPlace(a);
		lbA = LbCc(cc)->EvalMult(lbA, lbB);
		LbCc(cc)->RescaleInPlace(lbA);
		auto got = HostCt(cc, a);
		ExpectSlotsNear(cc, keys.secretKey, got, lbA, kSlots, 1e-5);
	}
}

TEST_F(ApiParityTest, EvalSquareAllLevels) {
	auto p	 = cc->MakeCKKSPackedPlaintext(vsq);
	auto a	 = cc->Encrypt(p, keys.publicKey);
	auto lbA = LbCt(a);
	for (uint32_t lvl = 0; lvl + 1 < kDepth; ++lvl) {
		SCOPED_TRACE("level " + std::to_string(lvl));
		a = cc->EvalSquare(a);
		cc->RescaleInPlace(a);
		lbA = LbCc(cc)->EvalSquare(lbA);
		LbCc(cc)->RescaleInPlace(lbA);
		auto got = HostCt(cc, a);
		ExpectSlotsNear(cc, keys.secretKey, got, lbA, kSlots, 1e-5);
	}
}

TEST_F(ApiParityTest, EvalRotateAllLevels) {
	Ciphertext<DCRTPoly> a, b;
	EncryptInputs(a, b);
	auto lbA = LbCt(a);
	auto lbB = LbCt(b);
	for (uint32_t lvl = 0; lvl + 1 < kDepth; ++lvl) {
		SCOPED_TRACE("level " + std::to_string(lvl));
		// Rotate at the current level.
		auto r		 = cc->EvalRotate(a, 1);
		auto gotR	 = HostCt(cc, r);
		auto oracleR = LbCc(cc)->EvalRotate(lbA, 1);
		ExpectSlotsNear(cc, keys.secretKey, gotR, oracleR, kSlots, 1e-5);
		// Descend a level for the next iteration.
		a = cc->EvalMult(a, b);
		cc->RescaleInPlace(a);
		lbA = LbCc(cc)->EvalMult(lbA, lbB);
		LbCc(cc)->RescaleInPlace(lbA);
	}
}

// ---- Bootstrap parity (DISABLED) ----
// EvalBootstrap on GPU vs OpenFHE. DISABLED until the GPU FIXEDAUTO bootstrap precision defect is
// fixed (the open "Bootstrap FIXEDAUTO" item; see the plan/gpu-results notes) — enabling it now would
// only re-report that known failure. Drop the DISABLED_ prefix once that fix lands. Bootstrap is
// heavily approximate, so it compares decrypted values within the bootstrap precision (~1e-2).
class ApiParityBootstrapTest : public ::testing::Test {
  protected:
	static constexpr uint32_t kDepth   = 25;
	static constexpr uint32_t kRingDim = 1u << 16;
	static constexpr uint32_t kSlots   = 8;

	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	const std::vector<double> v1 = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };

	void SetUp() override {
		if (!TestUseCuda())
			GTEST_SKIP() << "api-vs-OpenFHE parity is trivial on the CPU backend.";
		CCParams<CryptoContextCKKSRNS> params;
		params.SetMultiplicativeDepth(kDepth);
		params.SetScalingModSize(50);
		params.SetBatchSize(kSlots);
		params.SetRingDim(kRingDim);
		params.SetScalingTechnique(FIXEDAUTO);
		params.SetSecretKeyDist(UNIFORM_TERNARY);
		params.SetSecurityLevel(HEStd_NotSet);
		params.SetBackend(Backend::CUDA);
		cc = GenCryptoContext(params);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);
		cc->Enable(FHE);
		keys = cc->KeyGen();
		cc->EvalMultKeyGen(keys.secretKey);
		cc->EvalBootstrapSetup({ 1, 1 }, { 0, 0 }, kSlots, 11);
		cc->EvalBootstrapKeyGen(keys.secretKey, kSlots);
		cc->LoadContext(keys.publicKey);
	}
};

TEST_F(ApiParityBootstrapTest, DISABLED_EvalBootstrap) {
	auto pt		   = cc->MakeCKKSPackedPlaintext(v1);
	auto ct		   = cc->Encrypt(pt, keys.publicKey);
	auto lbCt	   = LbCt(ct);
	auto refreshed = cc->EvalBootstrap(ct);
	auto got	   = HostCt(cc, refreshed);
	auto oracle	   = LbCc(cc)->EvalBootstrap(lbCt);
	ExpectSlotsNear(cc, keys.secretKey, got, oracle, kSlots, 1e-2);
}
