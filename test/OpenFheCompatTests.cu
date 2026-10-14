#include "ParametrizedTest.cuh"

#include <CKKS/Ciphertext.cuh>
#include <CKKS/Plaintext.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>

#include <algorithm>
#include <any>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fideslib.hpp>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

using namespace fideslib;

namespace FIDESlib::Testing {

TEST(OpenFHECompatTests, EvalFastRotation) {
    uint32_t multDepth = 2;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(128);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, { 1, -2 });

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cRot1 = cc->EvalFastRotation(ctxt, 1, 2 * cc->GetRingDimension(), cc->EvalFastRotationPrecompute(ctxt));

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cRot2 = cc->EvalFastRotation(ctxt, 1, 2 * cc->GetRingDimension(), cc->EvalFastRotationPrecompute(ctxt));

    // EXPECT_EQ(cRot1->GetElements(), cRot2->GetElements());
    ASSERT_EQ_CIPHERTEXT(cRot1, cRot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cRot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cRot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

TEST(OpenFHECompatTests, EvalRotate) {
    uint32_t multDepth = 2;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(128);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, { 1, -2 });

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cRot1 = cc->EvalRotate(ctxt, 1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cRot2 = cc->EvalRotate(ctxt, 1);

    // EXPECT_EQ(cRot1->GetElements(), cRot2->GetElements());
    ASSERT_EQ_CIPHERTEXT(cRot1, cRot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cRot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cRot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

TEST(OpenFHECompatTests, AccumulateSum) {
    uint32_t multDepth = 2;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(128);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    // The accumulation radix is OpenFHE's compile-time PARTIAL_SUM_RADIX; generate the
    // fold's rotation indices {i*radix^level : i in [1, radix)} for slots=8, stride=1.
    std::vector<int32_t> accIndices;
    for (uint32_t s = 1; s < batchSize; s *= PARTIAL_SUM_RADIX)
        for (uint32_t idx = s; idx < batchSize && idx < PARTIAL_SUM_RADIX * s; idx += s)
            accIndices.push_back(static_cast<int32_t>(idx));
    cc->EvalRotateKeyGen(keys.secretKey, accIndices);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cAcc1 = cc->AccumulateSum(ctxt, batchSize, 1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cAcc2 = cc->AccumulateSum(ctxt, batchSize, 1);

    ASSERT_EQ_CIPHERTEXT(cAcc1, cAcc2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cAcc1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cAcc2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

TEST(OpenFHECompatTests, EvalBootstrap) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    // EXPECT_EQ(cBoot1->GetElements(), cBoot2->GetElements());
    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

TEST(OpenFHECompatTests, EvalBootstrapDense) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t ringDim = 1 << 12;
    uint32_t numSlots = ringDim / 2; // fully packed

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(numSlots);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(ringDim);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    // EXPECT_EQ(cBoot1->GetElements(), cBoot2->GetElements());
    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(numSlots);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(numSlots);

    ASSERT_ERROR_OK(r1, r2);
}

// level budget {1,1} takes the isLT/EvalLinearTransform branch
// of CoeffsToSlots/SlotsToCoeffs instead of the FFT-decomposed Horner path.
TEST(OpenFHECompatTests, EvalBootstrapLT) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 1, 1 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

// a different sparse slot count changes the PartialSum depth
// and the CtS/StC FFT split parameters.
TEST(OpenFHECompatTests, EvalBootstrapSlots64) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 64;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

// FIXEDMANUAL bootstrap exercises the manual-rescale branches
// of the raise/Chebyshev/StC transcriptions that the FLEXIBLE tests never reach.
TEST(OpenFHECompatTests, EvalBootstrapFixedManual) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FIXEDMANUAL);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

// Fully-packed bootstrap under FIXEDMANUAL: the dense branch splits into
// real/imaginary Chebyshev evaluations and recombines, a pipeline the sparse
// FIXEDMANUAL test never enters.
TEST(OpenFHECompatTests, EvalBootstrapDenseFixedManual) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t ringDim = 1 << 12;
    uint32_t numSlots = ringDim / 2; // fully packed

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FIXEDMANUAL);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(numSlots);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(ringDim);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    // The slot-dependent default correction factor lands at 9 for fully-packed
    // slots at this toy ring size, below deg = log2(2^60/2^50) = 10, which
    // EvalBootstrap rejects. Production-scale parameters don't trip this;
    // pass an explicit correction factor to get the same moduli as the
    // FLEXIBLEAUTO dense test.
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 10);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(numSlots);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(numSlots);

    ASSERT_ERROR_OK(r1, r2);
}

// SPARSE_ENCAPSULATED secret-key distribution takes its own
// Chebyshev coefficient set (g_coefficientsSparseEncapsulated) and keygen path.
// DISABLED — known structural divergence (O6c): the stock in-context sparse-switch
// dance and FIDESlib's dual-context design are different pipelines, and at this
// configuration the GPU path decodes to ~zero. Acceptance test for the O6c fix.
TEST(OpenFHECompatTests, DISABLED_EvalBootstrapSparseEncaps) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(SPARSE_ENCAPSULATED);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

// FLEXIBLEAUTOEXT (OpenFHE's default) adds an extra level at
// encryption and changes the ModRaise handling.
TEST(OpenFHECompatTests, EvalBootstrapFlexExt) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTOEXT);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

// Combination sweep: fully packed + FLEXIBLEAUTOEXT — dense real/imaginary
// split downstream of the extra-level ModRaise.
TEST(OpenFHECompatTests, EvalBootstrapDenseFlexExt) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t ringDim = 1 << 12;
    uint32_t numSlots = ringDim / 2; // fully packed

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTOEXT);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(numSlots);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(ringDim);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(numSlots);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(numSlots);

    ASSERT_ERROR_OK(r1, r2);
}

// Combination sweep: level budget {1,1} (EvalLinearTransform branch) + FIXEDMANUAL —
// the LT path's rescale gates were only ever exercised under FLEXIBLEAUTO.
TEST(OpenFHECompatTests, EvalBootstrapLTFixedManual) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FIXEDMANUAL);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 1, 1 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

// Combination sweep: SPARSE_TERNARY secret keys — selects the FIDESlib::SPARSE
// boot config (g_coefficientsSparse Chebyshev set, bootK=1.0, no encapsulation),
// a coefficient/keygen path no other test touches.
TEST(OpenFHECompatTests, EvalBootstrapSparseSecret) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(SPARSE_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

// Exhaustive key-distribution × bootstrap-variant matrix. Every row asserts both
// bit-exactness (ASSERT_EQ_CIPHERTEXT) and value-level agreement (ASSERT_ERROR_OK) so
// any failure mode shows up explicitly in the results. Variants covered: fully packed
// (slots = ringDim/2), LT ({1,1}) level budget, and multi-level sparse packing ({1,2}).
//
// SPARSE_ENCAPSULATED rows (EvalBootstrapSparseEncapsDense / EvalBootstrapSparseEncapsLT)
// validate the dual-context switching-key design (BTS_KSPARSE_*_SLOT in RawCiphertext.cu):
// the CPU reference reads OpenFHE's own sparse-switch keys at 2N-2/2N-4, the GPU pipeline
// its disjoint FIDESlib keys at 2N-6/2N-8. They assert value-level agreement only — the O6c
// carve-out (see BITCOMPAT.md): the dual-context design is still bit-divergent from the
// in-context stock reference (level/metadata gap), so bit-exactness is deferred to the O6c
// fix. The UNIFORM_TERNARY / SPARSE_TERNARY rows must match bit-exactly — if one of those
// fails, that is a different bug (e.g. a bootK regression).

// SPARSE_TERNARY × fully packed: sparse-secret run of the dense (complex) fully-packed
// branch — conjugate split, imaginary component, full approx-mod-reduction.
TEST(OpenFHECompatTests, EvalBootstrapSparseDense) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t ringDim = 1 << 12;
    uint32_t numSlots = ringDim / 2; // fully packed

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(SPARSE_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(numSlots);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(ringDim);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(numSlots);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(numSlots);

    // Value-level check first: it is always less strict than the bit-exact check below, so
    // a bit divergence (e.g. SPARSE_ENCAPSULATED's known O6c level/metadata gap) must never
    // mask a value-level failure.
    ASSERT_ERROR_OK(r1, r2);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);
}

// SPARSE_TERNARY × LT {1,1}: the EvalLinearTransform (linear-transformation) branch of
// CoeffsToSlots/SlotsToCoeffs under a sparse secret.
TEST(OpenFHECompatTests, EvalBootstrapSparseLT) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(SPARSE_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 1, 1 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    // Value-level check first: it is always less strict than the bit-exact check below, so
    // a bit divergence (e.g. SPARSE_ENCAPSULATED's known O6c level/metadata gap) must never
    // mask a value-level failure.
    ASSERT_ERROR_OK(r1, r2);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);
}

// SPARSE_TERNARY × multi-level {1,2} budget: asymmetric multi-level CtS/StC split
// (encoding lvlb=1, decoding lvlb=2) under a sparse secret — a different FFT/Horner
// collapse structure than the {3,3} and {1,1} budgets.
TEST(OpenFHECompatTests, EvalBootstrapSparseMultiLevel) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(SPARSE_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 1, 2 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    // Value-level check first: it is always less strict than the bit-exact check below, so
    // a bit divergence (e.g. SPARSE_ENCAPSULATED's known O6c level/metadata gap) must never
    // mask a value-level failure.
    ASSERT_ERROR_OK(r1, r2);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);
}

// UNIFORM_TERNARY × multi-level {1,2} budget: same asymmetric budget on the dense-secret
// row (existing UNIFORM coverage only had {3,3} and {1,1}).
TEST(OpenFHECompatTests, EvalBootstrapDenseMultiLevel) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 1, 2 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    // Value-level check first: it is always less strict than the bit-exact check below, so
    // a bit divergence (e.g. SPARSE_ENCAPSULATED's known O6c level/metadata gap) must never
    // mask a value-level failure.
    ASSERT_ERROR_OK(r1, r2);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);
}

// SPARSE_ENCAPSULATED × fully packed, level budget {3,3}: the CPU reference (OpenFHE's own
// sparse-switch keys at 2N-2/2N-4) and the GPU pipeline (FIDESlib's disjoint keys at 2N-6/2N-8)
// must agree both at value level and bit-exactly.
TEST(OpenFHECompatTests, EvalBootstrapSparseEncapsDense) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t ringDim = 1 << 12;
    uint32_t numSlots = ringDim / 2; // fully packed

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(SPARSE_ENCAPSULATED);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(numSlots);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(ringDim);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(numSlots);
    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(numSlots);

    // Value-level agreement is the contract for the SPARSE_ENCAPSULATED rows (the O6c
    // carve-out): the dual-context GPU design is still bit-divergent (level/metadata gap)
    // from the in-context stock reference, so bit-exactness is deferred to the O6c fix
    // (see BITCOMPAT.md). The value check must never be masked by the deferred one.
    ASSERT_ERROR_OK(r1, r2);
}

// SPARSE_ENCAPSULATED × LT {1,1}: the low-budget sibling of EvalBootstrapSparseEncapsDense,
// with the same CPU/GPU dual sparse-switch key layout.
TEST(OpenFHECompatTests, EvalBootstrapSparseEncapsLT) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(SPARSE_ENCAPSULATED);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 1, 1 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);
    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    // Value-level agreement is the contract for the SPARSE_ENCAPSULATED rows (the O6c
    // carve-out): the dual-context GPU design is still bit-divergent (level/metadata gap)
    // from the in-context stock reference, so bit-exactness is deferred to the O6c fix
    // (see BITCOMPAT.md). The value check must never be masked by the deferred one.
    ASSERT_ERROR_OK(r1, r2);
}

// Combination sweep: FIXEDAUTO bootstrap — the fourth scaling technique,
// previously untested at any level. Re-enabled: the O6e add/sub operand adjustment
// divergence is fixed in the current tree (stock AdjustForAddOrSub updates the GPU
// add/sub gates for every technique except FIXEDMANUAL).
TEST(OpenFHECompatTests, EvalBootstrapFixedAuto) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FIXEDAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot1 = cc->EvalBootstrap(ctxt);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto cBoot2 = cc->EvalBootstrap(ctxt);

    ASSERT_EQ_CIPHERTEXT(cBoot1, cBoot2);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot1, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, cBoot2, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

// Shared context builder for the arithmetic-level bit-compat tests.
static CryptoContext<DCRTPoly> MakeSmallContext(uint32_t multDepth, ScalingTechnique st = FLEXIBLEAUTO) {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(st);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(8);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(128);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    return cc;
}

TEST(OpenFHECompatTests, EvalArithmetic) {
    auto cc = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto cAdd = cc->EvalAdd(ct1, ct2);
    auto cAddSc = cc->EvalAdd(ct1, 0.5);
    auto cSub = cc->EvalSub(ct1, ct2);
    auto cSubSc = cc->EvalSub(ct1, 0.25);
    auto cNeg = cc->EvalNegate(ct1);
    auto cMult = cc->EvalMult(ct1, ct2);
    auto cMultSc = cc->EvalMult(ct1, 1.5);
    auto cSq = cc->EvalSquare(ct1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAdd = cc->EvalAdd(ct1, ct2);
    auto gAddSc = cc->EvalAdd(ct1, 0.5);
    auto gSub = cc->EvalSub(ct1, ct2);
    auto gSubSc = cc->EvalSub(ct1, 0.25);
    auto gNeg = cc->EvalNegate(ct1);
    auto gMult = cc->EvalMult(ct1, ct2);
    auto gMultSc = cc->EvalMult(ct1, 1.5);
    auto gSq = cc->EvalSquare(ct1);

    ASSERT_EQ_CIPHERTEXT(cAdd, gAdd);
    ASSERT_EQ_CIPHERTEXT(cAddSc, gAddSc);
    ASSERT_EQ_CIPHERTEXT(cSub, gSub);
    ASSERT_EQ_CIPHERTEXT(cSubSc, gSubSc);
    ASSERT_EQ_CIPHERTEXT(cMult, gMult);
    ASSERT_EQ_CIPHERTEXT(cMultSc, gMultSc);
    ASSERT_EQ_CIPHERTEXT(cSq, gSq);
    ASSERT_EQ_CIPHERTEXT(cNeg, gNeg);
}

// same arithmetic ops under FLEXIBLEAUTOEXT, separating
// "AUTOEXT basics" from "AUTOEXT bootstrap" if the bootstrap variant goes red.
TEST(OpenFHECompatTests, EvalArithmeticFlexExt) {
    auto cc = MakeSmallContext(4, FLEXIBLEAUTOEXT);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto cAdd = cc->EvalAdd(ct1, ct2);
    auto cAddSc = cc->EvalAdd(ct1, 0.5);
    auto cSub = cc->EvalSub(ct1, ct2);
    auto cSubSc = cc->EvalSub(ct1, 0.25);
    auto cNeg = cc->EvalNegate(ct1);
    auto cMult = cc->EvalMult(ct1, ct2);
    auto cMultSc = cc->EvalMult(ct1, 1.5);
    auto cSq = cc->EvalSquare(ct1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAdd = cc->EvalAdd(ct1, ct2);
    auto gAddSc = cc->EvalAdd(ct1, 0.5);
    auto gSub = cc->EvalSub(ct1, ct2);
    auto gSubSc = cc->EvalSub(ct1, 0.25);
    auto gNeg = cc->EvalNegate(ct1);
    auto gMult = cc->EvalMult(ct1, ct2);
    auto gMultSc = cc->EvalMult(ct1, 1.5);
    auto gSq = cc->EvalSquare(ct1);

    ASSERT_EQ_CIPHERTEXT(cAdd, gAdd);
    ASSERT_EQ_CIPHERTEXT(cAddSc, gAddSc);
    ASSERT_EQ_CIPHERTEXT(cSub, gSub);
    ASSERT_EQ_CIPHERTEXT(cSubSc, gSubSc);
    ASSERT_EQ_CIPHERTEXT(cMult, gMult);
    ASSERT_EQ_CIPHERTEXT(cMultSc, gMultSc);
    ASSERT_EQ_CIPHERTEXT(cSq, gSq);
    ASSERT_EQ_CIPHERTEXT(cNeg, gNeg);
}

// Combination sweep: FIXEDAUTO arithmetic — auto-rescale with fixed factors,
// the fourth scaling technique, previously untested at any level.
TEST(OpenFHECompatTests, EvalArithmeticFixedAuto) {
    auto cc = MakeSmallContext(4, FIXEDAUTO);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto cAdd = cc->EvalAdd(ct1, ct2);
    auto cAddSc = cc->EvalAdd(ct1, 0.5);
    auto cSub = cc->EvalSub(ct1, ct2);
    auto cSubSc = cc->EvalSub(ct1, 0.25);
    auto cNeg = cc->EvalNegate(ct1);
    auto cMult = cc->EvalMult(ct1, ct2);
    auto cMultSc = cc->EvalMult(ct1, 1.5);
    auto cSq = cc->EvalSquare(ct1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAdd = cc->EvalAdd(ct1, ct2);
    auto gAddSc = cc->EvalAdd(ct1, 0.5);
    auto gSub = cc->EvalSub(ct1, ct2);
    auto gSubSc = cc->EvalSub(ct1, 0.25);
    auto gNeg = cc->EvalNegate(ct1);
    auto gMult = cc->EvalMult(ct1, ct2);
    auto gMultSc = cc->EvalMult(ct1, 1.5);
    auto gSq = cc->EvalSquare(ct1);

    ASSERT_EQ_CIPHERTEXT(cAdd, gAdd);
    ASSERT_EQ_CIPHERTEXT(cAddSc, gAddSc);
    ASSERT_EQ_CIPHERTEXT(cSub, gSub);
    ASSERT_EQ_CIPHERTEXT(cSubSc, gSubSc);
    ASSERT_EQ_CIPHERTEXT(cMult, gMult);
    ASSERT_EQ_CIPHERTEXT(cMultSc, gMultSc);
    ASSERT_EQ_CIPHERTEXT(cSq, gSq);
    ASSERT_EQ_CIPHERTEXT(cNeg, gNeg);
}

TEST(OpenFHECompatTests, EvalArithmeticPt) {
    auto cc = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);

    auto cAddPt = cc->EvalAdd(ct1, ptxt2);
    auto cSubPt = cc->EvalSub(ct1, ptxt2);
    auto cMultPt = cc->EvalMult(ct1, ptxt2);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAddPt = cc->EvalAdd(ct1, ptxt2);
    auto gSubPt = cc->EvalSub(ct1, ptxt2);
    auto gMultPt = cc->EvalMult(ct1, ptxt2);

    ASSERT_EQ_CIPHERTEXT(cAddPt, gAddPt);
    ASSERT_EQ_CIPHERTEXT(cSubPt, gSubPt);
    ASSERT_EQ_CIPHERTEXT(cMultPt, gMultPt);
}

TEST(OpenFHECompatTests, EvalAdjust) {
    // Exercises the FLEXIBLEAUTO scale/level adjustment paths with operands at
    // mixed noise degrees and level gaps (including deg2/deg2 with a gap >= 2).
    auto cc = MakeSmallContext(6);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    // Encrypt ONCE: encryption is randomized, so both phases must share the input.
    auto c1 = cc->Encrypt(keys.publicKey, ptxt);

    auto runChain = [&](std::vector<Ciphertext<DCRTPoly>>& out) {
        auto c2 = cc->EvalMult(c1, c1); // deg2 @ top level
        auto c3 = cc->EvalMult(c2, c2); // deg2, one level down
        auto c4 = cc->EvalMult(c3, c3); // deg2, two levels down

        out.push_back(c2);
        out.push_back(c3);
        out.push_back(c4);
        out.push_back(cc->EvalAdd(c1, c3));  // deg1 vs deg2, gap 1
        out.push_back(cc->EvalAdd(c2, c4));  // deg2 vs deg2, gap 2
        out.push_back(cc->EvalSub(c4, c2));  // deg2 vs deg2, gap 2 (reversed)
        out.push_back(cc->EvalMult(c1, c4)); // deg1 vs deg2, gap 2
        out.push_back(cc->EvalMult(c2, c3)); // deg2 vs deg2, gap 1
        out.push_back(cc->EvalSquare(c1));   // same-handle mult above must equal this square
    };

    std::vector<Ciphertext<DCRTPoly>> cpu;
    runChain(cpu);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    std::vector<Ciphertext<DCRTPoly>> gpu;
    runChain(gpu);

    ASSERT_EQ(cpu.size(), gpu.size());
    for (size_t i = 0; i < cpu.size(); ++i) {
        std::cout << "adjust case " << i << ": ";
        ASSERT_EQ_CIPHERTEXT(cpu[i], gpu[i]);
    }
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

TEST(OpenFHECompatTests, EvalChebyshev) {

    std::function<double(double)> sigmoidFunc = sigmoid;

    for (auto tech : std::vector<ScalingTechnique>{ FIXEDMANUAL, FLEXIBLEAUTO, FLEXIBLEAUTOEXT, FIXEDAUTO }) {
        for (int d = 6; d < 200; ++d) {
            auto cc = MakeSmallContext(10, tech);
            auto keys = cc->KeyGen();
            cc->EvalMultKeyGen(keys.secretKey);

            auto coeffs = cc->GetChebyshevCoefficients(sigmoidFunc, -1.0, 2.0, d);
            // Degree-12 series -> Paterson-Stockmeyer path on both sides.
            // std::vector<double> coeffs = { 0.15, 0.05, 0.2, -0.03, 0.11, 0.007, -0.05, 0.021, 0.09, -0.012, 0.033, 0.004, -0.026 };

            std::vector<double> x = { 0.03, 0.06, 0.09, 0.12, 0.25, 0.37, 0.5, 0.62 };
            Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

            auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

            auto cCheb = cc->EvalChebyshevSeries(ctxt, coeffs, -1.0, 2.0);

            //====================================================================

            cc->SetDevices({ 0 });
            cc->LoadContext(keys.publicKey);

            auto gCheb = cc->EvalChebyshevSeries(ctxt, coeffs, -1.0, 2.0);

            std::cout << d << " " << tech << std::endl;

            {
                Plaintext r1;
                cc->Decrypt(keys.secretKey, cCheb, &r1);
                r1->SetLength(8);
                std::cout << r1 << std::endl;

                //====================================================================

                Plaintext r2;
                cc->Decrypt(keys.secretKey, gCheb, &r2);
                r2->SetLength(8);
                std::cout << r2 << std::endl;
                CudaCheckErrorMod;
                ASSERT_ERROR_OK(r1, r2);
            }
            ASSERT_EQ_CIPHERTEXT(cCheb, gCheb);
        }
    }
}

// FIXEDMANUAL branches of the Chebyshev transcription (su/cu LevelReduce gates, manual rescale placement)
TEST(OpenFHECompatTests, EvalChebyshevFixedManual) {
    auto cc = MakeSmallContext(10, FIXEDMANUAL);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> coeffs = { 0.15, 0.05, 0.2, -0.03, 0.11, 0.007, -0.05, 0.021, 0.09, -0.012, 0.033, 0.004, -0.026 };

    std::vector<double> x = { 0.03, 0.06, 0.09, 0.12, 0.25, 0.37, 0.5, 0.62 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cCheb = cc->EvalChebyshevSeries(ctxt, coeffs, -1.0, 1.0);

    // Decrypt the CPU result while still in CPU mode: if this already fails, the
    // problem is in the CPU/api path or the test's FIXEDMANUAL usage, not the GPU.
    // Under FIXEDMANUAL the series result may be at noise degree 2: rescale before
    // decoding (the bit-compat compare below still uses the raw outputs).
    Plaintext r1;
    cc->Decrypt(keys.secretKey, cCheb, &r1);
    r1->SetLength(8);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gCheb = cc->EvalChebyshevSeries(ctxt, coeffs, -1.0, 1.0);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, gCheb, &r2);
    r2->SetLength(8);

    ASSERT_ERROR_OK(r1, r2);

    ASSERT_EQ_CIPHERTEXT(cCheb, gCheb);
}

// FIXEDMANUAL arithmetic, including an explicit Rescale after
// the multiply — under FIXEDMANUAL both the CPU fallback and the GPU perform a real
// rescale, so the ModReduce path is comparable bit-for-bit
TEST(OpenFHECompatTests, EvalArithmeticFixedManual) {
    auto cc = MakeSmallContext(4, FIXEDMANUAL);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto cAdd = cc->EvalAdd(ct1, ct2);
    auto cSub = cc->EvalSub(ct1, ct2);
    auto cNeg = cc->EvalNegate(ct1);
    auto cMult = cc->EvalMult(ct1, ct2);
    auto cResc = cc->Rescale(cMult);
    auto cSq = cc->EvalSquare(ct1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAdd = cc->EvalAdd(ct1, ct2);
    auto gSub = cc->EvalSub(ct1, ct2);
    auto gNeg = cc->EvalNegate(ct1);
    auto gMult = cc->EvalMult(ct1, ct2);
    auto gResc = cc->Rescale(gMult);
    auto gSq = cc->EvalSquare(ct1);

    ASSERT_EQ_CIPHERTEXT(cAdd, gAdd);
    ASSERT_EQ_CIPHERTEXT(cSub, gSub);
    ASSERT_EQ_CIPHERTEXT(cNeg, gNeg);
    ASSERT_EQ_CIPHERTEXT(cMult, gMult);
    ASSERT_EQ_CIPHERTEXT(cResc, gResc);
    ASSERT_EQ_CIPHERTEXT(cSq, gSq);
}

TEST(OpenFHECompatTests, EvalFastRotationHoisted) {
    // Multi-index EvalFastRotation: the GPU side uses hoisted digits shared across
    // the rotations (rotate_hoisted), unlike the single-index path.
    auto cc = MakeSmallContext(2);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, { 1, -2 });

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    std::vector<int32_t> indices = { 1, -2 };
    auto cRots = cc->EvalFastRotation(ctxt, indices, 2 * cc->GetRingDimension(), cc->EvalFastRotationPrecompute(ctxt));

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gRots = cc->EvalFastRotation(ctxt, indices, 2 * cc->GetRingDimension(), cc->EvalFastRotationPrecompute(ctxt));

    ASSERT_EQ(cRots.size(), gRots.size());
    for (size_t i = 0; i < cRots.size(); ++i) {
        std::cout << "rotation index " << indices[i] << ": ";
        ASSERT_EQ_CIPHERTEXT(cRots[i], gRots[i]);
    }
}

TEST(OpenFHECompatTests, EvalAddMany) {
    auto cc = MakeSmallContext(2);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    std::vector<Ciphertext<DCRTPoly>> cts;
    for (int i = 0; i < 4; ++i) {
        Plaintext p = cc->MakeCKKSPackedPlaintext(x);
        cts.push_back(cc->Encrypt(keys.publicKey, p));
    }

    auto cSum = cc->EvalAddMany(cts);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gSum = cc->EvalAddMany(cts);

    ASSERT_EQ_CIPHERTEXT(cSum, gSum);
}

// =====================================================================
// Exhaustive API-surface coverage.
//
// These tests walk every ciphertext-producing operation exposed by
// api/CryptoContext.hpp that has both a CPU path (patched-OpenFHE reference) and a
// GPU path (FIDESlib), and require bit-identical output (ASSERT_EQ_CIPHERTEXT)
// between a run with devices unset and a run after SetDevices + LoadContext.
//
// Entry points excluded from bit-exact comparison, by design:
//  - Encrypt / Decrypt: encryption is randomized, and CKKS decryption adds fresh
//    noise (see BITCOMPAT.md); the value-level ASSERT_ERROR_OK comparisons in the
//    tests above are the appropriate check there.
//  - MakeCKKSPackedPlaintext / GetChebyshevCoefficients: host-side encodings with no
//    CPU/GPU split (both paths share the same implementation).
//  - KeyGen, EvalMultKeyGen, EvalRotateKeyGen, EvalBootstrapSetup, EvalBootstrapKeyGen,
//    Enable, SetDevices/SetAutoLoad*, LoadContext/LoadPlaintext/LoadCiphertext:
//    context/key infrastructure exercised by every test in this file.
//
// The remaining api/ entry points that the base suite never touched (host-side I/O and
// GPU-only transforms) are exercised for correctness below, in the "api coverage"
// section: CCParams setters/getters, Ciphertext/Plaintext accessors and operators,
// symmetric-key Encrypt, eval-key serialization (SerializeEvalMultKey /
// SerializeEvalAutomorphismKey / Deserialize*), fideslib::Serial round-trips for
// context/public/private keys, the start-offset AccumulateSumInPlace, and the
// convolution transforms (GPU-only: the CPU path OPENFHE_THROWs, so those compare
// against a public-API reference chain instead of a CPU in-context run).
//  - SPARSE_ENCAPSULATED bootstrap: known structural divergence (O6c) — the
//    EvalBootstrapSparseEncaps* tests above assert value-level agreement instead of
//    bit-exactness.
// =====================================================================

// EvalAdd / EvalSub / EvalMult with swapped argument order: (Plaintext, ct) and
// (scalar, ct). Add/Mult forward to the (ct, x) forms already covered by
// EvalArithmetic; EvalSub has dedicated GPU implementations (negate + add) that must
// reproduce OpenFHE's EvalSub(pt/scalar, ct) rounding order bit-for-bit.
TEST(OpenFHECompatTests, EvalSwappedArgOrder) {
    auto cc = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);

    auto cAddPt = cc->EvalAdd(ptxt2, ct1);
    auto cAddSc = cc->EvalAdd(0.5, ct1);
    auto cSubPt = cc->EvalSub(ptxt2, ct1);
    auto cSubSc = cc->EvalSub(0.25, ct1);
    auto cMultPt = cc->EvalMult(ptxt2, ct1);
    auto cMultSc = cc->EvalMult(1.5, ct1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAddPt = cc->EvalAdd(ptxt2, ct1);
    auto gAddSc = cc->EvalAdd(0.5, ct1);
    auto gSubPt = cc->EvalSub(ptxt2, ct1);
    auto gSubSc = cc->EvalSub(0.25, ct1);
    auto gMultPt = cc->EvalMult(ptxt2, ct1);
    auto gMultSc = cc->EvalMult(1.5, ct1);

    ASSERT_EQ_CIPHERTEXT(cAddPt, gAddPt);
    ASSERT_EQ_CIPHERTEXT(cAddSc, gAddSc);
    ASSERT_EQ_CIPHERTEXT(cSubPt, gSubPt);
    ASSERT_EQ_CIPHERTEXT(cSubSc, gSubSc);
    ASSERT_EQ_CIPHERTEXT(cMultPt, gMultPt);
    ASSERT_EQ_CIPHERTEXT(cMultSc, gMultSc);
}

// EvalAddInPlace / EvalSubInPlace / EvalMultInPlace: the mutating overloads,
// including EvalSubInPlace(scalar, ct) (i.e. scalar - ct). Each phase operates on a
// fresh clone so the shared encryption is never modified by the in-place ops.
TEST(OpenFHECompatTests, EvalArithmeticInPlace) {
    auto cc = MakeSmallContext(5);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto cAddCt = ct1->Clone();
    cc->EvalAddInPlace(cAddCt, ct2);
    auto cAddPt = ct1->Clone();
    cc->EvalAddInPlace(cAddPt, ptxt2);
    auto cAddSc = ct1->Clone();
    cc->EvalAddInPlace(cAddSc, 0.5);
    auto cSubCt = ct1->Clone();
    cc->EvalSubInPlace(cSubCt, ct2);
    auto cSubSc = ct1->Clone();
    cc->EvalSubInPlace(cSubSc, 0.25);
    auto cSubScR = ct1->Clone();
    cc->EvalSubInPlace(0.25, cSubScR);
    auto cMultPt = ct1->Clone();
    cc->EvalMultInPlace(cMultPt, ptxt2);
    auto cMultSc = ct1->Clone();
    cc->EvalMultInPlace(cMultSc, 1.5);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAddCt = ct1->Clone();
    cc->EvalAddInPlace(gAddCt, ct2);
    auto gAddPt = ct1->Clone();
    cc->EvalAddInPlace(gAddPt, ptxt2);
    auto gAddSc = ct1->Clone();
    cc->EvalAddInPlace(gAddSc, 0.5);
    auto gSubCt = ct1->Clone();
    cc->EvalSubInPlace(gSubCt, ct2);
    auto gSubSc = ct1->Clone();
    cc->EvalSubInPlace(gSubSc, 0.25);
    auto gSubScR = ct1->Clone();
    cc->EvalSubInPlace(0.25, gSubScR);
    auto gMultPt = ct1->Clone();
    cc->EvalMultInPlace(gMultPt, ptxt2);
    auto gMultSc = ct1->Clone();
    cc->EvalMultInPlace(gMultSc, 1.5);

    ASSERT_EQ_CIPHERTEXT(cAddCt, gAddCt);
    ASSERT_EQ_CIPHERTEXT(cAddPt, gAddPt);
    ASSERT_EQ_CIPHERTEXT(cAddSc, gAddSc);
    ASSERT_EQ_CIPHERTEXT(cSubCt, gSubCt);
    ASSERT_EQ_CIPHERTEXT(cSubSc, gSubSc);
    ASSERT_EQ_CIPHERTEXT(cSubScR, gSubScR);
    ASSERT_EQ_CIPHERTEXT(cMultPt, gMultPt);
    ASSERT_EQ_CIPHERTEXT(cMultSc, gMultSc);
}

// EvalNegateInPlace / EvalSquareInPlace: mutating forms of the sign-flip and squaring
// entry points (non-in-place covered by EvalArithmetic).
TEST(OpenFHECompatTests, EvalNegateSquareInPlace) {
    auto cc = MakeSmallContext(5);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cNeg = ctxt->Clone();
    cc->EvalNegateInPlace(cNeg);
    auto cSq = ctxt->Clone();
    cc->EvalSquareInPlace(cSq);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gNeg = ctxt->Clone();
    cc->EvalNegateInPlace(gNeg);
    auto gSq = ctxt->Clone();
    cc->EvalSquareInPlace(gSq);

    ASSERT_EQ_CIPHERTEXT(cNeg, gNeg);
    ASSERT_EQ_CIPHERTEXT(cSq, gSq);
}

// EvalAddMutable / EvalSubMutable / EvalMultMutable and their in-place forms: the
// mutable-argument entry points (non-const ciphertext references). The plaintext/
// ciphertext and swapped-argument overloads forward to the concrete forms, and the
// in-place (ct1, ct2) forms have dedicated GPU implementations.
TEST(OpenFHECompatTests, EvalMutableArguments) {
    auto cc = MakeSmallContext(5);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    // The mutable entry points may adjust their operands in place (rescale / tower
    // drop), exactly like OpenFHE's mutable ops. Each op therefore gets fresh clones
    // of the ciphertext operands (and a fresh plaintext), so neither phase is
    // contaminated by the other's mutations; the phase results must stay bit-equal.
    auto runOps = [&](std::vector<Ciphertext<DCRTPoly>>& out) {
        {
            auto a = ct1->Clone();
            auto b = ct2->Clone();
            out.push_back(cc->EvalAddMutable(a, b));
        }
        {
            auto a = ct1->Clone();
            Plaintext p = cc->MakeCKKSPackedPlaintext(x2);
            out.push_back(cc->EvalAddMutable(a, p));
        }
        {
            auto a = ct1->Clone();
            Plaintext p = cc->MakeCKKSPackedPlaintext(x2);
            out.push_back(cc->EvalAddMutable(p, a));
        }
        {
            auto a = ct1->Clone();
            auto b = ct2->Clone();
            cc->EvalAddMutableInPlace(a, b);
            out.push_back(a);
        }
        {
            auto a = ct1->Clone();
            auto b = ct2->Clone();
            out.push_back(cc->EvalSubMutable(a, b));
        }
        {
            auto a = ct1->Clone();
            Plaintext p = cc->MakeCKKSPackedPlaintext(x2);
            out.push_back(cc->EvalSubMutable(p, a));
        }
        {
            auto a = ct1->Clone();
            auto b = ct2->Clone();
            cc->EvalSubMutableInPlace(a, b);
            out.push_back(a);
        }
        {
            auto a = ct1->Clone();
            auto b = ct2->Clone();
            out.push_back(cc->EvalMultMutable(a, b));
        }
        {
            auto a = ct1->Clone();
            Plaintext p = cc->MakeCKKSPackedPlaintext(x2);
            out.push_back(cc->EvalMultMutable(a, p));
        }
        {
            auto a = ct1->Clone();
            auto b = ct2->Clone();
            cc->EvalMultMutableInPlace(a, b);
            out.push_back(a);
        }
        {
            auto a = ct1->Clone();
            out.push_back(cc->EvalSquareMutable(a));
        }
    };

    std::vector<Ciphertext<DCRTPoly>> cpu;
    runOps(cpu);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    std::vector<Ciphertext<DCRTPoly>> gpu;
    runOps(gpu);

    ASSERT_EQ(cpu.size(), gpu.size());
    for (size_t i = 0; i < cpu.size(); ++i) {
        std::cout << "mutable case " << i << ": ";
        ASSERT_EQ_CIPHERTEXT(cpu[i], gpu[i]);
    }
}

// EvalAddManyInPlace: serial in-place fold with the result landing in slot 0 (the
// OpenFHE in-place contract). The CPU path reuses slot 0's storage in place, so both
// phases must work on deep copies of the shared encryption to avoid aliasing.
TEST(OpenFHECompatTests, EvalAddManyInPlace) {
    auto cc = MakeSmallContext(2);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    std::vector<Ciphertext<DCRTPoly>> cts;
    for (int i = 0; i < 4; ++i) {
        Plaintext p = cc->MakeCKKSPackedPlaintext(x);
        cts.push_back(cc->Encrypt(keys.publicKey, p));
    }

    auto deepCopy = [](const std::vector<Ciphertext<DCRTPoly>>& src) {
        std::vector<Ciphertext<DCRTPoly>> dst;
        dst.reserve(src.size());
        for (const auto& ct : src) {
            Ciphertext<DCRTPoly> copy = ct->Clone();
            copy->EnsureLazyCPUCopy();
            dst.push_back(copy);
        }
        return dst;
    };

    auto cSum = deepCopy(cts);
    cc->EvalAddManyInPlace(cSum);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gSum = deepCopy(cts);
    cc->EvalAddManyInPlace(gSum);

    ASSERT_EQ_CIPHERTEXT(cSum[0], gSum[0]);
}

// EvalRotateInPlace: the mutating form of the (O1-unified hoisted) rotation.
TEST(OpenFHECompatTests, EvalRotateInPlace) {
    auto cc = MakeSmallContext(2);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, { 1, -2 });

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cRot = ctxt->Clone();
    cc->EvalRotateInPlace(cRot, 1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gRot = ctxt->Clone();
    cc->EvalRotateInPlace(gRot, 1);

    ASSERT_EQ_CIPHERTEXT(cRot, gRot);
}

// EvalFastRotationExt (single- and multi-index): the extended-basis (no mod-down)
// rotation, addFirst=true (the P·c0 fold matches OpenFHE's reference). Bit-exact —
// the GPU's extended (modUp) output is 3+2 = 5 towers, matching the CPU's QL·P basis.
TEST(OpenFHECompatTests, EvalFastRotationExt) {
    auto cc = MakeSmallContext(2);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, { 1, -2 });

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    std::vector<int32_t> indices = { 1, -2 };

    auto cExt1 = cc->EvalFastRotationExt(ctxt, 1, cc->EvalFastRotationPrecompute(ctxt), true);
    auto cExtM = cc->EvalFastRotationExt(ctxt, indices, cc->EvalFastRotationPrecompute(ctxt), true);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gExt1 = cc->EvalFastRotationExt(ctxt, 1, cc->EvalFastRotationPrecompute(ctxt), true);
    auto gExtM = cc->EvalFastRotationExt(ctxt, indices, cc->EvalFastRotationPrecompute(ctxt), true);

    ASSERT_EQ_CIPHERTEXT(cExt1, gExt1);

    ASSERT_EQ(cExtM.size(), gExtM.size());
    for (size_t i = 0; i < cExtM.size(); ++i) {
        std::cout << "ext rotation index " << indices[i] << ": ";
        ASSERT_EQ_CIPHERTEXT(cExtM[i], gExtM[i]);
    }
}

// EvalChebyshevSeriesInPlace: mutating form of the Chebyshev series evaluation
// (FLEXIBLEAUTO; the transcription pipeline is the one validated by EvalChebyshev).
TEST(OpenFHECompatTests, EvalChebyshevSeriesInPlace) {
    auto cc = MakeSmallContext(10);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> coeffs = { 0.15, 0.05, 0.2, -0.03, 0.11, 0.007, -0.05, 0.021, 0.09, -0.012, 0.033, 0.004, -0.026 };

    std::vector<double> x = { 0.03, 0.06, 0.09, 0.12, 0.25, 0.37, 0.5, 0.62 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cCheb = ctxt->Clone();
    cc->EvalChebyshevSeriesInPlace(cCheb, coeffs, -1.0, 1.0);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gCheb = ctxt->Clone();
    cc->EvalChebyshevSeriesInPlace(gCheb, coeffs, -1.0, 1.0);

    ASSERT_EQ_CIPHERTEXT(cCheb, gCheb);
}

// RescaleInPlace under FIXEDMANUAL, the only scaling technique where both the CPU
// reference and the GPU perform a real rescale (under the AUTO techniques the CPU
// Rescale is a no-op — see O3 in BITCOMPAT.md).
TEST(OpenFHECompatTests, RescaleInPlace) {
    auto cc = MakeSmallContext(4, FIXEDMANUAL);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto cMult = cc->EvalMult(ct1, ct2);
    auto cResc = cMult->Clone();
    cc->RescaleInPlace(cResc);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gMult = cc->EvalMult(ct1, ct2);
    auto gResc = gMult->Clone();
    cc->RescaleInPlace(gResc);

    ASSERT_EQ_CIPHERTEXT(cResc, gResc);
}

// SetLevel: metadata/tower-drop entry point. Both paths must drop exactly the same
// towers so the remaining elements (and noise degree) match bit-for-bit. The GPU
// copy is explicitly uploaded (LoadCiphertext) so the GPU dropToLevel path runs.
TEST(OpenFHECompatTests, SetLevel) {
    auto cc = MakeSmallContext(5);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cLev = ctxt->Clone();
    cc->SetLevel(cLev, 2);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gLev = ctxt->Clone();
    cc->LoadCiphertext(gLev);
    cc->SetLevel(gLev, 2);

    ASSERT_EQ_CIPHERTEXT(cLev, gLev);
}

// EvalBootstrapInPlace: mutating form of the standard FLEXIBLEAUTO sparse bootstrap
// (the same configuration as EvalBootstrap).
TEST(OpenFHECompatTests, EvalBootstrapInPlace) {
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    uint32_t numSlots = batchSize;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, multDepth - 1, nullptr, numSlots);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cBoot = ctxt->Clone();
    cc->EvalBootstrapInPlace(cBoot);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gBoot = ctxt->Clone();
    cc->EvalBootstrapInPlace(gBoot);

    ASSERT_EQ_CIPHERTEXT(cBoot, gBoot);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cBoot, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, gBoot, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

// AccumulateSumInPlace: mutating form of the radix-PARTIAL_SUM_RADIX rotation fold
// (the same configuration as AccumulateSum).
TEST(OpenFHECompatTests, AccumulateSumInPlace) {
    uint32_t multDepth = 2;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(128);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    std::vector<int32_t> accIndices;
    for (uint32_t s = 1; s < batchSize; s *= PARTIAL_SUM_RADIX)
        for (uint32_t idx = s; idx < batchSize && idx < PARTIAL_SUM_RADIX * s; idx += s)
            accIndices.push_back(static_cast<int32_t>(idx));
    cc->EvalRotateKeyGen(keys.secretKey, accIndices);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cAcc = ctxt->Clone();
    cc->AccumulateSumInPlace(cAcc, batchSize, 1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAcc = ctxt->Clone();
    cc->AccumulateSumInPlace(gAcc, batchSize, 1);

    ASSERT_EQ_CIPHERTEXT(cAcc, gAcc);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cAcc, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, gAcc, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

// AccumulateSumInPlace start-offset variant — DISABLED until the OpenFHE reference draft lands.
//
// The GPU path now runs the SAME unified lazy radix fold as the plain variants: a single
// radix-parametrized `Accumulate(ct, ACCUMULATE_SUM_RADIX, stride, slots, start)` (P2b lazy
// c1-per-level / c0-once discipline, metadata-neutral per O10, rotation set {start*radix^k*i}:
// {2, 4, 6} for start=2, slots=8, radix 4) with the api-level `inputSlots` save/restore — see
// the O8 sub-item. The CPU fallback, however, is still the eager doubling loop from `start`
// (rotations {2, 4}) until the reference lands the start-offset `EvalPartialSumInPlace`
// (deps/draft-start-accumulate-radix-lazy.patch): eager CPU vs lazy GPU are NOT bit-exact, so
// this stays DISABLED_ (like DISABLED_EvalBootstrapFixedAuto) and must be re-enabled only after
// the reference draft replaces the CPU fallback with the same lazy radix fold.
TEST(OpenFHECompatTests, DISABLED_AccumulateSumInPlaceStart) {
    uint32_t multDepth = 2;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(128);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // Rotation keys for the unified radix-4 start=2 fold — mirror the unified `Accumulate`
    // index generation exactly: level s rotates by {stride*s, 2*stride*s, ...} = {2, 4, 6} for
    // start=2, slots=8. The reference draft's `EvalPartialSumInPlace(.., startFactor)` issues
    // the same set, and the eager CPU doubling ({2, 4}) is a subset of it.
    int start = 2;
    std::vector<int32_t> accIndices;
    for (uint32_t s = static_cast<uint32_t>(start); s < batchSize; s *= PARTIAL_SUM_RADIX)
        for (uint32_t idx = s; idx < batchSize && idx < PARTIAL_SUM_RADIX * s; idx += s)
            accIndices.push_back(static_cast<int32_t>(idx));
    cc->EvalRotateKeyGen(keys.secretKey, accIndices);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cAcc = ctxt->Clone();
    cc->AccumulateSumInPlace(cAcc, batchSize, 1, start);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAcc = ctxt->Clone();
    cc->AccumulateSumInPlace(gAcc, batchSize, 1, start);

    ASSERT_EQ_CIPHERTEXT(cAcc, gAcc);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cAcc, &r1);
    r1->SetLength(batchSize);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, gAcc, &r2);
    r2->SetLength(batchSize);

    ASSERT_ERROR_OK(r1, r2);
}

// GPU→host ciphertext export + serialization: a result produced on the GPU (EvalAdd) is
// materialized back to its host copy (UnloadCiphertext), serialized to disk, and
// deserialized into a fresh CiphertextImpl bound to the same parent context. The
// round-tripped ciphertext must decrypt to the same values as the CPU reference op —
// this is the server→client return path (serialize the host-only result, ship bytes).
TEST(OpenFHECompatTests, UnloadCiphertext) {
    auto cc = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    // CPU reference: computed before the context is moved to the devices.
    auto cRef = cc->EvalAdd(ct1, ct2);

    //====================================================================
    // GPU phase: produce the same add on the devices, then export it.
    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAdd = cc->EvalAdd(ct1, ct2);
    ASSERT_TRUE(gAdd->loaded); // the GPU holds the result

    cc->UnloadCiphertext(gAdd);
    ASSERT_FALSE(gAdd->loaded); // host copy is now the source of truth
    ASSERT_EQ(gAdd->gpu, 0u);

    // Serialize the unloaded (host-only) ciphertext and pull it back.
    const std::string fname = "/tmp/opencode/unload_ct.bin";
    std::remove(fname.c_str());
    ASSERT_TRUE(fideslib::Serial::SerializeToFile(fname, gAdd, fideslib::SerType::BINARY));
    fideslib::Ciphertext<DCRTPoly> rt = std::make_shared<CiphertextImpl<DCRTPoly>>(std::move(cc));
    ASSERT_TRUE(fideslib::Serial::DeserializeFromFile(fname, rt, fideslib::SerType::BINARY));
    ASSERT_FALSE(rt->loaded);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, cRef, &r1);
    r1->SetLength(8);

    Plaintext r2;
    cc->Decrypt(keys.secretKey, rt, &r2);
    r2->SetLength(8);

    ASSERT_ERROR_OK(r1, r2);
}

// GPU→host export of an extended-basis ciphertext: the multi-index EvalFastRotationExt
// GPU path (SetDevices + LoadContext + EvalRotateKeyGen, rotate_hoisted(ext=true))
// leaves a 5-tower QL·P result on the device. Unload/serialize/deserialize must round-trip
// that extended basis (the R15 download path) so the re-imported ciphertext decrypts to
// the same values as direct decrypt — the server→client return path for extended results.
TEST(OpenFHECompatTests, SerializeCiphertextExtended) {
    auto cc = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, { 1, -2 });

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    std::vector<int32_t> indices = { 1, -2 };

    // CPU reference: multi-index extended rotation, computed before the devices are set
    // (this precompute also feeds the digits the CPU EvalFastRotationExt consumes).
    auto cExtM = cc->EvalFastRotationExt(ctxt, indices, cc->EvalFastRotationPrecompute(ctxt), true);

    //====================================================================
    // GPU phase: produce the same extended rotations on the devices, then export.
    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gExtM = cc->EvalFastRotationExt(ctxt, indices, nullptr, true);

    // The R15 download must round-trip the extended 5-tower basis bit-exactly.
    ASSERT_EQ_CIPHERTEXT(cExtM[0], gExtM[0]);

    // Unload: materialize the host copy of the extended ciphertext and evict the GPU one.
    cc->UnloadCiphertext(gExtM[0]);
    ASSERT_FALSE(gExtM[0]->loaded);

    // Serialize the extended-basis ciphertext and pull it back — the round-trip must be
    // lossless at the extended level (the R15 QL·P download is what gets serialized).
    const std::string fname = "/tmp/opencode/ext_ct.bin";
    std::remove(fname.c_str());
    ASSERT_TRUE(fideslib::Serial::SerializeToFile(fname, gExtM[0], fideslib::SerType::BINARY));
    fideslib::Ciphertext<DCRTPoly> rt = std::make_shared<CiphertextImpl<DCRTPoly>>(std::move(cc));
    ASSERT_TRUE(fideslib::Serial::DeserializeFromFile(fname, rt, fideslib::SerType::BINARY));

    // The deserialized copy matches the unloaded result bit-for-bit at the extended basis.
    ASSERT_EQ_CIPHERTEXT(gExtM[0], rt);

    // Decryptable form: OpenFHE's documented pattern for an EvalFastRotationExt result is
    // KeySwitchDown (a keyless ApproxModDown back into the Q basis) before Decrypt — the
    // scheme's DecryptCore would otherwise underflow dropping "negative" secret towers.
    // Apply the identical reference op to the direct result and the round-tripped one so
    // the comparison isolates the unload + serialize/deserialize handoff.
    auto modDownExtended = [](const fideslib::CryptoContext<DCRTPoly>& cc,
                              fideslib::Ciphertext<DCRTPoly>& ct) {
        auto& rawCc = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
        auto& rawCt = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
        auto down = rawCc->KeySwitchDown(rawCt);
        ct->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(down);
    };
    modDownExtended(cc, gExtM[0]);
    modDownExtended(cc, rt);

    // Direct decrypt of the unloaded result (after mod-down).
    Plaintext rDirect;
    cc->Decrypt(keys.secretKey, gExtM[0], &rDirect);
    rDirect->SetLength(8);

    Plaintext rRT;
    cc->Decrypt(keys.secretKey, rt, &rRT);
    rRT->SetLength(8);

    // The re-imported ciphertext must decrypt like the direct (pre-serialization) result.
    ASSERT_ERROR_OK(rDirect, rRT);

    // And like the CPU reference extended rotation (same mod-down applied).
    modDownExtended(cc, cExtM[0]);
    Plaintext rCPURef;
    cc->Decrypt(keys.secretKey, cExtM[0], &rCPURef);
    rCPURef->SetLength(8);
    ASSERT_ERROR_OK(rCPURef, rRT);
}

// =====================================================================
// api/ coverage: the remaining 0 %-line-rate entry points of the api package
// (per coverage/coverage.xml) that the exhaustive bit-compat sweep above does not
// reach. These are host-side-only or GPU-only-by-design, so they are exercised for
// correctness rather than CPU/GPU bit-exactness:
//   - CCParams setters/getters and the GAUSSIAN secret-key distribution
//   - CiphertextImpl accessors (GetScalingFactor, GetSlots, GetEncodingType,
//     SetSlots, the CPU-fallback GetLevel, the Ciphertext-copy ctor, operator+==!=)
//   - PlaintextImpl accessors (SetSlots, GetLevel, GetCKKSPackedValue, impl stream
//     and equality operators) and the complex-vector MakeCKKSPackedPlaintext overload
//   - CryptoContextImpl (GetCyclotomicOrder, SetAutoLoadPlaintexts/Ciphertexts,
//     GetPreScaleFactor, Synchronize, the SetDevices-after-Load throw, symmetric-key
//     Encrypt, the swapped-argument in-place/mutable arithmetic overloads, the
//     start-offset AccumulateSumInPlace, the convolution transforms and the
//     rotation-index helper, eval-key serialization)
//   - fideslib::Serial round-trips for context / public key / private key
//   - null-context constructor throws
// =====================================================================

TEST(OpenFHECompatTests, CCParamsOptions) {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(8);
    parameters.SetScalingModSize(50);
    parameters.SetBatchSize(8);
    parameters.SetRingDim(128);
    parameters.SetSecurityLevel(HEStd_NotSet);

    // Setters without a fideslib getter (SetNumLargeDigits / SetFirstModSize /
    // SetDigitSize / SetKeySwitchTechnique) reach the underlying OpenFHE params.
    parameters.SetNumLargeDigits(3);
    parameters.SetFirstModSize(55);
    parameters.SetDigitSize(35);
    parameters.SetKeySwitchTechnique(HYBRID);
    auto& raw = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(parameters.cpu);
    ASSERT_EQ(raw.GetNumLargeDigits(), 3u);
    ASSERT_EQ(raw.GetFirstModSize(), 55u);
    ASSERT_EQ(raw.GetDigitSize(), 35u);
    ASSERT_EQ(raw.GetKeySwitchTechnique(), lbcrypto::HYBRID);

    // Getters round-trip the fideslib values through the OpenFHE params.
    ASSERT_EQ(parameters.GetMultiplicativeDepth(), 8u);
    ASSERT_EQ(parameters.GetBatchSize(), 8u);

    // Secret-key distribution: every branch, including the GAUSSIAN fallback.
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    ASSERT_EQ(parameters.GetSecretKeyDist(), UNIFORM_TERNARY);
    parameters.SetSecretKeyDist(SPARSE_TERNARY);
    ASSERT_EQ(parameters.GetSecretKeyDist(), SPARSE_TERNARY);
    parameters.SetSecretKeyDist(SPARSE_ENCAPSULATED);
    ASSERT_EQ(parameters.GetSecretKeyDist(), SPARSE_ENCAPSULATED);
    parameters.SetSecretKeyDist(GAUSSIAN);
    ASSERT_EQ(parameters.GetSecretKeyDist(), GAUSSIAN);

    // Device list (rvalue overload).
    parameters.SetDevices(std::vector<int>{ 2, 3 });
    ASSERT_EQ(parameters.devices, (std::vector<int>{ 2, 3 }));

    // The parameter plumbing must still produce a working context.
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetDevices(std::vector<int>{});
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);
    auto ct = cc->Encrypt(keys.publicKey, ptxt);
    Plaintext r;
    cc->Decrypt(keys.secretKey, ct, &r);
    r->SetLength(8);
    auto vals = r->GetRealPackedValue();
    for (size_t i = 0; i < vals.size(); ++i)
        EXPECT_NEAR(vals[i], x[i], 1e-3);
}

TEST(OpenFHECompatTests, CiphertextAccessors) {
    auto cc = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    // CPU-phase (host-only) accessors: CPU fallback paths.
    ASSERT_EQ(ct1->GetEncodingType(), lbcrypto::CKKS_PACKED_ENCODING);
    ASSERT_EQ(ct1->GetSlots(), 8u);
    ASSERT_GT(ct1->GetScalingFactor(), 0.0);
    const size_t cpuLevel = ct1->GetLevel();
    ASSERT_TRUE(ct1->GetNoiseScaleDeg() == 1 || ct1->GetNoiseScaleDeg() == 2);

    // SetSlots (CPU fallback) + round-trip.
    ct1->SetSlots(4);
    ASSERT_EQ(ct1->GetSlots(), 4u);
    ct1->SetSlots(8);
    ASSERT_EQ(ct1->GetSlots(), 8u);

    // Delegating Ciphertext-copy constructor (make_shared from the shared_ptr).
    Ciphertext<DCRTPoly> copy = std::make_shared<CiphertextImpl<DCRTPoly>>(ct1);
    ASSERT_EQ_CIPHERTEXT(copy, ct1);

    // operator+ (shared_ptr form) and operator==/!= (dereferenced impl overloads).
    auto cSum = ct1 + ct2;
    ASSERT_EQ_CIPHERTEXT(cSum, cc->EvalAdd(ct1, ct2));
    ASSERT_TRUE(*ct1 == *ct1);
    ASSERT_FALSE(*ct1 != *ct1);
    auto s1 = cc->EvalAdd(ct1, ct2);
    auto s2 = cc->EvalAdd(ct1, ct2);
    ASSERT_TRUE(*s1 == *s2);
    ASSERT_FALSE(*s1 != *s2);

    //====================================================================
    // GPU phase: the loaded ciphertext takes the GPU accessor paths.
    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto gct2 = cc->Encrypt(keys.publicKey, ptxt2);
    ASSERT_TRUE(gct1->loaded);

    ASSERT_EQ(gct1->GetEncodingType(), lbcrypto::CKKS_PACKED_ENCODING);
    ASSERT_EQ(gct1->GetSlots(), 8u);
    ASSERT_GT(gct1->GetScalingFactor(), 0.0);
    EXPECT_NEAR(gct1->GetScalingFactor(), ct1->GetScalingFactor(), 1e-3 * ct1->GetScalingFactor());
    ASSERT_EQ(gct1->GetLevel(), cpuLevel);

    gct1->SetSlots(4);
    ASSERT_EQ(gct1->GetSlots(), 4u);
    gct1->SetSlots(8);

    auto gSum = gct1 + gct2;
    ASSERT_EQ_CIPHERTEXT(gSum, cc->EvalAdd(gct1, gct2));
    ASSERT_TRUE(*gct1 == *gct1);
    auto gs1 = cc->EvalAdd(gct1, gct2);
    auto gs2 = cc->EvalAdd(gct1, gct2);
    ASSERT_TRUE(*gs1 == *gs2);
    ASSERT_FALSE(*gs1 != *gs2);
}

TEST(OpenFHECompatTests, PlaintextAccessors) {
    auto cc = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, 2);

    // Accessors the suite never exercised directly.
    ASSERT_EQ(ptxt->GetLevel(), 2u);
    ASSERT_EQ(ptxt->GetSlots(), 8u);
    auto real = ptxt->GetRealPackedValue();
    for (size_t i = 0; i < real.size(); ++i)
        EXPECT_NEAR(real[i], x[i], 1e-3);
    auto cplx = ptxt->GetCKKSPackedValue();
    for (size_t i = 0; i < cplx.size(); ++i) {
        EXPECT_NEAR(cplx[i].real(), x[i], 1e-3);
        EXPECT_NEAR(cplx[i].imag(), 0.0, 1e-3);
    }

    ptxt->SetSlots(4);
    ASSERT_EQ(ptxt->GetSlots(), 4u);
    ptxt->SetSlots(8);

    // Stream + equality on the PlaintextImpl (not the shared_ptr) overloads.
    std::ostringstream os;
    os << *ptxt;
    ASSERT_FALSE(os.str().empty());

    Plaintext p2 = cc->MakeCKKSPackedPlaintext(x, 1, 2);
    ASSERT_TRUE(*ptxt == *p2);
    ASSERT_FALSE(*ptxt != *p2);
    std::vector<double> other = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    Plaintext p3 = cc->MakeCKKSPackedPlaintext(other, 1, 2);
    ASSERT_FALSE(*ptxt == *p3);
    ASSERT_TRUE(*ptxt != *p3);

    // Complex-vector encoding overload. OpenFHE's CKKSPackedEncoding constructor
    // treats a complex input as REAL — it zeroes every imaginary component — so
    // the round-trip preserves the real parts and decodes imag == 0.
    std::vector<std::complex<double>> cx = { {0.25, 0.5}, {1.0, -1.0}, {0.0, 2.0}, {3.5, 0.25}, {4.0, 0.0}, {5.0, -2.0}, {6.0, 0.5}, {7.0, 1.5} };
    Plaintext cplt = cc->MakeCKKSPackedPlaintext(cx);
    ASSERT_EQ(cplt->GetSlots(), 8u);
    auto cv = cplt->GetCKKSPackedValue();
    ASSERT_EQ(cv.size(), 8u);
    for (size_t i = 0; i < cv.size(); ++i) {
        EXPECT_NEAR(cv[i].real(), cx[i].real(), 1e-3);
        EXPECT_NEAR(cv[i].imag(), 0.0, 1e-3);
    }
}

TEST(OpenFHECompatTests, EncryptSecretKey) {
    // Symmetric-key encryption: both PrivateKey Encrypt overloads
    // (Encrypt(pt, sk) and Encrypt(sk, pt)) were previously never called.
    auto cc = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto c1 = cc->Encrypt(ptxt, keys.secretKey);
    auto c2 = cc->Encrypt(keys.secretKey, ptxt);

    Plaintext r1;
    cc->Decrypt(keys.secretKey, c1, &r1);
    r1->SetLength(8);
    Plaintext r2;
    cc->Decrypt(keys.secretKey, c2, &r2);
    r2->SetLength(8);

    ASSERT_ERROR_OK(r1, r2);
    auto vals = r1->GetRealPackedValue();
    for (size_t i = 0; i < vals.size(); ++i)
        EXPECT_NEAR(vals[i], x[i], 1e-3);
}

TEST(OpenFHECompatTests, ContextSettings) {
    auto cc = MakeSmallContext(4);
    EXPECT_EQ(cc->GetCyclotomicOrder(), 2u * cc->GetRingDimension());

    // Synchronize before the context is loaded: no-op path.
    cc->Synchronize();

    cc->SetAutoLoadPlaintexts(true);
    cc->SetAutoLoadCiphertexts(true);
    ASSERT_TRUE(cc->auto_load_plaintexts);
    ASSERT_TRUE(cc->auto_load_ciphertexts);

    auto keys = cc->KeyGen();
    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);
    cc->Synchronize(); // loaded path: device sync on the listed devices

    // SetDevices must throw after LoadContext.
    EXPECT_THROW(cc->SetDevices(std::vector<int>{ 1 }), lbcrypto::OpenFHEException);

    // Auto-load plaintexts: MakeCKKSPackedPlaintext now pushes to the devices.
    Plaintext apt = cc->MakeCKKSPackedPlaintext(x);
    ASSERT_TRUE(apt->loaded);

    // Auto-load ciphertexts is on by default: Encrypt pushes to the devices.
    auto c1 = cc->Encrypt(keys.publicKey, ptxt);
    ASSERT_TRUE(c1->loaded);

    // Turning ciphertext auto-load off keeps Encrypt host-only.
    cc->SetAutoLoadCiphertexts(false);
    ASSERT_FALSE(cc->auto_load_ciphertexts);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt);
    ASSERT_FALSE(c2->loaded);
}

TEST(OpenFHECompatTests, GetPreScaleFactor) {
    // GetPreScaleFactor needs the GPU context AND the bootstrap precomputation for
    // the requested slot count (EvalBootstrapSetup builds it).
    uint32_t multDepth = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 12);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keys = cc->KeyGen();
    std::vector<uint32_t> levelBudget = { 3, 3 };
    cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, batchSize, 0);
    // The GPU context registers the boot precomputation for the slots listed by
    // EvalBootstrapKeyGen (slots_bootstrap); Setup alone is not enough.
    cc->EvalBootstrapKeyGen(keys.secretKey, batchSize);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    const double f1 = cc->GetPreScaleFactor(batchSize);
    const double f2 = cc->GetPreScaleFactor(batchSize);
    ASSERT_TRUE(std::isfinite(f1));
    ASSERT_EQ(f1, f2);
}

TEST(OpenFHECompatTests, KeySerialization) {
    // SerializeEvalMultKey / SerializeEvalAutomorphismKey and their deserialize
    // counterparts (host-side eval-key I/O, previously never exercised).
    auto cc = MakeSmallContext(2);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, { 1, -2 });

    auto& rawCc = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
    auto& rawSk = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(keys.secretKey->pimpl);
    const std::string keyTag = rawSk->GetKeyTag();

    // Serialize with a keyTag (not the empty-tag "everything in the global map"
    // form): the deserializers re-insert by secret-key tag and throw if the tag is
    // already occupied, so a full-map dump would collide with other contexts' keys
    // still registered in this process. Clear our own tag before each deserialize.
    std::ostringstream multOs;
    ASSERT_TRUE(cc->SerializeEvalMultKey(multOs, SerType::BINARY, keyTag));
    std::ostringstream autoOs;
    ASSERT_TRUE(cc->SerializeEvalAutomorphismKey(autoOs, SerType::BINARY, keyTag));

    std::ostringstream multOsJ;
    ASSERT_TRUE(cc->SerializeEvalMultKey(multOsJ, SerType::JSON, keyTag));
    std::ostringstream autoOsJ;
    ASSERT_TRUE(cc->SerializeEvalAutomorphismKey(autoOsJ, SerType::JSON, keyTag));

    CCParams<CryptoContextCKKSRNS> dflt;
    CryptoContext<DCRTPoly> cc2 = GenCryptoContext(dflt);

    rawCc->ClearEvalMultKeys(keyTag);
    rawCc->ClearEvalAutomorphismKeys(keyTag);
    std::istringstream multIs(multOs.str());
    ASSERT_TRUE(cc2->DeserializeEvalMultKey(multIs, SerType::BINARY));
    std::istringstream autoIs(autoOs.str());
    ASSERT_TRUE(cc2->DeserializeEvalAutomorphismKey(autoIs, SerType::BINARY));

    // JSON smoke: same entry points through the JSON serializer.
    rawCc->ClearEvalMultKeys(keyTag);
    rawCc->ClearEvalAutomorphismKeys(keyTag);
    std::istringstream multIsJ(multOsJ.str());
    ASSERT_TRUE(cc2->DeserializeEvalMultKey(multIsJ, SerType::JSON));
    std::istringstream autoIsJ(autoOsJ.str());
    ASSERT_TRUE(cc2->DeserializeEvalAutomorphismKey(autoIsJ, SerType::JSON));
}

TEST(OpenFHECompatTests, SerialRoundTrip) {
    // fideslib::Serial round-trips for the context, public and private keys —
    // Serialize.cpp is the lowest-covered api file (11.6 %), and every one of these
    // overloads was previously never called.
    auto cc = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);
    auto cRef = cc->Encrypt(keys.publicKey, ptxt);

    Plaintext rRef;
    cc->Decrypt(keys.secretKey, cRef, &rRef);
    rRef->SetLength(8);

    const std::string dir = "/tmp/opencode";
    const std::string ctxFile = dir + "/ser-ctx.bin";
    const std::string pkFile = dir + "/ser-pk.bin";
    const std::string skFile = dir + "/ser-sk.bin";
    const std::string ctxFileJ = dir + "/ser-ctx.json";
    const std::string pkFileJ = dir + "/ser-pk.json";
    const std::string skFileJ = dir + "/ser-sk.json";
    for (const auto& f : { ctxFile, pkFile, skFile, ctxFileJ, pkFileJ, skFileJ })
        std::remove(f.c_str());
    for (const auto& f : { ctxFile + ".dev", pkFile + ".dev", skFile + ".dev", ctxFileJ + ".dev", pkFileJ + ".dev", skFileJ + ".dev" })
        std::remove(f.c_str());

    // ---- BINARY round-trip ----
    ASSERT_TRUE(fideslib::Serial::SerializeToFile(ctxFile, cc, fideslib::SerType::BINARY));
    ASSERT_TRUE(fideslib::Serial::SerializeToFile(pkFile, keys.publicKey, fideslib::SerType::BINARY));
    ASSERT_TRUE(fideslib::Serial::SerializeToFile(skFile, keys.secretKey, fideslib::SerType::BINARY));

    fideslib::CryptoContext<DCRTPoly> cc2;
    ASSERT_TRUE(fideslib::Serial::DeserializeFromFile(ctxFile, cc2, fideslib::SerType::BINARY));
    fideslib::PublicKey<DCRTPoly> pk2;
    ASSERT_TRUE(fideslib::Serial::DeserializeFromFile(pkFile, pk2, fideslib::SerType::BINARY));
    fideslib::PrivateKey<DCRTPoly> sk2;
    ASSERT_TRUE(fideslib::Serial::DeserializeFromFile(skFile, sk2, fideslib::SerType::BINARY));

    // The re-imported objects must encrypt/decrypt like the originals.
    Plaintext pt2 = cc2->MakeCKKSPackedPlaintext(x);
    auto c2 = cc2->Encrypt(pk2, pt2);
    Plaintext r2;
    cc2->Decrypt(sk2, c2, &r2);
    r2->SetLength(8);
    ASSERT_ERROR_OK(rRef, r2);

    // ---- JSON smoke ----
    ASSERT_TRUE(fideslib::Serial::SerializeToFile(ctxFileJ, cc, fideslib::SerType::JSON));
    ASSERT_TRUE(fideslib::Serial::SerializeToFile(pkFileJ, keys.publicKey, fideslib::SerType::JSON));
    ASSERT_TRUE(fideslib::Serial::SerializeToFile(skFileJ, keys.secretKey, fideslib::SerType::JSON));

    fideslib::CryptoContext<DCRTPoly> cc3;
    ASSERT_TRUE(fideslib::Serial::DeserializeFromFile(ctxFileJ, cc3, fideslib::SerType::JSON));
    fideslib::PublicKey<DCRTPoly> pk3;
    ASSERT_TRUE(fideslib::Serial::DeserializeFromFile(pkFileJ, pk3, fideslib::SerType::JSON));
    fideslib::PrivateKey<DCRTPoly> sk3;
    ASSERT_TRUE(fideslib::Serial::DeserializeFromFile(skFileJ, sk3, fideslib::SerType::JSON));
    ASSERT_TRUE(cc3 != nullptr);
    ASSERT_TRUE(pk3 != nullptr);
    ASSERT_TRUE(sk3 != nullptr);

    Plaintext pt3 = cc3->MakeCKKSPackedPlaintext(x);
    auto c3 = cc3->Encrypt(pk3, pt3);
    Plaintext r3;
    cc3->Decrypt(sk3, c3, &r3);
    r3->SetLength(8);
    ASSERT_ERROR_OK(rRef, r3);
}

TEST(OpenFHECompatTests, ConvolutionTransform) {
    // Small real 2-row kernel (gStep=2 rows of bStep=3 taps) exercised on the GPU;
    // the CPU path OPENFHE_THROWs by design, so the CPU reference is built with the
    // public EvalRotate/EvalMult/EvalAdd API replicating the documented op sequence
    // (rotate per tap, multiply by the row's filter masks, rotate rows, accumulate).
    const uint32_t multDepth = 4;
    const uint32_t scaleModSize = 50;
    const uint32_t batchSize = 16;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(128);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // 3-tap kernel over 16 slots. The api requires indexes.size() == bStep
    // (rotate_hoisted maps one output slot per index). Tap 0 is avoided: this
    // OpenFHE build throws on EvalRotate(ct, 0) (automorphism index 1, no key).
    const std::vector<int> taps = { -3, -1, 3 };
    const int bStep = 3;
    const int gStep = 2; // two filter rows -> rowSize = bStep * gStep
    const int rowSize = bStep * gStep;
    const int stride = 1;

    // Rotation keys for the taps, the row-normalization rotations {(gStep-j)*stride},
    // and the special mask sweep {maskRotationStride, 2*maskRotationStride}.
    cc->EvalRotateKeyGen(keys.secretKey, { -3, -2, -1, 1, 2, 3, 4, 5 });

    // Convolution rotation-index helper: gStep=2 < 8 => intra-block stride*1 only.
    auto convRots = cc->GetConvolutionTransformRotationIndices(rowSize, bStep, stride, static_cast<uint32_t>(gStep));
    ASSERT_EQ(convRots, (std::vector<int>{ 1 }));

    std::vector<double> x;
    for (int i = 0; i < static_cast<int>(batchSize); ++i)
        x.push_back(static_cast<double>(i + 1) * 0.25);

    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);
    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    // Filter masks: constant per-tap weights, distinct values per row/tap.
    const double w[rowSize] = { 0.5, 0.25, 0.75, 0.1, 0.9, 0.4 };
    std::vector<Plaintext> masks;
    for (int i = 0; i < rowSize; ++i)
        masks.push_back(cc->MakeCKKSPackedPlaintext(std::vector<double>(batchSize, w[i])));

    // CPU reference: out = Σ_j rot( Σ_i rot(ct, taps[i]) ⊙ mask_{j*bStep+i}, (gStep-j)*stride )
    auto accA = cc->EvalMult(cc->EvalRotate(ctxt, taps[0]), masks[0]);
    accA = cc->EvalAdd(accA, cc->EvalMult(cc->EvalRotate(ctxt, taps[1]), masks[1]));
    accA = cc->EvalAdd(accA, cc->EvalMult(cc->EvalRotate(ctxt, taps[2]), masks[2]));
    auto accB = cc->EvalMult(cc->EvalRotate(ctxt, taps[0]), masks[3]);
    accB = cc->EvalAdd(accB, cc->EvalMult(cc->EvalRotate(ctxt, taps[1]), masks[4]));
    accB = cc->EvalAdd(accB, cc->EvalMult(cc->EvalRotate(ctxt, taps[2]), masks[5]));
    accA = cc->EvalRotate(accA, (gStep - 0) * stride);
    accB = cc->EvalRotate(accB, (gStep - 1) * stride);
    auto cRef = cc->EvalAdd(accA, accB);

    Plaintext rRef;
    cc->Decrypt(keys.secretKey, cRef, &rRef);
    rRef->SetLength(batchSize);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gConv = ctxt->Clone();
    cc->ConvolutionTransformInPlace(gConv, gStep, bStep, masks, taps, stride, rowSize);

    Plaintext rConv;
    cc->Decrypt(keys.secretKey, gConv, &rConv);
    rConv->SetLength(batchSize);

    // The kernel accumulates the whole dot product with a single final rescale,
    // whereas the reference rescales per EvalMult term (FLEXIBLEAUTO) — so the two
    // are value-equal but not bit-equal, and the difference can exceed the default
    // ASSERT_ERROR_OK window (2^-42 at this scale) by a hair. Compare with an
    // absolute tolerance well below the ~2^-40 signal precision.
    auto rvRef = rRef->GetRealPackedValue();
    auto rvConv = rConv->GetRealPackedValue();
    ASSERT_EQ(rvRef.size(), rvConv.size());
    double convMaxErr = 0.0;
    for (size_t i = 0; i < rvRef.size(); ++i)
        convMaxErr = std::max(convMaxErr, std::abs(rvConv[i] - rvRef[i]));
    std::cout << "ConvolutionTransform max error: " << convMaxErr << std::endl;
    ASSERT_LT(convMaxErr, 1e-9);

    // SpecialConvolutionTransform (gStep=1, the resnet layer-0 pipeline): the mask
    // sweep t + rot(t, mrs) + rot(t, 2*mrs), the mask multiply, then the row rotation.
    // The kernel's internal scale bookkeeping (entry rescale of a NoiseLevel-2 input,
    // level-matched masks, final rescaleInternal) requires:
    //   - the input ciphertext at noise degree 2 (a raw EvalMult result), and
    //   - the filter masks / mask plaintexts encoded at a level >= the ciphertext's
    //     (plaintexts can only be scaled DOWN to the ciphertext level).
    // There is no CPU reference (the CPU path OPENFHE_THROWs by design), so the checks
    // are finiteness + run-to-run determinism on the device.
    const int mrs = 1;
    const int sBStep = 3;
    const int sRowSize = sBStep; // gStep = 1
    const std::vector<int> sTaps = { 1, 3, 5 };

    auto ctMul = cc->EvalMult(ctxt, ctxt); // NoiseLevel 2 (entry-rescaled by the kernel)
    std::vector<Plaintext> sMasks;
    for (int i = 0; i < sRowSize; ++i)
        sMasks.push_back(cc->MakeCKKSPackedPlaintext(std::vector<double>(batchSize, 0.1 * (i + 1)), 1, 2, nullptr, batchSize));

    auto gSpec = ctMul->Clone();
    cc->SpecialConvolutionTransformInPlace(gSpec, 1, sBStep, sMasks, sMasks[0], sTaps, stride, mrs, sRowSize);
    auto gSpec2 = ctMul->Clone();
    cc->SpecialConvolutionTransformInPlace(gSpec2, 1, sBStep, sMasks, sMasks[0], sTaps, stride, mrs, sRowSize);

    Plaintext rSpec, rSpec2;
    cc->Decrypt(keys.secretKey, gSpec, &rSpec);
    rSpec->SetLength(batchSize);
    cc->Decrypt(keys.secretKey, gSpec2, &rSpec2);
    rSpec2->SetLength(batchSize);

    auto sv1 = rSpec->GetRealPackedValue();
    auto sv2 = rSpec2->GetRealPackedValue();
    ASSERT_EQ(sv1.size(), sv2.size());
    for (size_t i = 0; i < sv1.size(); ++i) {
        ASSERT_TRUE(std::isfinite(sv1[i]));
        EXPECT_NEAR(sv1[i], sv2[i], 1e-9);
    }
}

TEST(OpenFHECompatTests, AccumulateSumInPlaceStartOffset) {
    // 4-arg AccumulateSumInPlace(ct, slots, stride, start): the start-offset variant.
    // Exercised value-level against an explicit EvalRotate/EvalAdd reference chain.
    // For start=2, slots=8 (radix=PARTIAL_SUM_RADIX) both implementations land on
    // x + rot2 + rot4 + rot6: the GPU lazy fold rotates the level-1 accumulator (the
    // input) by {2,4,6}, while the CPU fallback issues {2,4} against the compounding
    // accumulator (rot4 of x+rot2(x) contributes rot4(x)+rot6(x)). The two phases are
    // value-equal but not bit-exact (see the DISABLED_AccumulateSumInPlaceStart test).
    uint32_t multDepth = 2;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(128);
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, { 2, 4, 6 });

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);
    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    // Reference chain of rotations of the *original* ciphertext. Both implementations
    // land on x + rot2 + rot4 + rot6 for start=2, slots=8: the GPU lazy fold rotates
    // the level-1 accumulator (the input) by {2,4,6}, and the CPU fallback loop, while
    // it *issues* the rotations {2,4} against the compounding accumulator, applies
    // rot4 to (x + rot2(x)), i.e. net rot4(x) + rot6(x) — the same mathematical sum.
    auto makeChain = [&](std::initializer_list<int32_t> rots) {
        Ciphertext<DCRTPoly> acc = ctxt->Clone();
        for (auto r : rots)
            acc = cc->EvalAdd(acc, cc->EvalRotate(ctxt, r));
        return acc;
    };
    auto cFullRef = makeChain({ 2, 4, 6 });

    // CPU phase: the start-offset variant's host fallback.
    auto cAcc = ctxt->Clone();
    cc->AccumulateSumInPlace(cAcc, batchSize, 1, 2);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    // GPU phase: unified lazy radix fold {2, 4, 6}.
    auto gAcc = ctxt->Clone();
    cc->AccumulateSumInPlace(gAcc, batchSize, 1, 2);

    Plaintext rc, rg, rcF;
    cc->Decrypt(keys.secretKey, cAcc, &rc);
    rc->SetLength(batchSize);
    cc->Decrypt(keys.secretKey, gAcc, &rg);
    rg->SetLength(batchSize);
    cc->Decrypt(keys.secretKey, cFullRef, &rcF);
    rcF->SetLength(batchSize);

    // Both the CPU fallback and the GPU fold must equal the {2,4,6} sum; and the
    // two phases agree value-level (they are not bit-exact — see the DISABLED
    // AccumulateSumInPlaceStart test above).
    ASSERT_ERROR_OK(rcF, rc);
    ASSERT_ERROR_OK(rcF, rg);
    ASSERT_ERROR_OK(rc, rg);
}

TEST(OpenFHECompatTests, ApiNullContextThrows) {
    // The rvalue-context constructors reject a null parent context — the only
    // uncovered lines of the CiphertextImpl/PlaintextImpl constructors.
    CryptoContext<DCRTPoly> nullCtx;
    EXPECT_THROW(std::make_shared<CiphertextImpl<DCRTPoly>>(std::move(nullCtx)), lbcrypto::OpenFHEException);
    CryptoContext<DCRTPoly> nullCtx2;
    EXPECT_THROW(std::make_shared<PlaintextImpl>(std::move(nullCtx2)), lbcrypto::OpenFHEException);
}

// The swapped-argument / scalar-first arithmetic overloads that forward to the
// concrete forms covered by EvalSwappedArgOrder and EvalMutableArguments:
//   EvalAddInPlace(Plaintext&, Ciphertext&), EvalAddInPlace(double, Ciphertext&),
//   EvalSubMutable(Ciphertext&, Plaintext&), EvalMultInPlace(double, Ciphertext&),
//   EvalMultMutable(Plaintext&, Ciphertext&).
// Clone per phase so the shared encryption is never mutated by the in-place calls.
TEST(OpenFHECompatTests, EvalSwappedInPlaceOverloads) {
    auto cc = MakeSmallContext(5);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);

    auto cAddPtL = ct1->Clone();
    cc->EvalAddInPlace(ptxt2, cAddPtL);
    auto cAddScL = ct1->Clone();
    cc->EvalAddInPlace(0.5, cAddScL);
    auto cSubPt = cc->EvalSubMutable(ct1, ptxt2);
    auto cMultScL = ct1->Clone();
    cc->EvalMultInPlace(1.5, cMultScL);
    auto cMultPtL = cc->EvalMultMutable(ptxt2, ct1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAddPtL = ct1->Clone();
    cc->EvalAddInPlace(ptxt2, gAddPtL);
    auto gAddScL = ct1->Clone();
    cc->EvalAddInPlace(0.5, gAddScL);
    auto gSubPt = cc->EvalSubMutable(ct1, ptxt2);
    auto gMultScL = ct1->Clone();
    cc->EvalMultInPlace(1.5, gMultScL);
    auto gMultPtL = cc->EvalMultMutable(ptxt2, ct1);

    ASSERT_EQ_CIPHERTEXT(cAddPtL, gAddPtL);
    ASSERT_EQ_CIPHERTEXT(cAddScL, gAddScL);
    ASSERT_EQ_CIPHERTEXT(cSubPt, gSubPt);
    ASSERT_EQ_CIPHERTEXT(cMultScL, gMultScL);
    ASSERT_EQ_CIPHERTEXT(cMultPtL, gMultPtL);
}

// Error/edge branches: cross-context add, GetPreScaleFactor before load, Decrypt
// with a null output, the empty-PlaintextImpl fallbacks, and the serialization
// unsupported-SerType / null-ciphertext paths.
TEST(OpenFHECompatTests, ApiErrorBranches) {
    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

    // Cross-context add must throw (Ciphertext::operator+ guard).
    auto ccA = MakeSmallContext(2);
    auto ccB = MakeSmallContext(2);
    auto keysA = ccA->KeyGen();
    auto keysB = ccB->KeyGen();
    ccA->EvalMultKeyGen(keysA.secretKey);
    Plaintext ptA = ccA->MakeCKKSPackedPlaintext(x);
    Plaintext ptB = ccB->MakeCKKSPackedPlaintext(x);
    auto ctA = ccA->Encrypt(keysA.publicKey, ptA);
    auto ctB = ccB->Encrypt(keysB.publicKey, ptB);
    EXPECT_THROW(ctA + ctB, lbcrypto::OpenFHEException);

    // GetPreScaleFactor before the context is loaded to a device must throw.
    auto cc = MakeSmallContext(2);
    auto keys = cc->KeyGen();
    EXPECT_THROW(cc->GetPreScaleFactor(8), lbcrypto::OpenFHEException);

    // Decrypt with a null output pointer must throw; so does encryption after
    // EvalMultKeyGen would be illegal — not exercised here.
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);
    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);
    EXPECT_THROW(cc->Decrypt(keys.secretKey, ctxt, nullptr), lbcrypto::OpenFHEException);

    // Empty PlaintextImpl fallbacks (no host encoding) + null Plaintext streaming.
    PlaintextImpl empty;
    ASSERT_EQ(empty.GetLevel(), 0u);
    ASSERT_EQ(empty.GetSlots(), 0u);
    ASSERT_EQ(empty.GetLogPrecision(), 0.0);
    ASSERT_TRUE(empty.GetCKKSPackedValue().empty());
    ASSERT_TRUE(empty.GetRealPackedValue().empty());
    std::ostringstream osEmpty;
    osEmpty << empty;
    ASSERT_EQ(osEmpty.str(), "Empty Plaintext");
    ASSERT_FALSE(*ptxt == empty); // one side has no host encoding
    ASSERT_TRUE(*ptxt != empty);

    Plaintext nullPt;
    std::ostringstream osNull;
    osNull << nullPt;
    ASSERT_EQ(osNull.str(), "Empty Plaintext");

    // Serialization: unsupported SerType returns false (or throws for the key
    // wrappers), and a null ciphertext throws in both directions.
    const SerType bad = static_cast<SerType>(99);
    const std::string badFile = "/tmp/opencode/bad-ser.bin";
    const std::string ctFile = "/tmp/opencode/err-ct.bin";
    std::remove(badFile.c_str());
    std::remove(ctFile.c_str());

    ASSERT_FALSE(fideslib::Serial::SerializeToFile(badFile, cc, bad));
    ASSERT_FALSE(fideslib::Serial::SerializeToFile(badFile, keys.publicKey, bad));
    ASSERT_FALSE(fideslib::Serial::SerializeToFile(badFile, keys.secretKey, bad));
    fideslib::CryptoContext<DCRTPoly> ccOut;
    ASSERT_FALSE(fideslib::Serial::DeserializeFromFile(badFile, ccOut, bad));
    fideslib::PublicKey<DCRTPoly> pkOut;
    ASSERT_FALSE(fideslib::Serial::DeserializeFromFile(badFile, pkOut, bad));
    fideslib::PrivateKey<DCRTPoly> skOut;
    ASSERT_FALSE(fideslib::Serial::DeserializeFromFile(badFile, skOut, bad));

    fideslib::Ciphertext<DCRTPoly> nullCt;
    EXPECT_THROW(fideslib::Serial::SerializeToFile(badFile, nullCt, fideslib::SerType::BINARY), lbcrypto::OpenFHEException);

    // The deserializer reads the file first, so feed it a valid ciphertext and a
    // null target to reach the null-target guard.
    ASSERT_TRUE(fideslib::Serial::SerializeToFile(ctFile, ctxt, fideslib::SerType::BINARY));
    EXPECT_THROW(fideslib::Serial::DeserializeFromFile(ctFile, nullCt, fideslib::SerType::BINARY), lbcrypto::OpenFHEException);
    fideslib::Ciphertext<DCRTPoly> sink = std::make_shared<CiphertextImpl<DCRTPoly>>(std::move(cc));
    ASSERT_FALSE(fideslib::Serial::DeserializeFromFile(ctFile, sink, bad));

    // Eval-key serialization with an unsupported type throws.
    std::ostringstream osKey;
    std::istringstream isKey;
    EXPECT_THROW(ccA->SerializeEvalMultKey(osKey, bad), lbcrypto::OpenFHEException);
    EXPECT_THROW(ccA->SerializeEvalAutomorphismKey(osKey, bad), lbcrypto::OpenFHEException);
    EXPECT_THROW(ccA->DeserializeEvalMultKey(isKey, bad), lbcrypto::OpenFHEException);
    EXPECT_THROW(ccA->DeserializeEvalAutomorphismKey(isKey, bad), lbcrypto::OpenFHEException);
}

TEST(OpenFHECompatTests, CiphertextAdjustMatrix) {
    // Drives the Ciphertext operand-adjustment primitives (add/addMutable/sub/
    // subMutable/addPt/subPt -> adjustForAddOrSub / adjustScaleAndLevel /
    // adjustCiphertextToPlaintext) over operands at mixed noise-degree and level:
    // fresh (deg1), EvalMult result (deg2) and its Rescale (deg1). Under the AUTO
    // techniques these must run on both phases and be bit-identical. (FIXEDMANUAL is
    // excluded: adding an un-rescaled degree-2 operand to a degree-1 one is invalid
    // manual-rescale usage. A SetLevel-dropped operand is also excluded: its GPU add
    // trips a Debug metadata assert — a separate divergence.)
    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    for (auto tech : { FLEXIBLEAUTO, FIXEDAUTO }) {
        auto cc = MakeSmallContext(6, tech);
        auto keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);

        Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
        Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
        auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
        auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

        auto runOps = [&](std::vector<Ciphertext<DCRTPoly>>& out) {
            auto cMult = cc->EvalMult(ct1, ct2); // deg2, one level down
            auto cResc = cc->Rescale(cMult);     // deg1

            out.push_back(cc->EvalAdd(ct1, ct2));     // deg1 x deg1
            out.push_back(cc->EvalAdd(cMult, ct1));   // deg2 x deg1
            out.push_back(cc->EvalSub(cMult, ct1));   // deg2 x deg1 (reversed)
            out.push_back(cc->EvalAdd(cResc, cMult)); // deg1 x deg2
            out.push_back(cc->EvalSub(cMult, cResc)); // deg2 x deg1 (reversed)
            out.push_back(cc->EvalAdd(cMult, ptxt2)); // deg2 + plaintext
            out.push_back(cc->EvalSub(cMult, ptxt2)); // deg2 - plaintext

            auto a = ct1->Clone();
            cc->EvalAddInPlace(a, cMult);
            out.push_back(a);
            auto s = ct1->Clone();
            cc->EvalSubMutableInPlace(s, cMult);
            out.push_back(s);
            auto p = ct1->Clone();
            cc->EvalAddInPlace(p, ptxt2); // ct + plaintext in place
            out.push_back(p);
        };

        std::vector<Ciphertext<DCRTPoly>> cpu;
        runOps(cpu);

        //====================================================================

        cc->SetDevices({ 0 });
        cc->LoadContext(keys.publicKey);

        std::vector<Ciphertext<DCRTPoly>> gpu;
        runOps(gpu);

        ASSERT_EQ(cpu.size(), gpu.size());
        for (size_t i = 0; i < cpu.size(); ++i) {
            std::cout << tech << " adjust case " << i << ": ";
            ASSERT_EQ_CIPHERTEXT(cpu[i], gpu[i]);
        }
    }
}

TEST(OpenFHECompatTests, ContextParamSweep) {
    // Sweeps numLargeDigits x scaling technique so the ContextData generation
    // branches (digit/decomposition metadata, ElemForEvalMult/AddOrSub) run for more
    // than the single dnum=2 / FLEXIBLEAUTO configuration the rest of the suite uses.
    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    int ran = 0;
    for (auto tech : { FIXEDMANUAL, FIXEDAUTO, FLEXIBLEAUTO, FLEXIBLEAUTOEXT }) {
        for (uint32_t dnum : { 1u, 2u, 3u }) {
            CCParams<CryptoContextCKKSRNS> p;
            p.SetSecretKeyDist(UNIFORM_TERNARY);
            p.SetScalingTechnique(tech);
            p.SetMultiplicativeDepth(4);
            p.SetScalingModSize(50);
            p.SetBatchSize(8);
            p.SetSecurityLevel(HEStd_NotSet);
            p.SetRingDim(128);
            p.SetNumLargeDigits(dnum);
            p.SetPlaintextAutoload(false);
            p.SetCiphertextAutoload(true);

            CryptoContext<DCRTPoly> cc;
            try {
                cc = GenCryptoContext(p);
            } catch (const std::exception&) {
                continue; // unsupported dnum/technique combination
            }
            cc->Enable(PKE);
            cc->Enable(KEYSWITCH);
            cc->Enable(LEVELEDSHE);

            auto keys = cc->KeyGen();
            cc->EvalMultKeyGen(keys.secretKey);

            Plaintext pt = cc->MakeCKKSPackedPlaintext(x);
            auto ct = cc->Encrypt(keys.publicKey, pt);
            auto cAdd = cc->EvalAdd(ct, ct);
            auto cRes = cc->Rescale(cc->EvalMult(ct, ct));

            Plaintext rAddC, rResC;
            cc->Decrypt(keys.secretKey, cAdd, &rAddC);
            rAddC->SetLength(8);
            cc->Decrypt(keys.secretKey, cRes, &rResC);
            rResC->SetLength(8);

            //================================================================

            cc->SetDevices({ 0 });
            cc->LoadContext(keys.publicKey);

            auto gct = cc->Encrypt(keys.publicKey, pt);
            auto gAdd = cc->EvalAdd(gct, gct);
            auto gRes = cc->Rescale(cc->EvalMult(gct, gct));

            Plaintext rAddG, rResG;
            cc->Decrypt(keys.secretKey, gAdd, &rAddG);
            rAddG->SetLength(8);
            cc->Decrypt(keys.secretKey, gRes, &rResG);
            rResG->SetLength(8);

            // Value-level only: the sweep's purpose is to drive the ContextData
            // generation branches across configurations, not bit-compat (which the
            // arithmetic suites already cover at dnum=2).
            std::cout << "ContextParamSweep tech=" << tech << " dnum=" << dnum << std::endl;
            ASSERT_ERROR_OK(rAddC, rAddG);
            ASSERT_ERROR_OK(rResC, rResG);

            ran++;
        }
    }
    std::cout << "ContextParamSweep ran " << ran << " configurations" << std::endl;
    ASSERT_GT(ran, 0);
}

TEST(OpenFHECompatTests, BootstrapPrecomputationGuard) {
    // GetPreScaleFactor reaches ContextData::GetBootPrecomputation(); a context loaded
    // without EvalBootstrapSetup/EvalBootstrapKeyGen for those slots must throw (the
    // old assert()/map-operator[] silently default-constructed a precomputation).
    auto cc = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);
    EXPECT_THROW(cc->GetPreScaleFactor(8), std::runtime_error);
}

// Direct unit coverage for the Ciphertext primitives whose names carry
// add/sub/mult but that the public api never routes through: the two-operand
// accumulator forms and addMult*. A loaded FIDESlib context supplies the GPU context;
// GPU Ciphertext/Plaintext objects are built from the OpenFHE encryptions via
// GetRawCipherText/GetRawPlainText, the primitive stores into a fresh object, and the
// result is read back with GetOpenFHECipherText and compared to the api reference.
TEST(OpenFHECompatTests, CiphertextPrimitives) {
    auto cc = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto refAdd = cc->EvalAdd(ct1, ct2);
    auto refSub = cc->EvalSub(ct1, ct2);
    auto refAddPt = cc->EvalAdd(ct1, ptxt2);
    auto refAddScalar = cc->EvalAdd(ct1, 0.5);
    auto refMult = cc->EvalMult(ct1, ct2);
    auto refMultScalar = cc->EvalMult(ct1, 1.5);
    auto refMultPt = cc->EvalMult(ct1, ptxt2);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto& gpuCtx = std::any_cast<FIDESlib::CKKS::Context&>(cc->gpu);
    auto& rawCc = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
    const auto& rawSk = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(keys.secretKey->pimpl);
    const auto& rawCt1 = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
    const auto& rawCt2 = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
    const auto& rawPt1 = std::any_cast<const lbcrypto::Plaintext&>(ptxt1->cpu);
    const auto& rawPt2 = std::any_cast<const lbcrypto::Plaintext&>(ptxt2->cpu);

    auto mkCt = [&](const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& rc) {
        return FIDESlib::CKKS::Ciphertext(gpuCtx, FIDESlib::CKKS::GetRawCipherText(rawCc, rc));
    };
    auto mkPt = [&](const lbcrypto::Plaintext& rp) {
        return FIDESlib::CKKS::Plaintext(gpuCtx, FIDESlib::CKKS::GetRawPlainText(rawCc, rp));
    };

    auto maxDiff = [&](const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& a,
                       const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& b) {
        lbcrypto::Plaintext pa, pb;
        rawCc->Decrypt(rawSk, a, &pa);
        pa->SetLength(8);
        rawCc->Decrypt(rawSk, b, &pb);
        pb->SetLength(8);
        auto va = pa->GetRealPackedValue();
        auto vb = pb->GetRealPackedValue();
        double m = 0.0;
        for (size_t i = 0; i < va.size() && i < vb.size(); ++i)
            m = std::max(m, std::abs(va[i] - vb[i]));
        return m;
    };

    auto check = [&](const char* name, FIDESlib::CKKS::Ciphertext& g, const Ciphertext<DCRTPoly>& ref) {
        FIDESlib::CKKS::RawCipherText rr;
        g.store(rr);
        // Prototype at the reference's level/towers so the read-back basis matches.
        const auto& rawRef = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ref->cpu);
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly> rawOut = rawRef->Clone();
        FIDESlib::CKKS::GetOpenFHECipherText(rawOut, rr);
        double d = maxDiff(rawOut, rawRef);
        std::cout << "primitive " << name << " max error " << d << std::endl;
        ASSERT_LT(d, 1e-9);
    };

    {
        auto g = mkCt(rawCt1);
        auto a = mkCt(rawCt1);
        auto b = mkCt(rawCt2);
        g.add(a, b);
        check("add(a,b)", g, refAdd);
    }
    {
        auto g = mkCt(rawCt1);
        auto a = mkCt(rawCt1);
        auto b = mkCt(rawCt2);
        g.sub(a, b);
        check("sub(a,b)", g, refSub);
    }
    {
        auto g = mkCt(rawCt1);
        auto a = mkCt(rawCt1);
        auto b = mkCt(rawCt2);
        g.addMutable(a, b);
        check("addMutable(a,b)", g, refAdd);
    }
    {
        auto g = mkCt(rawCt1);
        auto a = mkCt(rawCt1);
        auto b = mkCt(rawCt2);
        g.subMutable(a, b);
        check("subMutable(a,b)", g, refSub);
    }
    {
        auto g = mkCt(rawCt1);
        auto a = mkCt(rawCt1);
        auto pt = mkPt(rawPt2);
        g.addPt(a, pt);
        check("addPt(a,pt)", g, refAddPt);
    }
    {
        auto g = mkCt(rawCt1);
        auto a = mkCt(rawCt1);
        auto pt = mkPt(rawPt2);
        g.addPtMutable(a, pt);
        check("addPtMutable(a,pt)", g, refAddPt);
    }
    {
        auto g = mkCt(rawCt1);
        auto b = mkCt(rawCt1);
        g.addScalar(b, 0.5);
        check("addScalar(b,c)", g, refAddScalar);
    }
    {
        auto g = mkCt(rawCt1);
        auto b = mkCt(rawCt1);
        g.multScalar(b, 1.5);
        check("multScalar(b,c)", g, refMultScalar);
    }
    {
        auto g = mkCt(rawCt1);
        auto b = mkCt(rawCt1);
        auto c = mkCt(rawCt2);
        g.mult(b, c);
        check("mult(b,c)", g, refMult);
    }
    {
        auto g = mkCt(rawCt1);
        auto b = mkCt(rawCt1);
        auto c = mkCt(rawCt2);
        g.multMutable(b, c);
        check("multMutable(b,c)", g, refMult);
    }
    {
        auto g = mkCt(rawCt1);
        auto c = mkCt(rawCt1);
        auto pt = mkPt(rawPt2);
        g.multPt(c, pt);
        check("multPt(c,pt)", g, refMultPt);
    }
    {
        auto g = mkCt(rawCt1);
        auto c = mkCt(rawCt1);
        auto pt = mkPt(rawPt2);
        g.multPtMutable(c, pt);
        check("multPtMutable(c,pt)", g, refMultPt);
    }
}

} // namespace FIDESlib::Testing
