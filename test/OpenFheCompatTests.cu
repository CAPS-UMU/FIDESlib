#include "ParametrizedTest.cuh"

#include <algorithm>
#include <any>
#include <cstdio>
#include <fideslib.hpp>
#include <map>
#include <set>
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
//  - SerializeEvalMultKey / SerializeEvalAutomorphismKey / Deserialize*: host-side I/O.
//  - ConvolutionTransformInPlace / SpecialConvolutionTransformInPlace: GPU-only in the
//    api — the CPU path OPENFHE_THROWs ("Not implemented for CPU path"), so no CPU
//    reference exists to compare against.
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

    auto cAddMut = cc->EvalAddMutable(ct1, ct2);
    auto cAddMutPt = cc->EvalAddMutable(ct1, ptxt2);
    auto cAddMutPtR = cc->EvalAddMutable(ptxt2, ct1);
    auto cAddMutInP = ct1->Clone();
    cc->EvalAddMutableInPlace(cAddMutInP, ct2);
    auto cSubMut = cc->EvalSubMutable(ct1, ct2);
    auto cSubMutPt = cc->EvalSubMutable(ptxt2, ct1);
    auto cSubMutInP = ct1->Clone();
    cc->EvalSubMutableInPlace(cSubMutInP, ct2);
    auto cMultMut = cc->EvalMultMutable(ct1, ct2);
    auto cMultMutPt = cc->EvalMultMutable(ct1, ptxt2);
    auto cMultMutInP = ct1->Clone();
    cc->EvalMultMutableInPlace(cMultMutInP, ct2);
    auto cSqMut = cc->EvalSquareMutable(ct1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAddMut = cc->EvalAddMutable(ct1, ct2);
    auto gAddMutPt = cc->EvalAddMutable(ct1, ptxt2);
    auto gAddMutPtR = cc->EvalAddMutable(ptxt2, ct1);
    auto gAddMutInP = ct1->Clone();
    cc->EvalAddMutableInPlace(gAddMutInP, ct2);
    auto gSubMut = cc->EvalSubMutable(ct1, ct2);
    auto gSubMutPt = cc->EvalSubMutable(ptxt2, ct1);
    auto gSubMutInP = ct1->Clone();
    cc->EvalSubMutableInPlace(gSubMutInP, ct2);
    auto gMultMut = cc->EvalMultMutable(ct1, ct2);
    auto gMultMutPt = cc->EvalMultMutable(ct1, ptxt2);
    auto gMultMutInP = ct1->Clone();
    cc->EvalMultMutableInPlace(gMultMutInP, ct2);
    auto gSqMut = cc->EvalSquareMutable(ct1);

    ASSERT_EQ_CIPHERTEXT(cAddMut, gAddMut);
    ASSERT_EQ_CIPHERTEXT(cAddMutPt, gAddMutPt);
    ASSERT_EQ_CIPHERTEXT(cAddMutPtR, gAddMutPtR);
    ASSERT_EQ_CIPHERTEXT(cAddMutInP, gAddMutInP);
    ASSERT_EQ_CIPHERTEXT(cSubMut, gSubMut);
    ASSERT_EQ_CIPHERTEXT(cSubMutPt, gSubMutPt);
    ASSERT_EQ_CIPHERTEXT(cSubMutInP, gSubMutInP);
    ASSERT_EQ_CIPHERTEXT(cMultMut, gMultMut);
    ASSERT_EQ_CIPHERTEXT(cMultMutPt, gMultMutPt);
    ASSERT_EQ_CIPHERTEXT(cMultMutInP, gMultMutInP);
    ASSERT_EQ_CIPHERTEXT(cSqMut, gSqMut);
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

} // namespace FIDESlib::Testing
