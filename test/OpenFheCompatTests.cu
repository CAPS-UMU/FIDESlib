#include "ParametrizedTest.cuh"

#include <algorithm>
#include <any>
#include <fideslib.hpp>
#include <map>
#include <set>
#include <vector>

using namespace fideslib;

namespace FIDESlib::Testing {

TEST(OpenFHECompatTests, EvalFastRotation) {
    uint32_t multDepth    = 2;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

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
    uint32_t multDepth    = 2;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

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
    uint32_t multDepth    = 2;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

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
    // Radix-2 (doubling) accumulation for slots=8, stride=1 rotates by the power-of-two set {1, 2, 4}.
    cc->EvalRotateKeyGen(keys.secretKey, { 1, 2, 4 });

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
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

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
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t ringDim      = 1 << 12;
    uint32_t numSlots     = ringDim / 2;  // fully packed

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
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

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
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 64;

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
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

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
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t ringDim      = 1 << 12;
    uint32_t numSlots     = ringDim / 2;  // fully packed

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
TEST(OpenFHECompatTests, DISABLED_EvalBootstrapSparseEncaps) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

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
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

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
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t ringDim      = 1 << 12;
    uint32_t numSlots     = ringDim / 2;  // fully packed

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
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

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
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

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


// Combination sweep: FIXEDAUTO bootstrap — the fourth scaling technique,
// previously untested at any level. DISABLED: known divergence (O6e) — the GPU
// add/sub operand adjustment only runs under FLEXIBLEAUTO/FLEXIBLEAUTOEXT,
// while stock AdjustForAddOrSub adjusts for every technique except FIXEDMANUAL,
// FIXEDAUTO included. Acceptance test for the O6e fix.
TEST(OpenFHECompatTests, DISABLED_EvalBootstrapFixedAuto) {
    uint32_t multDepth    = 25;
    uint32_t scaleModSize = 50;
    uint32_t batchSize    = 8;

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
    auto cc   = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto cAdd    = cc->EvalAdd(ct1, ct2);
    auto cAddSc  = cc->EvalAdd(ct1, 0.5);
    auto cSub    = cc->EvalSub(ct1, ct2);
    auto cSubSc  = cc->EvalSub(ct1, 0.25);
    auto cNeg    = cc->EvalNegate(ct1);
    auto cMult   = cc->EvalMult(ct1, ct2);
    auto cMultSc = cc->EvalMult(ct1, 1.5);
    auto cSq     = cc->EvalSquare(ct1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAdd    = cc->EvalAdd(ct1, ct2);
    auto gAddSc  = cc->EvalAdd(ct1, 0.5);
    auto gSub    = cc->EvalSub(ct1, ct2);
    auto gSubSc  = cc->EvalSub(ct1, 0.25);
    auto gNeg    = cc->EvalNegate(ct1);
    auto gMult   = cc->EvalMult(ct1, ct2);
    auto gMultSc = cc->EvalMult(ct1, 1.5);
    auto gSq     = cc->EvalSquare(ct1);

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
    auto cc   = MakeSmallContext(4, FLEXIBLEAUTOEXT);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto cAdd    = cc->EvalAdd(ct1, ct2);
    auto cAddSc  = cc->EvalAdd(ct1, 0.5);
    auto cSub    = cc->EvalSub(ct1, ct2);
    auto cSubSc  = cc->EvalSub(ct1, 0.25);
    auto cNeg    = cc->EvalNegate(ct1);
    auto cMult   = cc->EvalMult(ct1, ct2);
    auto cMultSc = cc->EvalMult(ct1, 1.5);
    auto cSq     = cc->EvalSquare(ct1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAdd    = cc->EvalAdd(ct1, ct2);
    auto gAddSc  = cc->EvalAdd(ct1, 0.5);
    auto gSub    = cc->EvalSub(ct1, ct2);
    auto gSubSc  = cc->EvalSub(ct1, 0.25);
    auto gNeg    = cc->EvalNegate(ct1);
    auto gMult   = cc->EvalMult(ct1, ct2);
    auto gMultSc = cc->EvalMult(ct1, 1.5);
    auto gSq     = cc->EvalSquare(ct1);

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
    auto cc   = MakeSmallContext(4, FIXEDAUTO);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto cAdd    = cc->EvalAdd(ct1, ct2);
    auto cAddSc  = cc->EvalAdd(ct1, 0.5);
    auto cSub    = cc->EvalSub(ct1, ct2);
    auto cSubSc  = cc->EvalSub(ct1, 0.25);
    auto cNeg    = cc->EvalNegate(ct1);
    auto cMult   = cc->EvalMult(ct1, ct2);
    auto cMultSc = cc->EvalMult(ct1, 1.5);
    auto cSq     = cc->EvalSquare(ct1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAdd    = cc->EvalAdd(ct1, ct2);
    auto gAddSc  = cc->EvalAdd(ct1, 0.5);
    auto gSub    = cc->EvalSub(ct1, ct2);
    auto gSubSc  = cc->EvalSub(ct1, 0.25);
    auto gNeg    = cc->EvalNegate(ct1);
    auto gMult   = cc->EvalMult(ct1, ct2);
    auto gMultSc = cc->EvalMult(ct1, 1.5);
    auto gSq     = cc->EvalSquare(ct1);

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
    auto cc   = MakeSmallContext(4);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);

    auto cAddPt  = cc->EvalAdd(ct1, ptxt2);
    auto cSubPt  = cc->EvalSub(ct1, ptxt2);
    auto cMultPt = cc->EvalMult(ct1, ptxt2);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAddPt  = cc->EvalAdd(ct1, ptxt2);
    auto gSubPt  = cc->EvalSub(ct1, ptxt2);
    auto gMultPt = cc->EvalMult(ct1, ptxt2);

    ASSERT_EQ_CIPHERTEXT(cAddPt, gAddPt);
    ASSERT_EQ_CIPHERTEXT(cSubPt, gSubPt);
    ASSERT_EQ_CIPHERTEXT(cMultPt, gMultPt);
}

TEST(OpenFHECompatTests, EvalAdjust) {
    // Exercises the FLEXIBLEAUTO scale/level adjustment paths with operands at
    // mixed noise degrees and level gaps (including deg2/deg2 with a gap >= 2).
    auto cc   = MakeSmallContext(6);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    // Encrypt ONCE: encryption is randomized, so both phases must share the input.
    auto c1 = cc->Encrypt(keys.publicKey, ptxt);

    auto runChain = [&](std::vector<Ciphertext<DCRTPoly>>& out) {
        auto c2 = cc->EvalMult(c1, c1);  // deg2 @ top level
        auto c3 = cc->EvalMult(c2, c2);  // deg2, one level down
        auto c4 = cc->EvalMult(c3, c3);  // deg2, two levels down

        out.push_back(c2);
        out.push_back(c3);
        out.push_back(c4);
        out.push_back(cc->EvalAdd(c1, c3));   // deg1 vs deg2, gap 1
        out.push_back(cc->EvalAdd(c2, c4));   // deg2 vs deg2, gap 2
        out.push_back(cc->EvalSub(c4, c2));   // deg2 vs deg2, gap 2 (reversed)
        out.push_back(cc->EvalMult(c1, c4));  // deg1 vs deg2, gap 2
        out.push_back(cc->EvalMult(c2, c3));  // deg2 vs deg2, gap 1
        out.push_back(cc->EvalSquare(c1));    // same-handle mult above must equal this square
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

TEST(OpenFHECompatTests, EvalChebyshev) {
    auto cc   = MakeSmallContext(10);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // Degree-12 series -> Paterson-Stockmeyer path on both sides.
    std::vector<double> coeffs = { 0.15, 0.05, 0.2, -0.03, 0.11, 0.007, -0.05, 0.021, 0.09, -0.012, 0.033, 0.004, -0.026 };

    std::vector<double> x = { 0.03, 0.06, 0.09, 0.12, 0.25, 0.37, 0.5, 0.62 };
    Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x);

    auto ctxt = cc->Encrypt(keys.publicKey, ptxt);

    auto cCheb = cc->EvalChebyshevSeries(ctxt, coeffs, -1.0, 1.0);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gCheb = cc->EvalChebyshevSeries(ctxt, coeffs, -1.0, 1.0);

    ASSERT_EQ_CIPHERTEXT(cCheb, gCheb);
}

// FIXEDMANUAL branches of the Chebyshev transcription (su/cu LevelReduce gates, manual rescale placement)
TEST(OpenFHECompatTests, EvalChebyshevFixedManual) {
    auto cc   = MakeSmallContext(10, FIXEDMANUAL);
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
    auto cChebR1 = cc->Rescale(cCheb);
    Plaintext r1;
    cc->Decrypt(keys.secretKey, cChebR1, &r1);
    r1->SetLength(8);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gCheb = cc->EvalChebyshevSeries(ctxt, coeffs, -1.0, 1.0);

    auto gChebR2 = cc->Rescale(gCheb);
    Plaintext r2;
    cc->Decrypt(keys.secretKey, gChebR2, &r2);
    r2->SetLength(8);

    ASSERT_ERROR_OK(r1, r2);

    ASSERT_EQ_CIPHERTEXT(cCheb, gCheb);
}

// FIXEDMANUAL arithmetic, including an explicit Rescale after
// the multiply — under FIXEDMANUAL both the CPU fallback and the GPU perform a real
// rescale, so the ModReduce path is comparable bit-for-bit
TEST(OpenFHECompatTests, EvalArithmeticFixedManual) {
    auto cc   = MakeSmallContext(4, FIXEDMANUAL);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    auto ct1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto ct2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto cAdd   = cc->EvalAdd(ct1, ct2);
    auto cSub   = cc->EvalSub(ct1, ct2);
    auto cNeg   = cc->EvalNegate(ct1);
    auto cMult  = cc->EvalMult(ct1, ct2);
    auto cResc  = cc->Rescale(cMult);
    auto cSq    = cc->EvalSquare(ct1);

    //====================================================================

    cc->SetDevices({ 0 });
    cc->LoadContext(keys.publicKey);

    auto gAdd   = cc->EvalAdd(ct1, ct2);
    auto gSub   = cc->EvalSub(ct1, ct2);
    auto gNeg   = cc->EvalNegate(ct1);
    auto gMult  = cc->EvalMult(ct1, ct2);
    auto gResc  = cc->Rescale(gMult);
    auto gSq    = cc->EvalSquare(ct1);

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
    auto cc   = MakeSmallContext(2);
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
    auto cc   = MakeSmallContext(2);
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


} // namespace FIDESlib::Testing
