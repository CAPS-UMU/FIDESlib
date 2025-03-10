//
// Created by seyda on 9/14/24.
//
#include <chrono>
#include <cmath>
#include <iomanip>
#include "CKKS/Context.cuh"
#include "CKKS/Limb.cuh"
#include "ConstantsGPU.cuh"
#include "gtest/gtest.h"

#include "CKKS/Ciphertext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "Math.cuh"
#include "ParametrizedTest.cuh"
#include "Rotation.cuh"
#include "openfhe/pke/openfhe.h"

using namespace FIDESlib;
using namespace lbcrypto;
using namespace std;
using namespace std::chrono;
using namespace chrono_literals;

namespace FIDESlib::Testing {
class RotationTests : public GeneralParametrizedTest {};

TEST_P(RotationTests, AutomorphTest_Single_Limb) {
    for (auto& i : cached_cc) {
        i.second.first->ClearEvalAutomorphismKeys();
        i.second.first->ClearEvalMultKeys();
        if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
            std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)
                ->m_bootPrecomMap.clear();
    }
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    auto keys = cc->KeyGen();

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};

    cc->EvalRotateKeyGen(keys.secretKey, {0, 1, 2, 3, 4, 5, 6, 7});

    auto batchSize = cc->GetCryptoParameters()->GetEncodingParams()->GetBatchSize();
    // inputs
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    // Encoding as plaintexts
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    // Encrypt the encoded vector
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c3 = cc->Encrypt(keys.publicKey, ptxt1);

    // Extract raw ciphertext
    int br = 0;  // br = 0: ct comes br, no br in the automorph kernel
    FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1, br);

    // Create ciphertext objects on GPUcc context
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw);

    for (int idx = 1; idx <= 4; ++idx) {
        // the rotation we want to see in the message vector
        //const int idx = 1;
        // calculation of the corresponding index in the ciphertext polynomial
        int index = 5;
        if (idx < 0) {
            index = FIDESlib::modinv(5, batchSize);
        }
        int k = FIDESlib::modpow(index, idx, GPUcc.N);

        auto c1_rot_CPU = c1->Clone();
        vector<uint32_t> vec(GPUcc.N);
        PrecomputeAutoMap(GPUcc.N, k, &vec);

        c1_rot_CPU->GetElements()[0] = c1_rot_CPU->GetElements()[0].AutomorphismTransform(k, vec);
        c1_rot_CPU->GetElements()[1] = c1_rot_CPU->GetElements()[1].AutomorphismTransform(k, vec);

        // Automorph can be done on a single limb in two ways: by calling automorph_multi with limb_count=1 or by calling automorph
        // const int limb_count = 1;
        // GPUct1_rot.automorph_multi(GPUct1, GPUct1_rot, k, limb_count, br);
        GPUct1.automorph(1, 0);
        FIDESlib::CKKS::RawCipherText raw_res2;
        GPUct1.store(GPUcc, raw_res2);
        auto c1_rot_GPU(c3);
        GetOpenFHECipherText(c1_rot_GPU, raw_res2, br);

        ASSERT_EQ_CIPHERTEXT(c1_rot_GPU, c1_rot_CPU);
    }
}

TEST_P(RotationTests, AutomorphTest_Multi_Limb) {
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    auto keys = cc->KeyGen();

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};

    cc->EvalRotateKeyGen(keys.secretKey, {0, 1, 2, 3, 4, 5, 6, 7});

    auto batchSize = cc->GetCryptoParameters()->GetEncodingParams()->GetBatchSize();
    // inputs
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    // Encoding as plaintexts
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    // Encrypt the encoded vector
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

    // Extract raw ciphertext
    int br = 0;  // br = 0: ct comes br, no br in the automorph kernel
    FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1, br);

    // Create ciphertext objects on GPUcc context
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw);

    // the rotation we want to see in the message vector
    const int idx = 1;
    // calculation of the corresponding index in the ciphertext polynomial
    int index = 5;
    if (idx < 0) {
        index = FIDESlib::modinv(5, batchSize);
    }
    int k = FIDESlib::modpow(index, idx, GPUcc.N);

    auto c1_rot_CPU = c1->Clone();
    vector<uint32_t> vec(GPUcc.N);
    PrecomputeAutoMap(GPUcc.N, k, &vec);
    c1_rot_CPU->GetElements()[0] = c1_rot_CPU->GetElements()[0].AutomorphismTransform(k, vec);
    c1_rot_CPU->GetElements()[1] = c1_rot_CPU->GetElements()[1].AutomorphismTransform(k, vec);

    GPUct1.automorph_multi(1, 0);

    FIDESlib::CKKS::RawCipherText raw_res2;
    GPUct1.store(GPUcc, raw_res2);
    auto c1_rot_GPU(c1);
    GetOpenFHECipherText(c1_rot_GPU, raw_res2, br);

    ASSERT_EQ_CIPHERTEXT(c1_rot_CPU, c1_rot_GPU);
}

// Define the parameter sets
//INSTANTIATE_TEST_SUITE_P(OpenFHEInterfaceTests, RotationTests,
//                         testing::Values(tparams64_13, tparams64_14, tparams64_15, tparams64_16));
INSTANTIATE_TEST_SUITE_P(OpenFHEInterfaceTests, RotationTests, testing::Values(TTALL64));
}  // namespace FIDESlib::Testing
