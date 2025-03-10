//
// Created by carlosad on 29/04/24.
//
#include <iomanip>

#include <gtest/gtest.h>
#include <openfhe/pke/openfhe.h>

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Limb.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ConstantsGPU.cuh"
#include "Math.cuh"
#include "ParametrizedTest.cuh"
#include "cpuNTT.hpp"
#include "cpuNTT_nega.hpp"

//#include "hook.h"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/CoeffsToSlots.cuh"

namespace FIDESlib::Testing {
class OpenFHEInterfaceTest : public GeneralParametrizedTest {};

/*
TEST_P(OpenFHEInterfaceTest, LoadCiphertext) {
    FIDESlib::CKKS::Context GPUcc{fideslibParams, generalTestParams.GPUs};

    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    auto keys = cc->KeyGen();

    // Step 3: Encoding and encryption of inputs

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vector
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

    // Extract raw ciphertext
    FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    assert(raw.numRes >= 1);
    assert(raw.numRes <= GPUcc.L + 1);

    // Create ciphertext objects on GPUcc context
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw);
    FIDESlib::CKKS::Ciphertext GPUct2(GPUcc);
    GPUct2.load(raw);

    // Optionally, load and check plaintext on GPUcc context
    FIDESlib::CKKS::RawPlainText rawpt = FIDESlib::CKKS::GetRawPlainText(cc, ptxt1);
    FIDESlib::CKKS::Plaintext GPUpt1(GPUcc, rawpt);
    FIDESlib::CKKS::Plaintext GPUpt2(GPUcc);
    GPUpt2.load(rawpt);
}

TEST_P(OpenFHEInterfaceTest, LoadStoreCiphertext) {
    FIDESlib::CKKS::Context GPUcc{fideslibParams, generalTestParams.GPUs};

    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    auto keys = cc->KeyGen();

    // Step 3: Encoding and encryption of inputs

    // Inputs
    // vector of c1 and c2, for loop running of evalAdd over vectors
    // will need to make it multithreaded
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.5};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);

    assert(raw.numRes >= 1);
    assert(raw.numRes <= GPUcc.L + 1);

    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw);

    FIDESlib::CKKS::Ciphertext GPUct2(GPUcc);
    GPUct2.load(raw);

    FIDESlib::CKKS::RawCipherText raw_res1;
    FIDESlib::CKKS::RawCipherText raw_res2;
    GPUct1.store(GPUcc, raw_res1);
    GPUct2.store(GPUcc, raw_res2);

    auto cRes1(c2);
    GetOpenFHECipherText(cRes1, raw_res1);
    auto cRes2(c2);
    GetOpenFHECipherText(cRes2, raw_res2);

    lbcrypto::Plaintext result1;
    lbcrypto::Plaintext result2;

    cc->Decrypt(keys.secretKey, cRes1, &result1);
    cc->Decrypt(keys.secretKey, cRes2, &result2);

    for (int j = 0; j < 2; ++j) {
        ASSERT_EQ(c1.get()->m_elements[j].m_vectors.size(), cRes1.get()->m_elements[j].m_vectors.size());

        for (size_t i = 0; i < c1.get()->m_elements[j].m_vectors.size(); ++i) {
            ASSERT_EQ(c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.size(),
                      cRes1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.size());
            ASSERT_EQ(c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data,
                      cRes1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data);
        }
    }

    for (int j = 0; j < 2; ++j) {
        ASSERT_EQ(c1.get()->m_elements[j].m_vectors.size(), cRes2.get()->m_elements[j].m_vectors.size());

        for (size_t i = 0; i < c1.get()->m_elements[j].m_vectors.size(); ++i) {
            ASSERT_EQ(c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.size(),
                      cRes2.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.size());
            ASSERT_EQ(c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data,
                      cRes2.get()->m_elements[j].m_vectors[i].m_values.get()->m_data);
        }
    }

    std::cout << "pt " << ptxt1;
    result1->SetLength(generalTestParams.batchSize);
    std::cout << "res1 " << result1;
    result2->SetLength(generalTestParams.batchSize);
    std::cout << "res2 " << result2;
}
 */

TEST_P(OpenFHEInterfaceTest, ExtractContextShowAdd) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};

    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x3 = {0.0};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
    lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

    std::cout << "Input x1: " << ptxt1 << std::endl;
    std::cout << "Input x2: " << ptxt2 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
    auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
    FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);

    // GPU add
    GPUct1.add(GPUct2);
    FIDESlib::CKKS::RawCipherText raw_res1;
    GPUct1.store(GPUcc, raw_res1);
    auto cResGPU(c3);

    GetOpenFHECipherText(cResGPU, raw_res1);
    lbcrypto::Plaintext resultGPU;
    cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

    // CPU add
    lbcrypto::Plaintext result;
    auto cAdd = cc->EvalAdd(c1, c2);
    cc->Decrypt(keys.secretKey, cAdd, &result);

    std::cout << "Add:\n";
    std::cout << "Result " << result;
    //result2->SetLength(batchSize);
    std::cout << "Result GPU " << resultGPU;

    ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

    FIDESlib::CKKS::RawPlainText rawpt = FIDESlib::CKKS::GetRawPlainText(cc, ptxt1);

    // CPU addPt
    auto cAddPt = cc->EvalAdd(cAdd, ptxt1);
    cc->Decrypt(keys.secretKey, cAddPt, &result);

    // GPU addPt
    FIDESlib::CKKS::Plaintext GPUpt1(GPUcc, rawpt);
    GPUct1.addPt(GPUpt1);
    GPUct1.store(GPUcc, raw_res1);

    GetOpenFHECipherText(cResGPU, raw_res1);
    cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

    std::cout << "AddPt:\n";
    std::cout << "Result " << result;
    //result2->SetLength(batchSize);
    std::cout << "Result GPU " << resultGPU;

    ASSERT_EQ_CIPHERTEXT(cAddPt, cResGPU);
}

TEST_P(OpenFHEInterfaceTest, ScalarAdd) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};

    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x3 = {0.0};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw1);

    // GPU add
    GPUct1.addScalar(2.0);
    GPUct2.addScalar(-2.0);
    FIDESlib::CKKS::RawCipherText raw_res1;
    GPUct1.store(GPUcc, raw_res1);
    auto cResGPU(c3);

    GetOpenFHECipherText(cResGPU, raw_res1);
    lbcrypto::Plaintext resultGPU;
    cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

    // CPU add
    lbcrypto::Plaintext result;
    auto cAdd = cc->EvalAdd(c1, 2.0);

    cc->Decrypt(keys.secretKey, cAdd, &result);

    std::cout << "Add:\n";
    std::cout << "Result " << result;
    //result2->SetLength(batchSize);
    std::cout << "Result GPU " << resultGPU;

    auto cSub = cc->EvalAdd(c1, -2.0);

    cc->Decrypt(keys.secretKey, cSub, &result);

    FIDESlib::CKKS::RawCipherText raw_res2;
    GPUct2.store(GPUcc, raw_res2);
    auto cResGPU2 = c3->Clone();
    GetOpenFHECipherText(cResGPU2, raw_res2);
    lbcrypto::Plaintext resultGPU2;
    cc->Decrypt(keys.secretKey, cResGPU2, &resultGPU2);
    std::cout << "Sub:\n";
    std::cout << "Result " << result;
    //result2->SetLength(batchSize);
    std::cout << "Result GPU " << resultGPU2;

    ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);
    ASSERT_EQ_CIPHERTEXT(cSub, cResGPU2);
}

TEST_P(OpenFHEInterfaceTest, ExtractContextRunNTT) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);

    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};

    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x3 = {0.0};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
    lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

    std::cout << "Input x1: " << ptxt1 << std::endl;
    std::cout << "Input x2: " << ptxt2 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
    auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::RawPlainText raw2 = FIDESlib::CKKS::GetRawPlainText(cc, ptxt2);

    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
    FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw1);
    FIDESlib::CKKS::Plaintext GPUpt2(GPUcc, raw2);

    c1->m_elements[0].SwitchFormat();
    //c1->m_elements[0].SwitchFormat();

    GPUct1.c0.INTT<ALGO_NATIVE>(1);
    //GPUct1.c0.NTT();

    FIDESlib::CKKS::RawCipherText raw_res1;
    GPUct1.store(GPUcc, raw_res1);
    auto cResGPU(c3);
    GetOpenFHECipherText(cResGPU, raw_res1);

    /*
        for(auto & k :c1->m_elements[0].m_vectors[0].m_values->m_data){
            k.m_value = FIDESlib::modprod(k.m_value, FIDESlib::host_constants.N, FIDESlib::host_constants.primes[0]);
        }
         */

    /*
            std::vector<uint64_t> v;
            for(auto & k :cResGPU->m_elements[0].m_vectors[0].m_values->m_data){
                v.push_back(k.m_value);
            }
            FIDESlib::bit_reverse_vector(v);
            FIDESlib::nega_fft2_forPrime(v, true, 0);
            for(int k = 0; k < v.size(); ++k){
                cResGPU->m_elements[0].m_vectors[0].m_values->m_data[k].m_value = v[k];
            }
        */
    for (int j = 0; j < 1; ++j) {
        ASSERT_EQ(c1.get()->m_elements[j].m_vectors.size(), cResGPU.get()->m_elements[j].m_vectors.size());

        for (size_t i = 0; i < c1.get()->m_elements[j].m_vectors.size(); ++i) {
            std::cout << "i = " << i << ", j = " << j << std::endl;
            ASSERT_EQ(c1.get()->m_elements[j].m_vectors[i].m_params->m_ciphertextModulus, GPUcc.prime[i].p);

            ASSERT_EQ(c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.size(),
                      cResGPU.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.size());

            for (int k = 0; k < GPUcc.N; ++k)
                if (c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data[k] !=
                    cResGPU.get()->m_elements[j].m_vectors[i].m_values.get()->m_data[k]) {
                    std::cout << std::hex << i << ":" << k << " "
                              << c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data[k] << " "
                              << cResGPU.get()->m_elements[j].m_vectors[i].m_values.get()->m_data[k] << std::endl;
                }

            //std::sort(c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.begin(),c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.end() );
            //std::sort(cResGPU.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.begin(),cResGPU.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.end() );

            ASSERT_EQ(c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data,
                      cResGPU.get()->m_elements[j].m_vectors[i].m_values.get()->m_data);
        }
    }
}

TEST_P(OpenFHEInterfaceTest, ExtractContextShowPtMult) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);

    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};

    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x3 = {0.0};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
    lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

    std::cout << "Input x1: " << ptxt1 << std::endl;
    std::cout << "Input x2: " << ptxt2 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
    auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::RawPlainText raw2 = FIDESlib::CKKS::GetRawPlainText(cc, ptxt2);

    FIDESlib::CKKS::Ciphertext GPUct1_(GPUcc, raw1);
    FIDESlib::CKKS::Ciphertext GPUct2_(GPUcc, raw1);
    FIDESlib::CKKS::Plaintext GPUpt2(GPUcc, raw2);

    // CPU ptMult

    auto cMultNoRes = cc->EvalMult(c1, ptxt2);
    //auto cAux = cc->EvalMult(c1, c1);
    lbcrypto::Plaintext result;
    cc->Decrypt(keys.secretKey, cMultNoRes, &result);

    std::cout << "MultPt:\n";
    std::cout << "Result " << result;

    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cMult;
    if (GPUcc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
        cMult = cc->Rescale(cMultNoRes);
    } else {
        cMult = cc->EvalMult(cMultNoRes, ptxt2);
    }

    cc->Decrypt(keys.secretKey, cMult, &result);

    std::cout << "MultPt:\n";
    std::cout << "Result " << result;

    for (int batch : FIDESlib::Testing::batch_configs) {
        fideslibParams.batch = batch;
        std::cout << "Batch " << batch << std::endl;
        GPUcc.batch = batch;
        cudaDeviceSynchronize();
        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc);
        FIDESlib::CKKS::Ciphertext GPUct2(GPUcc);
        GPUct1.copy(GPUct1_);
        GPUct2.copy(GPUct2_);

        // GPU ptMult
        GPUct1.multPt(GPUpt2, false);
        GPUct2.multPt(GPUpt2, true);
        {
            FIDESlib::CKKS::RawCipherText raw_res1;
            GPUct1.store(GPUcc, raw_res1);
            auto cResGPU(c3->Clone());
            GetOpenFHECipherText(cResGPU, raw_res1);
            lbcrypto::Plaintext resultGPU;

            ASSERT_EQ_CIPHERTEXT(cResGPU, cMultNoRes);
            cc->RescaleInPlace(cResGPU);
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

            std::cout << "Result GPU with OpenFHE rescale " << resultGPU;
        }

        if (GPUcc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
            GPUct1.rescale();
        } else {
            GPUct1.multPt(GPUpt2, false);
            GPUct2.multPt(GPUpt2, false);
        }

        {
            FIDESlib::CKKS::RawCipherText raw_res1;
            GPUct1.store(GPUcc, raw_res1);
            auto cResGPU(c3->Clone());
            GetOpenFHECipherText(cResGPU, raw_res1);

            lbcrypto::Plaintext resultGPU;
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
            std::cout << "Result GPU with rescale " << resultGPU;
            ASSERT_EQ_CIPHERTEXT(cMult, cResGPU);
            {
                const auto cryptoParams =
                    std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
                for (int i = 0; i < GPUcc.L; ++i) {
                    for (int j = 0; j <= GPUcc.L; ++j) {
                        if (i < j) {
                            ASSERT_EQ(FIDESlib::host_global.q_inv[j][i], cryptoParams->GetqlInvModq(GPUcc.L - j)[i]);
                        }
                    }
                }

                for (int i = 0; i < GPUcc.L; ++i) {
                    for (int j = 0; j <= GPUcc.L; ++j) {
                        if (i < j) {
                            ASSERT_EQ(FIDESlib::host_global.QlQlInvModqlDivqlModq[j][i],
                                      cryptoParams->GetQlQlInvModqlDivqlModq(GPUcc.L - j)[i]);
                        }
                    }
                }
            }

            FIDESlib::CKKS::RawCipherText raw_res2;
            GPUct2.store(GPUcc, raw_res2);
            auto cResGPU2(c3->Clone());
            GetOpenFHECipherText(cResGPU2, raw_res2);

            lbcrypto::Plaintext resultGPU2;
            cc->Decrypt(keys.secretKey, cResGPU2, &resultGPU2);

            std::cout << "Result GPU with fused ptmult" << resultGPU2;

            ASSERT_EQ_CIPHERTEXT(cMult, cResGPU2);
        }
    }
}

TEST_P(OpenFHEInterfaceTest, InitializeOpenFHE) {
    FIDESlib::CKKS::Context GPUcc{fideslibParams, generalTestParams.GPUs};

    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

    cc->EvalMultKeyGen(keys.secretKey);

    cc->EvalRotateKeyGen(keys.secretKey, {1, -2});

    // Step 3: Encoding and encryption of inputs

    // Inputs
    // vector of c1 and c2, for loop running of evalAdd over vectors
    // will need to make it multithreaded

    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    std::cout << "Input x1: " << ptxt1 << std::endl;
    std::cout << "Input x2: " << ptxt2 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto cAdd = cc->EvalAdd(c1, c2);
    lbcrypto::Plaintext result;
    std::cout.precision(8);
    std::cout << std::endl << "Results of homomorphic computations: " << std::endl;

    cc->Decrypt(keys.secretKey, cAdd, &result);
    result->SetLength(generalTestParams.batchSize);
    std::cout << "x1 = " << result;
    std::cout << "Estimated precision in bits: " << result->GetLogPrecision() << std::endl;
}

TEST_P(OpenFHEInterfaceTest, MultScalar) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, 0);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c3 = cc->Encrypt(keys.publicKey, ptxt1);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

    FIDESlib::CKKS::Ciphertext GPUct1_(GPUcc, raw1);

    lbcrypto::Plaintext result;
    auto cAdd = cc->EvalMult(c1, std::pow((double)2.0, (double)-7.0));

    cc->Decrypt(keys.secretKey, cAdd, &result);

    std::cout << "Mult:\n";
    std::cout << "Result " << result;

    for (int batch : FIDESlib::Testing::batch_configs) {
        fideslibParams.batch = batch;
        std::cout << "Batch " << batch << std::endl;
        GPUcc.batch = batch;

        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc);
        GPUct1.copy(GPUct1_);

        GPUct1.multScalar(std::pow((double)2.0, (double)-7.0), false);

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1.store(GPUcc, raw_res1);
        auto cResGPU(c3);

        GetOpenFHECipherText(cResGPU, raw_res1);
        lbcrypto::Plaintext resultGPU;
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

        std::cout << "Result GPU " << resultGPU;

        CudaCheckErrorMod;
        ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEInterfaceTest, Mult) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    kskEval.Initialize(GPUcc, rawKskEval);
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x3 = {0.0};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
    lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

    std::cout << "Input x1: " << ptxt1 << std::endl;
    std::cout << "Input x2: " << ptxt2 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
    auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

    lbcrypto::Plaintext result;
    auto cAdd = cc->EvalMult(c1, c2);

    cc->Decrypt(keys.secretKey, cAdd, &result);

    std::cout << "Mult:\n";
    std::cout << "Result " << result;

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);
    FIDESlib::CKKS::Ciphertext GPUct1_(GPUcc, raw1);
    FIDESlib::CKKS::Ciphertext GPUct2_(GPUcc, raw2);

    for (int batch : FIDESlib::Testing::batch_configs) {
        fideslibParams.batch = batch;
        std::cout << "Batch " << batch << std::endl;
        GPUcc.batch = batch;
        cudaDeviceSynchronize();

        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc);
        GPUct1.copy(GPUct1_);

        GPUct1.mult(GPUct2_, kskEval, false);

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1.store(GPUcc, raw_res1);
        auto cResGPU(c3);

        GetOpenFHECipherText(cResGPU, raw_res1);
        lbcrypto::Plaintext resultGPU;
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

        std::cout << "Result GPU " << resultGPU;

        CudaCheckErrorMod;
        ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEInterfaceTest, Square) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x3 = {0.0};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
    lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

    std::cout << "Input x1: " << ptxt1 << std::endl;
    std::cout << "Input x2: " << ptxt2 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

    FIDESlib::CKKS::Ciphertext GPUct1_(GPUcc, raw1);

    lbcrypto::Plaintext result;
    auto cAdd = cc->EvalSquare(c1);

    cc->Decrypt(keys.secretKey, cAdd, &result);

    std::cout << "Mult:\n";
    std::cout << "Result " << result;

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);

    kskEval.Initialize(GPUcc, rawKskEval);

    for (int batch : FIDESlib::Testing::batch_configs) {
        fideslibParams.batch = batch;
        std::cout << "Batch " << batch << std::endl;
        GPUcc.batch = batch;

        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc);
        GPUct1.copy(GPUct1_);
        GPUct1.square(kskEval, false);
        cudaDeviceSynchronize();

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1.store(GPUcc, raw_res1);
        auto cResGPU(c3);

        GetOpenFHECipherText(cResGPU, raw_res1);
        lbcrypto::Plaintext resultGPU;
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

        std::cout << "Result GPU " << resultGPU;

        CudaCheckErrorMod;
        ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEInterfaceTest, MultRescale) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x3 = {0.0};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
    lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

    std::cout << "Input x1: " << ptxt1 << std::endl;
    std::cout << "Input x2: " << ptxt2 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
    auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
    FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);

    lbcrypto::Plaintext result;
    auto cAdd = cc->EvalMult(c1, c2);
    cc->RescaleInPlace(cAdd);
    cc->Decrypt(keys.secretKey, cAdd, &result);

    std::cout << "Mult:\n";
    std::cout << "Result " << result;

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);

    kskEval.Initialize(GPUcc, rawKskEval);

    GPUct1.mult(GPUct2, kskEval, true);

    FIDESlib::CKKS::RawCipherText raw_res1;
    GPUct1.store(GPUcc, raw_res1);
    auto cResGPU(c3);

    GetOpenFHECipherText(cResGPU, raw_res1);
    lbcrypto::Plaintext resultGPU;
    cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

    std::cout << "Result GPU " << resultGPU;

    ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

    CudaCheckErrorMod;
}

TEST_P(OpenFHEInterfaceTest, Rotate) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    //cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, {1});

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    std::vector<double> x3 = {0.0};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    lbcrypto::Plaintext result;
    auto cAdd = cc->EvalRotate(c1, 1);

    cc->Decrypt(keys.secretKey, cAdd, &result);

    std::cout << "Rotate:\n";
    std::cout << "Result " << result;

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);

    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);

    kskEval.Initialize(GPUcc, rawKskEval);

    GPUct1.rotate(1, kskEval);

    FIDESlib::CKKS::RawCipherText raw_res1;
    GPUct1.store(GPUcc, raw_res1);
    auto cResGPU(c3);

    GetOpenFHECipherText(cResGPU, raw_res1);
    lbcrypto::Plaintext resultGPU;
    ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);
    cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

    std::cout << "Result GPU " << resultGPU;

    CudaCheckErrorMod;
}

TEST_P(OpenFHEInterfaceTest, Conjugate) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    //cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, {1});

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x3 = {0.0};

    auto FHE = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
    auto conjKey = FHE->ConjugateKeyGen(keys.secretKey);
    auto& evalKeyMap = cc->GetEvalAutomorphismKeyMap(keys.publicKey->GetKeyTag());
    evalKeyMap[GPUcc.N * 2 - 1] = conjKey;

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetConjugateKeySwitchKey(keys, cc);
    kskEval.Initialize(GPUcc, rawKskEval);
    FIDESlib::CKKS::Ciphertext GPUct2(GPUcc);
    GPUcc.AddRotationKey(2 * GPUcc.N - 1, std::move(kskEval));

    for (int i = 0; i <= GPUcc.L; ++i) {
        // Encoding as plaintexts
        lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, i);
        lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

        std::cout << "Input x1: " << ptxt1 << std::endl;

        // Encrypt the encoded vectors
        auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
        auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

        FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

        lbcrypto::Plaintext result;

        auto conj = FHE->Conjugate(c1, evalKeyMap);

        GPUct2.conjugate(GPUct1);

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct2.store(GPUcc, raw_res1);
        auto cResGPU(c3);

        GetOpenFHECipherText(cResGPU, raw_res1);
        lbcrypto::Plaintext resultGPU;
        ASSERT_EQ_CIPHERTEXT(conj, cResGPU);
        /*
        cc->Decrypt(keys.secretKey, conj, &result);
        std::cout << "Rotate:\n";
        std::cout << "Result " << result;
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
        std::cout << "Result GPU " << resultGPU;
*/
        CudaCheckErrorMod;
    }
    CudaCheckErrorMod;
}

TEST_P(OpenFHEInterfaceTest, HoistedRotate) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    //cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, {1, 2, 3, 4});

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    std::vector<double> x3 = {0.0};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto r1 = cc->Encrypt(keys.publicKey, ptxt3);
    auto r2 = cc->Encrypt(keys.publicKey, ptxt3);
    auto r3 = cc->Encrypt(keys.publicKey, ptxt3);
    auto r4 = cc->Encrypt(keys.publicKey, ptxt3);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, r1);
    FIDESlib::CKKS::Ciphertext GPUr1(GPUcc, raw2);
    FIDESlib::CKKS::Ciphertext GPUr2(GPUcc, raw2);
    FIDESlib::CKKS::Ciphertext GPUr3(GPUcc, raw2);
    FIDESlib::CKKS::Ciphertext GPUr4(GPUcc, raw2);

    lbcrypto::Plaintext result;
    auto cpu_r1 = cc->EvalRotate(c1, 1);
    auto cpu_r2 = cc->EvalRotate(c1, 2);
    auto cpu_r3 = cc->EvalRotate(c1, 3);
    auto cpu_r4 = cc->EvalRotate(c1, 4);

    std::cout << "Rotate:\n";
    cc->Decrypt(keys.secretKey, cpu_r1, &result);
    std::cout << "Result " << result;
    cc->Decrypt(keys.secretKey, cpu_r2, &result);
    std::cout << "Result " << result;
    cc->Decrypt(keys.secretKey, cpu_r3, &result);
    std::cout << "Result " << result;
    cc->Decrypt(keys.secretKey, cpu_r4, &result);
    std::cout << "Result " << result;

    FIDESlib::CKKS::KeySwitchingKey kskRot1(GPUcc);
    FIDESlib::CKKS::KeySwitchingKey kskRot2(GPUcc);
    FIDESlib::CKKS::KeySwitchingKey kskRot3(GPUcc);
    FIDESlib::CKKS::KeySwitchingKey kskRot4(GPUcc);

    FIDESlib::CKKS::RawKeySwitchKey rawkskRot1 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);
    FIDESlib::CKKS::RawKeySwitchKey rawkskRot2 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 2, cc);
    FIDESlib::CKKS::RawKeySwitchKey rawkskRot3 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 3, cc);
    FIDESlib::CKKS::RawKeySwitchKey rawkskRot4 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 4, cc);

    kskRot1.Initialize(GPUcc, rawkskRot1);
    kskRot2.Initialize(GPUcc, rawkskRot2);
    kskRot3.Initialize(GPUcc, rawkskRot3);
    kskRot4.Initialize(GPUcc, rawkskRot4);

    GPUct1.rotate_hoisted({&kskRot1, &kskRot2, &kskRot3, &kskRot4}, {1, 2, 3, 4}, {&GPUr1, &GPUr2, &GPUr3, &GPUct1});
    //GPUct1.rotate_hoisted({&kskRot1}, {1}, {&GPUr1});
    //GPUct1.rotate(2, kskRot2);

    FIDESlib::CKKS::RawCipherText raw_res1;

    auto cResGPU(c1);
    lbcrypto::Plaintext resultGPU;

    GPUr1.store(GPUcc, raw_res1);
    GetOpenFHECipherText(cResGPU, raw_res1);
    ASSERT_EQ_CIPHERTEXT(cpu_r1, cResGPU);
    cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
    std::cout << "Result GPU " << resultGPU;

    GPUr2.store(GPUcc, raw_res1);
    GetOpenFHECipherText(cResGPU, raw_res1);
    ASSERT_EQ_CIPHERTEXT(cpu_r2, cResGPU);
    cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
    std::cout << "Result GPU " << resultGPU;

    GPUr3.store(GPUcc, raw_res1);
    GetOpenFHECipherText(cResGPU, raw_res1);
    ASSERT_EQ_CIPHERTEXT(cpu_r3, cResGPU);
    cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
    std::cout << "Result GPU " << resultGPU;

    GPUct1.store(GPUcc, raw_res1);
    GetOpenFHECipherText(cResGPU, raw_res1);
    ASSERT_EQ_CIPHERTEXT(cpu_r4, cResGPU);
    cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
    std::cout << "Result GPU " << resultGPU;

    CudaCheckErrorMod;
}

TEST_P(OpenFHEInterfaceTest, MultAllLevels) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {4.0, 2.0, 1.3333333333, 1.0, 0.5, 0.3333333333, 0.25, 0.2};
    std::vector<double> x3 = {0.0};

    // Encoding as plaintexts

    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
    lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2_1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2_2 = cc->Encrypt(keys.publicKey, ptxt2);
    auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    kskEval.Initialize(GPUcc, rawKskEval);

    for (int i = 0; i < GPUcc.L - (GPUcc.rescaleTechnique == CKKS::Context::FLEXIBLEAUTOEXT); ++i) {
        ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, i);

        auto c2 = i % 2 == 0 ? c2_2 : c2_1;

        FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);
        FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);
        lbcrypto::Plaintext result;

        c1 = cc->EvalMult(c1, c2);
        cc->RescaleInPlace(c1);
        cc->Decrypt(keys.secretKey, c1, &result);

        std::cout << "Mult " << i << " levels used:\n";
        std::cout << "Result " << result;

        auto cResGPU(c3);

        GPUct1.mult(GPUct2, kskEval, GPUcc.rescaleTechnique == CKKS::Context::FIXEDMANUAL);

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1.store(GPUcc, raw_res1);

        GetOpenFHECipherText(cResGPU, raw_res1);

        lbcrypto::Plaintext resultGPU;

        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
        std::cout << "Result GPU " << resultGPU;
        ASSERT_EQ_CIPHERTEXT(c1, cResGPU);

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEInterfaceTest, RotateAllLevels) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    //cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, {1});

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    std::vector<double> x3 = {0.0};

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);
    kskEval.Initialize(GPUcc, rawKskEval);

    lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

    for (int i = 0; i <= GPUcc.L; ++i) {
        lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, i);
        // Encoding as plaintexts
        std::cout << "Input x1: " << ptxt1 << std::endl;

        // Encrypt the encoded vectors
        auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
        auto c3 = cc->Encrypt(keys.publicKey, ptxt3);
        FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

        lbcrypto::Plaintext result;
        auto cAdd = cc->EvalRotate(c1, 1);

        cc->Decrypt(keys.secretKey, cAdd, &result);

        std::cout << "Rotate " << i << " levels down:\n";
        std::cout << "Result " << result;

        GPUct1.rotate(1, kskEval);
        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1.store(GPUcc, raw_res1);
        auto cResGPU(c3);

        GetOpenFHECipherText(cResGPU, raw_res1);
        lbcrypto::Plaintext resultGPU;
        CudaCheckErrorMod;
        ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

        std::cout << "Result GPU " << resultGPU;
    }
}

TEST_P(OpenFHEInterfaceTest, SquareAllLevels) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    fideslibParams.batch = 3;
    std::cout << "Batch " << 3 << std::endl;
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);

    kskEval.Initialize(GPUcc, rawKskEval);

    for (int i = 0; i < GPUcc.L; ++i) {
        std::cout << "Dropped " << i << " levels.\n";
        // Encoding as plaintexts
        lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, i);

        std::cout << "Input x1: " << ptxt1 << std::endl;
        // Encrypt the encoded vectors
        auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
        auto c3 = cc->Encrypt(keys.publicKey, ptxt1);

        FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

        lbcrypto::Plaintext result;
        auto cAdd = cc->EvalSquare(c1);

        cc->Decrypt(keys.secretKey, cAdd, &result);

        std::cout << "Mult:\n";
        std::cout << "Result " << result;

        GPUct1.square(kskEval, false);

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1.store(GPUcc, raw_res1);
        auto cResGPU(c3);

        GetOpenFHECipherText(cResGPU, raw_res1);
        lbcrypto::Plaintext resultGPU;
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

        std::cout << "Result GPU " << resultGPU;

        CudaCheckErrorMod;
        ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEInterfaceTest, HoistedRotateAllLevels) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    //cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, {1, 2, 3, 4});

    fideslibParams.batch = 3;
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    std::vector<double> x3 = {0.0};
    lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

    FIDESlib::CKKS::KeySwitchingKey kskRot1(GPUcc);
    FIDESlib::CKKS::KeySwitchingKey kskRot2(GPUcc);
    FIDESlib::CKKS::KeySwitchingKey kskRot3(GPUcc);
    FIDESlib::CKKS::KeySwitchingKey kskRot4(GPUcc);

    FIDESlib::CKKS::RawKeySwitchKey rawkskRot1 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);
    FIDESlib::CKKS::RawKeySwitchKey rawkskRot2 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 2, cc);
    FIDESlib::CKKS::RawKeySwitchKey rawkskRot3 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 3, cc);
    FIDESlib::CKKS::RawKeySwitchKey rawkskRot4 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 4, cc);

    kskRot1.Initialize(GPUcc, rawkskRot1);
    kskRot2.Initialize(GPUcc, rawkskRot2);
    kskRot3.Initialize(GPUcc, rawkskRot3);
    kskRot4.Initialize(GPUcc, rawkskRot4);

    for (int i = 0; i <= GPUcc.L; ++i) {
        std::cout << "Dropped levels: " << i << std::endl;
        // Encoding as plaintexts
        lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, i);

        std::cout << "Input x1: " << ptxt1 << std::endl;

        // Encrypt the encoded vectors
        auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
        auto r1 = cc->Encrypt(keys.publicKey, ptxt1);
        auto r2 = cc->Encrypt(keys.publicKey, ptxt1);
        auto r3 = cc->Encrypt(keys.publicKey, ptxt1);
        auto r4 = cc->Encrypt(keys.publicKey, ptxt1);

        FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);

        FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, r1);
        FIDESlib::CKKS::Ciphertext GPUr1(GPUcc, raw2);
        FIDESlib::CKKS::Ciphertext GPUr2(GPUcc, raw2);
        FIDESlib::CKKS::Ciphertext GPUr3(GPUcc, raw2);
        FIDESlib::CKKS::Ciphertext GPUr4(GPUcc, raw2);

        lbcrypto::Plaintext result;
        auto cpu_r1 = cc->EvalRotate(c1, 1);
        auto cpu_r2 = cc->EvalRotate(c1, 2);
        auto cpu_r3 = cc->EvalRotate(c1, 3);
        auto cpu_r4 = cc->EvalRotate(c1, 4);

        std::cout << "Rotate:\n";
        cc->Decrypt(keys.secretKey, cpu_r1, &result);
        std::cout << "Result " << result;
        cc->Decrypt(keys.secretKey, cpu_r2, &result);
        std::cout << "Result " << result;
        cc->Decrypt(keys.secretKey, cpu_r3, &result);
        std::cout << "Result " << result;
        cc->Decrypt(keys.secretKey, cpu_r4, &result);
        std::cout << "Result " << result;

        GPUct1.rotate_hoisted({&kskRot1, &kskRot2, &kskRot3, &kskRot4}, {1, 2, 3, 4},
                              {&GPUr1, &GPUr2, &GPUr3, &GPUct1});
        //GPUct1.rotate_hoisted({&kskRot1}, {1}, {&GPUr1});
        //GPUct1.rotate(2, kskRot2);

        FIDESlib::CKKS::RawCipherText raw_res1;

        auto cResGPU(c1);
        lbcrypto::Plaintext resultGPU;

        GPUr1.store(GPUcc, raw_res1);
        GetOpenFHECipherText(cResGPU, raw_res1);
        ASSERT_EQ_CIPHERTEXT(cpu_r1, cResGPU);
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
        std::cout << "Result GPU " << resultGPU;

        GPUr2.store(GPUcc, raw_res1);
        GetOpenFHECipherText(cResGPU, raw_res1);
        ASSERT_EQ_CIPHERTEXT(cpu_r2, cResGPU);
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
        std::cout << "Result GPU " << resultGPU;

        GPUr3.store(GPUcc, raw_res1);
        GetOpenFHECipherText(cResGPU, raw_res1);
        ASSERT_EQ_CIPHERTEXT(cpu_r3, cResGPU);
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
        std::cout << "Result GPU " << resultGPU;

        GPUct1.store(GPUcc, raw_res1);
        GetOpenFHECipherText(cResGPU, raw_res1);
        ASSERT_EQ_CIPHERTEXT(cpu_r4, cResGPU);
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
        std::cout << "Result GPU " << resultGPU;

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEInterfaceTest, MatVecPt) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    // cc->Enable(KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};

    ///// PROBAR /////
    std::vector<double> x1 = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    std::vector<double> x[8] = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

    std::cout << "Input x1: " << ptxt1 << std::endl;
    //std::cout << "Input x2: " << ptxt2 << std::endl;
    // Encrypt the encoded vectors
    std::vector<lbcrypto::Plaintext> ptxt;
    for (int i = 0; i < 8; ++i) {
        ptxt.emplace_back(cc->MakeCKKSPackedPlaintext(x[i]));
    }
    std::cout << "Input x2: " << ptxt[0] << std::endl;
    using Cipher = lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>;
    std::vector<Cipher> ct;
    for (int i = 0; i < 8; ++i)
        ct.emplace_back(cc->Encrypt(keys.publicKey, ptxt1));

    if (1) {
        std::vector<FIDESlib::CKKS::Ciphertext> GPUct;
        for (int i = 0; i < 8; ++i) {
            FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ct[i]);
            GPUct.emplace_back(GPUcc, raw1);
        }

        std::vector<FIDESlib::CKKS::Plaintext> GPUpt;
        for (int i = 0; i < 8; ++i) {
            FIDESlib::CKKS::RawPlainText raw2 = FIDESlib::CKKS::GetRawPlainText(cc, ptxt[i]);
            GPUpt.emplace_back(GPUcc, raw2);
        }

        //////////////////////////////////////////////
        cc->GetScheme()->EvalMultInPlace(ct[0], ptxt[0]);
        for (int i = 1; i < 8; ++i) {
            //cc->EvalMultInPlace(ct[i], ptxt[i]);
            cc->GetScheme()->EvalMultInPlace(ct[i], ptxt[i]);
            cc->EvalAddInPlace(ct[0], ct[i]);
        }
        //cc->RescaleInPlace(ct[0]);

        lbcrypto::Plaintext resultCPU;
        cc->Decrypt(keys.secretKey, ct[0], &resultCPU);
        std::cout << "Result CPU " << resultCPU;
        GPUct[0].multPt(GPUpt[0], false);
        //GPUct[0].rescale();

        for (int i = 1; i < 8; ++i) {
            GPUct[i].multPt(GPUpt[i], false);
            //GPUct[i].rescale();
            GPUct[0].add(GPUct[i]);
        }
        //GPUct[0].rescale();
        {
            FIDESlib::CKKS::RawCipherText raw1;
            GPUct[0].store(GPUcc, raw1);

            //cc->RescaleInPlace(ct[1]); // The reference cyphertext has to be rescaled from or it bugs out.
            Cipher cResGPU(ct[1] /*cc->Encrypt(keys.publicKey, ptxt1)*/);
            FIDESlib::CKKS::GetOpenFHECipherText(cResGPU, raw1);
            lbcrypto::Plaintext resultGPU;
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
            std::cout << "Result GPU " << resultGPU;

            ASSERT_EQ_CIPHERTEXT(cResGPU, ct[0]);
        }
    }
    CudaCheckErrorMod;
}

TEST_P(OpenFHEInterfaceTest, MatVecPtScalar) {
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    // cc->Enable(KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};

    ///// PROBAR /////
    std::vector<double> x1 = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    std::vector<double> x[8] = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

    std::cout << "Input x1: " << ptxt1 << std::endl;
    //std::cout << "Input x2: " << ptxt2 << std::endl;

    using Cipher = lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>;
    std::vector<Cipher> ct;
    for (int i = 0; i < 8; ++i)
        ct.emplace_back(cc->Encrypt(keys.publicKey, ptxt1));

    if (1) {
        std::vector<FIDESlib::CKKS::Ciphertext> GPUct;
        for (int i = 0; i < 8; ++i) {
            FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ct[i]);
            GPUct.emplace_back(GPUcc, raw1);
        }
        /*
        std::vector<FIDESlib::CKKS::Plaintext> GPUpt;
        for (int i = 0; i < 8; ++i) {
            FIDESlib::CKKS::RawPlainText raw2 = FIDESlib::CKKS::GetRawPlainText(cc, ptxt[i]);
            GPUpt.emplace_back(GPUcc, raw2);
        }
*/
        //////////////////////////////////////////////
        cc->GetScheme()->EvalMultInPlace(ct[0], x[0][0]);
        for (int i = 1; i < 8; ++i) {
            //cc->EvalMultInPlace(ct[i], ptxt[i]);
            cc->GetScheme()->EvalMultInPlace(ct[i], x[i][0]);
            cc->EvalAddInPlace(ct[0], ct[i]);
        }
        //cc->RescaleInPlace(ct[0]);

        lbcrypto::Plaintext resultCPU;
        cc->Decrypt(keys.secretKey, ct[0], &resultCPU);
        std::cout << "Result CPU " << resultCPU;
        GPUct[0].multScalar(x[0][0], false);
        //GPUct[0].rescale();

        for (int i = 1; i < 8; ++i) {
            GPUct[i].multScalar(x[i][0], false);
            //GPUct[i].rescale();
            GPUct[0].add(GPUct[i]);
        }
        //GPUct[0].rescale();
        {
            FIDESlib::CKKS::RawCipherText raw1;
            GPUct[0].store(GPUcc, raw1);

            //cc->RescaleInPlace(ct[1]); // The reference cyphertext has to be rescaled from or it bugs out.
            Cipher cResGPU(ct[1] /*cc->Encrypt(keys.publicKey, ptxt1)*/);
            FIDESlib::CKKS::GetOpenFHECipherText(cResGPU, raw1);
            lbcrypto::Plaintext resultGPU;
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
            std::cout << "Result GPU " << resultGPU;

            ASSERT_EQ_CIPHERTEXT(cResGPU, ct[0]);
        }
    }
    CudaCheckErrorMod;
}

// Define the parameter sets
INSTANTIATE_TEST_SUITE_P(OpenFHEInterfaceTests, OpenFHEInterfaceTest, testing::Values(TTALL64));

class OpenFHEBootstrapTest : public GeneralParametrizedTest {};

TEST_P(OpenFHEBootstrapTest, ApproxModEval) {
    for (auto& i : cached_cc) {
        i.second.first->ClearEvalAutomorphismKeys();
        i.second.first->ClearEvalMultKeys();
        if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
            std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)
                ->m_bootPrecomMap.clear();
    }
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    // Encoding as plaintexts
    int slots = cc->GetRingDimension() / 2;
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, fideslibParams.L - 1, nullptr, slots);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

    std::cout << "Input x1: " << ptxt1 << std::endl;
    lbcrypto::Plaintext result;
    std::cout << "Setup Bootstrap" << std::endl;
    cc->EvalBootstrapSetup({5, 5}, {0, 0}, slots);

    std::cout << "Generate keys" << std::endl;
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};

    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);
    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    // coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475 0.00428562 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264 0.00211163 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128 -0.00695249 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494 -0.000989932 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05 -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10 -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
    std::cout << "Run bootstrap" << std::endl;
    auto [ctxtEnc, ctxtEnc_unused] = cc->GetScheme()->EvalBootstrapDensePartial(c1);

    constexpr bool previous = true;
    FIDESlib::CKKS::RawCipherText raw1;
    FIDESlib::CKKS::RawCipherText raw2;
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc);
    FIDESlib::CKKS::Ciphertext GPUct2(GPUcc);
    if constexpr (previous) {
        raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ctxtEnc);
        GPUct1.load(raw1);
        FIDESlib::CKKS::Ciphertext aux(GPUcc);
        cudaDeviceSynchronize();

        aux.conjugate(GPUct1);
        cudaDeviceSynchronize();
        cudaDeviceSynchronize();
        GPUct2.sub(GPUct1, aux);
        cudaDeviceSynchronize();
        GPUct1.add(aux);
        cudaDeviceSynchronize();

        multMonomial(GPUct2, 3 * 2 * GPUcc.N / 4);
        cudaDeviceSynchronize();
        //ctxt.copy(ctxtEncI);
        cudaDeviceSynchronize();
        if (GPUcc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
            GPUct1.rescale();
            GPUct2.rescale();
        }
    }

    auto FHE = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
    auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEnc->GetKeyTag());
    auto conj = FHE->Conjugate(ctxtEnc, evalKeyMap);
    //auto ctxtEncI = ctxtEnc;
    auto ctxtEncI = cc->EvalSub(ctxtEnc, conj);
    cc->EvalAddInPlace(ctxtEnc, conj);

    cc->GetScheme()->MultByMonomialInPlace(ctxtEncI, 3 * GPUcc.N * 2 / 4);

    if (ctxtEnc->GetNoiseScaleDeg() > 1) {
        cc->ModReduceInPlace(ctxtEnc);
        cc->ModReduceInPlace(ctxtEncI);
    }

    if constexpr (!previous) {
        raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ctxtEnc);
        GPUct1.load(raw1);
        raw2 = FIDESlib::CKKS::GetRawCipherText(cc, ctxtEncI);
        GPUct1.load(raw2);
    }

    {
        lbcrypto::Plaintext result;
        cc->Decrypt(keys.secretKey, ctxtEnc, &result);
        lbcrypto::Plaintext result2;
        cc->Decrypt(keys.secretKey, ctxtEncI, &result2);

        std::cout << "Starting point after coeffs to slots:\n";
        // std::cout << "Result " << result;
        // std::cout << "Result2 " << result2;
        {
            FIDESlib::CKKS::RawCipherText raw_res1;
            GPUct1.store(GPUcc, raw_res1);
            auto cResGPU = c2->Clone();
            GetOpenFHECipherText(cResGPU, raw_res1);
            lbcrypto::Plaintext resultGPU;
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
            //    std::cout << "Result GPU after cheby" << resultGPU;
            CudaCheckErrorMod;
            ASSERT_ERROR_OK(result, resultGPU);
            ASSERT_EQ_CIPHERTEXT(ctxtEnc, cResGPU);
        }
        {
            FIDESlib::CKKS::RawCipherText raw_res1;
            GPUct2.store(GPUcc, raw_res1);
            auto cResGPU = c2->Clone();
            GetOpenFHECipherText(cResGPU, raw_res1);
            lbcrypto::Plaintext resultGPU;
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
            //    std::cout << "Result GPU after cheby" << resultGPU;
            CudaCheckErrorMod;
            ASSERT_ERROR_OK(result2, resultGPU);
            ASSERT_EQ_CIPHERTEXT(ctxtEncI, cResGPU);
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Evaluate Chebyshev series for the sine wave
    ctxtEnc = cc->EvalChebyshevSeries(ctxtEnc, GPUcc.GetCoeffsChebyshev(), -1.0, 1.0);
    ctxtEncI = cc->EvalChebyshevSeries(ctxtEncI, GPUcc.GetCoeffsChebyshev(), -1.0, 1.0);

    /*
        for (const auto& i : ctxtEnc->GetElements().at(0).m_vectors) {
            std::cout << "(" << i.m_params->GetModulus() << ", " << i.m_values->at(0) << ") ";
        }
        std::cout << std::endl;

        for (const auto& i : ctxtEncI->GetElements().at(0).m_vectors) {
            std::cout << "(" << i.m_params->GetModulus() << ", " << i.m_values->at(0) << ") ";
        }
        std::cout << std::endl;
        */
    // Double-angle iterations
    if (true
        //(cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY) || (cryptoParams->GetSecretKeyDist() == SPARSE_TERNARY)
    ) {
        if (false) {
            // cryptoParams->GetScalingTechnique() != FIXEDMANUAL) {
            cc->GetScheme()->ModReduceInternalInPlace(ctxtEnc, lbcrypto::BASE_NUM_LEVELS_TO_DROP);
            cc->GetScheme()->ModReduceInternalInPlace(ctxtEncI, lbcrypto::BASE_NUM_LEVELS_TO_DROP);
        }
        uint32_t numIter;
        /*
              if (cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY)
                numIter = R_UNIFORM;
            else
                numIter = R_SPARSE;
                */
        numIter = GPUcc.GetDoubleAngleIts();
        lbcrypto::FHECKKSRNS::ApplyDoubleAngleIterations(ctxtEnc, numIter);
        lbcrypto::FHECKKSRNS::ApplyDoubleAngleIterations(ctxtEncI, numIter);

        for (const auto& i : ctxtEnc->GetElements().at(0).m_vectors) {
            std::cout << "(" << i.m_params->GetModulus() << ", " << i.m_values->at(0) << ") ";
        }
        std::cout << std::endl;

        for (const auto& i : ctxtEncI->GetElements().at(0).m_vectors) {
            std::cout << "(" << i.m_params->GetModulus() << ", " << i.m_values->at(0) << ") ";
        }
        std::cout << std::endl;
    }

    cc->GetScheme()->MultByMonomialInPlace(ctxtEncI, cc->GetRingDimension() / 2);
    cc->EvalAddInPlace(ctxtEnc, ctxtEncI);

    // scale the message back up after Chebyshev interpolation
    cc->GetScheme()->MultByIntegerInPlace(ctxtEnc, 1.0);

    for (const auto& i : ctxtEnc->GetElements().at(0).m_vectors) {
        std::cout << "(" << i.m_params->GetModulus() << ", " << i.m_values->at(0) << ") ";
    }
    std::cout << std::endl;

    ////////////////////////////////////////////////////////////////////

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);

    kskEval.Initialize(GPUcc, rawKskEval);

    for (int batch : FIDESlib::Testing::batch_configs) {
        fideslibParams.batch = batch;
        std::cout << "Batch " << batch << std::endl;
        GPUcc.batch = batch;
        cudaDeviceSynchronize();
        FIDESlib::CKKS::Ciphertext GPUct1_(GPUcc);
        FIDESlib::CKKS::Ciphertext GPUct2_(GPUcc);
        GPUct1_.copy(GPUct1);
        GPUct2_.copy(GPUct2);
        cudaDeviceSynchronize();

        FIDESlib::CKKS::approxModReduction(GPUct1_, GPUct2_, kskEval, 1.0);

        {
            lbcrypto::Plaintext result;
            cc->Decrypt(keys.secretKey, ctxtEnc, &result);

            FIDESlib::CKKS::RawCipherText raw_res1;
            GPUct1_.store(GPUcc, raw_res1);
            auto cResGPU = c2->Clone();
            GetOpenFHECipherText(cResGPU, raw_res1);
            lbcrypto::Plaintext resultGPU;
            CudaCheckErrorMod;
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

            // std::cout << "Result GPU after cheby" << resultGPU->GetStringValue().substr(0, 120);
            CudaCheckErrorMod;
            ASSERT_ERROR_OK(result, resultGPU);
            // ASSERT_EQ_CIPHERTEXT(ctxtEnc, cResGPU);
        }

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEBootstrapTest, ApproxModEvalSparse) {
    for (auto& i : cached_cc) {
        i.second.first->ClearEvalAutomorphismKeys();
        i.second.first->ClearEvalMultKeys();
        if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
            std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)
                ->m_bootPrecomMap.clear();
    }
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    lbcrypto::Plaintext result;
    std::cout << "Setup Bootstrap" << std::endl;
    cc->EvalBootstrapSetup({1, 1}, {0, 0}, 8);

    std::cout << "Generate keys" << std::endl;
    cc->EvalBootstrapKeyGen(keys.secretKey, 8);

    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, 8, GPUcc);
    //GPUcc.AddBootPrecomputation(8, )

    {
        FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
        FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
        kskEval.Initialize(GPUcc, rawKskEval);
        GPUcc.AddEvalKey(std::move(kskEval));
    }

    // coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475 0.00428562 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264 0.00211163 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128 -0.00695249 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494 -0.000989932 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05 -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10 -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
    //FIDESlib::CKKS::Bootstrap(GPUct1, 8);

    auto raised =
        std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->EvalBootstrapSetupOnly(c1, 1, 0);

    const std::shared_ptr<lbcrypto::CKKSBootstrapPrecom> precom =
        std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.find(8)->second;
    bool isLTBootstrap = (precom->m_paramsEnc[lbcrypto::CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1) &&
                         (precom->m_paramsDec[lbcrypto::CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1);
    auto ctxtEnc = (isLTBootstrap) ? std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)
                                         ->EvalLinearTransform(precom->m_U0hatTPre, raised)
                                   : std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)
                                         ->EvalCoeffsToSlots(precom->m_U0hatTPreFFT, raised);

    auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEnc->GetKeyTag());
    auto conj = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->Conjugate(ctxtEnc, evalKeyMap);
    cc->EvalAddInPlace(ctxtEnc, conj);

    const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    if (cryptoParams->GetScalingTechnique() == lbcrypto::FIXEDMANUAL) {
        while (ctxtEnc->GetNoiseScaleDeg() > 1) {
            cc->ModReduceInPlace(ctxtEnc);
        }
    } else {
        if (ctxtEnc->GetNoiseScaleDeg() == 2) {
            cc->GetScheme()->ModReduceInternalInPlace(ctxtEnc, lbcrypto::BASE_NUM_LEVELS_TO_DROP);
        }
    }
    /*{
            FIDESlib::CKKS::RawCipherText raw_res1;
            GPUct1.store(GPUcc, raw_res1);
            GetOpenFHECipherText(ctxtEnc, raw_res1);
        }*/
    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ctxtEnc);
    FIDESlib::CKKS::Ciphertext GPUct1_(GPUcc, raw1);
    {
        lbcrypto::Plaintext result;
        cc->Decrypt(keys.secretKey, ctxtEnc, &result);

        std::cout << "Starting point after coeffs to slots:\n";
        std::cout << "Result " << result;
    }
    /*
        FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ctxtEnc);
        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
        */
    std::cout << "coefficients ";
    for (auto& i : GPUcc.GetCoeffsChebyshev())
        std::cout << i << " ";

    // coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475 0.00428562 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264 0.00211163 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128 -0.00695249 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494 -0.000989932 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05 -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10 -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
    std::cout << std::endl;
    // Evaluate Chebyshev series for the sine wave
    ctxtEnc = cc->EvalChebyshevSeries(ctxtEnc, GPUcc.GetCoeffsChebyshev(), -1.0, 1.0);

    /*{
            lbcrypto::Plaintext result;
            cc->Decrypt(keys.secretKey, ctxtEnc, &result);

            std::cout << "After Chebyshev:\n";
            std::cout << "Result " << result;
        }*/

    // Double-angle iterations
    if (true  //(cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY) ||
        //(cryptoParams->GetSecretKeyDist() == SPARSE_TERNARY)
    ) {
        if (false  //cryptoParams->GetScalingTechnique() != FIXEDMANUAL
        ) {
            //algo->ModReduceInternalInPlace(ctxtEnc, BASE_NUM_LEVELS_TO_DROP);
        }
        uint32_t numIter;
        //if (cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY)
        //    numIter = R_UNIFORM;
        //else
        //    numIter = R_SPARSE;

        numIter = GPUcc.GetDoubleAngleIts();
        lbcrypto::FHECKKSRNS::ApplyDoubleAngleIterations(ctxtEnc, numIter);
    }

    // scale the message back up after Chebyshev interpolation
    cc->GetScheme()->MultByIntegerInPlace(ctxtEnc, 1.0);

    cc->Decrypt(keys.secretKey, ctxtEnc, &result);

    std::cout << "After ApproxModEval CPU:\n";
    std::cout << "Result " << result;

    for (int batch : FIDESlib::Testing::batch_configs) {
        fideslibParams.batch = batch;
        std::cout << "Batch " << batch << std::endl;
        GPUcc.batch = batch;
        cudaDeviceSynchronize();
        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc);
        GPUct1.copy(GPUct1_);

        FIDESlib::CKKS::approxModReductionSparse(GPUct1, GPUcc.GetEvalKey(), 1.0);

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1.store(GPUcc, raw_res1);

        {

            auto cResGPU = c2->Clone();
            GetOpenFHECipherText(cResGPU, raw_res1);
            lbcrypto::Plaintext resultGPU;
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
            std::cout << "Result GPU after ApproxModEval" << resultGPU;
            CudaCheckErrorMod;
            ASSERT_ERROR_OK(result, resultGPU);
            //ASSERT_EQ_CIPHERTEXT(ctxtEnc, cResGPU);
        }

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEBootstrapTest, LinearTransform) {
    for (auto& i : cached_cc) {
        i.second.first->ClearEvalAutomorphismKeys();
        i.second.first->ClearEvalMultKeys();
        if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
            std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)
                ->m_bootPrecomMap.clear();
    }
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    const int slots = 32;
    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, slots);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    lbcrypto::Plaintext result;
    std::cout << "Setup Bootstrap" << std::endl;
    cc->EvalBootstrapSetup({1, 1}, {0, 0}, slots);

    std::cout << "Generate keys" << std::endl;

    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

    // coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475 0.00428562 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264 0.00211163 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128 -0.00695249 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494 -0.000989932 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05 -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10 -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
    std::cout << "Run bootstrap start" << std::endl;
    auto FHE = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
    auto raised = FHE->EvalBootstrapSetupOnly(c1, 1, 0);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, raised);
    FIDESlib::CKKS::Ciphertext GPUct1_(GPUcc, raw1);
    {
        lbcrypto::Plaintext result;
        cc->Decrypt(keys.secretKey, raised, &result);

        std::cout << "Before linear transform:\n";
        std::cout << "Result " << result;

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1_.store(GPUcc, raw_res1);
        auto cResGPU = c2->Clone();
        GetOpenFHECipherText(cResGPU, raw_res1);
        lbcrypto::Plaintext resultGPU;
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
        std::cout << "Result GPU " << resultGPU;
        CudaCheckErrorMod;
        ASSERT_ERROR_OK(result, resultGPU);
        ASSERT_EQ_CIPHERTEXT(raised, cResGPU);
    }
    /*
        {
            {
                auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(raised->GetKeyTag());
                auto conj = FHE->Conjugate(raised, evalKeyMap);
                cc->EvalAddInPlace(raised, conj);
            }
            lbcrypto::Plaintext result;
            cc->Decrypt(keys.secretKey, raised, &result);

            std::cout << "Before linear transform:\n";
            std::cout << "Result " << result;

            //GPUct1.addPt(GPUcc.GetBootPrecomputation(8).A[0]);
            FIDESlib::CKKS::RawCipherText raw_res1;
            GPUct1.store(GPUcc, raw_res1);
            auto cResGPU = c2->Clone();
            GetOpenFHECipherText(cResGPU, raw_res1);

            auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cResGPU->GetKeyTag());
            auto conj = FHE->Conjugate(cResGPU, evalKeyMap);
            cc->EvalAddInPlace(cResGPU, conj);
            lbcrypto::Plaintext resultGPU;
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
            std::cout << "Result GPU " << resultGPU;
            CudaCheckErrorMod;
            ASSERT_EQ_CIPHERTEXT(raised, cResGPU);
        }
        */
    std::cout << "Run linear transform" << std::endl;

    auto ctxtEnc = FHE->EvalLinearTransform(FHE->m_bootPrecomMap.at(slots)->m_U0hatTPre, raised);

    cc->RescaleInPlace(ctxtEnc);

    auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEnc->GetKeyTag());
    auto conj = FHE->Conjugate(ctxtEnc, evalKeyMap);
    cc->EvalAddInPlace(ctxtEnc, conj);

    cc->Decrypt(keys.secretKey, ctxtEnc, &result);

    std::cout << "After linear transform:\n";
    std::cout << "Result " << result;

    for (int batch : FIDESlib::Testing::batch_configs) {
        fideslibParams.batch = batch;
        std::cout << "Batch " << batch << std::endl;
        GPUcc.batch = batch;
        cudaDeviceSynchronize();
        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc);
        GPUct1.copy(GPUct1_);

        std::cout << "Run linear transform GPU" << std::endl;
        /*
        FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
        FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
        kskEval.Initialize(GPUcc, rawKskEval);
        */

        FIDESlib::CKKS::EvalLinearTransform(GPUct1, slots, false);

        {
            FIDESlib::CKKS::RawCipherText raw_res1;
            GPUct1.store(GPUcc, raw_res1);
            auto cResGPU = c2->Clone();
            GetOpenFHECipherText(cResGPU, raw_res1);

            {
                auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cResGPU->GetKeyTag());
                auto conj = FHE->Conjugate(cResGPU, evalKeyMap);
                cc->EvalAddInPlace(cResGPU, conj);
            }

            /*
            while (ctxtEnc->GetNoiseScaleDeg() > 1) {
                cc->ModReduceInPlace(ctxtEnc);
            }
*/
            lbcrypto::Plaintext resultGPU;
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
            std::cout << "Result GPU after LT" << resultGPU;
            CudaCheckErrorMod;

            //ASSERT_EQ_CIPHERTEXT(ctxtEnc, cResGPU);
            ASSERT_ERROR_OK(result, resultGPU);
        }

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEBootstrapTest, CoeffsToSlots) {
    for (auto& i : cached_cc) {
        i.second.first->ClearEvalAutomorphismKeys();
        i.second.first->ClearEvalMultKeys();
        if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
            std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)
                ->m_bootPrecomMap.clear();
    }
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    // Encoding as plaintexts
    int slots = 16;
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, slots);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    lbcrypto::Plaintext result;
    std::cout << "Setup Bootstrap" << std::endl;
    cc->EvalBootstrapSetup({2, 2}, {0, 0}, slots);

    std::cout << "Generate keys" << std::endl;
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    // coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475 0.00428562 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264 0.00211163 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128 -0.00695249 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494 -0.000989932 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05 -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10 -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
    std::cout << "Run bootstrap start" << std::endl;
    auto FHE = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
    auto raised = FHE->EvalBootstrapSetupOnly(c1, 1, 0);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, raised);
    FIDESlib::CKKS::Ciphertext GPUct1_(GPUcc, raw1);
    {
        lbcrypto::Plaintext result;
        cc->Decrypt(keys.secretKey, raised, &result);

        std::cout << "Before linear transform:\n";
        std::cout << "Result " << result;

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1_.store(GPUcc, raw_res1);
        auto cResGPU = c2->Clone();
        GetOpenFHECipherText(cResGPU, raw_res1);
        lbcrypto::Plaintext resultGPU;
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
        std::cout << "Result GPU " << resultGPU;
        CudaCheckErrorMod;
        ASSERT_ERROR_OK(result, resultGPU);
        ASSERT_EQ_CIPHERTEXT(raised, cResGPU);
    }
    /*
        {
            {
                auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(raised->GetKeyTag());
                auto conj = FHE->Conjugate(raised, evalKeyMap);
                cc->EvalAddInPlace(raised, conj);
            }
            lbcrypto::Plaintext result;
            cc->Decrypt(keys.secretKey, raised, &result);

            std::cout << "Before linear transform:\n";
            std::cout << "Result " << result;

            //GPUct1.addPt(GPUcc.GetBootPrecomputation(8).A[0]);
            FIDESlib::CKKS::RawCipherText raw_res1;
            GPUct1.store(GPUcc, raw_res1);
            auto cResGPU = c2->Clone();
            GetOpenFHECipherText(cResGPU, raw_res1);

            auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cResGPU->GetKeyTag());
            auto conj = FHE->Conjugate(cResGPU, evalKeyMap);
            cc->EvalAddInPlace(cResGPU, conj);
            lbcrypto::Plaintext resultGPU;
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
            std::cout << "Result GPU " << resultGPU;
            CudaCheckErrorMod;
            ASSERT_EQ_CIPHERTEXT(raised, cResGPU);
        }
        */
    std::cout << "Run CoeffToSlot" << std::endl;

    auto ctxtEnc = FHE->EvalCoeffsToSlots(FHE->m_bootPrecomMap.at(slots)->m_U0hatTPreFFT, raised);

    cc->RescaleInPlace(ctxtEnc);

    auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEnc->GetKeyTag());
    auto conj = FHE->Conjugate(ctxtEnc, evalKeyMap);
    cc->EvalAddInPlace(ctxtEnc, conj);

    {
        cc->Decrypt(keys.secretKey, ctxtEnc, &result);

        std::cout << "After CoeffToSlot:\n";
        std::cout << "Result " << result;
    }

    std::cout << "Run CoeffToSlot GPU" << std::endl;
    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

    for (int batch : FIDESlib::Testing::batch_configs) {
        fideslibParams.batch = batch;
        std::cout << "Batch " << batch << std::endl;
        GPUcc.batch = batch;
        cudaDeviceSynchronize();
        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc);
        GPUct1.copy(GPUct1_);

        CudaCheckErrorMod;
        FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, false);
        if (GPUcc.rescaleTechnique == CKKS::Context::FIXEDMANUAL)
            GPUct1.rescale();
        CudaCheckErrorMod;
        {
            FIDESlib::CKKS::RawCipherText raw_res1;
            GPUct1.store(GPUcc, raw_res1);
            auto cResGPU = c2->Clone();
            GetOpenFHECipherText(cResGPU, raw_res1);

            {
                auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cResGPU->GetKeyTag());
                auto conj = FHE->Conjugate(cResGPU, evalKeyMap);
                cc->EvalAddInPlace(cResGPU, conj);
            }

            lbcrypto::Plaintext resultGPU;
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
            std::cout << "Result GPU after LT" << resultGPU;
            CudaCheckErrorMod;
            ASSERT_ERROR_OK(result, resultGPU);
            //ASSERT_EQ_CIPHERTEXT(ctxtEnc, cResGPU);
        }

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEBootstrapTest, SlotsToCoeffs) {
    for (auto& i : cached_cc) {
        i.second.first->ClearEvalAutomorphismKeys();
        i.second.first->ClearEvalMultKeys();
        if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
            std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)
                ->m_bootPrecomMap.clear();
    }
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    // Encoding as plaintexts
    int slots = 16;
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, slots);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    lbcrypto::Plaintext result;
    std::cout << "Setup Bootstrap" << std::endl;
    cc->EvalBootstrapSetup({2, 2}, {0, 0}, slots);

    std::cout << "Generate keys" << std::endl;
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    // coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475 0.00428562 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264 0.00211163 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128 -0.00695249 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494 -0.000989932 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05 -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10 -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
    std::cout << "Run bootstrap start" << std::endl;
    auto FHE = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
    auto raised = FHE->EvalBootstrapNoStC(c1, 1, 0);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, raised);
    FIDESlib::CKKS::Ciphertext GPUct1_(GPUcc, raw1);
    {
        lbcrypto::Plaintext result;
        cc->Decrypt(keys.secretKey, raised, &result);

        std::cout << "Before SlotsToCoeffs:\n";
        std::cout << "Result " << result;

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1_.store(GPUcc, raw_res1);
        auto cResGPU = c2->Clone();
        GetOpenFHECipherText(cResGPU, raw_res1);
        lbcrypto::Plaintext resultGPU;
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
        std::cout << "Result GPU " << resultGPU;
        CudaCheckErrorMod;
        ASSERT_ERROR_OK(result, resultGPU);
        ASSERT_EQ_CIPHERTEXT(raised, cResGPU);
    }

    std::cout << "Run SlotsToCoeffs" << std::endl;

    auto ctxtEnc = FHE->EvalSlotsToCoeffs(FHE->m_bootPrecomMap.at(slots)->m_U0PreFFT, raised);

    cc->RescaleInPlace(ctxtEnc);

    auto conj = cc->EvalRotate(ctxtEnc, slots);
    cc->EvalAddInPlace(ctxtEnc, conj);

    {
        cc->Decrypt(keys.secretKey, ctxtEnc, &result);

        std::cout << "After SlotsToCoeffs:\n";
        std::cout << "Result " << result;
    }

    std::cout << "Run SlotsToCoeffs GPU" << std::endl;
    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

    for (int batch : FIDESlib::Testing::batch_configs) {
        fideslibParams.batch = batch;
        std::cout << "Batch " << batch << std::endl;
        GPUcc.batch = batch;
        cudaDeviceSynchronize();
        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc);
        GPUct1.copy(GPUct1_);

        FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, true);

        {
            FIDESlib::CKKS::RawCipherText raw_res1;
            GPUct1.store(GPUcc, raw_res1);
            auto cResGPU = c2->Clone();
            GetOpenFHECipherText(cResGPU, raw_res1);

            {
                auto conj = cc->EvalRotate(cResGPU, slots);
                cc->EvalAddInPlace(cResGPU, conj);
            }

            lbcrypto::Plaintext resultGPU;
            cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
            std::cout << "Result GPU after SlotsToCoeffs" << resultGPU;
            CudaCheckErrorMod;
            ASSERT_ERROR_OK(result, resultGPU);
            //ASSERT_EQ_CIPHERTEXT(ctxtEnc, cResGPU);
        }

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEBootstrapTest, OpenFHEBootstrap) {
    for (auto& i : cached_cc) {
        i.second.first->ClearEvalAutomorphismKeys();
        i.second.first->ClearEvalMultKeys();
        if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
            std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)
                ->m_bootPrecomMap.clear();
    }
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

    cc->EvalMultKeyGen(keys.secretKey);

    int slots = 1 << 4;
    std::cout << "Setup Bootstrap" << std::endl;
    cc->EvalBootstrapSetup({2, 2}, {0, 0}, slots);

    std::cout << "Generate keys" << std::endl;
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, raw_param.L - 1, nullptr, slots);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto cAdd = cc->EvalBootstrap(c1);

    lbcrypto::Plaintext result;
    std::cout << cAdd->GetLevel() << "\n";
    cc->Decrypt(keys.secretKey, cAdd, &result);

    std::cout << "Result " << result;

    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};

    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

    ///////////////////////////////////////////////////////////7777

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    kskEval.Initialize(GPUcc, rawKskEval);
    GPUcc.AddEvalKey(std::move(kskEval));

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::Ciphertext GPUct_o(GPUcc, raw1);

    for (int batch : FIDESlib::Testing::batch_configs) {
        fideslibParams.batch = batch;
        std::cout << "Batch " << batch << std::endl;

        GPUcc.batch = batch;
        cudaDeviceSynchronize();

        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc);
        GPUct1.copy(GPUct_o);

        cudaDeviceSynchronize();

        FIDESlib::CKKS::Bootstrap(GPUct1, slots);

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1.store(GPUcc, raw_res1);
        auto cResGPU(c2);

        GetOpenFHECipherText(cResGPU, raw_res1);
        lbcrypto::Plaintext resultGPU;
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

        std::cout << "Result GPU " << resultGPU;

        CudaCheckErrorMod;
        ASSERT_ERROR_OK(result, resultGPU);
        //ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEBootstrapTest, OpenFHEBootstrapLT) {
    for (auto& i : cached_cc) {
        i.second.first->ClearEvalAutomorphismKeys();
        i.second.first->ClearEvalMultKeys();
        if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
            std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)
                ->m_bootPrecomMap.clear();
    }
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    int slots = 32;
    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, slots);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    {
        lbcrypto::Plaintext result;
        cc->Decrypt(keys.secretKey, c1, &result);

        std::cout << "Result " << result;
    }

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

    FIDESlib::CKKS::Ciphertext GPUct1_(GPUcc, raw1);

    lbcrypto::Plaintext result;
    std::cout << "Setup Bootstrap" << std::endl;
    cc->EvalBootstrapSetup({1, 1}, {0, 0}, slots);

    std::cout << "Generate keys" << std::endl;
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

    auto cAdd = cc->EvalBootstrap(c1);

    /*{
            cc->RescaleInPlace(cAdd);
            auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cAdd->GetKeyTag());
            auto conj = FHE->Conjugate(cAdd, evalKeyMap);
        }*/

    std::cout << cAdd->GetLevel() << "\n";
    cc->Decrypt(keys.secretKey, cAdd, &result);

    std::cout << "Result " << result;

    ///////////////////////////////////////////////////////////7777

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    kskEval.Initialize(GPUcc, rawKskEval);
    GPUcc.AddEvalKey(std::move(kskEval));

    for (int batch : FIDESlib::Testing::batch_configs) {
        fideslibParams.batch = batch;
        std::cout << "Batch " << batch << std::endl;
        GPUcc.batch = batch;

        cudaDeviceSynchronize();

        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc);
        GPUct1.copy(GPUct1_);

        FIDESlib::CKKS::Bootstrap(GPUct1, slots);

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1.store(GPUcc, raw_res1);
        auto cResGPU(c2);

        GetOpenFHECipherText(cResGPU, raw_res1);

        auto FHE = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
        /*
        {
            auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cResGPU->GetKeyTag());
            auto conj = FHE->Conjugate(cResGPU, evalKeyMap);
            cc->EvalAddInPlace(cResGPU, conj);
        }
        */
        lbcrypto::Plaintext resultGPU;
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

        std::cout << "Result GPU " << resultGPU;

        CudaCheckErrorMod;
        ASSERT_ERROR_OK(result, resultGPU);

        // ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

        CudaCheckErrorMod;
    }
}

TEST_P(OpenFHEBootstrapTest, OpenFHEBootstrapDense) {
    for (auto& i : cached_cc) {
        i.second.first->ClearEvalAutomorphismKeys();
        i.second.first->ClearEvalMultKeys();
        if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
            std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)
                ->m_bootPrecomMap.clear();
    }
    // Enable the features that you wish to use
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
    cc->EvalMultKeyGen(keys.secretKey);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};
    ///// PROBAR /////
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

    int slots = 1 << 15;
    // Encoding as plaintexts
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, slots);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

    std::cout << "Input x1: " << ptxt1 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

    FIDESlib::CKKS::Ciphertext GPUct1_(GPUcc, raw1);

    lbcrypto::Plaintext result;
    std::cout << "Setup Bootstrap" << std::endl;
    cc->EvalBootstrapSetup({5, 5}, {0, 0}, slots);

    std::cout << "Generate keys" << std::endl;
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);

    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

    auto cAdd = cc->EvalBootstrap(c1);

    /*{
            cc->RescaleInPlace(cAdd);
            auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cAdd->GetKeyTag());
            auto conj = FHE->Conjugate(cAdd, evalKeyMap);
        }*/

    std::cout << cAdd->GetLevel() << "\n";
    cc->Decrypt(keys.secretKey, cAdd, &result);

    std::cout << "Result " << result;

    ///////////////////////////////////////////////////////////7777

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    kskEval.Initialize(GPUcc, rawKskEval);
    GPUcc.AddEvalKey(std::move(kskEval));

    for (int batch : FIDESlib::Testing::batch_configs) {
        fideslibParams.batch = batch;
        std::cout << "Batch " << batch << std::endl;
        GPUcc.batch = batch;
        cudaDeviceSynchronize();

        FIDESlib::CKKS::Ciphertext GPUct1(GPUcc);
        GPUct1.copy(GPUct1_);

        FIDESlib::CKKS::Bootstrap(GPUct1, slots);

        FIDESlib::CKKS::RawCipherText raw_res1;
        GPUct1.store(GPUcc, raw_res1);
        auto cResGPU(c2);

        GetOpenFHECipherText(cResGPU, raw_res1);

        lbcrypto::Plaintext resultGPU;
        cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

        std::cout << "Levels: " << cResGPU->GetLevel() << std::endl;
        std::cout << "Result GPU " << resultGPU;

        CudaCheckErrorMod;
        ASSERT_ERROR_OK(result, resultGPU);

        //ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

        CudaCheckErrorMod;
    }
}

INSTANTIATE_TEST_SUITE_P(OpenFHEBootstrapTests, OpenFHEBootstrapTest, testing::Values(TTALL64BOOT));
}  // namespace FIDESlib::Testing