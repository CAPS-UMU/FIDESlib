//
// Created by carlosad on 24/04/24.
//
#include <bit>
#include <cassert>
#include "CKKS/Context.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "Math.cuh"

using namespace lbcrypto;

/**
* Converts a vector of polynomial limbs to a single flattened array
*/
std::vector<std::vector<uint64_t>> FIDESlib::CKKS::GetRawArray(
    std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>> polys) {
    // total size is r * N
    int numRes = polys.size();
    int numElements = (*polys[0].m_values).GetLength();

    std::vector<std::vector<uint64_t>> flattened(numRes, std::vector<uint64_t>(numElements));

    // Fill the array
    for (int r = 0; r < numRes; ++r) {
        for (int i = 0; i < numElements; i++) {
            flattened[r][i] = (*polys[r].m_values)[i].ConvertToInt();
        }
    }
    return flattened;
};

/**
* Gets the moduli from a vector of polynomial limbs and returns a single array
*/
static std::vector<uint64_t> GetModuli(std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>> polys) {
    int numRes = polys.size();
    std::vector<uint64_t> moduli(numRes);
    for (int r = 0; r < numRes; r++) {
        moduli[r] = polys[r].GetModulus().ConvertToInt();
    }
    return moduli;
};

/**
* Converts a ciphertext from openFHE into the RawCiphertext format
*/
FIDESlib::CKKS::RawCipherText FIDESlib::CKKS::GetRawCipherText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                                               lbcrypto::Ciphertext<DCRTPoly> ct, int REV) {
    RawCipherText result;  //{ .cc = cc };
    //result.originalCipherText=ct;
    result.numRes = ct->GetElements()[0].GetAllElements().size();
    result.N = (*(ct->GetElements()[0].GetAllElements())[0].m_values).GetLength();
    result.sub_0 = GetRawArray(ct->GetElements()[0].GetAllElements());
    result.sub_1 = GetRawArray(ct->GetElements()[1].GetAllElements());

    // We read (hopefully) in eval form, and OpenFHE should be bit_reversed, so REVERSE == 0.
    // Changed the REV variable to an argument to the function.
    // purpose: profiling Automorph kernel with both bit-reversed and normal order ciphertexts.
    if (!REV) {
        for (auto& i : result.sub_0)
            bit_reverse_vector(i);
        for (auto& i : result.sub_1)
            bit_reverse_vector(i);
    }
    result.moduli = GetModuli(ct->GetElements()[0].GetAllElements());
    result.format = ct->GetElements()[0].GetFormat();

    result.Noise = ct->m_scalingFactor;
    result.NoiseLevel = ct->m_noiseScaleDeg;

    return result;
};

/**
* Converts a ciphertext from the RawCiphertext format back to the OpenFHE ciphertext format*/
void FIDESlib::CKKS::GetOpenFHECipherText(lbcrypto::Ciphertext<DCRTPoly> result, RawCipherText raw, int REV) {

    assert(result->GetElements()[0].GetAllElements().size() >= raw.numRes);
    // Changed the REV variable to an argument to the function.
    // purpose: profiling Automorph kernel with both bit-reversed and normal order ciphertexts.
    if (!REV) {
        for (auto& i : raw.sub_0)
            bit_reverse_vector(i);
        for (auto& i : raw.sub_1)
            bit_reverse_vector(i);
    }
    DCRTPoly sub_0 = result->GetElements().at(0);
    DCRTPoly sub_1 = result->GetElements().at(1);
    auto& dcrt_0 = sub_0.GetAllElements();
    auto& dcrt_1 = sub_1.GetAllElements();
    result->SetLevel(result->GetLevel() + result->GetElements().at(0).GetNumOfElements() - raw.numRes);
    dcrt_0.resize(raw.numRes);
    dcrt_1.resize(raw.numRes);
    for (int r = 0; r < raw.numRes; r++) {
        for (int i = 0; i < raw.N; i++) {
            (*dcrt_0.at(r).m_values).at(i).SetValue(raw.sub_0.at(r).at(i));
            (*dcrt_1.at(r).m_values).at(i).SetValue(raw.sub_1.at(r).at(i));
        }
    }

    //sub_0.m_vectors=dcrt_0;
    //sub_1.m_vectors=dcrt_1;
    for (size_t i = sub_0.m_params->m_params.size(); i > sub_0.m_vectors.size(); --i) {
        DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_0.m_params);
        newP->PopLastParam();
        sub_0.m_params.reset(newP);
    }

    for (size_t i = sub_1.m_params->m_params.size(); i > sub_1.m_vectors.size(); --i) {
        DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_1.m_params);
        newP->PopLastParam();
        sub_1.m_params.reset(newP);
    }

    std::vector<lbcrypto::DCRTPoly> ct_new = {sub_0, sub_1};
    result->SetElements(ct_new);

    result->m_scalingFactor = raw.Noise;
    result->m_noiseScaleDeg = raw.NoiseLevel;
}

void FIDESlib::CKKS::GetOpenFHEPlaintext(lbcrypto::Plaintext result, RawPlainText raw, int REV) {

    assert(result->GetElement<DCRTPoly>().GetAllElements().size() >= raw.numRes);
    // Changed the REV variable to an argument to the function.
    // purpose: profiling Automorph kernel with both bit-reversed and normal order ciphertexts.
    if (!REV) {
        for (auto& i : raw.sub_0)
            bit_reverse_vector(i);
    }
    DCRTPoly sub_0 = result->GetElement<DCRTPoly>();
    auto& dcrt_0 = sub_0.GetAllElements();
    result->SetLevel(result->GetLevel() + result->GetElement<DCRTPoly>().GetNumOfElements() - raw.numRes);
    dcrt_0.resize(raw.numRes);
    for (int r = 0; r < raw.numRes; r++) {
        for (int i = 0; i < raw.N; i++) {
            (*dcrt_0.at(r).m_values).at(i).SetValue(raw.sub_0.at(r).at(i));
        }
    }

    //sub_0.m_vectors=dcrt_0;
    //sub_1.m_vectors=dcrt_1;
    for (size_t i = sub_0.m_params->m_params.size(); i > sub_0.m_vectors.size(); --i) {
        DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_0.m_params);
        newP->PopLastParam();
        sub_0.m_params.reset(newP);
    }

    result->encodedVectorDCRT = sub_0;
    result->scalingFactor = raw.Noise;
    result->noiseScaleDeg = raw.NoiseLevel;

    /*
    std::cout << result << std::endl;

    bool ok = result->Decode();
    std::cout << ok << " " << result << std::endl;
    */
}

FIDESlib::CKKS::RawPlainText FIDESlib::CKKS::GetRawPlainText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                                             lbcrypto::Plaintext pt) {
    RawPlainText result;  //{.cc = cc};
    result.originalPlainText = pt;
    result.numRes = pt->GetElement<DCRTPoly>().GetAllElements().size();
    result.N = (*(pt->GetElement<DCRTPoly>().GetAllElements())[0].m_values).GetLength();
    result.sub_0 = GetRawArray(pt->GetElement<DCRTPoly>().GetAllElements());
    result.moduli = GetModuli(pt->GetElement<DCRTPoly>().GetAllElements());

    result.format = pt->GetElement<DCRTPoly>().GetFormat();

    if constexpr (REVERSE) {
        for (auto& i : result.sub_0)
            bit_reverse_vector(i);
    }

    result.Noise = pt->GetScalingFactor();
    result.NoiseLevel = pt->GetNoiseScaleDeg();

    return result;
}

FIDESlib::CKKS::RawParams FIDESlib::CKKS::GetRawParams(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc) {
    RawParams result;
    result.N = cc->GetRingDimension();
    result.logN = std::bit_width((uint32_t)result.N) - 1;
    result.L = cc->params->m_params->m_params.size() - 1;
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    result.scalingTechnique = cryptoParams->GetScalingTechnique();
    //result.qbit = cc->params->m_params->m_params->;
    //auto aux = cc->GetCryptoParameters()->GetParamsPK()->GetParamPartition();

    for (auto& i : cc->params->m_params->m_params) {
        result.moduli.push_back(i->m_ciphertextModulus.m_value);
        result.root_of_unity.push_back(i->m_rootOfUnity.m_value);
        result.cyclotomic_order.push_back(i->m_cyclotomicOrder);
    }

    // intnat::ChineseRemainderTransformFTTNat<intnat::NativeVector>::m_rootOfUnityReverseTableByModulus
    for (size_t i = 0; i < result.moduli.size(); ++i) {
        using namespace intnat;
        //  NumberTheoreticTransformNat<NativeVector>().PreCompute
        using FFT = ChineseRemainderTransformFTTNat<NativeVector>;
        auto mapSearch = FFT::m_rootOfUnityReverseTableByModulus.find(result.moduli[i]);
        if (mapSearch == FFT::m_rootOfUnityReverseTableByModulus.end() ||
            mapSearch->second.GetLength() != (size_t)result.N /*CycloOrderHf*/) {
            assert("OpenFHE has not generated the NTT tables we want yet :(" == nullptr);
            //PreCompute(result.root_of_unity[i], result.N << 2, result.moduli[i]);
        } else {
            int size = FFT::m_rootOfUnityReverseTableByModulus[result.moduli[i]].GetLength();

            for (int k = 0; k < size; ++k) {
                result.psi[i].push_back(FFT::m_rootOfUnityReverseTableByModulus[result.moduli[i]].at(k).m_value);
                result.psi_inv[i].push_back(
                    FFT::m_rootOfUnityInverseReverseTableByModulus[result.moduli[i]].at(k).m_value);
            }
            result.N_inv.push_back(FFT::m_cycloOrderInverseTableByModulus.at(result.moduli[i]).at(result.logN).m_value);
        }
    }

    //    intnat::NumberTheoreticTransformNat<intnat::NativeVector>().
    //    mubintvec<ubint<unsigned long>>;

    result.ModReduceFactor.resize(result.L + 1);
    for (size_t i = 0; i < result.ModReduceFactor.size(); ++i) {
        result.ModReduceFactor[/*result.L - */ i] = cryptoParams->GetModReduceFactor(i);
    }
    result.ScalingFactorReal.resize(result.L + 1);
    for (size_t i = 0; i < result.ScalingFactorReal.size(); ++i) {
        result.ScalingFactorReal[result.L - i] = cryptoParams->GetScalingFactorReal(i);
    }

    result.ScalingFactorRealBig.resize(result.L + 1);
    for (size_t i = 0; i < result.ScalingFactorRealBig.size(); ++i) {
        result.ScalingFactorRealBig[result.L - i] = cryptoParams->GetScalingFactorRealBig(i);
    }

    {
        auto& src = cryptoParams->m_QlQlInvModqlDivqlModq;
        auto& dest = result.m_QlQlInvModqlDivqlModq;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i].resize(src[i].size());
            for (size_t j = 0; j < src[i].size(); ++j) {
                dest[i][j] = src[i][j].m_value;
            }
        }
    }

    /// Key Switching precomputations !!!
    result.dnum = cryptoParams->GetNumPartQ();
    result.K = cryptoParams->m_paramsP->m_params.size();
    assert(cryptoParams->GetNumPartQ() == cryptoParams->GetNumberOfQPartitions());
    cryptoParams->GetNumPerPartQ();

    {
        auto& src = cryptoParams->GetParamsP()->m_params;
        for (auto& i : src) {
            result.SPECIALmoduli.push_back(i->m_ciphertextModulus.m_value);
            result.SPECIALroot_of_unity.push_back(i->m_rootOfUnity.m_value);
            result.SPECIALcyclotomic_order.push_back(i->m_cyclotomicOrder);
        }
    }

    {
        auto& src = cryptoParams->m_paramsPartQ;
        for (auto& i : src) {
            result.PARTITIONmoduli.emplace_back();
            for (auto& j : i->m_params) {
                result.PARTITIONmoduli.back().push_back(j->m_ciphertextModulus.m_value);
            }
        }
    }

    {
        auto& src = cryptoParams->GetPHatInvModp();
        auto& dest = result.PHatInvModp;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i] = src[i].m_value;
        }
    }

    {
        auto& src = cryptoParams->GetPInvModq();
        auto& dest = result.PInvModq;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i] = src[i].m_value;
        }
    }

    {
        auto& src = cryptoParams->GetPHatModq();
        auto& dest = result.PHatModq;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i].resize(src[i].size());
            for (size_t j = 0; j < src[i].size(); ++j) {
                dest[i][j] = src[i][j].m_value;
            }
        }
    }

    {
        auto& dest = result.PartQlHatInvModq;
        auto& src = cryptoParams->m_PartQlHatInvModq;
        dest.resize(src.size());
        for (size_t k = 0; k < dest.size(); ++k) {
            dest[k].resize(src[k].size());
            for (size_t i = 0; i < dest[k].size(); ++i) {
                dest[k][i].resize(src[k][i].size());
                for (size_t j = 0; j < src[k][i].size(); ++j) {
                    dest[k][i][j] = src[k][i][j].m_value;
                }
            }
        }
    }

    {
        auto& dest = result.PartQlHatModp;
        auto& src = cryptoParams->m_PartQlHatModp;
        dest.resize(result.dnum);
        dest.resize(src.size());

        for (size_t k = 0; k < dest.size(); ++k) {
            dest[k].resize(src[k].size());
            for (size_t i = 0; i < dest[k].size(); ++i) {
                dest[k][i].resize(src[k][i].size());
                for (size_t j = 0; j < src[k][i].size(); ++j) {
                    dest[k][i][j].resize(src[k][i][j].size());
                    for (size_t l = 0; l < src[k][i][j].size(); ++l) {
                        dest[k][i][j][l] = src[k][i][j][l].m_value;
                    }
                }
            }
        }
    }

    if (cc->GetScheme()->m_FHE) {
        if (cryptoParams->GetSecretKeyDist() == SPARSE_TERNARY) {
            result.coefficientsCheby = lbcrypto::FHECKKSRNS::g_coefficientsSparse;
            // k = K_SPARSE;
            result.bootK = 1.0;  // do not divide by k as we already did it during precomputation
        } else {
            result.coefficientsCheby = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
            //coefficients = g_coefficientsUniform;
            result.bootK = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)
                               ->K_UNIFORM;  // lbcrypto::FHECKKSRNS::K_UNIFORM;
        }

        if (cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY)
            result.doubleAngleIts = lbcrypto::FHECKKSRNS::R_UNIFORM;
        else
            result.doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
    }

    result.p = cryptoParams->GetPlaintextModulus();

    return result;
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetEvalKeySwitchKey(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys) {
    std::vector<std::vector<std::vector<uint64_t>>> a_moduli;
    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> a;
    std::vector<std::vector<std::vector<uint64_t>>> b;
    //const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());

    auto& keyMap = lbcrypto::CryptoContextImpl<DCRTPoly>::s_evalMultKeyMap;
    if (keyMap.find(keys.secretKey->GetKeyTag()) != keyMap.end()) {
        const std::vector<EvalKey<DCRTPoly>>& key = keyMap[keys.secretKey->GetKeyTag()];
        const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key.at(0));

        for (auto a_raw = ek.get()->m_rKey; auto& i : a_raw) {
            std::vector<std::vector<std::vector<uint64_t>>> a_inner;
            std::vector<std::vector<uint64_t>> a_inner_moduli;
            for (auto& j : i) {
                auto v = GetRawArray(j.m_vectors);
                a_inner_moduli.emplace_back();
                auto& a_aux = a_inner_moduli.back();
                for (auto& p : j.m_params->m_params) {
                    a_aux.push_back(p->m_ciphertextModulus.m_value);
                }
                a_inner.push_back(v);
            }
            a.push_back(a_inner);
            a_moduli.push_back(a_inner_moduli);
        }
        //    auto b_raw = key->get_dcrtKeys();

        /*
            for (const auto & i : b_raw) {
                auto v = GetRawArray(i.m_vectors);
                b.push_back(v);
            }
        */
    } else {
        assert("EvalKey is not present !!!" == nullptr);
    }

    return RawKeySwitchKey(std::move(a_moduli), std::move(a), std::move(b));
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetRotationKeySwitchKey(
    const KeyPair<lbcrypto::DCRTPoly>& keys, int index, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc) {
    std::vector<std::vector<std::vector<uint64_t>>> a_moduli;
    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> a;
    std::vector<std::vector<std::vector<uint64_t>>> b;
    //const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());

    auto& keyMap = lbcrypto::CryptoContextImpl<DCRTPoly>::s_evalAutomorphismKeyMap;
    if (keyMap.find(keys.secretKey->GetKeyTag()) != keyMap.end()) {
        auto& keyMap2 = keyMap[keys.secretKey->GetKeyTag()];
        uint32_t x = FIDESlib::modpow(5, index, cc->GetRingDimension() * 2);
        if (keyMap2->find(x) != keyMap2->end()) {
            const auto& key = keyMap2->at(x);
            assert(key != nullptr);
            const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key);

            for (auto a_raw = ek.get()->m_rKey; auto& i : a_raw) {
                std::vector<std::vector<std::vector<uint64_t>>> a_inner;
                std::vector<std::vector<uint64_t>> a_inner_moduli;
                for (auto& j : i) {
                    auto v = GetRawArray(j.m_vectors);
                    a_inner_moduli.emplace_back();
                    auto& a_aux = a_inner_moduli.back();
                    for (auto& p : j.m_params->m_params) {
                        a_aux.push_back(p->m_ciphertextModulus.m_value);
                    }
                    a_inner.push_back(v);
                }
                a.push_back(a_inner);
                a_moduli.push_back(a_inner_moduli);
            }
            //    auto b_raw = key->get_dcrtKeys();

            /*
                for (const auto & i : b_raw) {
                    auto v = GetRawArray(i.m_vectors);
                    b.push_back(v);
                }
            */
        } else {
            assert("RotKey is not present for rotation !!!" == nullptr);
            std::cout << "RotKey is not present for rotation " << index << "!!!" << std::endl;
        }
    } else {
        assert("RotKey is not present !!!" == nullptr);
        std::cout << "RotKey is not present !!!" << std::endl;
    }
    return RawKeySwitchKey(std::move(a_moduli), std::move(a), std::move(b));
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetConjugateKeySwitchKey(
    const KeyPair<lbcrypto::DCRTPoly>& keys, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc) {
    std::vector<std::vector<std::vector<uint64_t>>> a_moduli;
    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> a;
    std::vector<std::vector<std::vector<uint64_t>>> b;
    //const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());

    auto& keyMap2 = cc->GetEvalAutomorphismKeyMap(keys.publicKey->GetKeyTag());

    if (keyMap2.find(2 * cc->GetRingDimension() - 1) != keyMap2.end()) {
        const auto& key = keyMap2.at(2 * cc->GetRingDimension() - 1);
        assert(key != nullptr);
        const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key);
        //std::cout << std::endl << "Clave " << ek->GetKeyTag() << "\n";
        for (auto a_raw = ek.get()->m_rKey; auto& i : a_raw) {
            std::vector<std::vector<std::vector<uint64_t>>> a_inner;
            std::vector<std::vector<uint64_t>> a_inner_moduli;
            for (auto& j : i) {
                auto v = GetRawArray(j.m_vectors);
                a_inner_moduli.emplace_back();
                auto& a_aux = a_inner_moduli.back();
                for (auto& p : j.m_params->m_params) {
                    a_aux.push_back(p->m_ciphertextModulus.m_value);
                }
                a_inner.push_back(v);
            }
            a.push_back(a_inner);
            a_moduli.push_back(a_inner_moduli);
        }
        //    auto b_raw = key->get_dcrtKeys();

        /*
                for (const auto & i : b_raw) {
                    auto v = GetRawArray(i.m_vectors);
                    b.push_back(v);
                }
            */
    } else {
        assert("RotKey is not present for rotation !!!" == nullptr);
        std::cout << "RotKey is not present for conjugation!!!" << std::endl;
    }
    return RawKeySwitchKey(std::move(a_moduli), std::move(a), std::move(b));
}

#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"

constexpr bool remove_extension = true;

void FIDESlib::CKKS::AddBootstrapPrecomputation(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
                                                const KeyPair<lbcrypto::DCRTPoly>& keys, int slots,
                                                FIDESlib::CKKS::Context& GPUcc) {
    FIDESlib::CKKS::BootstrapPrecomputation result;

    auto precom =
        std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.find(slots)->second;

    if (precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1 &&
        precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1) {
        result.LT.slots = slots;
        result.LT.bStep = (precom->m_dim1 == 0) ? ceil(sqrt(slots)) : precom->m_dim1;

        if (1) {  // extended limbs computation
            auto auxA = precom->m_U0hatTPre;
            auto auxInvA = precom->m_U0Pre;

            result.LT.A.clear();
            for (int i = 0; i < auxA.size(); ++i) {
                RawPlainText raw = GetRawPlainText(cc, auxA.at(i));
                result.LT.A.emplace_back(GPUcc, raw);
                if constexpr (remove_extension)
                    result.LT.A.back().c0.freeSpecialLimbs();
                if (0) {
                    RawPlainText raw2;
                    result.LT.A.back().store(raw2);
                    lbcrypto::Plaintext pt = auxA.at(0);
                    GetOpenFHEPlaintext(pt, raw2);
                }

                // result.A.back().moddown();

                if (0) {
                    RawPlainText raw2;
                    result.LT.A.back().store(raw2);
                    lbcrypto::Plaintext pt = auxA.at(0);
                    GetOpenFHEPlaintext(pt, raw2);
                }
            }

            result.LT.invA.clear();
            for (int i = 0; i < auxInvA.size(); ++i) {
                RawPlainText raw = GetRawPlainText(cc, auxInvA.at(i));
                result.LT.invA.emplace_back(GPUcc, raw);
                if constexpr (remove_extension)
                    result.LT.invA.back().c0.freeSpecialLimbs();
                // result.invA.back().moddown();
            }
        } else {
            std::vector<lbcrypto::ConstPlaintext> auxA;
            std::vector<lbcrypto::ConstPlaintext> auxInvA;

            std::vector<std::vector<std::complex<double>>> U0(slots, std::vector<std::complex<double>>(slots));
            std::vector<std::vector<std::complex<double>>> U1(slots, std::vector<std::complex<double>>(slots));
            std::vector<std::vector<std::complex<double>>> U0hatT(slots, std::vector<std::complex<double>>(slots));
            std::vector<std::vector<std::complex<double>>> U1hatT(slots, std::vector<std::complex<double>>(slots));

            uint32_t m = 4 * slots;
            bool isSparse = (2 * GPUcc.N != m) ? true : false;

            // computes indices for all primitive roots of unity
            std::vector<uint32_t> rotGroup(slots);
            uint32_t fivePows = 1;
            for (uint32_t i = 0; i < slots; ++i) {
                rotGroup[i] = fivePows;
                fivePows *= 5;
                fivePows %= m;
            }

            // computes all powers of a primitive root of unity exp(2 * M_PI/m)
            std::vector<std::complex<double>> ksiPows(m + 1);
            for (uint32_t j = 0; j < m; ++j) {
                double angle = 2.0 * M_PI * j / m;
                ksiPows[j].real(cos(angle));
                ksiPows[j].imag(sin(angle));
            }
            ksiPows[m] = ksiPows[0];

            for (size_t i = 0; i < slots; i++) {
                for (size_t j = 0; j < slots; j++) {
                    U0[i][j] = ksiPows[(j * rotGroup[i]) % m];
                    U0hatT[j][i] = std::conj(U0[i][j]);
                    U1[i][j] = std::complex<double>(0, 1) * U0[i][j];
                    U1hatT[j][i] = std::conj(U1[i][j]);
                }
            }

            NativeInteger q =
                cc->GetCryptoParameters()->GetElementParams()->GetParams()[0]->GetModulus().ConvertToInt();
            double qDouble = q.ConvertToDouble();

            uint128_t factor = ((uint128_t)1 << ((uint32_t)std::round(std::log2(qDouble))));
            double pre = qDouble / factor;
            double k = (std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters())
                            ->GetSecretKeyDist() == SPARSE_TERNARY)
                           ? std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->K_SPARSE
                           : 1.0;
            double scaleEnc = pre / k;
            double scaleDec = 1 / pre;

            uint32_t L0 = cc->GetCryptoParameters()->GetElementParams()->GetParams().size();
            // for FLEXIBLEAUTOEXT we do not need extra modulus in auxiliary plaintexts
            if (std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters())
                    ->GetScalingTechnique() == FLEXIBLEAUTOEXT)
                L0 -= 1;
            uint32_t lEnc = L0 - precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] - 1;
            uint32_t approxModDepth =
                std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)
                    ->GetModDepthInternal(
                        std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters())
                            ->GetSecretKeyDist());
            uint32_t depthBT = approxModDepth + precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] +
                               precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
            uint32_t lDec = L0 - depthBT;

            std::shared_ptr<FHECKKSRNS> fhe = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);

            if (!isSparse) {
                auxA = fhe->EvalLinearTransformPrecompute(*cc, U0hatT, scaleEnc, lEnc, false);
                auxInvA = fhe->EvalLinearTransformPrecompute(*cc, U0, scaleDec, lDec, false);
            } else {
                auxA = fhe->EvalLinearTransformPrecompute(*cc, U0hatT, U1hatT, 0, scaleEnc, lEnc, false);
                auxInvA = fhe->EvalLinearTransformPrecompute(*cc, U0, U1, 1, scaleDec, lDec, false);
            }

            result.LT.A.clear();
            for (int i = 0; i < auxA.size(); ++i) {
                RawPlainText raw = GetRawPlainText(cc, auxA.at(i));
                result.LT.A.emplace_back(GPUcc, raw);
            }

            result.LT.invA.clear();
            for (int i = 0; i < auxInvA.size(); ++i) {
                RawPlainText raw = GetRawPlainText(cc, auxInvA.at(i));
                result.LT.invA.emplace_back(GPUcc, raw);
            }
        }

        for (int i = 1; i < result.LT.bStep; ++i) {
            KeySwitchingKey ksk(GPUcc);
            RawKeySwitchKey rksk = GetRotationKeySwitchKey(keys, i, cc);
            ksk.Initialize(GPUcc, rksk);
            GPUcc.AddRotationKey(i, std::move(ksk));
        }

        for (int i = result.LT.bStep; i < result.LT.slots; i += result.LT.bStep) {
            KeySwitchingKey ksk(GPUcc);
            RawKeySwitchKey rksk = GetRotationKeySwitchKey(keys, i, cc);
            ksk.Initialize(GPUcc, rksk);
            GPUcc.AddRotationKey(i, std::move(ksk));
        }
    } else {

        {  // CoeffToSlots metadata
            uint32_t M = cc->GetCyclotomicOrder();
            uint32_t N = cc->GetRingDimension();
            int32_t levelBudget = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
            int32_t layersCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_COLL];
            int32_t remCollapse = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_REM];
            int32_t numRotations = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
            int32_t b = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP];
            int32_t g = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP];
            int32_t numRotationsRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
            int32_t bRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP_REM];
            int32_t gRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

            int32_t stop = -1;
            int32_t flagRem = 0;

            auto algo = cc->GetScheme();

            if (remCollapse != 0) {
                stop = 0;
                flagRem = 1;
            }

            // precompute the inner and outer rotations
            result.CtS.resize(levelBudget);
            for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {
                if (flagRem == 1 && i == 0) {
                    // remainder corresponds to index 0 in encoding and to last index in decoding
                    result.CtS[i].bStep = gRem;
                    result.CtS[i].gStep = bRem;
                    result.CtS[i].slots = numRotationsRem;
                    result.CtS[i].rotIn.resize(numRotationsRem + 1);
                } else {
                    result.CtS[i].bStep = g;
                    result.CtS[i].gStep = b;
                    result.CtS[i].slots = numRotations;
                    result.CtS[i].rotIn.resize(numRotations + 1);
                }
            }

            for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {
                result.CtS[i].rotOut.resize(b + bRem);
            }

            for (int32_t s = levelBudget - 1; s > stop; s--) {
                for (int32_t j = 0; j < g; j++) {
                    result.CtS[s].rotIn[j] = ReduceRotation((j - int32_t((numRotations + 1) / 2) + 1) *
                                                                (1 << ((s - flagRem) * layersCollapse + remCollapse)),
                                                            slots);
                }

                for (int32_t i = 0; i < b; i++) {
                    result.CtS[s].rotOut[i] =
                        ReduceRotation((g * i) * (1 << ((s - flagRem) * layersCollapse + remCollapse)), M / 4);
                }
            }

            if (flagRem) {
                for (int32_t j = 0; j < gRem; j++) {
                    result.CtS[stop].rotIn[j] = ReduceRotation((j - int32_t((numRotationsRem + 1) / 2) + 1), slots);
                }

                for (int32_t i = 0; i < bRem; i++) {
                    result.CtS[stop].rotOut[i] = ReduceRotation((gRem * i), M / 4);
                }
            }
        }
        {  // SlotToCoeff metadata
            uint32_t M = cc->GetCyclotomicOrder();
            uint32_t N = cc->GetRingDimension();

            int32_t levelBudget = precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
            int32_t layersCollapse = precom->m_paramsDec[CKKS_BOOT_PARAMS::LAYERS_COLL];
            int32_t remCollapse = precom->m_paramsDec[CKKS_BOOT_PARAMS::LAYERS_REM];
            int32_t numRotations = precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
            int32_t b = precom->m_paramsDec[CKKS_BOOT_PARAMS::BABY_STEP];
            int32_t g = precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP];
            int32_t numRotationsRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
            int32_t bRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::BABY_STEP_REM];
            int32_t gRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

            auto algo = cc->GetScheme();

            int32_t flagRem = 0;

            if (remCollapse != 0) {
                flagRem = 1;
            }

            // precompute the inner and outer rotations

            result.StC.resize(levelBudget);
            for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {

                if (flagRem == 1 && i == uint32_t(levelBudget - 1)) {
                    // remainder corresponds to index 0 in encoding and to last index in decoding
                    result.StC[i].bStep = gRem;
                    result.StC[i].gStep = bRem;
                    result.StC[i].slots = numRotationsRem;
                    result.StC[i].rotIn.resize(numRotationsRem + 1);
                } else {
                    result.StC[i].bStep = g;
                    result.StC[i].gStep = b;
                    result.StC[i].slots = numRotations;
                    result.StC[i].rotIn.resize(numRotations + 1);
                }
            }

            for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {
                result.StC.at(i).rotOut.resize(b + bRem);
            }

            for (int32_t s = 0; s < levelBudget - flagRem; s++) {
                for (int32_t j = 0; j < g; j++) {
                    result.StC.at(s).rotIn.at(j) =
                        ReduceRotation((j - int32_t((numRotations + 1) / 2) + 1) * (1 << (s * layersCollapse)), M / 4);
                }

                for (int32_t i = 0; i < b; i++) {
                    result.StC.at(s).rotOut.at(i) = ReduceRotation((g * i) * (1 << (s * layersCollapse)), M / 4);
                }
            }

            if (flagRem) {
                int32_t s = levelBudget - flagRem;
                for (int32_t j = 0; j < gRem; j++) {
                    result.StC.at(s).rotIn.at(j) = ReduceRotation(
                        (j - int32_t((numRotationsRem + 1) / 2) + 1) * (1 << (s * layersCollapse)), M / 4);
                }

                for (int32_t i = 0; i < bRem; i++) {
                    result.StC.at(s).rotOut.at(i) = ReduceRotation((gRem * i) * (1 << (s * layersCollapse)), M / 4);
                }
            }
        }

        auto& A = precom->m_U0hatTPreFFT;
        auto& invA = precom->m_U0PreFFT;

        for (int i = 0; i < A.size(); ++i) {
            for (int j = 0; j < A.at(i).size(); ++j) {
                RawPlainText raw = GetRawPlainText(cc, A.at(i).at(j));
                result.CtS.at(i).A.emplace_back(GPUcc, raw);
                if constexpr (remove_extension)
                    result.CtS.at(i).A.back().c0.freeSpecialLimbs();
                CudaCheckErrorMod;
            }
        }

        for (int i = 0; i < invA.size(); ++i) {
            for (int j = 0; j < invA.at(i).size(); ++j) {
                RawPlainText raw = GetRawPlainText(cc, invA.at(i).at(j));
                result.StC.at(i).A.emplace_back(GPUcc, raw);
                if constexpr (remove_extension)
                    result.StC.at(i).A.back().c0.freeSpecialLimbs();
            }
        }

        for (auto& i : result.StC) {
            for (auto& j : i.rotIn) {
                if (j && !GPUcc.HasRotationKey(j)) {
                    KeySwitchingKey ksk(GPUcc);
                    RawKeySwitchKey rksk = GetRotationKeySwitchKey(keys, j, cc);
                    ksk.Initialize(GPUcc, rksk);
                    GPUcc.AddRotationKey(j, std::move(ksk));
                }
            }
            for (auto& j : i.rotOut) {
                if (j && !GPUcc.HasRotationKey(j)) {
                    KeySwitchingKey ksk(GPUcc);
                    RawKeySwitchKey rksk = GetRotationKeySwitchKey(keys, j, cc);
                    ksk.Initialize(GPUcc, rksk);
                    GPUcc.AddRotationKey(j, std::move(ksk));
                }
            }
        }

        for (auto& i : result.CtS) {
            for (auto& j : i.rotIn) {
                if (j && !GPUcc.HasRotationKey(j)) {
                    KeySwitchingKey ksk(GPUcc);
                    RawKeySwitchKey rksk = GetRotationKeySwitchKey(keys, j, cc);
                    ksk.Initialize(GPUcc, rksk);
                    GPUcc.AddRotationKey(j, std::move(ksk));
                }
            }
            for (auto& j : i.rotOut) {
                if (j && !GPUcc.HasRotationKey(j)) {
                    KeySwitchingKey ksk(GPUcc);
                    RawKeySwitchKey rksk = GetRotationKeySwitchKey(keys, j, cc);
                    ksk.Initialize(GPUcc, rksk);
                    GPUcc.AddRotationKey(j, std::move(ksk));
                }
            }
        }
        std::reverse(result.CtS.begin(), result.CtS.end());
    }

    if (GPUcc.N / 2 != slots) {
        for (uint32_t j = 1; j < GPUcc.N / (2 * slots); j <<= 1) {
            KeySwitchingKey ksk(GPUcc);
            RawKeySwitchKey rksk = GetRotationKeySwitchKey(keys, j * slots, cc);
            ksk.Initialize(GPUcc, rksk);
            GPUcc.AddRotationKey(j * slots, std::move(ksk));
        }
    }

    KeySwitchingKey ksk(GPUcc);
    RawKeySwitchKey rksk = GetConjugateKeySwitchKey(keys, cc);
    ksk.Initialize(GPUcc, rksk);
    GPUcc.AddRotationKey(GPUcc.N * 2 - 1, std::move(ksk));

    result.correctionFactor =
        std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_correctionFactor;

    GPUcc.AddBootPrecomputation(slots, std::move(result));
}
