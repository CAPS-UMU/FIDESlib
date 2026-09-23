//
// Created by carlosad on 24/04/24.
//
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/openfhe-interface/ParameterSwitch.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "Math.cuh"
#include <bit>
#include <cassert>
using namespace lbcrypto;

/**
 * Converts a vector of polynomial limbs to a single flattened array
 */
std::vector<std::vector<uint64_t>> FIDESlib::CKKS::GetRawArray(std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>> polys) {
    // total size is r * N
    int numRes = polys.size();
    int numElements = (polys[0].GetValues() /*.m_values*/).GetLength();

    std::vector<std::vector<uint64_t>> flattened(numRes, std::vector<uint64_t>(numElements));

    // Fill the array
    for (int r = 0; r < numRes; ++r) {
        for (int i = 0; i < numElements; i++) {
            flattened[r][i] = (polys[r].GetValues() /*.m_values*/)[i].ConvertToInt();
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
FIDESlib::CKKS::RawCipherText FIDESlib::CKKS::GetRawCipherText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, lbcrypto::Ciphertext<DCRTPoly> ct, int REV) {
    RawCipherText result; //{ .cc = cc };
    // result.originalCipherText=ct;
    result.numRes = ct->GetElements()[0].GetAllElements().size();
    result.N = ((ct->GetElements()[0].GetAllElements())[0].GetValues() /*.m_values*/).GetLength();
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

    result.Noise = ct->GetScalingFactor();      // m_scalingFactor;
    result.NoiseLevel = ct->GetNoiseScaleDeg(); // m_noiseScaleDeg;
    result.keyid = ct->GetKeyTag();             // keyTag;
    result.slots = ct->GetSlots();

    return result;
};

/**
 * Converts a ciphertext from the RawCiphertext format back to the OpenFHE ciphertext format*/
void FIDESlib::CKKS::GetOpenFHECipherText(lbcrypto::Ciphertext<DCRTPoly> result, RawCipherText raw, int REV) {

    int size = result->GetElements()[0].GetAllElements().size();
    if (size < raw.numRes) {
        raw.numRes = size;
        raw.sub_0.resize(size);
        raw.sub_1.resize(size);
    }
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

    // sub_0.m_vectors=dcrt_0;
    // sub_1.m_vectors=dcrt_1;
    for (size_t i = sub_0.GetParams()->GetParams() /*m_params->m_params*/.size(); i > sub_0.GetAllElements() /*.m_vectors*/.size(); --i) {
        DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_0.GetParams() /*.m_params*/);
        newP->PopLastParam();
        sub_0.m_params.reset(newP);
    }

    for (size_t i = sub_1.GetParams()->GetParams().size(); i > sub_1.GetAllElements().size(); --i) {
        DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_1.GetParams());
        newP->PopLastParam();
        sub_1.m_params.reset(newP);
    }

    std::vector<lbcrypto::DCRTPoly> ct_new = { sub_0, sub_1 };
    result->SetElements(ct_new);

    result->SetScalingFactor(raw.Noise);      // Getm_scalingFactor*/ = raw.Noise;
    result->SetNoiseScaleDeg(raw.NoiseLevel); // /*m_noiseScaleDeg*/ = raw.NoiseLevel;
    result->SetKeyTag(raw.keyid);
    result->SetSlots(raw.slots);
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

    // sub_0.m_vectors=dcrt_0;
    // sub_1.m_vectors=dcrt_1;
    for (size_t i = sub_0.GetParams()->GetParams().size(); i > sub_0.GetAllElements().size(); --i) {
        DCRTPoly::Params* newP = new DCRTPoly::Params(*sub_0.GetParams());
        newP->PopLastParam();
        sub_0.m_params.reset(newP);
    }

    result->GetElement<DCRTPoly>() /*encodedVectorDCRT*/ = sub_0;
    result->SetScalingFactor(raw.Noise);      // scalingFactor = raw.Noise;
    result->SetNoiseScaleDeg(raw.NoiseLevel); // noiseScaleDeg = raw.NoiseLevel;
    result->SetSlots(raw.slots);
    /*
    std::cout << result << std::endl;
bool ok = result->Decode();
    std::cout << ok << " " << result << std::endl;
    */
}

FIDESlib::CKKS::RawPlainText FIDESlib::CKKS::GetRawPlainText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, lbcrypto::Plaintext pt) {
    RawPlainText result; //{.cc = cc};
    result.originalPlainText = pt;
    result.numRes = pt->GetElement<DCRTPoly>().GetAllElements().size();
    result.N = ((pt->GetElement<DCRTPoly>().GetAllElements())[0].GetValues() /*.m_values*/).GetLength();
    result.sub_0 = GetRawArray(pt->GetElement<DCRTPoly>().GetAllElements());
    result.moduli = GetModuli(pt->GetElement<DCRTPoly>().GetAllElements());

    result.format = pt->GetElement<DCRTPoly>().GetFormat();

    if constexpr (REVERSE) {
        for (auto& i : result.sub_0)
            bit_reverse_vector(i);
    }

    result.Noise = pt->GetScalingFactor();
    result.NoiseLevel = pt->GetNoiseScaleDeg();
    result.slots = pt->GetSlots();

    return result;
}

FIDESlib::CKKS::RawPlainText FIDESlib::CKKS::GetRawPlainText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, ReadOnlyPlaintext pt) {
    RawPlainText result; //{.cc = cc};
    // result.originalPlainText = pt;
    result.numRes = pt->GetElement<DCRTPoly>().GetAllElements().size();
    result.N = ((pt->GetElement<DCRTPoly>().GetAllElements())[0].GetValues() /*.m_values*/).GetLength();
    result.sub_0 = GetRawArray(pt->GetElement<DCRTPoly>().GetAllElements());
    result.moduli = GetModuli(pt->GetElement<DCRTPoly>().GetAllElements());

    result.format = pt->GetElement<DCRTPoly>().GetFormat();

    if constexpr (REVERSE) {
        for (auto& i : result.sub_0)
            bit_reverse_vector(i);
    }

    result.Noise = pt->GetScalingFactor();
    result.NoiseLevel = pt->GetNoiseScaleDeg();
    result.slots = pt->GetSlots();

    return result;
}

FIDESlib::CKKS::RawParams FIDESlib::CKKS::GetRawParams(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, FIDESlib::BOOT_CONFIG boot_conf) {
    RawParams result;
    result.N = cc->GetRingDimension();
    result.logN = std::bit_width((uint32_t)result.N) - 1;
    // cc->GetCryptoParameters()
    auto& elem_params = cc->GetCryptoParameters()->GetElementParams();
    result.L = elem_params->GetParams().size() - 1;
    // result.L = cc->params->m_params->m_params.size() - 1;
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    result.scalingTechnique = cryptoParams->GetScalingTechnique();
    // result.qbit = cc->params->m_params->m_params->;
    // auto aux = cc->GetCryptoParameters()->GetParamsPK()->GetParamPartition();

    for (auto& i : /*cc->params->m_params->m_params*/
         cc->GetCryptoParameters()->GetElementParams()->GetParams()) {
        result.moduli.push_back(i->GetModulus().ConvertToInt<uint64_t>() /*m_ciphertextModulus.m_value*/);
        result.root_of_unity.push_back(i->GetRootOfUnity().ConvertToInt<uint64_t>() /*m_rootOfUnity.m_value*/);
        result.cyclotomic_order.push_back(i->GetCyclotomicOrder() /*m_cyclotomicOrder*/);
    }

    for (size_t i = 0; i < result.moduli.size(); ++i) {
        using namespace intnat;
        using FFT = ChineseRemainderTransformFTTNat<NativeVector>;
        auto tables = FFT::GetTables(NativeInteger(result.root_of_unity[i]), result.N << 1, NativeInteger(result.moduli[i]));
        const size_t size = tables->rootReverse.GetLength();
        assert(size == (size_t)result.N);

        for (size_t k = 0; k < size; ++k) {
            result.psi[i].push_back(tables->rootReverse.at(k).ConvertToInt<uint64_t>() /*.m_value*/);
            result.psi_inv[i].push_back(tables->rootInverseReverse.at(k).ConvertToInt<uint64_t>() /*m_value*/);
        }
        result.N_inv.push_back(tables->cycloOrderInverse.at(result.logN).ConvertToInt<uint64_t>() /*.m_value*/);
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

    result.ScalingFactorRealBig.resize(result.L + 1, 0);
    for (size_t i = 0; i < result.ScalingFactorRealBig.size() - 1; ++i) {
        result.ScalingFactorRealBig[result.L - i] = cryptoParams->GetScalingFactorRealBig(i);
    }

    /// Key Switching precomputations !!!
    result.dnum = cryptoParams->GetNumPartQ();
    result.K = cryptoParams->GetParamsP()->GetParams().size();
    assert(cryptoParams->GetNumPartQ() == cryptoParams->GetNumberOfQPartitions());
    cryptoParams->GetNumPerPartQ();

    {
        auto& src = cryptoParams->GetParamsP()->m_params;
        for (auto& i : src) {
            result.SPECIALmoduli.push_back(i->GetModulus().ConvertToInt<uint64_t>() /* m_ciphertextModulus.m_value*/);
            result.SPECIALroot_of_unity.push_back(i->GetRootOfUnity().ConvertToInt<uint64_t>() /*m_rootOfUnity.m_value*/);
            result.SPECIALcyclotomic_order.push_back(i->GetCyclotomicOrder() /*m_cyclotomicOrder*/);
        }
    }

    {
        auto& src = cryptoParams->m_paramsPartQ;
        for (auto& i : src) {
            result.PARTITIONmoduli.emplace_back();
            for (auto& j : i->GetParams()) {
                result.PARTITIONmoduli.back().push_back(j->GetModulus().ConvertToInt<uint64_t>() /*m_ciphertextModulus.m_value*/);
            }
        }
    }

    {
        auto& src = cryptoParams->GetPHatInvModp();
        auto& dest = result.PHatInvModp;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i] = src[i].ConvertToInt<uint64_t>();
        }
    }

    {
        auto& src = cryptoParams->GetPInvModq();
        auto& dest = result.PInvModq;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i] = src[i].ConvertToInt<uint64_t>(); // m_value;
        }
    }

    {
        auto& src = cryptoParams->GetPHatModq();
        auto& dest = result.PHatModq;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i].resize(src[i].size());
            for (size_t j = 0; j < src[i].size(); ++j) {
                dest[i][j] = src[i][j].ConvertToInt<uint64_t>(); // m_value;
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
                    dest[k][i][j] = src[k][i][j].ConvertToInt<uint64_t>(); // m_value;
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
                        dest[k][i][j][l] = src[k][i][j][l].ConvertToInt<uint64_t>(); // m_value;
                    }
                }
            }
        }
    }

    if (cc->GetScheme()->m_FHE) {
        // The API layer (CryptoContextImpl::LoadContext) maps the OpenFHE secret-key
        // distribution to one FIDESlib bootstrap configuration (ENCAPS for
        // SPARSE_ENCAPSULATED, SPARSE for SPARSE_TERNARY, UNIFORM for UNIFORM_TERNARY),
        // which selects the Chebyshev set, the double-angle count and the encapsulation
        // machinery below. Internally FIDESlib can run any key type through the
        // (sparse-)encapsulated bootstrap — only the key material (switching keys) differs
        // per distribution, since the CtS/StC plaintext matrices are always computed
        // host-side by OpenFHE's EvalBootstrapSetup. Those plaintexts bake in a 1/K
        // division (scaleEnc = pre/K: K = K_SPARSE for SPARSE_TERNARY,
        // K_SPARSE_ENCAPSULATED for SPARSE_ENCAPSULATED, K = 1.0 for UNIFORM_TERNARY), so
        // at runtime FHECKKSRNS::EvalBootstrap rescales the mod-reduction input by
        // pre/(k*N) with k = 1.0 for sparse keys and k = K_UNIFORM for the dense (uniform)
        // key. bootK must mirror that runtime k — keyed off the actual secret-key
        // distribution, never a hardcoded constant.
        const auto fhe = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
        const auto keyDist = cryptoParams->GetSecretKeyDist();
        const bool isSparseKey = (keyDist == lbcrypto::SPARSE_TERNARY || keyDist == lbcrypto::SPARSE_ENCAPSULATED);
        const double bootK = isSparseKey ? 1.0 : static_cast<double>(fhe->K_UNIFORM);

        if (boot_conf == FIDESlib::ENCAPS) {
            result.coefficientsCheby = lbcrypto::FHECKKSRNS::g_coefficientsSparseEncapsulated; // degree 32
            // k adjustment for the (sparse-)encapsulated mod-reduction. The CtS/StC matrices are
            // computed host-side by EvalBootstrapSetup and bake scaleEnc = pre/K_pre(dist) with
            // K_pre(SPARSE_TERNARY)=K_SPARSE, K_pre(SPARSE_ENCAPSULATED)=K_SPARSE_ENCAPSULATED,
            // K_pre(UNIFORM_TERNARY)=1. The GPU pipeline then rescales by pre/(k*N); to reach the
            // encapsulated convention baked * (1/k) = 1/K_SPARSE_ENCAPSULATED the runtime k must be
            // K_SPARSE_ENCAPSULATED * K_pre(dist) (missing 1/K must be un-baked): uniform keys need
            // k = 16, sparse-ternary keys k = 16/K_SPARSE, encapsulated keys k = 1 (the OpenFHE
            // runtime k for those keys, cf. FHECKKSRNS::EvalBootstrap). bootK is keyed off the
            // ACTUAL secret-key distribution — never a hardcoded constant.
            result.bootK = keyDist == lbcrypto::SPARSE_TERNARY ?
                static_cast<double>(lbcrypto::FHECKKSRNS::K_SPARSE_ENCAPSULATED) / lbcrypto::FHECKKSRNS::K_SPARSE :
                keyDist == lbcrypto::UNIFORM_TERNARY ?
                static_cast<double>(lbcrypto::FHECKKSRNS::K_SPARSE_ENCAPSULATED) :
                1.0;
            result.sparse_encaps = true;
            result.doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
        } else if (boot_conf == FIDESlib::SPARSE) {
            result.coefficientsCheby = lbcrypto::FHECKKSRNS::g_coefficientsSparse;
            result.bootK = bootK;
            result.doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
            result.sparse_encaps = false;
        } else if (boot_conf == FIDESlib::UNIFORM) {
            // result.coefficientsCheby = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
            result.coefficientsCheby = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
            result.bootK = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->K_UNIFORM; // lbcrypto::FHECKKSRNS::K_UNIFORM;
            result.doubleAngleIts = lbcrypto::FHECKKSRNS::R_UNIFORM;
            result.sparse_encaps = false;
        }
    }

    result.p = cryptoParams->GetPlaintextModulus();

    return result;
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetKeySwitchKey(std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> ek) {

    std::vector<std::vector<std::vector<uint64_t>>> a_moduli;
    std::vector<std::vector<std::vector<std::vector<uint64_t>>>> a;
    std::vector<std::vector<std::vector<uint64_t>>> b;
    std::string keytag;

    // for (auto a_raw = ek.get()->m_rKey; auto& i : a_raw) {
    for (/*auto a_raw = ek.get()->m_rKey;*/ auto& i : { ek->GetAVector(), ek->GetBVector() }) {
        std::vector<std::vector<std::vector<uint64_t>>> a_inner;
        std::vector<std::vector<uint64_t>> a_inner_moduli;
        for (auto& j : i) {
            auto v = GetRawArray(j.GetAllElements() /*.m_vectors*/);
            a_inner_moduli.emplace_back();
            auto& a_aux = a_inner_moduli.back();
            for (auto& p : j.GetParams()->GetParams() /*m_params->m_params*/) {
                a_aux.push_back(p->GetModulus().ConvertToInt<uint64_t>() /* m_ciphertextModulus.m_value*/);
            }
            a_inner.push_back(v);
        }
        a.push_back(a_inner);
        a_moduli.push_back(a_inner_moduli);
    }
    keytag = ek->GetKeyTag();

    return RawKeySwitchKey(std::move(a_moduli), std::move(a), std::move(b), std::move(keytag));
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetEvalKeySwitchKey(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys) {

    // const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());

    auto& keyMap = keys.publicKey->GetCryptoContext()->GetAllEvalMultKeys();
    // lbcrypto::CryptoContextImpl<DCRTPoly>::s_evalMultKeyMap;
    if (keyMap.find(keys.secretKey->GetKeyTag()) != keyMap.end()) {
        const std::vector<EvalKey<DCRTPoly>>& key = keyMap[keys.secretKey->GetKeyTag()];
        const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key.at(0));
        return GetKeySwitchKey(ek);
    } else {
        assert("EvalKey is not present !!!" == nullptr);
    }

    return RawKeySwitchKey{};
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetEvalKeySwitchKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
    auto& keyMap = cc->GetAllEvalMultKeys();
    // lbcrypto::CryptoContextImpl<DCRTPoly>::s_evalMultKeyMap;
    if (keyMap.find(publicKey->GetKeyTag()) != keyMap.end()) {
        const std::vector<EvalKey<DCRTPoly>>& key = keyMap[publicKey->GetKeyTag()];
        const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key.at(0));

        return GetKeySwitchKey(ek);
    } else {
        assert("EvalKey is not present !!!" == nullptr);
    }

    return RawKeySwitchKey{};
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetRotationKeySwitchKey(const KeyPair<lbcrypto::DCRTPoly>& keys,
                                                                        int index,
                                                                        lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc) {
    auto& keyMap = cc->GetAllEvalAutomorphismKeys();

    if (keyMap.find(keys.secretKey->GetKeyTag()) != keyMap.end()) {
        auto& keyMap2 = keyMap[keys.secretKey->GetKeyTag()];
        uint32_t x = FIDESlib::modpow(5, index, cc->GetRingDimension() * 2);
        if (keyMap2->find(x) == keyMap2->end()) {
            cc->EvalAtIndexKeyGen(keys.secretKey, { index });
        }
        {
            const auto& key = keyMap2->at(x);
            assert(key != nullptr);
            const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key);

            return GetKeySwitchKey(ek);
        }
    } else {
        assert("RotKey is not present !!!" == nullptr);
        std::cout << "RotKey is not present !!!" << std::endl;
    }
    return RawKeySwitchKey{};
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetRotationKeySwitchKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int index) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
    auto& keyMap = cc->GetAllEvalAutomorphismKeys();

    if (keyMap.find(publicKey->GetKeyTag()) != keyMap.end()) {
        auto& keyMap2 = keyMap[publicKey->GetKeyTag()];
        uint32_t x = FIDESlib::modpow(5, index, cc->GetRingDimension() * 2);

        {
            const auto& key = keyMap2->at(x);
            assert(key != nullptr);
            const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key);

            return GetKeySwitchKey(ek);
        }
    } else {
        assert("RotKey is not present !!!" == nullptr);
        std::cout << "RotKey is not present !!!" << std::endl;
    }
    return RawKeySwitchKey{};
}

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetConjugateKeySwitchKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
    auto& keyMap2 = cc->GetEvalAutomorphismKeyMap(publicKey->GetKeyTag());

    if (keyMap2.find(2 * cc->GetRingDimension() - 1) != keyMap2.end()) {
        const auto& key = keyMap2.at(2 * cc->GetRingDimension() - 1);
        assert(key != nullptr);
        const auto ek = std::dynamic_pointer_cast<EvalKeyRelinImpl<DCRTPoly>>(key);
        // std::cout << std::endl << "Clave " << ek->GetKeyTag() << "\n";
        return GetKeySwitchKey(ek);
    } else {
        assert("RotKey is not present for rotation !!!" == nullptr);
        std::cout << "RotKey is not present for conjugation!!!" << std::endl;
    }
    return RawKeySwitchKey{};
}

#include "CKKS/BootstrapPrecomputation.cuh"

#include "CKKS/KeySwitchingKey.cuh"

#include "CKKS/LimbPartition.cuh"

#include "CKKS/RNSPoly.cuh"

std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>
FIDESlib::CKKS::GenRotationKeys(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, std::vector<int> indexes) {
    return GenRotationKeys(keys.secretKey, indexes);
}

std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>
FIDESlib::CKKS::GenRotationKeys(const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& keys, std::vector<int> indexes) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = keys->GetCryptoContext();
    std::set<int> indexes2(indexes.begin(), indexes.end());
    std::vector<int> indexes3;
    for (int i : indexes2) {
        if (i) {
            indexes3.emplace_back(i);
        }
    }
    auto evalKeys = cc->GetScheme()->EvalAtIndexKeyGen(keys, indexes3);
    CryptoContextImpl<lbcrypto::DCRTPoly>::InsertEvalAutomorphismKey(evalKeys, keys->GetKeyTag());
    return evalKeys;
    // std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)
    //     ->cc->EvalAtIndexKeyGen(keys.secretKey, indexes3);
}

void FIDESlib::CKKS::AddRotationKeys(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, FIDESlib::CKKS::Context& GPUcc, std::vector<int> indexes) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
    std::set<int> indexes2(indexes.begin(), indexes.end());
    std::vector<int> indexes3;
    for (int i : indexes2) {
        if (i && !GPUcc->HasRotationKey(i, publicKey->GetKeyTag())) {
            indexes3.emplace_back(i);
        }
    }
    for (int i : indexes3) {
        auto clave_rotacion = FIDESlib::CKKS::GetRotationKeySwitchKey(publicKey, i);
        // std::cout << "Load rotation key " << i << std::endl;
        FIDESlib::CKKS::KeySwitchingKey clave_rotacion_gpu(GPUcc);
        clave_rotacion_gpu.Initialize(clave_rotacion);
        GPUcc->AddRotationKey(i, std::move(clave_rotacion_gpu));
    }
}

void FIDESlib::CKKS::GenAndAddRotationKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                           const KeyPair<lbcrypto::DCRTPoly>& keys,
                                           FIDESlib::CKKS::Context& GPUcc,
                                           std::vector<int> indexes) {

    GenRotationKeys(keys, indexes);
    AddRotationKeys(keys.publicKey, GPUcc, indexes);
}

constexpr bool remove_extension = false;
[[maybe_unused]] constexpr bool MAKE_CTS_LT_FRIENDLY = true;
[[maybe_unused]] constexpr bool MAKE_STC_LT_FRIENDLY = true;

std::vector<int> FIDESlib::CKKS::GetBootstrapIndexes(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
                                                     int slots,
                                                     FIDESlib::CKKS::BootstrapPrecomputation* result_) {
    // ContextData& GPUcc = *GPUcc_;
    std::vector<int> indexes;
    BootstrapPrecomputation result;
    auto precom = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.find(slots)->second;

    uint32_t M = cc->GetCyclotomicOrder();
    uint32_t N = cc->GetRingDimension();

    int slots_red = std::min((int)M / 4, slots * 2);

    if (precom->m_paramsEnc.lvlb /* [CKKS_BOOT_PARAMS::LEVEL_BUDGET]*/ == 1 && precom->m_paramsDec.lvlb /*[CKKS_BOOT_PARAMS::LEVEL_BUDGET]*/ == 1) {

        result.LT.slots = slots;
        result.LT.bStep = (precom->m_paramsEnc.g /*m_dim1*/ == 0) ? ceil(sqrt(slots)) : precom->m_paramsEnc.g /*m_dim1*/;

        for (int i = 1; i < result.LT.bStep; ++i) {
            indexes.push_back(ReduceRotation(i, slots_red));
        }

#if AFFINE_LT
        indexes.push_back(ReduceRotation(result.LT.bStep, slots_red));
#else
        for (int i = result.LT.bStep; i < result.LT.slots; i += result.LT.bStep) {
            indexes.push_back(i);
        }
#endif
    } else {
        {
            // CoeffToSlots metadata

            int32_t levelBudget = precom->m_paramsEnc.lvlb;                // [CKKS_BOOT_PARAMS::LEVEL_BUDGET];
            int32_t layersCollapse = precom->m_paramsEnc.layersCollapse;   //[CKKS_BOOT_PARAMS::LAYERS_COLL];
            int32_t remCollapse = precom->m_paramsEnc.remCollapse;         // [CKKS_BOOT_PARAMS::LAYERS_REM];
            int32_t numRotations = precom->m_paramsEnc.numRotations;       // [CKKS_BOOT_PARAMS::NUM_ROTATIONS];
            int32_t b = precom->m_paramsEnc.b;                             // [CKKS_BOOT_PARAMS::BABY_STEP];
            int32_t g = precom->m_paramsEnc.g;                             //[CKKS_BOOT_PARAMS::GIANT_STEP];
            int32_t numRotationsRem = precom->m_paramsEnc.numRotationsRem; //[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
            int32_t bRem = precom->m_paramsEnc.bRem;                       // [CKKS_BOOT_PARAMS::BABY_STEP_REM];
            int32_t gRem = precom->m_paramsEnc.gRem;                       //[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

            int32_t stop = -1;
            int32_t flagRem = 0;

            auto algo = cc->GetScheme();

            if (remCollapse != 0) {
                stop = 0;
                flagRem = 1;
            }

            // precompute the inner and outer rotations
            {
                result.CtS.resize(levelBudget);
                for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {
                    if (flagRem == 1 && i == 0) {
                        // remainder corresponds to index 0 in encoding and to last index in decoding
                        result.CtS[i].bStep = gRem;
                        result.CtS[i].gStep = bRem;
                        result.CtS[i].slots = numRotationsRem;
                        result.CtS[i].rotIn.resize(gRem);
                        result.CtS[i].rotOut.resize(bRem);
                    } else {
                        result.CtS[i].bStep = g;
                        result.CtS[i].gStep = b;
                        result.CtS[i].slots = numRotations;
                        result.CtS[i].rotIn.resize(g);
                        result.CtS[i].rotOut.resize(b);
                    }
                }

                for (int32_t s = levelBudget - 1; s > stop; s--) {
                    for (int32_t j = 0; j < g; j++) {
                        result.CtS[s].rotIn[j] = ReduceRotation((j) * (1 << ((s - flagRem) * layersCollapse + remCollapse)), slots_red);
                    }

                    for (int32_t i = 0; i < b; i++) {
                        result.CtS[s].rotOut[i] = ReduceRotation((g * i) * (1 << ((s - flagRem) * layersCollapse + remCollapse)), slots_red);
                    }
                }

                if (flagRem) {
                    for (int32_t j = 0; j < gRem; j++) {
                        result.CtS[stop].rotIn[j] = ReduceRotation((j), slots_red);
                    }

                    for (int32_t i = 0; i < bRem; i++) {
                        result.CtS[stop].rotOut[i] = ReduceRotation((gRem * i), slots_red);
                    }
                }
            }

            // std::cout << g << " " << b << " " << gRem << " " << bRem << std::endl;
        }

        {
            // SlotToCoeff metadata
            uint32_t M = cc->GetCyclotomicOrder();
            uint32_t N = cc->GetRingDimension();

            int32_t levelBudget = precom->m_paramsDec.lvlb;                // [CKKS_BOOT_PARAMS::LEVEL_BUDGET];
            int32_t layersCollapse = precom->m_paramsDec.layersCollapse;   //[CKKS_BOOT_PARAMS::LAYERS_COLL];
            int32_t remCollapse = precom->m_paramsDec.remCollapse;         // [CKKS_BOOT_PARAMS::LAYERS_REM];
            int32_t numRotations = precom->m_paramsDec.numRotations;       // [CKKS_BOOT_PARAMS::NUM_ROTATIONS];
            int32_t b = precom->m_paramsDec.b;                             // [CKKS_BOOT_PARAMS::BABY_STEP];
            int32_t g = precom->m_paramsDec.g;                             //[CKKS_BOOT_PARAMS::GIANT_STEP];
            int32_t numRotationsRem = precom->m_paramsDec.numRotationsRem; //[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
            int32_t bRem = precom->m_paramsDec.bRem;                       // [CKKS_BOOT_PARAMS::BABY_STEP_REM];
            int32_t gRem = precom->m_paramsDec.gRem;                       //[CKKS_BOOT_PARAMS::GIANT_STEP_REM];
            auto algo = cc->GetScheme();

            int32_t flagRem = 0;

            if (remCollapse != 0) {
                flagRem = 1;
            }

            // precompute the inner and outer rotations
            {
                result.StC.resize(levelBudget);
                for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {

                    if (flagRem == 1 && i == uint32_t(levelBudget - 1)) {
                        // remainder corresponds to index 0 in encoding and to last index in decoding
                        result.StC[i].bStep = gRem;
                        result.StC[i].gStep = bRem;
                        result.StC[i].slots = numRotationsRem;
                        result.StC[i].rotIn.resize(gRem);
                        result.StC.at(i).rotOut.resize(bRem);
                    } else {
                        result.StC[i].bStep = g;
                        result.StC[i].gStep = b;
                        result.StC[i].slots = numRotations;
                        result.StC[i].rotIn.resize(g);
                        result.StC.at(i).rotOut.resize(b);
                    }
                }

                for (int32_t s = 0; s < levelBudget - flagRem; s++) {
                    for (int32_t j = 0; j < g; j++) {
                        result.StC.at(s).rotIn.at(j) = ReduceRotation((j) * (1 << (s * layersCollapse)), slots_red);
                    }

                    for (int32_t i = 0; i < b; i++) {
                        result.StC.at(s).rotOut.at(i) = ReduceRotation((g * i) * (1 << (s * layersCollapse)), slots_red);
                    }
                }

                if (flagRem) {
                    int32_t s = levelBudget - flagRem;
                    for (int32_t j = 0; j < gRem; j++) {
                        result.StC.at(s).rotIn.at(j) = ReduceRotation((j) * (1 << (s * layersCollapse)), slots_red);
                    }

                    for (int32_t i = 0; i < bRem; i++) {
                        result.StC.at(s).rotOut.at(i) = ReduceRotation((gRem * i) * (1 << (s * layersCollapse)), slots_red);
                    }
                }
            }

            // std::cout << g << " " << b << " " << gRem << " " << bRem << std::endl;
        }

        std::reverse(result.CtS.begin(), result.CtS.end());

        {
            int acc_offset = 0;
            for (size_t i = 0; i < result.CtS.size(); i++) {
                acc_offset += (result.CtS[i].slots / 2) * (result.CtS[i].bStep > 1 ?
                    result.CtS[i].rotIn[1] :
                    (result.CtS[i].gStep > 1 ? result.CtS[i].rotOut[1] : 0));
            }
            indexes.emplace_back(ReduceRotation(-acc_offset, slots_red));

            auto& j = result.CtS.back().rotOut[0];
            j = ReduceRotation(-acc_offset, slots_red);
        }

        {
            int acc_offset = 0;
            for (size_t i = 0; i < result.StC.size(); i++) {
                acc_offset += (result.StC[i].slots / 2) * (result.StC[i].bStep > 1 ?
                    result.StC[i].rotIn[1] :
                    (result.StC[i].gStep > 1 ? result.StC[i].rotOut[1] : 0));
            }
            indexes.emplace_back(ReduceRotation(-acc_offset, slots_red));

            auto& j = result.StC.back().rotOut[0];
            j = ReduceRotation(-acc_offset, slots_red);
        }

        for (auto& v : { &result.CtS, &result.StC }) {
            for (auto& i : *v) {
                for (auto& j : i.rotIn) {
                    indexes.push_back(ReduceRotation(j, slots_red));
                }
#if AFFINE_LT
                // We do not include rotOut[0], it is later set to 0 but the last that is set to acc_offset
                for (auto& j : {
                         /*i.rotOut[0],*/ i.rotOut.size() > 1 ? i.rotOut[1] /*- i.rotOut[0]*/ : 0 }) {
                    indexes.push_back(ReduceRotation(j, slots_red));
                }
#else
                for (auto& j : i.rotOut) {
                    if (j && !GPUcc.HasRotationKey(j)) {
                        indexes.push_back(j);
                    }
                }
#endif
            }
        }
    }

    /*
    int slots_transform = std::min((int)slots * 2, (int)cc->GetCyclotomicOrder() / 4);
    for (auto& i : indexes) {
        auto j_ = i % slots_transform;
        if (j_ < 0)
            j_ += slots_transform;
        if (j_ > slots_transform / 2)
            j_ += cc->GetCyclotomicOrder() / 4 - slots_transform;
        i = j_;
    }
    */

    if (cc->GetRingDimension() / 2 != static_cast<uint32_t>(slots)) {
        int bStep = PARTIAL_SUM_RADIX;
        result.accumulate_bStep = bStep;
        std::vector<int> rotations = GetAccumulateRotationIndices(bStep, slots, cc->GetRingDimension() / 2 / slots);
        for (auto idx : rotations) {
            indexes.push_back(idx);
        }
    }

    if (result_)
        *result_ = std::move(result);
    return indexes;
}

void FIDESlib::CKKS::GenBootstrapKeys(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, int slots, bool SSE) {
    GenBootstrapKeys(keys.secretKey, slots, SSE);
}

// FIDESlib sparse-encapsulation auxiliary key slots in the OpenFHE automorphism key map.
// The indices are even, so they can never collide with the odd 5^k mod 2N rotation keys. OpenFHE
// reserves 2N-2/2N-4 for ITS OWN SPARSE_ENCAPSULATED sparse-switch keys, whose format (the
// 3-limb q0*P' KeySwitchGenSparse key + a hybrid KeySwitchGen) is incompatible with FIDESlib's
// switchable-context keys; FHECKKSRNS::EvalBootstrap reads exactly those two slots (KeySwitchSparse
// via evalKeyMap.at(2N-4) and KeySwitchInPlace via .at(2N-2)) and crashes on a foreign key, so
// FIDESlib must NOT reuse them. The next two even indices down (2N-6, 2N-8) are unused by both
// OpenFHE and FIDESlib rotation conventions.
static constexpr uint32_t BTS_KSPARSE_TO_SLOT = 6;   // 2N-6: switch to   the sparse (tug) key
static constexpr uint32_t BTS_KSPARSE_FROM_SLOT = 8; // 2N-8: switch back to the real key

void FIDESlib::CKKS::GenBootstrapKeys(const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& keys, int slots, bool SSE) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = keys->GetCryptoContext();
    std::vector<int> indexes = GetBootstrapIndexes(cc, slots, nullptr);
    cc->EvalMultKeyGen(keys);

    auto evalKeys = GenRotationKeys(keys, GetBootstrapIndexes(cc, slots, nullptr));
    auto conjKey = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->ConjugateKeyGen(keys);

    (*evalKeys)[cc->GetCyclotomicOrder() - 1] = conjKey;

    CryptoContextImpl<lbcrypto::DCRTPoly>::InsertEvalAutomorphismKey(evalKeys, keys->GetKeyTag());
    // Reinsert all keys to add the particular conj key and sse keys

    if (SSE) {
        auto cc_switch = CKKS::createSwitchableContextBasedOnContext(cc, 1, 1, cc->GetRingDimension() / 2);

        // auto keys_switch = cc_switch->KeyGen();
        auto [swtch, sk_sparse] = CKKS::createContextSwitchingKeys(cc, cc_switch, keys, 32);
        (*evalKeys)[cc->GetCyclotomicOrder() - BTS_KSPARSE_TO_SLOT] = swtch.first;
        (*evalKeys)[cc->GetCyclotomicOrder() - BTS_KSPARSE_FROM_SLOT] = swtch.second;

        // The GPU pipeline materializes the two auxiliary slots in AddBootstrapKeys and expects
        // the FIDESlib switching keys generated above (createContextSwitchingKeys over the
        // switchable context) there. CryptoContextImpl::InsertEvalAutomorphismKey does NOT
        // replace indices that are already present in the map (despite its "silently replaces"
        // doc comment), so force the FIDESlib keys into the actual automorphism map. The slots
        // are FIDESlib-exclusive (see BTS_KSPARSE_*_SLOT), so nothing pre-exists there and this
        // write simply makes insertion order-independent.
        auto& keyMap = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetEvalAutomorphismKeyMap(keys->GetKeyTag());
        keyMap[cc->GetCyclotomicOrder() - BTS_KSPARSE_TO_SLOT] = swtch.first;
        keyMap[cc->GetCyclotomicOrder() - BTS_KSPARSE_FROM_SLOT] = swtch.second;

        // We can discard sk_sparse and cc_switch
    }
}

void FIDESlib::CKKS::AddBootstrapKeys(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int slots, FIDESlib::CKKS::Context& GPUcc_) {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();
    FIDESlib::CKKS::BootstrapPrecomputation& result = GPUcc_->GetBootPrecomputation(slots);

    ContextData& GPUcc = *GPUcc_;
    std::vector<int> indexes = GetBootstrapIndexes(cc, slots, nullptr);

    // std::cout << "Add eval key" << std::endl;
    {
        KeySwitchingKey ksk(GPUcc_);
        RawKeySwitchKey rksk = GetEvalKeySwitchKey(publicKey);
        ksk.Initialize(rksk);
        GPUcc.AddEvalKey(std::move(ksk));
    }

    // std::cout << "Add conjugate key" << std::endl;
    {
        KeySwitchingKey ksk(GPUcc_);
        RawKeySwitchKey rksk = GetConjugateKeySwitchKey(publicKey);
        ksk.Initialize(rksk);
        GPUcc.AddRotationKey(GPUcc.N * 2 - 1, std::move(ksk));
    }
    // std::cout << "Add rotation keys" << std::endl;

    AddRotationKeys(publicKey, GPUcc_, indexes);

    if (GPUcc.param.raw->sparse_encaps) {
        auto cc_switch = CKKS::createSwitchableContextBasedOnContext(cc, 1, 1, cc->GetRingDimension() / 2);

        auto& evalKeys = cc->GetEvalAutomorphismKeyMap(publicKey->GetKeyTag());

        FIDESlib::CKKS::RawParams raw_param2 = FIDESlib::CKKS::GetRawParams(cc_switch);
        // FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), devices};
        FIDESlib::CKKS::Context cc_switch_ = CKKS::GenCryptoContextGPU(GPUcc.param.adaptTo(raw_param2), GPUcc.GPUid);
        FIDESlib::CKKS::ContextData& GPUcc2 = *cc_switch_;

        // std::cout << "Add atob key" << std::endl;
        FIDESlib::CKKS::KeySwitchingKey ksk_atob(cc_switch_);

        {
            std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> res =
                std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(evalKeys[2 * GPUcc.N - BTS_KSPARSE_TO_SLOT]);
            assert(res != nullptr);
            FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetKeySwitchKey(res);

            // Key-tag normalization (sparse-encapsulation switch keys).
            //
            // The sparse-switch dance is internal to FIDESlib — from the outside the ciphertext
            // always stays under the main key — so GetSecretSwitchingKey must find both the
            // "to" and the "from" directional keys indexed by the main key's tag. The keys
            // materialized here come from FIDESlib's own GenBootstrapKeys (slots 2N-6/2N-8,
            // see BTS_KSPARSE_*_SLOT), whose createContextSwitchingKeys products already carry
            // the main key tag; this normalization re-asserts the invariant at materialization
            // time, so the lookups hold whatever the automorphism map ended up holding.
            //
            // (Historical context: FIDESlib previously reused OpenFHE's own SPARSE_ENCAPSULATED
            // slots 2N-2/2N-4. FHECKKSRNS::EvalBootstrapKeyGen authors ITS keys there —
            // KeySwitchGenSparse(privateKey, skNew) whose target is a tag-less internal tug key,
            // hence a foreign auto-generated tag that never equaled the ciphertext tag — and
            // CryptoContextImpl::InsertEvalAutomorphismKey does not replace existing indices, so
            // the GPU consumed OpenFHE's 3-limb q0*P' keys and the switch-back lookup missed.
            // The CPU bootstrap reads those same two slots and crashes on a foreign key, which is
            // why FIDESlib moved its auxiliary keys to exclusive indices.)
            rawKskEval.keyid = publicKey->GetKeyTag();
            ksk_atob.Initialize(rawKskEval);
        }
        // std::cout << "Add btoa key" << std::endl;
        FIDESlib::CKKS::KeySwitchingKey ksk_btoa(GPUcc_);
        {
            std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> res =
                std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(evalKeys[2 * GPUcc.N - BTS_KSPARSE_FROM_SLOT]);
            FIDESlib::CKKS::RawKeySwitchKey rawKskEval2 = FIDESlib::CKKS::GetKeySwitchKey(res);

            // See the key-tag normalization comment above.
            rawKskEval2.keyid = publicKey->GetKeyTag();
            ksk_btoa.Initialize(rawKskEval2);
        }

        CKKS::AddSecretSwitchingKey(std::move(ksk_atob), std::move(ksk_btoa));

        result.sparse_context = cc_switch_;
    }

    std::cout << "Rotation keys loaded: " << GPUcc.precom.keys.begin()->second.rot_keys.size() << " ~ "
        << 2 * ((long long)GPUcc.precom.keys.begin()->second.rot_keys.size() * GPUcc.dnum * (GPUcc.L + GPUcc.K + 1) * GPUcc.N * 8 / (1 << 20)) << "MB"
        << std::endl;
}

void FIDESlib::CKKS::AddBootstrapPlaintexts(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
                                            int slots,
                                            FIDESlib::CKKS::Context& GPUcc_,
                                            FIDESlib::CKKS::BootstrapPrecomputation& result) {
    ContextData& GPUcc = *GPUcc_;
    auto precom = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.find(slots)->second;

    if (precom->m_paramsEnc.lvlb /*[CKKS_BOOT_PARAMS::LEVEL_BUDGET]*/ == 1 && precom->m_paramsDec.lvlb /*[CKKS_BOOT_PARAMS::LEVEL_BUDGET]*/ == 1) {

        if (!GPUcc.HasBootPrecomputation(slots)) {
            if constexpr (1) {
                // extended limbs computation
                auto auxA = precom->m_U0hatTPre;
                auto auxInvA = precom->m_U0Pre;

                result.LT.A.clear();
                for (uint32_t i = 0; i < auxA.size(); ++i) {
                    RawPlainText raw = GetRawPlainText(cc, auxA.at(i));
                    result.LT.A.emplace_back(GPUcc_, raw);
                    if constexpr (remove_extension)
                        result.LT.A.back().c0.freeSpecialLimbs();
                }

                result.LT.invA.clear();
                for (uint32_t i = 0; i < auxInvA.size(); ++i) {
                    RawPlainText raw = GetRawPlainText(cc, auxInvA.at(i));
                    result.LT.invA.emplace_back(GPUcc_, raw);
                    if constexpr (remove_extension)
                        result.LT.invA.back().c0.freeSpecialLimbs();
                }
            }
        }

    } else {

        if (!GPUcc.HasBootPrecomputation(slots)) {
            uint32_t M = cc->GetCyclotomicOrder();
            auto& A = precom->m_U0hatTPreFFT;
            auto& invA = precom->m_U0PreFFT;

            for (uint32_t i = 0; i < A.size(); ++i) {
                for (uint32_t j = 0; j < A.at(A.size() - 1 - i).size(); ++j) {
                    RawPlainText raw = GetRawPlainText(cc, A.at(A.size() - 1 - i).at(j));
                    result.CtS.at(i).A.emplace_back(GPUcc_, raw);
                    if constexpr (remove_extension)
                        result.CtS.at(i).A.back().c0.freeSpecialLimbs();
                }
            }

            for (uint32_t i = 0; i < invA.size(); ++i) {
                for (uint32_t j = 0; j < invA.at(i).size(); ++j) {
                    RawPlainText raw = GetRawPlainText(cc, invA.at(i).at(j));
                    result.StC.at(i).A.emplace_back(GPUcc_, raw);
                    if constexpr (remove_extension)
                        result.StC.at(i).A.back().c0.freeSpecialLimbs();
                }
            }
        }
    }
}

void FIDESlib::CKKS::AddBootstrapPrecomputation(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int slots, FIDESlib::CKKS::Context& GPUcc_) {
    ContextData& GPUcc = *GPUcc_;
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = publicKey->GetCryptoContext();

    if (!GPUcc.HasBootPrecomputation(slots)) {
        FIDESlib::CKKS::BootstrapPrecomputation result_;

        FIDESlib::CKKS::BootstrapPrecomputation& result = GPUcc.HasBootPrecomputation(slots) ? GPUcc_->GetBootPrecomputation(slots) : result_;

        std::vector<int> indexes = GetBootstrapIndexes(cc, slots, &result);

        AddBootstrapPlaintexts(cc, slots, GPUcc_, result);

        result.correctionFactor = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_correctionFactor;

        if (GPUcc.param.raw->sparse_encaps) {
            result.sparse_encaps = true;
        }

        GPUcc.AddBootPrecomputation(slots, std::move(result));
    }

    AddBootstrapKeys(publicKey, slots, GPUcc_);
}

void FIDESlib::CKKS::AddBootstrapPrecomputation(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
                                                const KeyPair<lbcrypto::DCRTPoly>& keys,
                                                int slots,
                                                FIDESlib::CKKS::Context& GPUcc_) {

    GenBootstrapKeys(keys, slots, GPUcc_->param.raw->sparse_encaps);

    AddBootstrapPrecomputation(keys.publicKey, slots, GPUcc_);
}
