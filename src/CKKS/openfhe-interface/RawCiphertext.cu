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
    if (sub_0.GetAllElements().size() < static_cast<size_t>(raw.numRes)) {
        // Shell too shallow: rebuild the elements on the full modulus chain so the payload is not
        // silently truncated (the old code clamped raw.numRes to the shell's tower count and lost
        // the extra towers of the bootstrapped result). The chain comes from the scheme's element
        // params (the shell's own element params are already trimmed to its tower count); it is
        // trimmed back to raw.numRes below, and SetLevel already yields L0 - raw.numRes from the
        // shell's level/tower count.
        const auto chain = result->GetCryptoParameters()->GetElementParams();
        sub_0 = DCRTPoly(chain, result->GetElements().at(0).GetFormat(), /*initializeElementToZero=*/true);
        sub_1 = DCRTPoly(chain, result->GetElements().at(1).GetFormat(), /*initializeElementToZero=*/true);
    }
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

    // Changed the REV variable to an argument to the function.
    // purpose: profiling Automorph kernel with both bit-reversed and normal order ciphertexts.
    if (!REV) {
        for (auto& i : raw.sub_0)
            bit_reverse_vector(i);
    }
    DCRTPoly sub_0 = result->GetElement<DCRTPoly>();
    if (sub_0.GetAllElements().size() < static_cast<size_t>(raw.numRes)) {
        // A plaintext element carries only its own (already trimmed) params, so it cannot be grown
        // to the payload's tower count; fail loudly instead of the old assert/UB.
        OPENFHE_THROW("GetOpenFHEPlaintext: shell plaintext has fewer towers (" +
            std::to_string(sub_0.GetAllElements().size()) + ") than the payload (" +
            std::to_string(raw.numRes) + "); encode the shell at a deeper level");
    }
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

FIDESlib::CKKS::RawParams FIDESlib::CKKS::GetRawParams(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, FIDESlib::BOOT_CONFIG boot_conf,
                                                       bool sparseKS) {
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
    assert(cryptoParams->GetNumPartQ() == cryptoParams->GetNumberOfQPartitions());

    // Sparse secret encapsulation: when sparseKS is requested and the source context is
    // SPARSE_ENCAPSULATED, the special basis of this (switchable) context is the auxiliary
    // basis P' of the sparse key switch, with the sparse PHatInv/PHat/PInv tables, rather
    // than the ordinary hybrid P. K then counts the P' primes.
    const bool useSparseP = sparseKS && (cryptoParams->GetSparseKSParamsP() != nullptr) &&
                            !cryptoParams->GetSparseKSParamsP()->GetParams().empty();

    {
        auto& src = useSparseP ? cryptoParams->GetSparseKSParamsP()->m_params : cryptoParams->GetParamsP()->m_params;
        for (auto& i : src) {
            result.SPECIALmoduli.push_back(i->GetModulus().ConvertToInt<uint64_t>() /* m_ciphertextModulus.m_value*/);
            result.SPECIALroot_of_unity.push_back(i->GetRootOfUnity().ConvertToInt<uint64_t>() /*m_rootOfUnity.m_value*/);
            result.SPECIALcyclotomic_order.push_back(i->GetCyclotomicOrder() /*m_cyclotomicOrder*/);
        }
    }
    result.K = result.SPECIALmoduli.size();

    {
        for (uint32_t part = 0; part < cryptoParams->GetNumPartQ(); ++part) {
            const auto& i = cryptoParams->GetParamsPartQ(part);
            result.PARTITIONmoduli.emplace_back();
            for (auto& j : i->GetParams()) {
                result.PARTITIONmoduli.back().push_back(j->GetModulus().ConvertToInt<uint64_t>() /*m_ciphertextModulus.m_value*/);
            }
        }
    }

    {
        auto& src = useSparseP ? cryptoParams->GetSparseKSPHatInvModp() : cryptoParams->GetPHatInvModp();
        auto& dest = result.PHatInvModp;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i] = src[i].ConvertToInt<uint64_t>();
        }
    }

    {
        auto& src = useSparseP ? cryptoParams->GetSparseKSPInvModq() : cryptoParams->GetPInvModq();
        auto& dest = result.PInvModq;
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i] = src[i].ConvertToInt<uint64_t>(); // m_value;
        }
    }

    {
        auto& dest = result.PHatModq;
        if (useSparseP) {
            // GetSparseKSPHatModq is [q_i][p'_j] (target Ql major); the GPU ModDown2 kernel
            // indexes the matrix as [source p'_j][target q_i] (like the hybrid GetPHatModq),
            // so transpose it here.
            auto& src = cryptoParams->GetSparseKSPHatModq();
            const size_t nQ = src.size();
            const size_t nP = nQ > 0 ? src[0].size() : 0;
            dest.assign(nP, std::vector<uint64_t>(nQ));
            for (size_t q = 0; q < nQ; ++q)
                for (size_t p = 0; p < nP; ++p)
                    dest[p][q] = src[q][p].ConvertToInt<uint64_t>();
        } else {
            auto& src = cryptoParams->GetPHatModq();
            dest.resize(src.size());
            for (size_t i = 0; i < src.size(); ++i) {
                dest[i].resize(src[i].size());
                for (size_t j = 0; j < src[i].size(); ++j) {
                    dest[i][j] = src[i][j].ConvertToInt<uint64_t>(); // m_value;
                }
            }
        }
    }

    // Exact-HPS mod-down overflow correction [a*P']_{q_i} and 1/p'_j for the sparse key switch.
    if (useSparseP) {
        auto& src = cryptoParams->GetSparseKSAlphaPModq();
        auto& dest = result.SparseAlphaPModq;
        dest.resize(src.size());
        for (size_t a = 0; a < src.size(); ++a) {
            dest[a].resize(src[a].size());
            for (size_t i = 0; i < src[a].size(); ++i)
                dest[a][i] = src[a][i].ConvertToInt<uint64_t>();
        }
        result.SparseKSpInv = cryptoParams->GetSparseKSpInv();

        auto& aql = cryptoParams->GetSparseKSAlphaQlModp();
        auto& daql = result.SparseAlphaQlModp;
        daql.resize(aql.size());
        for (size_t a = 0; a < aql.size(); ++a) {
            daql[a].resize(aql[a].size());
            for (size_t j = 0; j < aql[a].size(); ++j)
                daql[a][j] = aql[a][j].ConvertToInt<uint64_t>();
        }
        auto& qlParams = cryptoParams->GetSparseKSParamsQ()->GetParams();
        result.SparseQlInv.resize(qlParams.size());
        for (size_t i = 0; i < qlParams.size(); ++i)
            result.SparseQlInv[i] = 1.0 / qlParams[i]->GetModulus().ConvertToDouble();
    }

    {
        auto& dest = result.PartQlHatInvModq;
        dest.resize(cryptoParams->GetNumPartQ());
        for (uint32_t k = 0; k < cryptoParams->GetNumPartQ(); ++k) {
            const uint32_t sizePartQk = cryptoParams->GetParamsPartQ(k)->GetParams().size();
            dest[k].resize(sizePartQk);
            for (uint32_t s = 0; s < sizePartQk; ++s) {
                const auto& src = cryptoParams->GetPartQlHatInvModq(k, s);
                dest[k][s].resize(src.size());
                for (size_t j = 0; j < src.size(); ++j) {
                    dest[k][s][j] = src[j].ConvertToInt<uint64_t>(); // m_value;
                }
            }
        }
    }

    {
        auto& dest = result.PartQlHatModp;
        const uint32_t sizeQ = static_cast<uint32_t>(result.L) + 1;
        const uint32_t numPartQ = cryptoParams->GetNumPartQ();
        const uint32_t alpha = static_cast<uint32_t>(std::ceil(static_cast<double>(sizeQ) / numPartQ));
        dest.resize(sizeQ);
        for (uint32_t lvl = 0; lvl < sizeQ; ++lvl) {
            const uint32_t beta = static_cast<uint32_t>(std::ceil(static_cast<double>(lvl + 1) / alpha));
            dest[lvl].resize(beta);
            for (uint32_t k = 0; k < beta; ++k) {
                const auto& src = cryptoParams->GetPartQlHatModp(lvl, k);
                dest[lvl][k].resize(src.size());
                for (size_t i = 0; i < src.size(); ++i) {
                    dest[lvl][k][i].resize(src[i].size());
                    for (size_t j = 0; j < src[i].size(); ++j) {
                        dest[lvl][k][i][j] = src[i][j].ConvertToInt<uint64_t>(); // m_value;
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
        const auto keyDist = cryptoParams->GetSecretKeyDist();
        const double foldedK = lbcrypto::FHECKKSRNS::GetModRaiseOverflowBound(
            cryptoParams, lbcrypto::FHECKKSRNS::K_UNIFORM);
        (void)foldedK;  // folded into scaleEnc by OpenFHE; FIDESlib's k cancels it, so k = 1.0
        const double bootK = 1.0;

        if (boot_conf == FIDESlib::ENCAPS) {
            result.coefficientsCheby = lbcrypto::GetBootTableCoeffSparseEncapsulated(); // degree 32
            // k adjustment for the (sparse-)encapsulated mod-reduction. The CtS/StC matrices are
            // computed host-side by EvalBootstrapSetup and fold scaleEnc = pre/K_pre(dist) with
            // K_pre(SPARSE_TERNARY)=K_SPARSE, K_pre(SPARSE_ENCAPSULATED)=K_SPARSE_ENCAPSULATED,
            // K_pre(UNIFORM_TERNARY)=K_UNIFORM. The GPU pipeline rescales by pre/(k*N) and the
            // (sparse-)encapsulated coefficients interpolate the K_SPARSE_ENCAPSULATED convention,
            // so the runtime k re-normalizes dist -> encapsulated: K_SPARSE_ENCAPSULATED for a
            // UNIFORM key, K_SPARSE_ENCAPSULATED/K_SPARSE for a SPARSE_TERNARY key and 1 for a
            // SPARSE_ENCAPSULATED key (table and folded matrix agree). bootK is keyed off the
            // ACTUAL secret-key distribution — never a hardcoded constant.
            result.bootK = keyDist == lbcrypto::SPARSE_TERNARY ? static_cast<double>(lbcrypto::FHECKKSRNS::K_SPARSE_ENCAPSULATED) / lbcrypto::FHECKKSRNS::K_SPARSE :
                keyDist == lbcrypto::UNIFORM_TERNARY ? static_cast<double>(lbcrypto::FHECKKSRNS::K_SPARSE_ENCAPSULATED) :
                                                       1.0;
            result.sparse_encaps = true;
            result.doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
        } else if (boot_conf == FIDESlib::SPARSE) {
            result.coefficientsCheby = lbcrypto::GetBootTableCoeffSparse();
            result.bootK = bootK;
            result.doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
            result.sparse_encaps = false;
        } else if (boot_conf == FIDESlib::UNIFORM) {
            result.coefficientsCheby = lbcrypto::GetBootTableCoeffUniform();
            result.bootK = bootK;  // 1.0: K is folded into scaleEnc by OpenFHE (see above)
            result.doubleAngleIts = lbcrypto::FHECKKSRNS::R_UNIFORM;
            result.sparse_encaps = false;
        }
    }

    result.p = cryptoParams->GetPlaintextModulus();

    return result;
}

// ---------------------------------------------------------------------------
// Host-side arithmetic helpers
// ---------------------------------------------------------------------------

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

FIDESlib::CKKS::RawKeySwitchKey FIDESlib::CKKS::GetRotationKeySwitchKey(const KeyPair<lbcrypto::DCRTPoly>& keys, int index, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc) {
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

std::vector<int> FIDESlib::CKKS::GetBootstrapIndexes(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, int slots, FIDESlib::CKKS::BootstrapPrecomputation* result_) {
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
                acc_offset += (result.CtS[i].slots / 2) * (result.CtS[i].bStep > 1 ? result.CtS[i].rotIn[1] : (result.CtS[i].gStep > 1 ? result.CtS[i].rotOut[1] : 0));
            }
            indexes.emplace_back(ReduceRotation(-acc_offset, slots_red));

            auto& j = result.CtS.back().rotOut[0];
            j = ReduceRotation(-acc_offset, slots_red);
        }

        {
            int acc_offset = 0;
            for (size_t i = 0; i < result.StC.size(); i++) {
                acc_offset += (result.StC[i].slots / 2) * (result.StC[i].bStep > 1 ? result.StC[i].rotIn[1] : (result.StC[i].gStep > 1 ? result.StC[i].rotOut[1] : 0));
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
                for (auto& j : { /*i.rotOut[0],*/ i.rotOut.size() > 1 ? i.rotOut[1] /*- i.rotOut[0]*/ : 0 }) {
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
// Use OpenFHE's own FHECKKSRNS::EvalBootstrapKeyGen sparse slots: KeySwitchGenSparse at 2N-4 and
// the switch-back KeySwitchGen at 2N-2. Both use OpenFHE's internal sparse secret, which is the
// only way the GPU result can be bit-identical to the CPU EvalBootstrap reference.
static constexpr uint32_t BTS_KSPARSE_TO_SLOT = 4;   // 2N-4: switch to   the sparse (tug) key
static constexpr uint32_t BTS_KSPARSE_FROM_SLOT = 2; // 2N-2: switch back to the real key

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
        // The sparse switching keys are authored by OpenFHE's own FHECKKSRNS::EvalBootstrapKeyGen
        // (called by the api before this function) at automorphism slots 2N-4 (KeySwitchGenSparse)
        // and 2N-2 (KeySwitchGen). AddBootstrapKeys reads exactly those slots. Using OpenFHE's
        // internal sparse secret is required for bit-exactness against the CPU EvalBootstrap.
        (void)evalKeys;
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

        // Use the MAIN context's sparse basis P' and exact-HPS tables, not the switchable
        // context's hybrid P: the sparse GHS switching key generated by GenBootstrapKeys lives
        // over the main Ql*P', so the GPU context must expose exactly that special basis. (The
        // switchable context deliberately has a single Q limb q0, which is the same first prime
        // as the main context, so reinterpretContext still matches at level 0.)
        {
            FIDESlib::CKKS::RawParams sparseRaw = FIDESlib::CKKS::GetRawParams(cc, FIDESlib::ENCAPS, /*sparseKS=*/true);
            raw_param2.SPECIALmoduli = sparseRaw.SPECIALmoduli;
            raw_param2.SPECIALroot_of_unity = sparseRaw.SPECIALroot_of_unity;
            raw_param2.SPECIALcyclotomic_order = sparseRaw.SPECIALcyclotomic_order;
            raw_param2.PHatInvModp = sparseRaw.PHatInvModp;
            raw_param2.PHatModq = sparseRaw.PHatModq;
            raw_param2.PInvModq = sparseRaw.PInvModq;
            raw_param2.SparseAlphaPModq = sparseRaw.SparseAlphaPModq;
            raw_param2.SparseKSpInv = sparseRaw.SparseKSpInv;
            raw_param2.K = sparseRaw.K;
        }

        FIDESlib::CKKS::Context cc_switch_ = CKKS::GenCryptoContextGPU(GPUcc.param.adaptTo(raw_param2), GPUcc.GPUid);
        FIDESlib::CKKS::ContextData& GPUcc2 = *cc_switch_;

        // atob: dense -> sparse, over Ql*P', exact HPS mod-up/down (OpenFHE KeySwitchSparse).
        FIDESlib::CKKS::KeySwitchingKey ksk_atob(cc_switch_);
        {
            std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> res =
                std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(evalKeys[2 * GPUcc.N - BTS_KSPARSE_TO_SLOT]);
            assert(res != nullptr);
            FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetKeySwitchKey(res);
            rawKskEval.keyid = publicKey->GetKeyTag();
            ksk_atob.InitializeSparse(rawKskEval);
        }

        // btoa: sparse -> dense, ordinary HYBRID key over the main context's Q*P.
        FIDESlib::CKKS::KeySwitchingKey ksk_btoa(GPUcc_);
        {
            std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> res =
                std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(evalKeys[2 * GPUcc.N - BTS_KSPARSE_FROM_SLOT]);
            FIDESlib::CKKS::RawKeySwitchKey rawKskEval2 = FIDESlib::CKKS::GetKeySwitchKey(res);
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

void FIDESlib::CKKS::AddBootstrapPlaintexts(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, int slots, FIDESlib::CKKS::Context& GPUcc_, FIDESlib::CKKS::BootstrapPrecomputation& result) {
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

        // The reference evaluators branch on the datatype in exactly one bootstrap-relevant
        // place: the sparsely packed SlotsToCoeffs fold runs only for REAL. Record the COMPLEX
        // datatype so the GPU mirrors that (Bootstrap.cu skips the fold when this is set).
        result.isComplex = (cc->GetCKKSDataType() == lbcrypto::COMPLEX);

        if (GPUcc.param.raw->sparse_encaps) {
            result.sparse_encaps = true;
        }

        // StC-first (decode-first) bootstrap: record the OpenFHE setup flag and the
        // decode level budget needed to replicate the EvalBootstrapStCFirst depletion
        // (expectedLevel = L0 - (lvlbDec + 2 + (FLEXIBLEAUTOEXT ? 1 : 0))).
        auto precom = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.find(slots)->second;
        result.btsfirstboot = precom->BTSlotsEncoding;
        result.lvlbDec = static_cast<int>(precom->m_paramsDec.lvlb);

        GPUcc.AddBootPrecomputation(slots, std::move(result));
    }

    AddBootstrapKeys(publicKey, slots, GPUcc_);
}

void FIDESlib::CKKS::AddBootstrapPrecomputation(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, const KeyPair<lbcrypto::DCRTPoly>& keys, int slots, FIDESlib::CKKS::Context& GPUcc_) {

    GenBootstrapKeys(keys, slots, GPUcc_->param.raw->sparse_encaps);

    AddBootstrapPrecomputation(keys.publicKey, slots, GPUcc_);
}

// ---------------------------------------------------------------------------
// bootstrap-precompute fingerprint helpers (test/diagnostic)
// ---------------------------------------------------------------------------

namespace {
uint64_t FpData(const std::vector<std::vector<uint64_t>>& sub) {
    uint64_t h = 1469598103934665603ULL;
    for (const auto& limb : sub) {
        for (uint64_t x : limb) {
            h ^= x;
            h *= 1099511628211ULL;
        }
    }
    return h;
}
} // namespace

std::vector<FIDESlib::CKKS::BootPrecomputeFingerprint> FIDESlib::CKKS::FingerprintBootPrecomputeCPU(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, int slots)
{
    std::vector<BootPrecomputeFingerprint> out;
    const auto precom =
        std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.find(slots)->second;
    const bool isLT = (precom->m_paramsEnc.lvlb == 1) && (precom->m_paramsDec.lvlb == 1);

    auto push = [&](const ReadOnlyPlaintext& pt) {
        RawPlainText raw = GetRawPlainText(const_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc), pt);
        // Replicate the GPU download's stride compression (RNSPoly::loadConstant, compress=true):
        // limbs longer than 2*slots are sub-sampled every (size/2/slots)-th value. Only then do
        // the CPU-side fingerprints equal the GPU-stored residues.
        const uint32_t slots = raw.slots > 0 ? static_cast<uint32_t>(raw.slots) : 1;
        std::vector<std::vector<uint64_t>> sub = raw.sub_0;
        for (auto& limb : sub) {
            const size_t stride = limb.size() / 2 / slots;
            if (stride > 1) {
                std::vector<uint64_t> aux(2 * slots);
                for (uint32_t j = 0; j < 2 * slots; ++j)
                    aux[j] = limb[static_cast<size_t>(j) * stride];
                limb = std::move(aux);
            }
        }
        BootPrecomputeFingerprint f;
        f.level         = static_cast<int>(pt->GetLevel());
        f.towers        = raw.numRes;
        f.scalingFactor = raw.Noise;
        f.fp            = FpData(sub);
        out.push_back(f);
    };

    if (isLT) {
        // Same order as AddBootstrapPlaintexts: m_U0hatTPre first (reversed) then m_U0Pre.
        for (size_t i = 0; i < precom->m_U0hatTPre.size(); ++i)
            push(precom->m_U0hatTPre[precom->m_U0hatTPre.size() - 1 - i]);
        for (size_t i = 0; i < precom->m_U0Pre.size(); ++i)
            push(precom->m_U0Pre[i]);
    }
    else {
        for (size_t i = 0; i < precom->m_U0hatTPreFFT.size(); ++i)
            for (size_t j = 0; j < precom->m_U0hatTPreFFT[precom->m_U0hatTPreFFT.size() - 1 - i].size(); ++j)
                push(precom->m_U0hatTPreFFT[precom->m_U0hatTPreFFT.size() - 1 - i][j]);
        for (size_t i = 0; i < precom->m_U0PreFFT.size(); ++i)
            for (size_t j = 0; j < precom->m_U0PreFFT[i].size(); ++j)
                push(precom->m_U0PreFFT[i][j]);
    }
    return out;
}

std::vector<FIDESlib::CKKS::BootPrecomputeFingerprint> FIDESlib::CKKS::FingerprintBootPrecomputeGPU(
    FIDESlib::CKKS::Context& GPUcc_, int slots)
{
    std::vector<BootPrecomputeFingerprint> out;
    const auto& pre = GPUcc_->GetBootPrecomputation(slots);

    auto push = [&](const Plaintext& pt) {
        std::vector<std::vector<uint64_t>> data;
        const_cast<Plaintext&>(pt).c0.store(data);
        BootPrecomputeFingerprint f;
        f.level         = pt.c0.getLevel();
        f.towers        = static_cast<int>(data.size());
        f.scalingFactor = pt.NoiseFactor;
        f.fp            = FpData(data);
        out.push_back(f);
    };

    for (const auto& lt : pre.CtS)
        for (const auto& pt : lt.A)
            push(pt);
    for (const auto& lt : pre.StC)
        for (const auto& pt : lt.A)
            push(pt);
    return out;
}
