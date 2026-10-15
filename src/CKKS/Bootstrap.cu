//
// Created by carlosad on 4/12/24.
//

#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/Context.cuh"
#include <stdexcept>
#include <string>
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

using namespace FIDESlib::CKKS;

constexpr bool PRINT = false;

void FIDESlib::CKKS::BootstrapCPUraise(Ciphertext& ctxt,
    const int slots,
    std::shared_ptr<lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<expdtype>>>>>& CPUcc,
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys,
    const bool prescaled) {
    CudaNvtxRange r(std::string{ sc::current().function_name() });

    FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    Ciphertext aux(cc_);
    bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

    /////////////////////////////////////////////////////////////////////
    // NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q; // q.ConvertToDouble();

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p; // cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    /*
#if NATIVEINT != 128
    if (deg > static_cast<int32_t>(m_correctionFactor)) {
        OPENFHE_THROW("Degree [" + std::to_string(deg) + "] must be less than or equal to the correction factor [" +
                      std::to_string(m_correctionFactor) + "].");
    }
#endif
    */
    uint32_t correction = cc.GetBootPrecomputation(slots).correctionFactor - deg;
    if constexpr (PRINT)
        std::cout << cc.GetBootPrecomputation(slots).correctionFactor << " " << deg << std::endl;
    double post = std::pow(2, static_cast<double>(deg));

    double pre = 1. / post;
    uint64_t scalar = std::llround(post);

    //////////////////////////////////////////////////////////////////////

    {

        ModRaise(ctxt, slots, correction, prescaled);

        //------------------------------------------------------------------------------
        // SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
        //------------------------------------------------------------------------------

        // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
        double k = cc.GetBootK();

        double constantEvalMult = pre * (1.0 / (k * cc.N));

        if constexpr (PRINT)
            std::cout << "mult: " << constantEvalMult << std::endl;
        ctxt.multScalar(constantEvalMult, false);

        if constexpr (PRINT) {
            std::cout << "Raise scaled ";
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(1));
                }
            }
            std::cout << std::endl;
        }

        ////////////////////////////////////////////////////////////////

        Accumulate(ctxt, cc.GetBootPrecomputation(slots).accumulate_bStep, slots, cc.N / 2 / slots);
        // The fold is metadata-neutral (O10 — `Accumulate` no longer narrows `ctxt.slots` after a
        // full fold). Spell out the sparse re-interpretation this call site needs here, so the
        // downstream CtS rotations keep normalizing indices through `ctxt.slots` exactly as before.
        if (cc.N / 2 == ctxt.slots)
            ctxt.slots = slots;
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescaleInternal();
    }

    //   std::cout << "LT" << std::endl;

    if (isLT) {
        EvalLinearTransform(ctxt, slots, false);
    } else {
        // EvalCoeffsToSlots rotates modulo min(M/4, 2*slots); a sparse slot count's effective
        // index space is 2*slots, so normalyzeIndex must see it.
        if (cc.N / 2 != slots)
            ctxt.slots = 2 * slots;
        EvalCoeffsToSlots(ctxt, slots, false);
        ctxt.slots = slots;
    }
    //  std::cout << "ModRed" << std::endl;

    if (cc.N / 2 == slots) {
        aux.conjugate(ctxt);
        Ciphertext ctxtEncI(cc_);
        ctxtEncI.sub(ctxt, aux);
        ctxt.add(aux);
        ctxtEncI.multMonomial(3 * 2 * cc.N / 4);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescaleInternal();
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxtEncI.rescaleInternal();

        approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(ctxt.keyID), scalar);
    } else {
        aux.conjugate(ctxt);
        ctxt.add(aux);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescaleInternal();
        approxModReductionSparse(ctxt, scalar);
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescaleInternal();
    }

    //  std::cout << "LT" << std::endl;

    if (isLT) {
        EvalLinearTransform(ctxt, slots, true);
    } else {
        // Same as the encode: the trailing decode's rotation index space is 2*slots for a sparse
        // slot count.
        if (cc.N / 2 != slots)
            ctxt.slots = 2 * slots;
        EvalCoeffsToSlots(ctxt, slots, true);
        ctxt.slots = slots;
    }

    if (cc.N / 2 != slots) {
        aux.rotate(ctxt, slots);
        ctxt.add(aux);
    }

    uint64_t corFactor = (uint64_t)1 << std::llround(correction);
    multIntScalar(ctxt, corFactor);
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "End bootstrap ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(2));
            }
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
    }
}

// The single-iteration bootstrap pipeline. Bit-compat transcription of the
// numIterations == 1 body of OpenFHE FHECKKSRNS::EvalBootstrap (mod-raise first).
namespace FIDESlib::CKKS {
static void BootstrapSingle(Ciphertext& ctxt, const int slots, const bool prescaled) {
    CudaNvtxRange r(std::string{ sc::current().function_name() });

    constexpr bool SPARSE_PLAINTEXT_BOOT = true;
    assert(slots >= ctxt.slots);
    int old_slots = ctxt.slots;

    FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    Ciphertext aux(cc_);
    bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

    /////////////////////////////////////////////////////////////////////
    // NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q; // q.ConvertToDouble();

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p; // cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    /*
#if NATIVEINT != 128
    if (deg > static_cast<int32_t>(m_correctionFactor)) {
        OPENFHE_THROW("Degree [" + std::to_string(deg) + "] must be less than or equal to the correction factor [" +
                      std::to_string(m_correctionFactor) + "].");
    }
#endif
    */
    uint32_t correction = cc.GetBootPrecomputation(slots).correctionFactor - deg;
    if constexpr (PRINT)
        std::cout << cc.GetBootPrecomputation(slots).correctionFactor << " " << deg << std::endl;
    double post = std::pow(2, static_cast<double>(deg));

    double pre = 1. / post;
    uint64_t scalar = std::llround(post);

    //////////////////////////////////////////////////////////////////////
    bool sparse_encaps = cc.GetBootPrecomputation(slots).sparse_encaps;

    {
        ModRaise(ctxt, slots, correction, prescaled, sparse_encaps);
        //------------------------------------------------------------------------------
        // SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
        //------------------------------------------------------------------------------

        // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
        double k = cc.GetBootK();

        // TO-DO: The 1/32 scale will be pre-applied with OpenFHE v1.4, so remove it from here
        double constantEvalMult = pre * (1.0 / (k * cc.N));

        /*
        if (sparse_encaps) {
            constantEvalMult = pre * (1.0 / (k * cc.N) / 32);
        }
        */

        if constexpr (PRINT)
            std::cout << "mult: " << constantEvalMult << std::endl;
        ctxt.multScalar(constantEvalMult, false);
        if (MODRAISE_WITH_P0) {
            ctxt.rescaleInternal();
            // ctxt.multScalar(1.0, false); // TODO remove
            // ctxt.rescale();
        }

        if constexpr (PRINT) {
            std::cout << "Raise scaled ";
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(1));
                }
            }
            std::cout << std::endl;
        }

        ////////////////////////////////////////////////////////////////

        // The sparse switch-back is performed inside ModRaise (right after the raise), matching
        // OpenFHE's ModRaiseInPlace order; see ModRaise.
        Accumulate(ctxt, cc.GetBootPrecomputation(slots).accumulate_bStep, slots, cc.N / 2 / slots);
    }

    ctxt.slots = cc.N / 2 == slots ? slots : 2 * slots;

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescaleInternal();
    }

    //   std::cout << "LT" << std::endl;

    if (isLT) {
        EvalLinearTransform(ctxt, slots, false);
    } else {
        // EvalCoeffsToSlots rotates modulo min(M/4, 2*slots); a sparse slot count's effective
        // index space is 2*slots.
        if (cc.N / 2 != slots)
            ctxt.slots = 2 * slots;
        EvalCoeffsToSlots(ctxt, slots, false);
        ctxt.slots = slots;
    }

    //  std::cout << "ModRed" << std::endl;

    if (cc.N / 2 == slots) {
        aux.conjugate(ctxt);
        Ciphertext ctxtEncI(cc_);
        ctxtEncI.sub(ctxt, aux);
        ctxt.add(aux);
        ctxtEncI.multMonomial(3 * 2 * cc.N / 4);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescaleInternal();
        else if (ctxt.NoiseLevel == 2)
            ctxt.rescaleInternal(); // FIDESlib bit-compat: mirror OpenFHE's ModReduceInternal before the Chebyshev series
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxtEncI.rescaleInternal();
        else if (ctxtEncI.NoiseLevel == 2)
            ctxtEncI.rescaleInternal(); // FIDESlib bit-compat: same for the imaginary component
        approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(ctxt.keyID), scalar);
    } else {
        aux.conjugate(ctxt);
        if (SPARSE_PLAINTEXT_BOOT) {
        }
        ctxt.add(aux);

        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescaleInternal();
        else if (ctxt.NoiseLevel == 2)
            ctxt.rescaleInternal(); // FIDESlib bit-compat: mirror OpenFHE's ModReduceInternal before the Chebyshev series
        approxModReductionSparse(ctxt, scalar);
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescaleInternal();
    }

    //  std::cout << "LT" << std::endl;

    if (isLT) {
        EvalLinearTransform(ctxt, slots, true);
    } else {
        // Same as the encode: the trailing decode's rotation index space is 2*slots for a sparse
        // slot count.
        if (cc.N / 2 != slots)
            ctxt.slots = 2 * slots;
        EvalCoeffsToSlots(ctxt, slots, true);
        ctxt.slots = slots;
    }

    if (cc.N / 2 != slots) {
        ctxt.slots = 2 * slots;
        aux.rotate(ctxt, slots);
        ctxt.add(aux);
        ctxt.slots = slots;
    }

    uint64_t corFactor = (uint64_t)1 << std::llround(correction);
    multIntScalar(ctxt, corFactor);
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "End bootstrap ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(2));
            }
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
    }

    ctxt.slots = old_slots;
}
} // namespace FIDESlib::CKKS

namespace FIDESlib::CKKS {

// Shared numIterations == 2 body of both bootstrap flavors. Bit-compat
// transcription of the corresponding branches of FHECKKSRNS::EvalBootstrap
// (mod-raise first) / EvalBootstrapStCFirst (decode first), compositeDegree == 1:
// bootstrap twice (input and bootstrapping error) and recombine so the residual
// error is cancelled. Levels are in FIDESlib space (highest active limb index).
// `single` is the single-iteration pipeline and `noiseAwareModReduce` selects the
// post-bootstrap tower drop: the mod-raise-first flavor passes
// ModReduceInternalInPlace(ct, compositeDegree) (one tower), the decode-first one
// ModReduceInternalInPlace(ct, compositeDegree * (noiseDeg - 1)).
static void BootstrapIterativeImpl(Ciphertext& ctxt,
    const int slots,
    const bool prescaled,
    const uint32_t precision,
    void (*single)(Ciphertext&, int, bool),
    bool noiseAwareModReduce) {
    ContextData& cc = ctxt.cc;
    const Context& cc_ = ctxt.cc_;
    const uint32_t powerOfTwoModulus = 1u << precision;
    const int32_t initLevel = ctxt.getLevel(); // initSizeQ = initLevel + 1 towers

    const auto modReduceAfterBootstrap = [&](Ciphertext& ct) {
        if (noiseAwareModReduce) {
            for (int d = ct.NoiseLevel - 1; d > 0; --d)
                ct.rescaleInternal();
        } else {
            ct.rescaleInternal();
        }
    };

    // Step 3: bootstrap the initial ciphertext, then ModReduceInternalInPlace and
    // scale up by powerOfTwoModulus.
    Ciphertext ctInitial(cc_);
    ctInitial.copy(ctxt);
    single(ctInitial, slots, prescaled);
    modReduceAfterBootstrap(ctInitial);
    multIntScalar(ctInitial, powerOfTwoModulus); // MultByIntegerInPlace(ctInitialBootstrap, 2^precision)

    // If we start with more towers than we obtain from bootstrapping, return the
    // original ciphertext unchanged (OpenFHE: bootstrappingSizeQ <= initSizeQ).
    if (ctInitial.getLevel() <= initLevel) {
        return;
    }

    // Step 2: clone the input, and (FIXEDMANUAL only) first drop noiseDeg - 1
    // towers; then scale both up by powerOfTwoModulus, leaving the extended top
    // tower implicitly zero.
    Ciphertext ctScaledUp(cc_);
    ctScaledUp.copy(ctxt);
    if (cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
        for (int d = ctScaledUp.NoiseLevel - 1; d > 0; --d) {
            ctScaledUp.rescaleInternal(); // ModReduceInternalInPlace(ctScaledUp, noiseDeg - 1)
        }
    }
    multIntScalar(ctScaledUp, powerOfTwoModulus);

    // Step 5: FIXEDAUTO only — mod the bootstrapped ciphertext down onto the
    // input's tower count (OpenFHE: DropLastElements + SetLevel).
    Ciphertext ctScaledDown(cc_);
    ctScaledDown.copy(ctInitial);
    if (cc.rescaleTechnique == CKKS::FIXEDAUTO) {
        ctScaledDown.dropToLevel(initLevel);
    }

    // Steps 6-7: the bootstrapping error is scaledDown - scaledUp. EvalSub's
    // AdjustForAddOrSubInPlace aligns level/scale first; FIDESlib's sub() is a
    // bit-exact transcription of that alignment.
    Ciphertext ctBootstrappingError(cc_);
    ctBootstrappingError.copy(ctScaledDown);
    ctBootstrappingError.sub(ctScaledUp);

    // Step 8: bootstrap the error, then ModReduceInternalInPlace.
    single(ctBootstrappingError, slots, prescaled);
    modReduceAfterBootstrap(ctBootstrappingError);

    // Step 9: subtract the bootstrapped error from the initial bootstrap.
    ctxt.copy(ctInitial);
    ctxt.sub(ctBootstrappingError);

    // Step 10: scale back down by powerOfTwoModulus (EvalMultInPlace(1/2^precision)).
    ctxt.multScalarNoPrecheck(1.0 / static_cast<double>(powerOfTwoModulus), false);
}

// OpenFHE FHECKKSRNS::EvalBootstrapStCFirst, lines 1434-1479: bring the input
// down to the level where the decode-first SlotsToCoeffs precomputation lives
// (OpenFHE expectedLevel = L0 - (lvlbDec + 2 + (FLEXIBLEAUTOEXT ? 1 : 0)), which
// is lvlbDec + 1 + ext in FIDESlib level space), or throw when the input is
// already deeper than that.
static void DepleteForStCFirst(Ciphertext& ctxt, const int slots) {
    ContextData& cc = ctxt.cc;
    const int expectedLevel = cc.GetBootPrecomputation(slots).lvlbDec + 1 +
        (cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT ? 1 : 0);

    if (ctxt.getLevel() < expectedLevel + ctxt.NoiseLevel - 1) {
        throw std::runtime_error("Not enough levels to perform Bootstrapping.");
    }

    // ModReduceInternalInPlace(ct, noiseDeg - 1).
    if (ctxt.NoiseLevel == 2) {
        ctxt.rescaleInternal();
    }
    const int32_t ctxtLvl = ctxt.getLevel();

    if (cc.rescaleTechnique == CKKS::FIXEDMANUAL || cc.rescaleTechnique == CKKS::FIXEDAUTO) {
        // LevelReduceInternalInPlace(ct, expectedLevel - ctxtLevel) when needed.
        if (ctxtLvl > expectedLevel) {
            ctxt.dropToLevel(expectedLevel, /*skip_adjust=*/true);
        }
        return;
    }

    // FLEXIBLEAUTO / FLEXIBLEAUTOEXT (OpenFHE lines 1457-1479 / 1772-1785). NOTE: this is NOT the
    // AdjustLevelsAndDepthInPlace pattern — the correction uses the *inverted* ratio scf1/scf2/scf,
    // so Ciphertext::adjustScaleAndLevel must NOT be used here (EvalBootstrapStCFirstAboveTarget is
    // the coverage that guards this). OpenFHE (compositeDegree == 1):
    //   scf2      = GetScalingFactor();
    //   scf1      = GetScalingFactorRealBig(expectedLevel - compositeDegree);
    //   scf       = GetScalingFactorReal(ctxtLevel);
    //   targetScf = GetScalingFactorReal(expectedLevel);
    // FIDESlib stores these inverted: ScalingFactorReal[k] == GetScalingFactorReal(L - k)
    // (RawCiphertext.cu), and OpenFHE ctxtLevel == L - ctxtLvl, so
    //   scf1 -> ScalingFactorRealBig[expectedLevel + compositeDegree], scf -> ScalingFactorReal[ctxtLvl].
    constexpr uint32_t compositeDegree = 1;
    if (ctxtLvl > expectedLevel) {
        double scf2 = ctxt.NoiseFactor;
        double scf1 = cc.param.ScalingFactorRealBig[expectedLevel + compositeDegree]; // GetScalingFactorRealBig(expectedLevel - 1)
        double scf = cc.param.ScalingFactorReal[ctxtLvl];                             // GetScalingFactorReal(ctxtLevel)
        double targetScf = cc.param.ScalingFactorReal[expectedLevel];                 // GetScalingFactorReal(expectedLevel)
        ctxt.multScalarNoPrecheck(scf1 / scf2 / scf, false);      // EvalMult(ct, scf1/scf2/scf)
        if (ctxt.getLevel() - (ctxt.NoiseLevel - 1) > expectedLevel) {
            // LevelReduceInternalInPlace(ct, expectedLevel - ctxtLevel - (noiseDeg - 1)).
            ctxt.dropToLevel(expectedLevel + ctxt.NoiseLevel - 1, /*skip_adjust=*/true);
        }
        for (int d = 1; d < ctxt.NoiseLevel; ++d) {
            ctxt.rescaleInternal(); // ModReduceInternalInPlace(ct, noiseDeg - 1)
        }
        ctxt.NoiseFactor = targetScf; // SetScalingFactor(targetScf)
    } else {
        for (int d = 1; d < ctxt.NoiseLevel; ++d) {
            ctxt.rescaleInternal();
        }
    }
}

// The decode-first single-iteration pipeline: bit-compat transcription of
// FHECKKSRNS::EvalBootstrapStCFirst (numIterations == 1, lines 1434-1688).
// Unlike the mod-raise-first flavor, the SlotsToCoeffs decode runs on the
// *depleted* input before the modulus raise, and there is no trailing decode
// nor sparse doubling; the pipeline ends in the slots domain.
static void BootstrapStCSingle(Ciphertext& ctxt, const int slots, const bool prescaled) {
    CudaNvtxRange r(std::string{ sc::current().function_name() });

    assert(slots >= ctxt.slots);
    const int old_slots = ctxt.slots;

    Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    Ciphertext aux(cc_);
    const bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

    // Same prologue as the mod-raise-first flavor (q/p/deg/correction/pre/scalar).
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q; // q.ConvertToDouble();
    const auto p = cc.param.raw->p; // cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);
    int32_t deg = std::round(std::log2(qDouble / powP));
    uint32_t correction = cc.GetBootPrecomputation(slots).correctionFactor - deg;
    double post = std::pow(2, static_cast<double>(deg));
    double pre = 1. / post;
    uint64_t scalar = std::llround(post);
    bool sparse_encaps = cc.GetBootPrecomputation(slots).sparse_encaps;

    // Deplete the input to the decode level, then run the decode (SlotsToCoeffs)
    // FIRST, before the modulus raise.
    DepleteForStCFirst(ctxt, slots);
    if (isLT) {
        EvalLinearTransform(ctxt, slots, true);
    } else {
        // EvalSlotsToCoeffs rotates modulo min(M/4, 2*slots); a sparse slot
        // count's effective domain is 2*slots, so normalyzeIndex must see it. Restore the semantic
        // slot count afterwards (the reference keeps the input `slots`).
        if (cc.N / 2 != slots)
            ctxt.slots = 2 * slots;
        EvalCoeffsToSlots(ctxt, slots, true);
        ctxt.slots = slots;
    }
    // OpenFHE EvalBootstrapStCFirst: the sparsely packed case doubles the decoded message. The
    // reference guards this fold with `GetCKKSDataType() == REAL` (ckksrns-fhe.cpp, the StC-first
    // SlotsToCoeffs fold), so a COMPLEX precomputation deliberately skips it.
    if (cc.N / 2 != slots && !cc.GetBootPrecomputation(slots).isComplex) {
        ctxt.slots = 2 * slots;
        aux.rotate(ctxt, slots);
        ctxt.add(aux);
        ctxt.slots = slots;
    }

    // The remainder coincides with the mod-raise-first pipeline
    // (ModRaise -> normalize -> PartialSum -> CtS encode -> sine).
    ModRaise(ctxt, slots, correction, prescaled, sparse_encaps);

    double k = cc.GetBootK();
    double constantEvalMult = pre * (1.0 / (k * cc.N));
    ctxt.multScalar(constantEvalMult, false);
    if (MODRAISE_WITH_P0) {
        ctxt.rescaleInternal();
    }
    // The sparse switch-back is performed inside ModRaise (right after the raise), matching
    // OpenFHE's ModRaiseInPlace order; see ModRaise.
    Accumulate(ctxt, cc.GetBootPrecomputation(slots).accumulate_bStep, slots, cc.N / 2 / slots);

    ctxt.slots = cc.N / 2 == slots ? slots : 2 * slots;

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescaleInternal();
    }

    if (isLT) {
        EvalLinearTransform(ctxt, slots, false);
    } else {
        EvalCoeffsToSlots(ctxt, slots, false);
    }

    // conj + add, then the single-part approximate mod-reduction: EvalBootstrapStCFirst
    // evaluates the Chebyshev series over one ciphertext for REAL data (the OpenFHE
    // default), with no ctxtEncI component.
    aux.conjugate(ctxt);
    ctxt.add(aux);
    if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
        ctxt.rescaleInternal();
    else if (ctxt.NoiseLevel == 2)
        ctxt.rescaleInternal(); // OpenFHE ModReduceInternalInPlace before the Chebyshev series
    approxModReductionSparse(ctxt, scalar);

    // Decode-first flavor: no trailing SlotsToCoeffs nor sparse doubling; scale
    // the message back up to its original scale (OpenFHE: MultByInteger(corFactor)).
    uint64_t corFactor = (uint64_t)1 << std::llround(correction);
    multIntScalar(ctxt, corFactor);

    ctxt.slots = old_slots;
}

} // namespace FIDESlib::CKKS

void FIDESlib::CKKS::Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled) {
    Bootstrap(ctxt, slots, prescaled, /*numIterations=*/1, /*precision=*/0);
}

void FIDESlib::CKKS::Bootstrap(Ciphertext& ctxt,
    const int slots,
    const bool prescaled,
    const uint32_t numIterations,
    const uint32_t precision) {
    CudaNvtxRange r(std::string{ sc::current().function_name() });

    const auto& pre = ctxt.cc.GetBootPrecomputation(slots);
    if (pre.isFBT || pre.isFEFunc) {
        throw std::invalid_argument(std::string("CKKS Bootstrap: slots=") + std::to_string(slots) +
                                    " is configured for a reserved mode, not the standard bootstrap.");
    }

    // OpenFHE FHECKKSRNS::EvalBootstrap dispatches to the decode-first pipeline
    // before validating numIterations when the setup used BTSlotsEncoding.
    if (pre.btsfirstboot) {
        BootstrapStCFirst(ctxt, slots, prescaled, numIterations, precision);
        return;
    }

    // OpenFHE supports exactly 1 or 2 iterations.
    if (numIterations != 1 && numIterations != 2) {
        throw std::invalid_argument("CKKS Bootstrapping only supported for 1 or 2 iterations.");
    }
    if (numIterations == 1) {
        BootstrapSingle(ctxt, slots, prescaled);
        return;
    }
    BootstrapIterativeImpl(ctxt, slots, prescaled, precision, &BootstrapSingle, /*noiseAwareModReduce=*/false);
}

void FIDESlib::CKKS::BootstrapStCFirst(Ciphertext& ctxt, const int slots, const bool prescaled) {
    BootstrapStCFirst(ctxt, slots, prescaled, /*numIterations=*/1, /*precision=*/0);
}

void FIDESlib::CKKS::BootstrapStCFirst(Ciphertext& ctxt,
    const int slots,
    const bool prescaled,
    const uint32_t numIterations,
    const uint32_t precision) {
    CudaNvtxRange r(std::string{ sc::current().function_name() });

    const auto& pre = ctxt.cc.GetBootPrecomputation(slots);
    if (pre.isFBT || pre.isFEFunc) {
        throw std::invalid_argument(
            "CKKS BootstrapStCFirst: slots=" + std::to_string(slots) +
            " is configured for a reserved mode, not the standard bootstrap.");
    }
    if (!pre.btsfirstboot) {
        throw std::invalid_argument(
            "CKKS BootstrapStCFirst: slots=" + std::to_string(slots) +
            " was not set up with EvalBootstrapSetup(btsfirstboot=true); the SlotsToCoeffs-first pipeline "
            "needs the decode-first precomputation.");
    }

    // OpenFHE FHECKKSRNS::EvalBootstrapStCFirst supports exactly 1 or 2 iterations.
    if (numIterations != 1 && numIterations != 2) {
        throw std::invalid_argument("CKKS Bootstrapping only supported for 1 or 2 iterations.");
    }
    if (numIterations == 1) {
        BootstrapStCSingle(ctxt, slots, prescaled);
        return;
    }
    BootstrapIterativeImpl(ctxt, slots, prescaled, precision, &BootstrapStCSingle, /*noiseAwareModReduce=*/true);
}

double FIDESlib::CKKS::GetPreScaleFactor(Context& cc_, int slots) {
    ContextData& cc = *cc_;
    SetCurrentContext(cc_);
    /////////////////////////////////////////////////////////////////////
    // NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q; // q.ConvertToDouble();

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p; // cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    /*
    #if NATIVEINT != 128
        if (deg > static_cast<int32_t>(m_correctionFactor)) {
            OPENFHE_THROW("Degree [" + std::to_string(deg) + "] must be less than or equal to the correction factor [" +
                          std::to_string(m_correctionFactor) + "].");
        }
    #endif
        */
    uint32_t correction = cc.GetBootPrecomputation(slots).correctionFactor - deg;

    double res = 0.0;
    if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
        uint32_t lvl = cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT;
        double targetSF = cc.param.ScalingFactorReal[cc.L - lvl];
        double sourceSF = cc.param.ScalingFactorReal[1]; // ciphertext->GetScalingFactor();
        uint32_t numTowers = 2;                          // ciphertext->GetElements()[0].GetNumOfElements();
        double modToDrop = static_cast<double>(cc.prime.at(numTowers - 1).p);
        // cryptoParams->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();
        //  in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
        //  a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
        //  So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
        //  for NATIVEINT = 128.
        //  Scaling down the message by a correction factor to emulate using a larger q0.
        //  This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF);
        double pow = std::pow((double)2.0, (double)-1.0 * (double)correction);
        adjustmentFactor *= pow;
        if constexpr (PRINT)
            std::cout << adjustmentFactor << std::endl;
        res = adjustmentFactor;
    } else {
        // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
        // Scaling down the message by a correction factor to emulate using a larger q0.
        // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        res = std::pow((double)2.0, (double)-1.0 * (double)correction);
    }

    return res;
}

void FIDESlib::CKKS::ModRaise(Ciphertext& ctxt, const int slots, const uint32_t correction, const bool prescaled, const bool sparse_encaps) {
    CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
    ContextData& cc = ctxt.cc;
    //------------------------------------------------------------------------------
    // RAISING THE MODULUS
    //------------------------------------------------------------------------------

    if (!prescaled) {
        assert(ctxt.getLevel() - ctxt.NoiseLevel + 1 >= 1);
    } else {
        assert(ctxt.getLevel() - ctxt.NoiseLevel + 1 == 0);
    }
    // In FLEXIBLEAUTO, raising the ciphertext to a larger number
    // of towers is a bit more complex, because we need to adjust
    // it's scaling factor to the one that corresponds to the level
    // it's being raised to.
    // Increasing the modulus
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "Initial ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb)
                SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
        CudaCheckErrorMod;
    }
    if (ctxt.NoiseLevel == 2)
        ctxt.rescaleInternal();
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "Initial 2 ";
        CudaCheckErrorMod;
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb)
                SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
        std::cout << correction << std::endl;
        std::cout << std::pow((double)2.0, (double)-1.0 * (double)correction) << std::endl;
        cudaDeviceSynchronize();
        CudaCheckErrorMod;
    }

    double targetSF = ctxt.NoiseFactor;
    if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
        uint32_t lvl = cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT;
        if (MODRAISE_WITH_P0) {
            targetSF = cc.param.ScalingFactorReal[cc.L - lvl]; // Will multiply with scalar scaled by P_0 and rescaled down by P_0.
        } else {
            targetSF = cc.param.ScalingFactorReal[cc.L - lvl];
        }
        double sourceSF = ctxt.NoiseFactor;       // ciphertext->GetScalingFactor();
        uint32_t numTowers = ctxt.getLevel() + 1; // ciphertext->GetElements()[0].GetNumOfElements();
        double modToDrop = static_cast<double>(cc.prime.at(numTowers - 1).p);
        // cryptoParams->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();

        // in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
        // a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
        // So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
        // for NATIVEINT = 128.
        // Scaling down the message by a correction factor to emulate using a larger q0.
        // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF);
        double pow = std::pow((double)2.0, (double)-1.0 * (double)correction);
        adjustmentFactor *= pow;
        if constexpr (PRINT)
            std::cout << adjustmentFactor << std::endl;

        if (!prescaled) {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            ctxt.multScalar(adjustmentFactor);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            // cc->EvalMultInPlace(ciphertext, adjustmentFactor);
            ctxt.rescaleInternal();
            ctxt.dropToLevel(0, true);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
        } else {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Prescale path ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            if (ctxt.NoiseLevel == 2) {
                ctxt.dropToLevel(1, true);
                ctxt.rescaleInternal();
            } else {
                ctxt.dropToLevel(0, true);
            }
        }
        ctxt.NoiseFactor = targetSF;
    } else {
        // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
        // Scaling down the message by a correction factor to emulate using a larger q0.
        // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        if (!prescaled) {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            ctxt.multScalar(std::pow((double)2.0, (double)-1.0 * (double)correction), false);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            ctxt.rescaleInternal();
            ctxt.dropToLevel(0);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
        } else {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Prescale path ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            if (ctxt.NoiseLevel == 2) {
                ctxt.dropToLevel(1);
                ctxt.rescaleInternal();
            } else {
                ctxt.dropToLevel(0);
            }
        }
    }

    if (sparse_encaps) {
        auto& sparse_context = cc.GetBootPrecomputation(slots).sparse_context;
        auto sparse_context_use = sparse_context.lock();
        Ciphertext sparse_ctxt(sparse_context_use);
        auto& atob = CKKS::GetSecretSwitchingKey(ctxt.cc_, sparse_context_use, ctxt.keyID);

        sparse_ctxt.reinterpretContext(ctxt);
        sparse_ctxt.keySwitchSparse(atob);
        ctxt.reinterpretContext(sparse_ctxt);
        ctxt.NoiseFactor = targetSF;
    }

    //   std::cout << "Boot start " << std::endl;
    // auto ctxtDCRT = raised->GetElements();
    if constexpr (PRINT) {
        std::cout << "Adjustment 1: ";
        CudaCheckErrorMod;
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }

    ctxt.c0.INTT(cc.batch, true);

    if constexpr (PRINT) {
        CudaCheckErrorMod;
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    //   std::cout << "Grow" << std::endl;
    ctxt.c0.grow(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT));
    if (MODRAISE_WITH_P0) {
        ctxt.c0.generateSpecialLimbs(false, true);
        ctxt.c0.setLevel(cc.L + 1);
    }
    //   std::cout << "Broadcast" << std::endl;
    if constexpr (PRINT) {
        CudaCheckErrorMod;
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    ctxt.c0.broadcastLimb0();
    if constexpr (PRINT) {
        CudaCheckErrorMod;
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    ctxt.c0.NTT(cc.batch, true);
    // std::cout << cc.batch << std::endl;
    if constexpr (PRINT) {
        std::cout << "ModRaise ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    ctxt.c1.INTT(cc.batch, true);
    if constexpr (PRINT) {
        std::cout << "Adjustment c1 ";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    //  std::cout << "Grow" << std::endl;
    ctxt.c1.grow(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT));
    if (MODRAISE_WITH_P0) {
        ctxt.c1.generateSpecialLimbs(false, true);
        ctxt.c1.setLevel(cc.L + 1);
    }
    //  std::cout << "Broadcast" << std::endl;
    if constexpr (PRINT) {
        std::cout << "Adjustment c1  ";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    ctxt.c1.broadcastLimb0();
    if constexpr (PRINT) {
        std::cout << "Adjustment c1";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    ctxt.c1.NTT(cc.batch, true);
    if constexpr (PRINT) {
        std::cout << "Adjustment c1";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }

    // Sparse secret encapsulation: switch back to the real key immediately after the modulus
    // raise (OpenFHE ModRaiseInPlace does KeySwitchInPlace(2N-2) here, before the caller's
    // pre/(k*N) scalar multiplication and before the rest of the bootstrap). Doing it earlier
    // (before the caller's multScalar) matches the reference's rounding/scale trajectory.
    if (sparse_encaps) {
        auto& sparse_context = cc.GetBootPrecomputation(slots).sparse_context;
        auto sparse_context_use = sparse_context.lock();
        auto& btoa = CKKS::GetSecretSwitchingKey(sparse_context_use, ctxt.cc_, ctxt.keyID);
        ctxt.keySwitch(btoa);
    }

    ctxt.slots = cc.N / 2;
}
