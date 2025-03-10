//
// Created by carlosad on 4/12/24.
//

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/Context.cuh"

using namespace FIDESlib::CKKS;

constexpr bool PRINT = false;

void FIDESlib::CKKS::Bootstrap(Ciphertext& ctxt, const int slots) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});

    Context& cc = ctxt.cc;
    /////////////////////////////////////////////////////////////////////
    //NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q;  //q.ConvertToDouble();

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
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

    //------------------------------------------------------------------------------
    // RAISING THE MODULUS
    //------------------------------------------------------------------------------

    // In FLEXIBLEAUTO, raising the ciphertext to a larger number
    // of towers is a bit more complex, because we need to adjust
    // it's scaling factor to the one that corresponds to the level
    // it's being raised to.
    // Increasing the modulus

    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();
    if constexpr (PRINT) {
        std::cout << "Initial ";
        for (auto& j : ctxt.c0.GPU) {
            for (auto& i : j.limb)
                SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
        std::cout << correction << std::endl;
        std::cout << std::pow((double)2.0, (double)-1.0 * (double)correction) << std::endl;
    }

    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        uint32_t lvl = cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT;
        double targetSF = cc.param.ScalingFactorReal[cc.L - lvl];
        double sourceSF = ctxt.NoiseFactor;        // ciphertext->GetScalingFactor();
        uint32_t numTowers = ctxt.getLevel() + 1;  // ciphertext->GetElements()[0].GetNumOfElements();
        double modToDrop = static_cast<double>(cc.prime.at(numTowers - 1).p);
        //cryptoParams->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();

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

        ctxt.multScalar(adjustmentFactor);
        //cc->EvalMultInPlace(ciphertext, adjustmentFactor);
        ctxt.rescale();
        //algo->ModReduceInternalInPlace(ciphertext, BASE_NUM_LEVELS_TO_DROP);
        ctxt.NoiseFactor = targetSF;
        //ciphertext->SetScalingFactor(targetSF);
    } else {  // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
        // Scaling down the message by a correction factor to emulate using a larger q0.
        // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        ctxt.multScalar(std::pow((double)2.0, (double)-1.0 * (double)correction), false);
        ctxt.rescale();
    }
    // auto ctxtDCRT = raised->GetElements();
    if constexpr (PRINT) {
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU)
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        std::cout << std::endl;
    }

    ctxt.c0.INTT(cc.batch);
    if constexpr (PRINT) {
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU)
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();
    ctxt.c0.grow(cc.L - (cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT), false);
    cudaDeviceSynchronize();
    ctxt.c0.broadcastLimb0();
    cudaDeviceSynchronize();
    if constexpr (PRINT) {
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU)
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        std::cout << std::endl;
    }
    ctxt.c0.NTT(cc.batch);
    ctxt.c1.INTT(cc.batch);
    cudaDeviceSynchronize();
    ctxt.c1.grow(cc.L - (cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT), false);
    cudaDeviceSynchronize();
    ctxt.c1.broadcastLimb0();
    cudaDeviceSynchronize();
    ctxt.c1.NTT(cc.batch);
    if constexpr (PRINT) {
        std::cout << "ModRaise ";
        for (auto& j : ctxt.c0.GPU)
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        std::cout << std::endl;
    }
    //------------------------------------------------------------------------------
    // SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
    //------------------------------------------------------------------------------

    // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
    double k = cc.GetBootK();

    double constantEvalMult = pre * (1.0 / (k * cc.N));

    if constexpr (PRINT)
        std::cout << "mult: " << constantEvalMult << std::endl;
    CudaCheckErrorMod;
    ctxt.multScalar(constantEvalMult, false);

    if constexpr (PRINT) {
        std::cout << "Raise scaled ";
        for (auto& j : ctxt.c0.GPU)
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        std::cout << std::endl;
    }

    ////////////////////////////////////////////////////////////////

    bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;
    Ciphertext aux(cc);

    if (cc.N / 2 != slots) {
        for (int j = 1; j < cc.N / (2 * slots); j <<= 1) {
            aux.rotate(ctxt, j * slots, cc.GetRotationKey(j * slots));
            ctxt.add(aux);
        }
    }
    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    if (isLT) {
        cudaDeviceSynchronize();
        EvalLinearTransform(ctxt, slots, false);
        cudaDeviceSynchronize();
    } else
        EvalCoeffsToSlots(ctxt, slots, false);

    if (cc.N / 2 == slots) {
        cudaDeviceSynchronize();
        aux.conjugate(ctxt);
        cudaDeviceSynchronize();
        Ciphertext ctxtEncI(cc);
        cudaDeviceSynchronize();
        ctxtEncI.sub(ctxt, aux);
        cudaDeviceSynchronize();
        ctxt.add(aux);
        cudaDeviceSynchronize();
        multMonomial(ctxtEncI, 3 * 2 * cc.N / 4);
        cudaDeviceSynchronize();
        //ctxt.copy(ctxtEncI);
        cudaDeviceSynchronize();
        ctxt.rescale();
        ctxtEncI.rescale();
        approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(), scalar);
        cudaDeviceSynchronize();
    } else {
        /*
        if (ctxt.NoiseLevel == 2) {

        }
*/
        cudaDeviceSynchronize();
        aux.conjugate(ctxt);
        cudaDeviceSynchronize();
        ctxt.add(aux);
        cudaDeviceSynchronize();

        if (cc.rescaleTechnique == Context::FIXEDMANUAL)
            ctxt.rescale();

        approxModReductionSparse(ctxt, cc.GetEvalKey(), scalar);
        if constexpr (PRINT)
            std::cout << "Scalar last " << scalar << std::endl;
        cudaDeviceSynchronize();
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    if (isLT)
        EvalLinearTransform(ctxt, slots, true);
    else
        EvalCoeffsToSlots(ctxt, slots, true);

    if (cc.N / 2 != slots) {
        aux.rotate(ctxt, slots, cc.GetRotationKey(slots));
        ctxt.add(aux);
    }

    uint64_t corFactor = (uint64_t)1 << std::llround(correction);
    multIntScalar(ctxt, corFactor);
    if constexpr (PRINT) {
        for (auto& j : ctxt.c0.GPU)
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(2));
            }
        std::cout << std::endl;
    }
}
