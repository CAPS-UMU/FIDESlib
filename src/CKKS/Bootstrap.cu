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

using namespace FIDESlib::CKKS;

constexpr bool PRINT = false;

void FIDESlib::CKKS::BootstrapCPUraise(
    Ciphertext& ctxt, const int slots,
    std::shared_ptr<
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<expdtype>>>>>& CPUcc,
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys, const bool prescaled) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});

    Context& cc = ctxt.cc;
    Ciphertext aux(cc);
    bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

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

    if constexpr (1) {

        FIDESlib::CKKS::RawCipherText raw_res1;
        ctxt.store(cc, raw_res1);

        std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
        lbcrypto::Plaintext ptxt1 = CPUcc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);
        auto c1 = CPUcc->Encrypt(keys.publicKey, ptxt1);
        //auto cResGPU = c1->Clone();
        GetOpenFHECipherText(c1, raw_res1);

        auto FHE = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(CPUcc->GetScheme()->m_FHE);
        auto raised = FHE->EvalBootstrapSetupOnly(c1, 1, 0);

        FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(CPUcc, raised);
        ctxt.load(raw1);
    } else {

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

        /*
        if (cc.N / 2 != slots) {
            for (int j = 1; j < cc.N / (2 * slots); j <<= 1) {
                aux.copy(ctxt);
                aux.rotate(j * slots, cc.GetRotationKey(j * slots));
                ctxt.add(aux);
            }
        }
        */
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    //   std::cout << "LT" << std::endl;

    if (isLT) {
        EvalLinearTransform(ctxt, slots, false);
    } else {
        EvalCoeffsToSlots(ctxt, slots, false);
    }
    //  std::cout << "ModRed" << std::endl;

    if (cc.N / 2 == slots) {
        aux.conjugate(ctxt);
        Ciphertext ctxtEncI(cc);
        ctxtEncI.sub(ctxt, aux);
        ctxt.add(aux);
        multMonomial(ctxtEncI, 3 * 2 * cc.N / 4);
        ctxt.rescale();
        ctxtEncI.rescale();
        approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(), scalar);
    } else {
        aux.conjugate(ctxt);
        ctxt.add(aux);
        if (cc.rescaleTechnique == Context::FIXEDMANUAL)
            ctxt.rescale();
        approxModReductionSparse(ctxt, cc.GetEvalKey(), scalar);
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    //  std::cout << "LT" << std::endl;

    if (isLT) {
        EvalLinearTransform(ctxt, slots, true);
    } else {
        EvalCoeffsToSlots(ctxt, slots, true);
    }

    if (cc.N / 2 != slots) {
        aux.rotate(ctxt, slots, cc.GetRotationKey(slots));
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

void FIDESlib::CKKS::Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});

    Context& cc = ctxt.cc;
    Ciphertext aux(cc);
    bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

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

        /*
        if (cc.N / 2 != slots) {
            for (int j = 1; j < cc.N / (2 * slots); j <<= 1) {
                aux.copy(ctxt);
                aux.rotate(j * slots, cc.GetRotationKey(j * slots));
                ctxt.add(aux);
            }
        }
        */
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    //   std::cout << "LT" << std::endl;

    if (isLT) {
        EvalLinearTransform(ctxt, slots, false);
    } else {
        EvalCoeffsToSlots(ctxt, slots, false);
    }
    //  std::cout << "ModRed" << std::endl;

    if (cc.N / 2 == slots) {
        aux.conjugate(ctxt);
        Ciphertext ctxtEncI(cc);
        ctxtEncI.sub(ctxt, aux);
        ctxt.add(aux);
        multMonomial(ctxtEncI, 3 * 2 * cc.N / 4);
        ctxt.rescale();
        ctxtEncI.rescale();
        approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(), scalar);
    } else {
        aux.conjugate(ctxt);
        ctxt.add(aux);
        if (cc.rescaleTechnique == Context::FIXEDMANUAL)
            ctxt.rescale();
        approxModReductionSparse(ctxt, cc.GetEvalKey(), scalar);
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    //  std::cout << "LT" << std::endl;

    if (isLT) {
        EvalLinearTransform(ctxt, slots, true);
    } else {
        EvalCoeffsToSlots(ctxt, slots, true);
    }

    if (cc.N / 2 != slots) {
        aux.rotate(ctxt, slots, cc.GetRotationKey(slots));
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
double FIDESlib::CKKS::GetPreScaleFactor(Context& cc, int slots) {
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

    double res = 0.0;
    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        uint32_t lvl = cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT;
        double targetSF = cc.param.ScalingFactorReal[cc.L - lvl];
        double sourceSF = cc.param.ScalingFactorReal[1];  // ciphertext->GetScalingFactor();
        uint32_t numTowers = 2;                           // ciphertext->GetElements()[0].GetNumOfElements();
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
        res = adjustmentFactor;
    } else {  // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
              // Scaling down the message by a correction factor to emulate using a larger q0.
              // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        res = std::pow((double)2.0, (double)-1.0 * (double)correction);
    }

    return res;
}

void FIDESlib::CKKS::ModRaise(Ciphertext& ctxt, const int slots, const uint32_t correction, const bool prescaled) {
    Context& cc = ctxt.cc;
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
        ctxt.rescale();
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
            //cc->EvalMultInPlace(ciphertext, adjustmentFactor);
            ctxt.rescale();
            ctxt.dropToLevel(1);
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
                ctxt.rescale();
            } else {
                ctxt.dropToLevel(0);
            }
        }
        ctxt.NoiseFactor = targetSF;
    } else {  // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
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
            ctxt.rescale();
            ctxt.dropToLevel(1);
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
                ctxt.rescale();
            } else {
                ctxt.dropToLevel(0);
            }
        }
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
    ctxt.c0.grow(cc.L - (cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT));
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
    ctxt.c1.grow(cc.L - (cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT));
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
}
