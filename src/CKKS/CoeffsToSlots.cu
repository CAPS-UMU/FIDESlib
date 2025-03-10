//
// Created by carlosad on 27/11/24.
//

#include <ranges>
#include <vector>
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Plaintext.cuh"

using namespace FIDESlib::CKKS;

void FIDESlib::CKKS::EvalLinearTransform(Ciphertext& ctxt, int slots, bool decode) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});

    /*
    auto pair = m_bootPrecomMap.find(slots);
    if (pair == m_bootPrecomMap.end()) {
        std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
                             std::string(" slots were not generated") +
                             std::string(" Need to call EvalBootstrapSetup and EvalBootstrapKeyGen to proceed"));
        OPENFHE_THROW(errorMsg);
    }
    const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;

    auto cc = ct->GetCryptoContext();
    */
    Context& cc = ctxt.cc;
    // Computing the baby-step bStep and the giant-step gStep.
    uint32_t bStep = cc.GetBootPrecomputation(slots).LT.bStep;
    uint32_t gStep = ceil(static_cast<double>(slots) / bStep);

    uint32_t M = cc.N * 2;
    uint32_t N = cc.N;

    // computes the NTTs for each CRT limb (for the hoisted automorphisms used
    // later on)
    //auto digits = cc->EvalFastRotationPrecompute(ct);

    std::vector<Ciphertext> fastRotation;

    for (int i = 0; i < bStep - 1; ++i)
        fastRotation.emplace_back(cc);

    std::vector<Ciphertext*> fastRotationPtr;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr.push_back(&fastRotation[i - 1]);
        keys.push_back(&cc.GetRotationKey(i));
        indexes.push_back(i);
    }

    if (0) {
        ctxt.rotate_hoisted(keys, indexes, fastRotationPtr);
    } else {
        for (int i = 0; i < bStep - 1; ++i) {
            fastRotation[i].rotate(ctxt, i + 1, cc.GetRotationKey(i + 1));
            //cudaDeviceSynchronize();
        }
        cudaDeviceSynchronize();
    }
    Ciphertext result(cc);
    Ciphertext inner(cc);
    std::vector<Plaintext>& A = decode ? cc.GetBootPrecomputation(slots).LT.invA : cc.GetBootPrecomputation(slots).LT.A;

    for (uint32_t j = 0; j < gStep; j++) {

        inner.multPt(ctxt, A[bStep * j], false);
        for (uint32_t i = 1; i < bStep; i++) {
            if (bStep * j + i < slots) {
                inner.addMultPt(fastRotation[i - 1], A[bStep * j + i], false);
            }
        }
        // Does rotate -> rescale work???
        //inner.rescale();
        if (j == 0) {
            result.copy(inner);
        } else {
            inner.rotate(bStep * j, cc.GetRotationKey(bStep * j));
            result.add(inner);
        }
    }

    ctxt.copy(result);
    cudaDeviceSynchronize();
}

void FIDESlib::CKKS::EvalCoeffsToSlots(Ciphertext& ctxt, int slots, bool decode) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});

    Context& cc = ctxt.cc;
    //  No need for Encrypted Bit Reverse
    Ciphertext& result = ctxt;
    // hoisted automorphisms
    if (result.NoiseLevel == 2)
        result.rescale();
    std::vector<Ciphertext> auxiliar;

    Ciphertext outer(cc);
    Ciphertext inner(cc);

    int steps = 0;
    for (BootstrapPrecomputation::LTstep& step :
         (decode ? cc.GetBootPrecomputation(slots).StC : cc.GetBootPrecomputation(slots).CtS)) {
        // computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)

        std::vector<Ciphertext*> fastRotationPtr;
        std::vector<int> indexes;
        std::vector<KeySwitchingKey*> keys;

        for (int i = 0; i < step.bStep; ++i) {
            if (i >= auxiliar.size()) {
                auxiliar.emplace_back(cc);
            }
        }
        for (int i = 0; i < step.bStep; ++i) {
            fastRotationPtr.push_back(&auxiliar[i]);
            keys.push_back(step.rotIn[i] ? &cc.GetRotationKey(step.rotIn[i]) : nullptr);
            indexes.push_back(step.rotIn[i]);
        }

        result.rotate_hoisted(keys, indexes, fastRotationPtr);
        for (int32_t i = 0; i < step.gStep; i++) {

            // for the first iteration with j=0:
            int32_t G = step.bStep * i;
            inner.multPt(auxiliar[0], step.A[G], false);
            // continue the loop
            for (int32_t j = 1; j < step.bStep; j++) {
                if ((G + j) != step.slots) {
                    inner.addMultPt(auxiliar[j], step.A[G + j], false);
                }
            }

            if (i == 0) {
                outer.copy(inner);
            } else {
                if (step.rotOut[i] != 0) {
                    inner.rotate(step.rotOut[i], cc.GetRotationKey(step.rotOut[i]));
                }
                outer.add(inner);
            }
        }

        steps++;
        if (steps != (decode ? cc.GetBootPrecomputation(slots).StC : cc.GetBootPrecomputation(slots).CtS).size())
            outer.rescale();
        result.copy(outer);
    }

    CudaCheckErrorMod;
}
