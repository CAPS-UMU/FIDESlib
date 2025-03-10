//
// Created by carlosad on 27/11/24.
//

#ifndef GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
#define GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH

#include <vector>
#include "Plaintext.cuh"
namespace FIDESlib::CKKS {

class BootstrapPrecomputation {
   public:
    struct {
        int slots = -1;
        int bStep = -1;
        std::vector<Plaintext> A;
        std::vector<Plaintext> invA;
    } LT;

    struct LTstep {
        int slots = -1;
        int bStep = -1;
        int gStep = -1;
        std::vector<Plaintext> A;
        std::vector<int> rotIn;
        std::vector<int> rotOut;
    };

    std::vector<LTstep> StC;
    std::vector<LTstep> CtS;
    uint32_t correctionFactor;
};

}  // namespace FIDESlib::CKKS

#endif  //GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
