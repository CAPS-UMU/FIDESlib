//
// Created by carlosad on 27/11/24.
//

#ifndef GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
#define GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH

#define AFFINE_LT true

#include "Plaintext.cuh"
#include <vector>

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
    int accumulate_bStep = 4;
    uint32_t correctionFactor;
    bool sparse_encaps{ false };
    bool btsfirstboot{ false };
    /// CKKSDataType == COMPLEX for this precomputation. The reference's evaluators branch on the
    /// datatype in exactly one place that touches the bootstrap: the sparsely packed SlotsToCoeffs
    /// fold is applied only for REAL. COMPLEX sparse/LT precomputations carry this so the GPU skips it.
    bool isComplex{ false };
    int lvlbDec{ -1 };

    // Bookkeeping flags for the shared per-slot bootstrap precomputation. They are never set here
    // (this branch has no functional setup path) and only feed the inert mode guards in Bootstrap.cu.
    bool isFBT{ false };
    bool isFEFunc{ false };

    std::weak_ptr<ContextData> sparse_context;
};
} // namespace FIDESlib::CKKS

#endif // GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH