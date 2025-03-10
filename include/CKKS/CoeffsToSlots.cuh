//
// Created by carlosad on 27/11/24.
//

#ifndef GPUCKKS_COEFFSTOSLOTS_CUH
#define GPUCKKS_COEFFSTOSLOTS_CUH
#include "forwardDefs.cuh"

namespace FIDESlib::CKKS {
void EvalLinearTransform(Ciphertext& ctxt, int slots, bool decode);

void EvalCoeffsToSlots(Ciphertext& ctxt, int slots, bool decode);
}  // namespace FIDESlib::CKKS
#endif  //GPUCKKS_COEFFSTOSLOTS_CUH
