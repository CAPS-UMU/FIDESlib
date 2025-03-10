//
// Created by carlosad on 4/12/24.
//

#ifndef GPUCKKS_BOOTSTRAP_CUH
#define GPUCKKS_BOOTSTRAP_CUH

#include "forwardDefs.cuh"

namespace FIDESlib::CKKS {
void Bootstrap(Ciphertext& ctxt, const int slots);
}

#endif  //GPUCKKS_BOOTSTRAP_CUH
