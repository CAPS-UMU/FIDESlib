//
// Created by carlosad on 12/11/24.
//

#ifndef GPUCKKS_APPROXMODEVAL_CUH
#define GPUCKKS_APPROXMODEVAL_CUH

#include "Ciphertext.cuh"

namespace FIDESlib::CKKS {
void multMonomial(Ciphertext& ctxt, int power);

void approxModReduction(Ciphertext& ctxtEnc, Ciphertext& ctxtEncI, const KeySwitchingKey& keySwitchingKey,
                        uint64_t post);

void multIntScalar(Ciphertext& ctxt, uint64_t op);

void approxModReductionSparse(Ciphertext& ctxtEnc, const KeySwitchingKey& keySwitchingKey, uint64_t post);

}  // namespace FIDESlib::CKKS

#endif  //GPUCKKS_APPROXMODEVAL_CUH
