//
// Created by carlosad on 12/11/24.
//

#ifndef GPUCKKS_APPROXMODEVAL_CUH
#define GPUCKKS_APPROXMODEVAL_CUH

#include "CKKS/forwardDefs.cuh"
#include <cinttypes>
#include <vector>

namespace FIDESlib::CKKS {
/** OpenFHE: for degree 5 or less uses naïve implementation, on FIDESlib, its always Patterson Stockmayer
 *  I suggest to only use range [-1, 1]
 * */
void evalChebyshevSeries(Ciphertext& ctxt, std::vector<double>& coefficients, double lower_bound = -1.0, double upper_bound = 1.0);

void approxModReduction(Ciphertext& ctxtEnc, Ciphertext& ctxtEncI, const KeySwitchingKey& keySwitchingKey, uint64_t post);

void multIntScalar(Ciphertext& ctxt, uint64_t op);

void approxModReductionSparse(Ciphertext& ctxtEnc, uint64_t post);

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_APPROXMODEVAL_CUH