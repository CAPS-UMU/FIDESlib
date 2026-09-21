//
// Created by carlosad on 4/12/24.
//

#ifndef GPUCKKS_BOOTSTRAP_CUH
#define GPUCKKS_BOOTSTRAP_CUH

#include "forwardDefs.cuh"
#include "pke/openfhe.h"

namespace FIDESlib::CKKS {
void BootstrapCPUraise(Ciphertext& ctxt,
  const int slots,
  std::shared_ptr<lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<expdtype>>>>>& CPUcc,
  lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys,
  const bool prescaled);
/**
 * @brief Bootstraps `ctxt` in place (refreshes its level budget).
 *
 * @pre ctxt.getLevel() >= ctxt.NoiseLevel (levels count RNS limbs minus one): a NoiseLevel-1 input needs at least
 *      two limbs (level >= 1) and a NoiseLevel-2 input (pending rescale) at least three (level >= 2), because the
 *      modulus raise rescales once (twice for NoiseLevel 2) before raising. With `prescaled == true` the input must
 *      instead satisfy level == NoiseLevel - 1. Violations throw std::invalid_argument.
 * @param ctxt      Ciphertext to bootstrap; on return it is at the post-bootstrap level with NoiseLevel 1.
 * @param slots     Number of slots the bootstrap precomputation was generated for (>= ctxt.slots).
 * @param prescaled Whether the caller already applied the modulus-raise prescaling.
 */
void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled = false);
double GetPreScaleFactor(Context& cc, int slots);
void ModRaise(Ciphertext& ctxt, const int slots, const uint32_t correction, const bool prescaled = false, bool sparse_encaps = false);
} // namespace FIDESlib::CKKS

#endif // GPUCKKS_BOOTSTRAP_CUH
