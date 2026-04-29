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
// void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled = false);
void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled = false);
double GetPreScaleFactor(Context& cc, int slots);
void ModRaise(Ciphertext& ctxt, const int slots, const uint32_t correction, const bool prescaled = false, bool sparse_encaps = false);
} // namespace FIDESlib::CKKS

#endif // GPUCKKS_BOOTSTRAP_CUH
