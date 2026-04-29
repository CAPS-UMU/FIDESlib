//
// Created by carlosad on 14/09/25.
//

#ifndef PARAMETERSWITCH_CUH
#define PARAMETERSWITCH_CUH

#include "pke/openfhe.h"

namespace FIDESlib {
namespace CKKS {

lbcrypto::CryptoContext<lbcrypto::DCRTPoly> createSwitchableContextBasedOnContext(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, int limbs, int digits, int hamming_weight);

std::pair<std::pair<std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>, std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>>,
  std::shared_ptr<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>>
createContextSwitchingKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cca,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& ccb,
  const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& a,
  int hamming_weight_b);

std::pair<std::pair<std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>, std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>>,
  std::shared_ptr<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>>
createContextSwitchingKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cca,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& ccb,
  const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& a,
  int hamming_weight_b);

} // namespace CKKS
} // namespace FIDESlib

#endif // PARAMETERSWITCH_CUH
