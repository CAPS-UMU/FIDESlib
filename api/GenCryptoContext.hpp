#ifndef API_GENCRYPTOCONTEXT_HPP
#define API_GENCRYPTOCONTEXT_HPP

#include "CCParams.hpp"
#include "CryptoContext.hpp"

namespace fideslib {
/// @brief CKKS Context builder function.
/// @param params Parameter set.
/// @return CKKS context.
CryptoContext<DCRTPoly> GenCryptoContext(CCParams<CryptoContextCKKSRNS>& params);
} // namespace fideslib

#endif // API_GENCRYPTOCONTEXT_HPP