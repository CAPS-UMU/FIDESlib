#ifndef API_KEYPAIR_HPP
#define API_KEYPAIR_HPP

#include "Definitions.hpp"
#include "PrivateKey.hpp"
#include "PublicKey.hpp"

namespace fideslib {
/// @brief Class for managing a public/private key pair.
/// @tparam T Underlying representation type.
template <typename T> class KeyPair;

/// @brief Specialization of KeyPair for the DCRTPoly representation.
template <> class KeyPair<DCRTPoly> {
  public:
    KeyPair() = default;
    ~KeyPair() = default;

    // ---- Copy ----

    KeyPair(const KeyPair&) = delete;
    KeyPair& operator=(const KeyPair&) = default;

    // ---- Move ----

    KeyPair(KeyPair&&) = default;
    KeyPair& operator=(KeyPair&&) = default;

    // ---- Keys ----

    PublicKey<DCRTPoly> publicKey;
    PrivateKey<DCRTPoly> secretKey;
};
} // namespace fideslib

#endif // API_KEYPAIR_HPP