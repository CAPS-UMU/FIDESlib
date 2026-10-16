#ifndef API_PUBLICKEY_HPP
#define API_PUBLICKEY_HPP

#include <any>
#include <memory>

#include "Definitions.hpp"

namespace fideslib {
/// @brief Public key for the CKKS-RNS scheme.
/// @tparam T Underlying representation type.
template <typename T> class PublicKeyImpl;

/// @brief Shared pointer alias for PublicKeyImpl.
template <typename T> using PublicKey = std::shared_ptr<PublicKeyImpl<T>>;

/// @brief Specialization of PublicKey for the DCRTPoly representation.
template <> class PublicKeyImpl<DCRTPoly> {
  public:
    PublicKeyImpl() = default;
    ~PublicKeyImpl() = default;

    // ---- Copy ----

    PublicKeyImpl(const PublicKeyImpl&);
    PublicKeyImpl& operator=(const PublicKeyImpl&) = delete;

    // ---- Move ----

    PublicKeyImpl(PublicKeyImpl&&) = delete;
    PublicKeyImpl& operator=(PublicKeyImpl&&) = delete;

    // ---- Internal State ----

    std::any pimpl;
};
} // namespace fideslib
#endif // API_PUBLICKEY_HPP