#ifndef API_PRIVATEKEY_HPP
#define API_PRIVATEKEY_HPP

#include <any>
#include <memory>

namespace fideslib {
/// @brief Private key for the CKKS-RNS scheme.
/// @tparam T Underlying representation type.
template <typename T> class PrivateKeyImpl;

/// @brief Shared pointer alias for PrivateKeyImpl.
template <typename T> using PrivateKey = std::shared_ptr<PrivateKeyImpl<T>>;

/// @brief Specialization of PrivateKey for the DCRTPoly representation.
template <> class PrivateKeyImpl<DCRTPoly> {
  public:
    PrivateKeyImpl() = default;
    ~PrivateKeyImpl() = default;

    // ---- Copy ----

    PrivateKeyImpl(const PrivateKeyImpl&);
    PrivateKeyImpl& operator=(const PrivateKeyImpl&) = delete;

    // ---- Move ----

    PrivateKeyImpl(PrivateKeyImpl&&) = delete;
    PrivateKeyImpl& operator=(PrivateKeyImpl&&) = delete;

    // ---- Internal State ----

    std::any pimpl;
};
} // namespace fideslib

#endif // API_PRIVATEKEY_HPP