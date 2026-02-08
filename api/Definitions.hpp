#ifndef API_DEFINITIONS_HPP
#define API_DEFINITIONS_HPP

#include <cinttypes>
#include <memory>

namespace fideslib {

/// @brief Phantom type for the CKKS-RNS scheme.
typedef uint32_t CryptoContextCKKSRNS;

/// @brief Phantom type for the DCRTPoly representation.
typedef uint32_t DCRTPoly;

/// @brief Ciphertext representation for the CKKS-RNS scheme.
/// @tparam T Underlying representation type.
template <typename T> class CiphertextImpl;

/// @brief Shared pointer alias for CiphertextImpl.
/// @tparam T Underlying representation type.
template <typename T> using Ciphertext = std::shared_ptr<CiphertextImpl<T>>;

/// @brief Class for managing a cryptographic context.
/// @tparam T Underlying representation type.
template <typename T> class CryptoContextImpl;

/// @brief Shared pointer alias for CryptoContextImpl.
/// @tparam T Underlying representation type.
template <typename T> using CryptoContext = std::shared_ptr<CryptoContextImpl<T>>;

/// @brief Plaintext representation for the CKKS-RNS scheme.
class PlaintextImpl;

/// @brief Shared pointer alias for PlaintextImpl.
using Plaintext = std::shared_ptr<PlaintextImpl>;

/// @brief Enumeration of supported PKE scheme features.
enum PKESchemeFeature {
	PKE			 = 0x01,
	KEYSWITCH	 = 0x02,
	PRE			 = 0x04,
	LEVELEDSHE	 = 0x08,
	ADVANCEDSHE	 = 0x10,
	MULTIPARTY	 = 0x20,
	FHE			 = 0x40,
	SCHEMESWITCH = 0x80,
};

/// @brief Result structure for decryption operations.
struct DecryptResult {
	bool isValid;
	uint32_t messageLength;
};

/// @brief Enumeration of supported scaling techniques.
enum ScalingTechnique {
    FIXEDMANUAL = 0,
    FIXEDAUTO,
    FLEXIBLEAUTO,
    FLEXIBLEAUTOEXT,
};

/// @brief Enumeration of supported key switching techniques.
enum KeySwitchTechnique {
    INVALID_KS_TECH = 0,
    //BV,
    HYBRID = 2,
};

/// @brief Enumeration of supported secret key distributions.
enum SecretKeyDist {
    GAUSSIAN            = 0,
    UNIFORM_TERNARY     = 1,
    SPARSE_TERNARY      = 2,
    SPARSE_ENCAPSULATED = 3,  
};

/// @brief Enumeration of supported security levels.
enum SecurityLevel {
    HEStd_128_classic,
    HEStd_192_classic,
    HEStd_256_classic,
    HEStd_128_quantum,
    HEStd_192_quantum,
    HEStd_256_quantum,
    HEStd_NotSet,
};

enum CKKSDataType {
    REAL = 0,
    COMPLEX,
};

} // namespace fideslib

#endif