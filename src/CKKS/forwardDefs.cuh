//
// Created by carlosad on 4/11/24.
//

#ifndef GPUCKKS_FORWARDDEFS_CUH
#define GPUCKKS_FORWARDDEFS_CUH

#include <concepts>
#include <memory>

namespace FIDESlib::CKKS {
// class Context;
using KeyHash = std::string;
class ContextData;
using Context = std::shared_ptr<ContextData>;
class Ciphertext;
class KeySwitchingKey;
class Plaintext;
class RNSPoly;
template <typename T> class Limb;
class Parameters;
class BootstrapPrecomputation;

template <typename T>
concept CiphertextPtr = std::same_as<T, std::shared_ptr<Ciphertext>> || std::same_as<T, Ciphertext*>;
template <typename T>
concept PlaintextPtr = std::same_as<T, std::shared_ptr<Plaintext>> || std::same_as<T, Plaintext*>;

template <CiphertextPtr ptrT> class CiphertextBatch;
template <PlaintextPtr ptrT> class PlaintextBatch;
} // namespace FIDESlib::CKKS

namespace FIDESlib {
enum ALGO { ALGO_NATIVE = 0, ALGO_NONE = 1, ALGO_SHOUP = 3, ALGO_BARRETT = 4, ALGO_BARRETT_FP64 = 5 };

constexpr ALGO DEFAULT_ALGO = ALGO_BARRETT;

enum BOOT_CONFIG { UNIFORM = 0, UNIFORM_2 = 1, SPARSE = 2, ENCAPS = 3, ENCAPS_2 = 4 };
} // namespace FIDESlib

// namespace FIDESlib::CKKS

#endif // GPUCKKS_FORWARDDEFS_CUH
