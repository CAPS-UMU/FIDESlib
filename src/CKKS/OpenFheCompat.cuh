//
// OpenFHE floating-point-codegen compatibility probe.
//
// FIDESlib reproduces OpenFHE's Chebyshev and functional-bootstrapping arithmetic bit-for-bit:
// the GPU pipeline is validated against the CPU reference under ASSERT_EQ_CIPHERTEXT. That
// equality only holds when OpenFHE's floating-point codegen matches FIDESlib's — in particular
// when *neither* side contracts multiply+add into FMA. The shared helpers are header-only
// templates (e.g. LongDivisionChebyshev, ComputeDegreesPS, the scaling-factor math), so they are
// compiled into OpenFHE's CPU translation units *and* into FIDESlib's GPU translation units. If
// OpenFHE is built with machine-specific optimizations (WITH_NATIVEOPT -> -march=native, or an
// explicit -mfma/-mavx2/-mavx512f), those two copies round differently in the last bits and the
// CPU/GPU comparison fails — value-correct, but not bit-exact.
//
// See docs / commit history: an OpenFHE installed from the fork's `build/` directory
// (WITH_NATIVEOPT=ON) makes the bit-exact CPU/GPU compatibility tests fail, while the
// WITH_NATIVEOPT=OFF install passes.
//
#ifndef FIDESLIB_CKKS_OPENFHECOMPAT_CUH
#define FIDESLIB_CKKS_OPENFHECOMPAT_CUH

#include <string>

namespace FIDESlib::CKKS {

/// @brief True when the linked OpenFHE was built with FMA-enabled floating-point codegen
///        (WITH_NATIVEOPT / -march=native, or -mfma/-mavx2/-mavx512*).
bool OpenFheHasFma();

/// @brief Human-readable list of the FMA-enabling flags detected in the OpenFHE build
///        (empty string when none). Useful in failure messages and tests.
std::string OpenFheFmaFlags();

/// @brief Startup probe: fails fast (throws std::runtime_error) when the linked OpenFHE was
///        built with FMA-enabled floating-point codegen, which silently breaks FIDESlib's
///        bit-exact CPU/GPU agreement. Idempotent (runs the check at most once). Set the
///        environment variable FIDESLIB_ALLOW_OPENFHE_FMA=1 to downgrade the failure to a
///        warning.
void CheckOpenFheFpCompat();

}  // namespace FIDESlib::CKKS

#endif  // FIDESLIB_CKKS_OPENFHECOMPAT_CUH
