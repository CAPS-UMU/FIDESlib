#include "CKKS/OpenFheCompat.cuh"

// Pulls in OpenFHE's generated core/config_core.h, which records WITH_NATIVEOPT (the
// -march=native/-mcpu=native build option).
#include <openfhe.h>

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace FIDESlib::CKKS {
namespace {

// Collects every indication that OpenFHE's (or this translation unit's) floating-point code
// was compiled with FMA contraction enabled. Empty => safe.
std::string DetectOpenFheFmaFlags() {
    std::string flags;
    [[maybe_unused]] auto add = [&flags](const char* what) {
        if (!flags.empty())
            flags += ", ";
        flags += what;
    };

    // OpenFHE's own build configuration: WITH_NATIVEOPT adds -march=native / -mcpu=native.
#ifdef WITH_NATIVEOPT
    add("OpenFHE WITH_NATIVEOPT (-march=native)");
#endif

    // FMA-enabling flags found by CMake in the installed OpenFHE's exported compile flags
    // (catches explicit -mfma/-mavx2/-mavx512f that WITH_NATIVEOPT would not record).
#ifdef FIDESLIB_OPENFHE_FMA_FLAGS
    add("OpenFHE flags: " FIDESLIB_OPENFHE_FMA_FLAGS);
#endif

    // If this translation unit itself was compiled with FMA, the two sides cannot be assumed
    // to round identically either.
#ifdef __FMA__
    add("FIDESlib compiled with __FMA__");
#endif

    return flags;
}

std::once_flag probeOnce;

}  // namespace

bool OpenFheHasFma() {
    return !DetectOpenFheFmaFlags().empty();
}

std::string OpenFheFmaFlags() {
    return DetectOpenFheFmaFlags();
}

void CheckOpenFheFpCompat() {
    std::call_once(probeOnce, [] {
        const std::string flags = DetectOpenFheFmaFlags();
        if (flags.empty())
            return;

        const std::string message =
            "FIDESlib: the linked OpenFHE was built with FMA-enabled floating-point codegen [" +
            flags +
            "]. FIDESlib reproduces OpenFHE's Chebyshev / functional-bootstrapping arithmetic "
            "bit-for-bit, which requires identical non-FMA codegen: with FMA enabled the CPU "
            "reference and the GPU path disagree in the last bits (value-correct but not "
            "bit-exact). Rebuild/install OpenFHE with -DWITH_NATIVEOPT=OFF (the default) and "
            "without -mfma/-mavx2/-mavx512f. Set FIDESLIB_ALLOW_OPENFHE_FMA=1 to downgrade this "
            "to a warning.";

        const char* allow = std::getenv("FIDESLIB_ALLOW_OPENFHE_FMA");
        if (allow != nullptr && std::string(allow) != "0") {
            std::cerr << "WARNING: " << message << std::endl;
            return;
        }
        throw std::runtime_error(message);
    });
}

}  // namespace FIDESlib::CKKS
