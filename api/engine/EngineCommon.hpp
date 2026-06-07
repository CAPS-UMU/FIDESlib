#ifndef API_ENGINE_COMMON_HPP
#define API_ENGINE_COMMON_HPP

#include <cstdint>

#include "Definitions.hpp"

namespace fideslib {

// Number of mod-evaluation levels the bootstrap mod-reduction Chebyshev approximation consumes for
// the given secret-key distribution. Both engines pass this to OpenFHE's EvalBootstrapSetup so the
// CPU and CUDA paths configure mod-evaluation identically. Defined in EngineCommon.cpp.
int32_t bootstrapModEvalLevels(SecretKeyDist keyDist);

} // namespace fideslib

#endif // API_ENGINE_COMMON_HPP
