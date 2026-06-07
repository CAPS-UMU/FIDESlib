#ifndef API_BACKEND_HPP
#define API_BACKEND_HPP

#include <memory>

namespace fideslib {

/// @brief Execution backend, chosen at runtime. The enum is always declared;
/// availability depends on what was compiled in (see IsBackendAvailable).
enum class Backend {
	CPU,  ///< OpenFHE on the host. Always available.
	CUDA, ///< FIDESlib CUDA implementation. Available iff built with CUDA.
};

class Engine;

/// @brief Whether a backend was compiled into this build.
bool IsBackendAvailable(Backend backend);

/// @brief Construct the engine for a backend; throws if it was not compiled in.
std::unique_ptr<Engine> MakeEngine(Backend backend);

} // namespace fideslib

#endif
