#include "engine/Backend.hpp"

#include "engine/Engine.hpp"
#include "engine/cpu/OpenFheEngine.hpp"
#include "engine/cuda/CudaEngine.hpp"

#include <stdexcept>

namespace fideslib {

bool IsBackendAvailable(Backend backend) {
	switch (backend) {
	case Backend::CPU: return true;
	case Backend::CUDA: return true;
	}
	return false;
}

std::unique_ptr<Engine> MakeEngine(Backend backend) {
	switch (backend) {
	case Backend::CPU: return std::make_unique<OpenFheEngine>();
	case Backend::CUDA: return std::make_unique<CudaEngine>();
	}
	throw std::runtime_error("unknown backend");
}

} // namespace fideslib
