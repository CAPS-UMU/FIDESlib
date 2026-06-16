#include "engine/EngineCommon.hpp"

#include <openfhe.h>

#include <vector>

namespace fideslib {

int32_t bootstrapModEvalLevels(SecretKeyDist keyDist) {
	std::vector<double> coeffchebyshev;
	int doubleAngleIts = 3;
	if (keyDist == SPARSE_ENCAPSULATED) {
		coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsSparseEncapsulated;
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
	} else if (keyDist == SPARSE_TERNARY) {
		coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsSparse;
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
	} else if (keyDist == UNIFORM_TERNARY) {
		coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_UNIFORM;
	} else {
		OPENFHE_THROW("Unsupported key distribution");
	}
	return static_cast<int>(lbcrypto::GetMultiplicativeDepthByCoeffVector(coeffchebyshev, false)) + doubleAngleIts;
}

} // namespace fideslib
