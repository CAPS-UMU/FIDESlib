//
// Created by seyda on 5/19/25.
//

#include "PolyApprox.cuh"

#include "CKKS/AccumulateBroadcast.cuh"

namespace FIDESlib::CKKS {

std::vector<double> get_chebyshev_coefficients(const std::function<double(double)>& func, const double a, const double b, const uint32_t degree) {
	if (!degree) {
		OPENFHE_THROW("The degree of approximation can not be zero");
	}

	const size_t coeffTotal{ degree + 1 };
	const double bMinusA = 0.5 * (b - a);
	const double bPlusA	 = 0.5 * (b + a);
	const double PiByDeg = M_PI / static_cast<double>(coeffTotal);
	std::vector<double> functionPoints(coeffTotal);
	for (size_t i = 0; i < coeffTotal; ++i)
		functionPoints[i] = func(std::cos(PiByDeg * (i + 0.5)) * bMinusA + bPlusA);

	const double multFactor = 2.0 / static_cast<double>(coeffTotal);
	std::vector<double> coefficients(coeffTotal);
	for (size_t i = 0; i < coeffTotal; ++i) {
		for (size_t j = 0; j < coeffTotal; ++j)
			coefficients[i] += functionPoints[j] * std::cos(PiByDeg * i * (j + 0.5));
		coefficients[i] *= multFactor;
	}
	return coefficients;
}

} // namespace FIDESlib::CKKS
