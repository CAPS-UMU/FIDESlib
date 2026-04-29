//
// Created by carlosad on 17/03/24.
//

#ifndef FIDESLIB_CKKS_PARAMETERS_CUH
#define FIDESLIB_CKKS_PARAMETERS_CUH

#include "LimbUtils.cuh"
#include "Math.cuh"
#include <optional>
#include <vector>

#include "openfhe-interface/RawCiphertext.cuh"
#undef duration

namespace FIDESlib::CKKS {

struct RawParams;

class Parameters {
  public:
	int logN, L, dnum, K = -1;
	std::vector<PrimeRecord> primes;
	std::vector<PrimeRecord> Sprimes;
	std::vector<double> ModReduceFactor;
	std::vector<double> ScalingFactorReal;
	std::vector<double> ScalingFactorRealBig;
	lbcrypto::ScalingTechnique scalingTechnique;
	std::optional<RawParams> raw;
	int batch = 100;

	bool operator<(const Parameters& b) const {

		if (logN < b.logN) {
			return true;
		} else if (logN > b.logN) {
			return false;
		}

		if (L < b.L) {
			return true;
		} else if (L > b.L) {
			return false;
		}

		if (dnum < b.dnum) {
			return true;
		} else if (dnum > b.dnum) {
			return false;
		}

		if (K < b.K) {
			return true;
		} else if (K > b.K) {
			return false;
		}

		if (scalingTechnique < b.scalingTechnique) {
			return true;
		} else if (scalingTechnique > b.scalingTechnique) {
			return false;
		}

		if (primes.size() < b.primes.size()) {
			return true;
		} else if (primes.size() > b.primes.size()) {
			return false;
		}

		for (size_t i = 0; i < primes.size(); i++) {
			if (b.primes[i].p > primes[i].p) {
				return true;
			} else if (b.primes[i].p < primes[i].p) {
				return false;
			}
		}

		if (Sprimes.size() < b.Sprimes.size()) {
			return true;
		} else if (Sprimes.size() > b.Sprimes.size()) {
			return false;
		}

		for (size_t i = 0; i < Sprimes.size(); i++) {
			if (b.Sprimes[i].p > Sprimes[i].p) {
				return true;
			} else if (b.Sprimes[i].p < Sprimes[i].p) {
				return false;
			}
		}

		if (this->raw.has_value() < b.raw.has_value()) {
			return true;
		}
		return false;
	}

	bool operator==(const Parameters& b) const {
		return !(*this < b) && !(b < *this);
	}

	Parameters adaptTo(RawParams& raw) const;
};

} // namespace FIDESlib::CKKS
#endif // FIDESLIB_CKKS_PARAMETERS_CUH
