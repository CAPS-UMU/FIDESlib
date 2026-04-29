//
// Created by carlosad on 24/12/25.
//
#include "CKKS/Parameters.cuh"

namespace FIDESlib::CKKS {
Parameters Parameters::adaptTo(RawParams& raw) const {
	//: q
	// std::cout << "Adapt in" << std::endl;

	std::vector<PrimeRecord> new_primes;
	for (auto i : raw.moduli) {
		new_primes.push_back(PrimeRecord{ .p = i, .type = U64 });
	}
	std::vector<PrimeRecord> new_SPECIALprimes;
	for (auto i : raw.SPECIALmoduli) {
		new_SPECIALprimes.push_back(PrimeRecord{ .p = i, .type = U64 });
	}

	Parameters res{ .logN	  = raw.logN,
		.L					  = raw.L,
		.dnum				  = raw.dnum,
		.K					  = raw.K,
		.primes				  = std::move(new_primes),
		.Sprimes			  = std::move(new_SPECIALprimes),
		.ModReduceFactor	  = raw.ModReduceFactor,
		.ScalingFactorReal	  = raw.ScalingFactorReal,
		.ScalingFactorRealBig = raw.ScalingFactorRealBig,
		.scalingTechnique	  = raw.scalingTechnique,
		.raw				  = raw,
		.batch				  = batch };
	// std::cout << "Adapt out" << std::endl;
	return res;
}
} // namespace FIDESlib::CKKS