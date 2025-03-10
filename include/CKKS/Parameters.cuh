//
// Created by carlosad on 17/03/24.
//

#ifndef FIDESLIB_CKKS_PARAMETERS_CUH
#define FIDESLIB_CKKS_PARAMETERS_CUH

#include <optional>
#include <vector>
#include "LimbUtils.cuh"
#include "Math.cuh"
#include "openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib::CKKS {

class Parameters {
   public:
    int logN, L, dnum, K = -1;
    std::vector<PrimeRecord> primes;
    std::vector<PrimeRecord> Sprimes;
    std::vector<double> ModReduceFactor;
    std::vector<double> ScalingFactorReal;
    std::vector<double> ScalingFactorRealBig;
    lbcrypto::ScalingTechnique scalingTechnique;
    RawParams* raw = nullptr;
    int batch = 1;

    Parameters adaptTo(RawParams& raw) const {

        std::vector<PrimeRecord> new_primes;
        for (auto i : raw.moduli) {
            new_primes.push_back(PrimeRecord{.p = i, .type = U64});
        }
        std::vector<PrimeRecord> new_SPECIALprimes;
        for (auto i : raw.SPECIALmoduli) {
            new_SPECIALprimes.push_back(PrimeRecord{.p = i, .type = U64});
        }

        Parameters res{.logN = raw.logN,
                       .L = raw.L,
                       .dnum = raw.dnum,
                       .K = raw.K,
                       .primes = std::move(new_primes),
                       .Sprimes = std::move(new_SPECIALprimes),
                       .ModReduceFactor = raw.ModReduceFactor,
                       .ScalingFactorReal = raw.ScalingFactorReal,
                       .ScalingFactorRealBig = raw.ScalingFactorRealBig,
                       .scalingTechnique = raw.scalingTechnique,
                       .raw = &raw,
                       .batch = batch};
        return res;
    }
};

}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_CKKS_PARAMETERS_CUH
