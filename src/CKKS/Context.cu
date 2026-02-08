//
// Created by carlosad on 2/05/24.
//
#include <source_location>
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"

#include "../parallel_for.hpp"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"

#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

namespace FIDESlib::CKKS {

std::atomic_uint64_t next_uid = 0;
constexpr bool SPLIT_SPECIAL = true;

std::map<Parameters, std::shared_ptr<ContextData>> map_param_context;
Context currentContext;

//std::map<std::pair<Parameters, Parameters>, std::shared_ptr<std::map<KeyHash, KeySwitchingKey>>> map_param_switch;
std::vector<std::pair<std::pair<Parameters, Parameters>, std::unique_ptr<std::map<KeyHash, KeySwitchingKey>>>>
    map_param_switch;

/* Communicate internally if ContextData created succesfully, on unsuccessful creation,
 * its param field contains a "normalized" version for caching */
bool OK = false;

ContextData::ContextData(const Parameters& param_, const std::vector<int>& devs, const int secBits)
    : my_range(loc, LIFETIME),
      param((CudaNvtxStart(std::string{sc::current().function_name()}.substr()), param_)),
      precom(),
      logN(param.logN),
      N(1 << logN),
      rescaleTechnique(translateRescalingTechnique(param.scalingTechnique)),
      L(param.L),
      logQ(computeLogQ(L, param.primes)),
      batch(param.batch),
      GPUid(devs),
      dnum((validateDnum(GPUid, param.dnum) /*, param.dnum*/)),
      GPUdigits(generateGPUdigits(dnum, GPUid)),
      prime((param.primes.resize(L + 1), param.primes)),
      meta{generateMeta(GPUid, dnum, GPUdigits, prime, param)},
      logQ_d(computeLogQ_d(dnum, meta, prime)),
      K(computeK(logQ_d, param.Sprimes, param)),
      logP(computeLogQ(K - 1, param.Sprimes)),
      specialPrime((param.Sprimes.resize(K), param.Sprimes)),
      specialMeta(generateSpecialMeta(meta, specialPrime, L + 1, GPUid)),
      splitSpecialMeta(generateSplitSpecialMeta(specialMeta.at(0), GPUid)),
      decompMeta(generateDecompMeta(meta, GPUdigits, GPUid, L)),
      digitMeta(generateDigitMeta(meta, splitSpecialMeta, specialMeta.at(0), GPUdigits, GPUid)),
      gatherMeta(generateGatherMeta(meta, L)),
      limbGPUid(generateLimbGPUid(meta, L)),
      digitGPUid(generateDigitGPUid(meta, L, dnum)),
      GPUrank(GPUid.size())
//top_limb(devs.size())
{
    if (map_param_context.contains(param)) {
        OK = false;
        return;
    }

    //auto& constants = precom.constants;
    //auto& globals = precom.globals;
    auto [constants, globals] = SetupConstants<Parameters>(prime, meta, specialPrime, specialMeta.at(0), decompMeta,
                                                           digitMeta, GPUdigits, GPUid, N, param);

    precom.constants = constants;
    precom.globals = std::move(globals);

    PrepareNCCLCommunication();

    // CheckBitSecurity();
    int bits = 0;
    for (auto& j : {prime, specialPrime})
        for (auto& i : j)
            bits += i.bits;

    for (int dev : GPUid) {
        cudaSetDevice(dev);
        cudaMemPool_t mp;
        cudaDeviceGetDefaultMemPool(&mp, dev);
        uint64_t threshold = UINT64_MAX;  //5l * 1024l * 1024l * 1024l;  // One Gigabyte of memory
        cudaMemPoolSetAttribute(mp, cudaMemPoolAttrReleaseThreshold, &threshold);
        CudaCheckErrorModNoSync;
    }

    OK = true;
    CudaNvtxStop();
}

std::vector<dim3> ContextData::generateLimbGPUid(const std::vector<std::vector<LimbRecord>>& meta, const int L) {
    std::vector<dim3> res(L + 1, 0);
    for (int i = 0; i < static_cast<int>(meta.size()); ++i) {
        for (size_t j = 0; j < meta.at(i).size(); ++j) {
            res.at(meta[i][j].id) = {static_cast<uint32_t>(i), static_cast<uint32_t>(j), 0};
        }
    }
    return res;
}

std::vector<std::vector<std::vector<LimbRecord>>> ContextData::generateDigitMeta(
    const std::vector<std::vector<LimbRecord>>& meta, const std::vector<std::vector<LimbRecord>>& splitSpecialMeta,
    const std::vector<LimbRecord>& specialMeta, const std::vector<std::vector<int>>& digitGPUid,
    const std::vector<int>& GPUid) {
    std::vector<std::vector<std::vector<LimbRecord>>> digitMeta(meta.size());

    for (size_t i = 0; i < digitGPUid.size(); ++i) {
        cudaSetDevice(GPUid[i]);
        for (int d : digitGPUid.at(i)) {
            digitMeta[i].emplace_back();

            if constexpr (SPLIT_SPECIAL) {
                for (auto& l : splitSpecialMeta.at(i)) {
                    digitMeta[i].back().emplace_back(LimbRecord{.id = l.id, .type = l.type, .digit = l.digit});
                    digitMeta[i].back().back().stream.init();
                }
            } else {
                for (auto& l : specialMeta) {
                    digitMeta[i].back().emplace_back(LimbRecord{.id = l.id, .type = l.type, .digit = l.digit});
                    digitMeta[i].back().back().stream.init();
                }
            }

            for (auto& l : meta.at(i)) {
                if (l.digit != d) {
                    digitMeta[i].back().emplace_back(LimbRecord{.id = l.id, .type = l.type, .digit = l.digit});
                    digitMeta[i].back().back().stream.init();
                }
            }

            /*
            std::sort(
                digitMeta[i].back().begin() + specialMeta.size(), digitMeta[i].back().end(),
                [](LimbRecord& a, LimbRecord& b) { return a.digit < b.digit || (a.digit == b.digit && a.id < b.id); });
            */
        }
    }
    return digitMeta;
}

std::vector<std::vector<std::vector<LimbRecord>>> ContextData::generateDecompMeta(
    const std::vector<std::vector<LimbRecord>>& meta, const std::vector<std::vector<int>> digitGPUid,
    const std::vector<int>& GPUid, int L) {
    std::vector<std::vector<std::vector<LimbRecord>>> decompMeta(meta.size());

    for (size_t i = 0; i < digitGPUid.size(); ++i) {
        cudaSetDevice(GPUid[i]);
        for (int d : digitGPUid.at(i)) {
            decompMeta[i].emplace_back();

            for (int primeid = 0; primeid <= L; ++primeid) {
                for (auto& m : meta) {
                    for (auto& l : m) {
                        if (l.id == primeid && l.digit == d) {
                            decompMeta[i].back().push_back(LimbRecord{.id = l.id, .type = l.type, .digit = l.digit});
                            decompMeta[i].back().back().stream.init();
                        }
                    }
                }
            }
        }
    }

    return decompMeta;
}

bool ContextData::isValidPrimeId(const int i) const {
    return (i >= 0 && i < L + 1 + K);
}

int ContextData::computeLogQ(const int L, std::vector<PrimeRecord>& primes) {
    int res = 0;
    assert(L <= (int)primes.size());
    for (int i = 0; i <= L; ++i) {
        res += (primes[i].bits == -1) ? (primes[i].bits = (int)std::bit_width(primes[i].p)) : primes[i].bits;
    }
    return res;
}

const int& ContextData::validateDnum(const std::vector<int>& GPUid, const int& dnum) {
    return dnum;
}

int findDigitOnParam(const Parameters& param, uint64_t modulus) {
    for (size_t i = 0; i < param.raw->PARTITIONmoduli.size(); ++i) {
        for (uint64_t j : param.raw->PARTITIONmoduli.at(i)) {
            if (modulus == j)
                return i;
        }
    }
    return -1;
}

std::vector<std::vector<LimbRecord>> ContextData::generateMeta(const std::vector<int>& GPUid, const int dnum,
                                                               const std::vector<std::vector<int>> digitGPUid,
                                                               const std::vector<PrimeRecord>& prime,
                                                               const Parameters& param) {
    int devs = GPUid.size();
    std::vector<std::vector<LimbRecord>> meta(devs);

    //for (int i = 0; i < devs; ++i) {
    // cudaSetDevice(GPUid.at(i));
    // meta.at(i).resize((prime.size() + devs - i - 1) / devs);
    //}

    if constexpr (0) {
        int threshhold1 = (prime.size() / 2 + devs - 1) / devs;
        int threshhold2 = threshhold1 * devs;
        int dev = 0;
        for (int i = 0; i < (int)prime.size(); ++i) {
            int digit_ = !param.raw ? i % dnum : findDigitOnParam(param, prime.at(i).p);

            if (i < threshhold1) {
            } else if (i < threshhold2) {
                dev = (dev + 1) % devs;
                if (dev == 0)
                    dev = (dev + 1) % devs;
            } else {
                dev = (dev + 1) % devs;
            }
            /*{
            int dev = -1;
            for (size_t j = 0; j < digitGPUid.size(); ++j) {
                for (auto& k : digitGPUid.at(j))
                    if (k == digit)
                        dev = j;
            }
        }*/

            cudaSetDevice(GPUid[dev]);

            meta[dev].push_back(
                LimbRecord{.id = i,
                           .type = (prime[i].type ? *(prime[i].type) : (prime[i].bits <= 30 ? U32 : U64)),
                           .digit = digit_});
            meta[dev].back().stream.init();
            // std::cout << "i: " << i << " gpu:" << dev << std::endl;
        }
    } else {
        for (int i = 0; i < (int)prime.size(); ++i) {
            int digit_ = !param.raw ? i % dnum : findDigitOnParam(param, prime.at(i).p);

            int dev = i % GPUid.size();
            /*{
            int dev = -1;
            for (size_t j = 0; j < digitGPUid.size(); ++j) {
                for (auto& k : digitGPUid.at(j))
                    if (k == digit)
                        dev = j;
            }
        }*/

            cudaSetDevice(GPUid[dev]);

            meta[dev].push_back(
                LimbRecord{.id = i,
                           .type = (prime[i].type ? *(prime[i].type) : (prime[i].bits <= 30 ? U32 : U64)),
                           .digit = digit_});
            meta[dev].back().stream.init();
        }
    }

    return meta;
}

std::vector<int> ContextData::computeLogQ_d(const int dnum, const std::vector<std::vector<LimbRecord>>& meta,
                                            const std::vector<PrimeRecord>& prime) {
    std::vector<int> logQ_d(dnum, 0);

    for (auto& i : meta)
        for (auto& j : i)
            logQ_d.at(j.digit) += prime.at(j.id).bits;

    return logQ_d;
}

const int& ContextData::computeK(const std::vector<int>& logQ_d, std::vector<PrimeRecord>& Sprimes, Parameters& param) {

    size_t res = 0;
    int logMaxD = *std::max_element(logQ_d.begin(), logQ_d.end());
    int bits = 0;
    for (; bits < logMaxD && res < Sprimes.size(); ++res) {
        bits += (Sprimes.at(res).bits <= 0) ? (Sprimes.at(res).bits = (int)std::bit_width(Sprimes.at(res).p)) - 1
                                            : Sprimes.at(res).bits - 1;
    }

    if (param.K != -1) {
        return param.K;
    }

    assert(bits >= logMaxD);
    return param.K = res;
}

std::vector<std::vector<LimbRecord>> ContextData::generateSpecialMeta(const std::vector<std::vector<LimbRecord>>& meta,
                                                                      const std::vector<PrimeRecord>& specialPrime,
                                                                      const int ID0, const std::vector<int>& GPUid) {
    std::vector<std::vector<LimbRecord>> specialMeta(GPUid.size());

    for (size_t d = 0; d < GPUid.size(); ++d) {
        specialMeta.at(d).resize(specialPrime.size());
        cudaSetDevice(GPUid[d]);
        for (int i = 0; i < (int)specialPrime.size(); ++i) {
            specialMeta.at(d).at(i).id = ID0 + i;
            specialMeta.at(d).at(i).type =
                (specialPrime[i].type ? *(specialPrime[i].type) : (specialPrime[i].bits <= 30 ? U32 : U64));
            specialMeta.at(d).at(i).stream.init();
        }
    }

    return specialMeta;
}

std::vector<LimbRecord> ContextData::generateGatherMeta(const std::vector<std::vector<LimbRecord>>& meta, int L) {
    std::vector<LimbRecord> gatherMeta(L + 1);

    /*TODO: generate streams for each device*/
    for (int i = 0; i <= L; ++i) {
        for (size_t j = 0; j < meta.size(); ++j) {
            for (size_t k = 0; k < meta.at(j).size(); ++k) {
                if (meta[j][k].id == i) {
                    gatherMeta.at(i).id = i;
                    gatherMeta.at(i).digit = meta[j][k].digit;
                    gatherMeta.at(i).type = meta[j][k].type;
                }
            }
        }
    }

    return gatherMeta;
}

std::vector<std::vector<int>> ContextData::generateGPUdigits(const int dnum, const std::vector<int>& devs) {
    std::vector<std::vector<int>> res(devs.size());
    for (int d = 0; d < dnum; ++d) {
        for (int gpu = 0; gpu < devs.size(); ++gpu) {
            res[gpu].push_back(d);
        }
    }
    return res;
}

RNSPoly& ContextData::getKeySwitchAux() {
    if (key_switch_aux == nullptr)
        key_switch_aux = std::make_unique<RNSPoly>(*this, L, false);

    key_switch_aux->generateDecompAndDigit(false);
    key_switch_aux->generateSpecialLimbs(false, false);
    return *key_switch_aux;
}

RNSPoly& ContextData::getKeySwitchAux2() {
    if (key_switch_aux2 == nullptr)
        key_switch_aux2 = std::make_unique<RNSPoly>(*this, L, false);
    key_switch_aux2->generateDecompAndDigit(false);
    key_switch_aux2->generateSpecialLimbs(false, false);
    return *key_switch_aux2;
}

RNSPoly& ContextData::getModdownAux(const int num) {
    if (moddown_aux[num % moddown_aux.size()] == nullptr)
        moddown_aux[num % moddown_aux.size()] = std::make_unique<RNSPoly>(*this, L, false);
    moddown_aux[num % moddown_aux.size()]->generateSpecialLimbs(false, true);
    return *moddown_aux[num % moddown_aux.size()];
}
std::vector<uint64_t> ContextData::ElemForEvalMult(int level, const double operand, int level_in) {

    uint32_t numTowers = level + 1;
    std::vector<lbcrypto::DCRTPoly::Integer> moduli(numTowers);
    for (usint i = 0; i < numTowers; i++) {
        moduli[i] = prime[i].p;
    }

    double scFactor;
    if (level_in == -1 || level_in == level) {
        scFactor = param.ScalingFactorReal[level];
    } else {
        /** Lets handle scale changes more efficiently!*/
        assert(level > 0);
        double scFactorIn = param.ScalingFactorReal[level_in];
        double scFactorOut = param.ScalingFactorReal[level - 1];
        double rescalingFactor = param.ModReduceFactor[level];
        scFactor = scFactorOut * rescalingFactor / scFactorIn;

        assert(abs(param.ScalingFactorReal[level - 1] * rescalingFactor -
                   param.ScalingFactorReal[level] * param.ScalingFactorReal[level]) < 1e-9);
        assert(abs(scFactorIn * scFactor / rescalingFactor - scFactorOut) < 1e-9);
    }

    typedef int128_t DoubleInteger;
    int32_t MAX_BITS_IN_WORD_LOCAL = 125;

    int32_t logApprox = 0;
    const double res = std::fabs(operand * scFactor);
    if (res > 0) {
        int32_t logSF = static_cast<int32_t>(std::ceil(std::log2(res)));
        int32_t logValid = (logSF <= MAX_BITS_IN_WORD_LOCAL) ? logSF : MAX_BITS_IN_WORD_LOCAL;
        logApprox = logSF - logValid;
    }
    double approxFactor = pow(2, logApprox);

    DoubleInteger large = static_cast<DoubleInteger>(operand / approxFactor * scFactor + 0.5);
    DoubleInteger large_abs = (large < 0 ? -large : large);
    DoubleInteger bound = (uint64_t)1 << 63;

    std::vector<lbcrypto::DCRTPoly::Integer> factors(numTowers);

    if (large_abs > bound) {
        for (usint i = 0; i < numTowers; i++) {
            DoubleInteger reduced = large % moduli[i].ConvertToInt();

            factors[i] = (reduced < 0) ? static_cast<uint64_t>(reduced + moduli[i].ConvertToInt())
                                       : static_cast<uint64_t>(reduced);
        }
    } else {
        int64_t scConstant = static_cast<int64_t>(large);
        for (usint i = 0; i < numTowers; i++) {
            int64_t reduced = scConstant % static_cast<int64_t>(moduli[i].ConvertToInt());

            factors[i] = (reduced < 0) ? reduced + moduli[i].ConvertToInt() : reduced;
        }
    }

    // Scale back up by approxFactor within the CRT multiplications.
    if (logApprox > 0) {
        int32_t logStep = (logApprox <= lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP)
                              ? logApprox
                              : lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP;
        lbcrypto::DCRTPoly::Integer intStep = uint64_t(1) << logStep;
        std::vector<lbcrypto::DCRTPoly::Integer> crtApprox(numTowers, intStep);
        logApprox -= logStep;

        while (logApprox > 0) {
            int32_t logStep = (logApprox <= lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP)
                                  ? logApprox
                                  : lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP;
            lbcrypto::DCRTPoly::Integer intStep = uint64_t(1) << logStep;
            std::vector<lbcrypto::DCRTPoly::Integer> crtSF(numTowers, intStep);
            crtApprox = lbcrypto::CKKSPackedEncoding::CRTMult(crtApprox, crtSF, moduli);
            logApprox -= logStep;
        }
        factors = lbcrypto::CKKSPackedEncoding::CRTMult(factors, crtApprox, moduli);
    }

    std::vector<uint64_t> result(numTowers);
    for (int i = 0; i < result.size(); ++i) {
        result[i] = factors[i].ConvertToInt();
        result[i] = result[i] % prime[i].p;
    }

    return result;
}

std::ostream& operator<<(std::ostream& o, const uint128_t& x) {
    if (x == std::numeric_limits<uint128_t>::min())
        return o << "0";
    if (x < 10)
        return o << (char)(x + '0');
    return o << x / 10 << (char)(x % 10 + '0');
}

std::vector<uint64_t> ContextData::ElemForEvalAddOrSub(const int level, const double operand, const int noise_deg) {
    usint sizeQl = level + 1;
    std::vector<lbcrypto::DCRTPoly::Integer> moduli(sizeQl);
    for (usint i = 0; i < sizeQl; i++) {
        moduli[i] = prime[i].p;
    }

    //double scFactor = param.ScalingFactorReal.at(level);
    double scFactor = 0;
    if (this->rescaleTechnique == FLEXIBLEAUTOEXT && level == L) {
        scFactor =
            param.ScalingFactorRealBig.at(level);  // cryptoParams->GetScalingFactorRealBig(ciphertext->GetLevel());
    } else {
        scFactor = param.ScalingFactorReal.at(level);  //cryptoParams->GetScalingFactorReal(ciphertext->GetLevel());
    }

    int32_t logApprox = 0;
    const double res = std::fabs(operand * scFactor);
    if (res > 0) {
        int32_t logSF = static_cast<int32_t>(std::ceil(std::log2(res)));
        int32_t logValid = (logSF <= lbcrypto::LargeScalingFactorConstants::MAX_BITS_IN_WORD)
                               ? logSF
                               : lbcrypto::LargeScalingFactorConstants::MAX_BITS_IN_WORD;
        logApprox = logSF - logValid;
    }
    double approxFactor = pow(2, logApprox);

    lbcrypto::DCRTPoly::Integer scConstant = static_cast<uint64_t>(operand * scFactor / approxFactor + 0.5);
    std::vector<lbcrypto::DCRTPoly::Integer> crtConstant(sizeQl, scConstant);

    // Scale back up by approxFactor within the CRT multiplications.
    if (logApprox > 0) {
        int32_t logStep = (logApprox <= lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP)
                              ? logApprox
                              : lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP;
        lbcrypto::DCRTPoly::Integer intStep = uint64_t(1) << logStep;
        std::vector<lbcrypto::DCRTPoly::Integer> crtApprox(sizeQl, intStep);
        logApprox -= logStep;

        while (logApprox > 0) {
            int32_t logStep = (logApprox <= lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP)
                                  ? logApprox
                                  : lbcrypto::LargeScalingFactorConstants::MAX_LOG_STEP;
            lbcrypto::DCRTPoly::Integer intStep = uint64_t(1) << logStep;
            std::vector<lbcrypto::DCRTPoly::Integer> crtSF(sizeQl, intStep);
            crtApprox = lbcrypto::CKKSPackedEncoding::CRTMult(crtApprox, crtSF, moduli);
            logApprox -= logStep;
        }
        crtConstant = lbcrypto::CKKSPackedEncoding::CRTMult(crtConstant, crtApprox, moduli);
    }

    // In FLEXIBLEAUTOEXT mode at level 0, we don't use the depth to calculate the scaling factor,
    // so we return the value before taking the depth into account.
    if (this->rescaleTechnique == FLEXIBLEAUTOEXT && level == L) {
        std::vector<uint128_t> result(sizeQl);
        for (int i = 0; i < result.size(); ++i) {
            result[i] = crtConstant[i].ConvertToInt<uint128_t>();
        }

        for (int i = 0; i < result.size(); ++i) {
            result[i] = result[i] % prime[i].p;
        }

        std::vector<uint64_t> result2(crtConstant.size());
        for (int i = 0; i < result.size(); ++i) {
            result2[i] = result[i];
        }

        return result2;
    }

    lbcrypto::DCRTPoly::Integer intScFactor = static_cast<uint64_t>(scFactor + 0.5);
    std::vector<lbcrypto::DCRTPoly::Integer> crtScFactor(sizeQl, intScFactor);

    for (usint i = 1; i < noise_deg; i++) {
        crtConstant = lbcrypto::CKKSPackedEncoding::CRTMult(crtConstant, crtScFactor, moduli);
    }

    std::vector<uint128_t> result(sizeQl);
    for (int i = 0; i < result.size(); ++i) {
        result[i] = crtConstant[i].ConvertToInt<uint128_t>();
    }

    for (int i = 0; i < result.size(); ++i) {
        result[i] = result[i] % prime[i].p;
    }

    std::vector<uint64_t> result2(crtConstant.size());
    for (int i = 0; i < result.size(); ++i) {
        result2[i] = result[i];
    }

    return result2;
}
std::vector<double>& ContextData::GetCoeffsChebyshev() {
    assert(param.raw);
    return param.raw->coefficientsCheby;
}
int ContextData::GetDoubleAngleIts() {
    assert(param.raw);
    return param.raw ? param.raw->doubleAngleIts : 3;
}

int ContextData::GetBootK() {
    assert(param.raw);
    return param.raw ? param.raw->bootK : 1;
}

bool ContextData::HasBootPrecomputation(int slots) {

    return precom.boot.contains(slots);
}
BootstrapPrecomputation& ContextData::GetBootPrecomputation(int slots) {
    if (!precom.boot.contains(slots))
        assert("No precomputation." == nullptr);
    return precom.boot[slots];
}

KeySwitchingKey& ContextData::GetRotationKey(int index, const KeyHash& keyID, int slots) {
    if (index != 2 * N - 1) {  // Handle conjugate key independently
        index = index % (N / 2);
        if (index < 0)
            index += this->N / 2;
        if (index > slots / 2)
            index += N / 2 - slots;

        if (!precom.keys.at(keyID).rot_keys.contains(index)) {
            if (slots != -1 && slots != N / 2) {
                //std::cout << "Looking for alternative key modulo-slot compatible to " << index << std::endl;

                for (int i = 1; i < N / 2 / slots; ++i) {
                    int index_ = (index + i * slots) % (N / 2);
                    if (precom.keys.at(keyID).rot_keys.contains(index_)) {
                        return precom.keys.at(keyID).rot_keys.at(index_);
                    }
                }
            }
            std::cout << "Rotation index " << index << "/ " << index - slots << "not found." << std::endl;
        }
    }

    return precom.keys.at(keyID).rot_keys.at(index);
}
void ContextData::AddRotationKey(int index, KeySwitchingKey&& ksk) {
    //index = index % (cc.N / 2);
    while (index < 0)
        index += this->N / 2;
    if (!precom.keys.contains(ksk.keyID))
        precom.keys[ksk.keyID] = Precomputations::KeyPrecomputations{};
    precom.keys.at(ksk.keyID).rot_keys.emplace(index, std::move(ksk));
}
bool ContextData::HasRotationKey(int index, const KeyHash& keyID) {
    //index = index % (cc.N / 2);
    while (index < 0)
        index += this->N / 2;
    if (!precom.keys.contains(keyID))
        precom.keys[keyID] = Precomputations::KeyPrecomputations{};
    return precom.keys.at(keyID).rot_keys.contains(index);
}

void ContextData::AddEvalKey(KeySwitchingKey&& ksk) {
    if (!precom.keys.contains(ksk.keyID))
        precom.keys[ksk.keyID] = Precomputations::KeyPrecomputations{};
    std::unique_ptr<KeySwitchingKey> key = std::make_unique<KeySwitchingKey>(std::move(ksk));
    std::unique_ptr<KeySwitchingKey>& dest_key = precom.keys.at(key->keyID).eval_key;
    dest_key = std::move(key);
}
KeySwitchingKey& ContextData::GetEvalKey(const KeyHash& keyID) {
    assert(precom.keys.contains(keyID));
    assert(precom.keys[keyID].eval_key);
    return *precom.keys.at(keyID).eval_key;
}

void ContextData::AddBootPrecomputation(int slots, BootstrapPrecomputation&& precomp) {
    {
        std::cout << "Adding bootstrap precomputation to GPU for " << slots << " slots.\n"

                  << "Plaintexts loaded: "
                  << (precomp.CtS.size() == 0 ? (precomp.LT.A.size() + precomp.LT.invA.size())
                                              : (precomp.StC.size() * precomp.StC.at(0).A.size() +
                                                 precomp.CtS.size() * precomp.CtS.at(0).A.size()))
                  << " ~ "
                  << (precomp.CtS.size() == 0
                          ? (precomp.LT.A.size() * (precomp.LT.A.at(0).c0.getLevel() +
                                                    precomp.LT.A.at(0).c0.isModUp() * specialMeta[0].size()) +
                             precomp.LT.invA.size() * (precomp.LT.invA.at(0).c0.getLevel() +
                                                       precomp.LT.invA.at(0).c0.isModUp() * specialMeta[0].size()))
                          : (precomp.StC.size() * precomp.StC.at(0).A.size() *
                                 (1 + precomp.StC.at(0).A.at(0).c0.getLevel() +
                                  precomp.StC.at(0).A.at(0).c0.isModUp() * specialMeta[0].size()) +
                             precomp.CtS.size() * precomp.CtS.at(0).A.size() *
                                 (1 + precomp.CtS.at(0).A.at(0).c0.getLevel() +
                                  precomp.CtS.at(0).A.at(0).c0.isModUp() * specialMeta[0].size()))) *
                         N * 8 / (1 << 20)
                  << "MB\n";
    }

    precom.boot.emplace(slots, std::move(precomp));
}

FIDESlib::CKKS::RESCALE_TECHNIQUE ContextData::translateRescalingTechnique(lbcrypto::ScalingTechnique technique) {
    return technique == lbcrypto::ScalingTechnique::FIXEDAUTO         ? FIDESlib::CKKS::FIXEDAUTO
           : technique == lbcrypto::ScalingTechnique::FIXEDMANUAL     ? FIDESlib::CKKS::FIXEDMANUAL
           : technique == lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT ? FIDESlib::CKKS::FLEXIBLEAUTOEXT
           : technique == lbcrypto::ScalingTechnique::FLEXIBLEAUTO    ? FIDESlib::CKKS::FLEXIBLEAUTO
                                                                      : FIDESlib::CKKS::NO_RESCALE;
}

void ContextData::PrepareNCCLCommunication() {

    if (GPUid.size() > 1) {

        std::set<int> ids;
        for (int i : GPUid)
            ids.insert(i);
        int num_ranks = ids.size();

        bool p2p = true;
        for (auto& i : ids) {
            cudaSetDevice(i);
            for (auto& j : ids) {
                if (i != j) {
                    int canAccessPeer;
                    cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                    if (canAccessPeer)
                        cudaDeviceEnablePeerAccess(j, 0);
                    else
                        p2p = false;

                    cudaDeviceCanAccessPeer(&canAccessPeer, j, i);
                    if (canAccessPeer)
                        cudaDeviceEnablePeerAccess(j, 0);
                    else
                        p2p = false;
                }
            }
        }
        this->canP2P = p2p;
        std::cout << "GPU P2P? " << this->canP2P << std::endl;

        /*
        if (GPUid.size() > 1) {
            if (this->canP2P)
                std::cout << "P2P (Nvlink?) detected" << std::endl;
            else
                std::cout << "NO P2P" << std::endl;
        }*/
#ifdef NCCL
        NCCLCHECK(ncclGetUniqueId(&communicatorID));
        GPUrank.resize(GPUid.size());
        ncclGroupStart();
        for (int i = 0; i < GPUid.size(); i++) {
            cudaSetDevice(GPUid[i]);
            if (precom.dev_to_communicator[GPUid[i]] == nullptr) {
                NCCLCHECK(ncclCommInitRank(GPUrank.data() + i, num_ranks, communicatorID, i));
                precom.dev_to_communicator[GPUid[i]] = GPUrank.data() + i;
            } else {
                GPUrank[i] = *precom.dev_to_communicator[GPUid[i]];
            }
        }
        ncclGroupEnd();
#endif
        cudaDeviceSynchronize();
    }

    top_limb_stream.resize(2 * GPUid.size());
    top_limb_stream2.resize(2 * GPUid.size());
    top_limb_buffer.resize(2 * GPUid.size());
    top_limb_buffer2.resize(2 * GPUid.size());
    top_limb_buffer_handle.resize(2 * GPUid.size());
    top_limb_buffer2_handle.resize(2 * GPUid.size());
    for (size_t i = 0; i < 2 * GPUid.size(); ++i) {
        int g = i % GPUid.size();
        cudaSetDevice(GPUid[g]);

        top_limb_stream[g].init(100);
        top_limb_stream2[g].init(100);

        if (GPUid.size() > 1) {
#ifdef NCCL
            NCCLCHECK(ncclMemAlloc((void**)&top_limb_buffer[i], sizeof(uint64_t) * N));
            NCCLCHECK(
                ncclCommRegister(GPUrank[g], top_limb_buffer[i], sizeof(uint64_t) * N, &top_limb_buffer_handle[i]));
            NCCLCHECK(ncclMemAlloc((void**)&top_limb_buffer2[i], sizeof(uint64_t) * N));
            NCCLCHECK(
                ncclCommRegister(GPUrank[g], top_limb_buffer2[i], sizeof(uint64_t) * N, &top_limb_buffer2_handle[i]));
#else
            cudaMalloc((void**)&top_limb_buffer[i], sizeof(uint64_t) * N);
            cudaMalloc((void**)&top_limb_buffer2[i], sizeof(uint64_t) * N);
#endif
        } else {
            cudaMalloc((void**)&top_limb_buffer[i], sizeof(uint64_t) * N);
            cudaMalloc((void**)&top_limb_buffer2[i], sizeof(uint64_t) * N);
        }
        //cudaDeviceSynchronize();
        top_limbptr.emplace_back(top_limb_stream[g], 1, GPUid[g], (void**)&top_limb_buffer[i]);
        top_limbptr2.emplace_back(top_limb_stream2[g], 1, GPUid[g], (void**)&top_limb_buffer2[i]);
        gatherStream.resize(GPUid.size());

        for (size_t i = 0; i < GPUid.size(); ++i) {
            gatherStream[i].resize(GPUid.size());
            for (size_t j = 0; j < GPUid.size(); ++j) {
                cudaSetDevice(GPUid[j]);
                gatherStream[i][j].init(100);
            }
        }

        /* for (int i = 0; i < dnum; ++i) {
            key_switch_digits.emplace_back(*this, L, true);
            key_switch_digits.back().generateSpecialLimbs();
        }*/

        CudaCheckErrorModNoSync;
    }

    digitStream.resize(dnum);
    digitStreamForMemcpyPeer.resize(dnum);
    for (int i = 0; i < dnum; ++i) {
        digitStream[i].resize(GPUid.size());
        digitStreamForMemcpyPeer[i].resize(GPUid.size());
        for (size_t j = 0; j < GPUid.size(); ++j) {
            cudaSetDevice(GPUid[j]);
            digitStream[i][j].init(100);
            digitStreamForMemcpyPeer[i][j].resize(GPUid.size());
            for (size_t k = 0; k < GPUid.size(); ++k) {
                digitStreamForMemcpyPeer[i][j][k].init(100);
            }
        }
    }

    digitStream2.resize(dnum);
    for (int i = 0; i < dnum; ++i) {
        digitStream2[i].resize(GPUid.size());
        for (size_t j = 0; j < GPUid.size(); ++j) {
            cudaSetDevice(GPUid[j]);
            digitStream2[i][j].init();
        }
    }
}

const std::vector<int> ContextData::generateDigitGPUid(std::vector<std::vector<LimbRecord>>& meta, const int L,
                                                       const int dnum) {
    std::vector<int> res(dnum);
    for (size_t i = 0; i < meta.size(); ++i) {
        for (auto& j : meta[i]) {
            res[j.digit] = i;
        }
    }
    return res;
}

std::vector<std::vector<LimbRecord>> ContextData::generateSplitSpecialMeta(std::vector<LimbRecord>& specialMeta,
                                                                           const std::vector<int> GPUid) {
    std::vector<std::vector<LimbRecord>> res(GPUid.size());

    int init = 0;
    for (int i = 0; i < GPUid.size(); ++i) {
        cudaSetDevice(GPUid[i]);
        int num = (specialMeta.size() - init) / (GPUid.size() - i);
        for (int j = init; j < init + num; ++j) {
            res[i].emplace_back(
                LimbRecord{.id = specialMeta[j].id, .type = specialMeta[j].type, .digit = specialMeta[j].digit});
            res[i].back().stream.init();
        }
        init += num;
    }
    return res;
}

ContextData::~ContextData() {
    CudaCheckErrorMod;
    key_switch_aux.reset(nullptr);
    //   CudaCheckErrorMod;
    key_switch_aux2.reset(nullptr);
    //   CudaCheckErrorMod;
    for (auto& i : moddown_aux) {
        i.reset(nullptr);
    }
    //   CudaCheckErrorMod;
    precom.auxPoly.clear();
    //   CudaCheckErrorMod;
    precom.monomialCache.clear();
    //   CudaCheckErrorMod;
    precom.boot.clear();
    //   CudaCheckErrorMod;
    for (size_t i = 0; i < top_limbptr.size(); ++i) {
        int g = i % GPUid.size();
        cudaSetDevice(GPUid[g]);
        if (GPUid.size() > 1) {
#ifdef NCCL
            if (top_limb_buffer_handle[i])
                NCCLCHECK(ncclCommDeregister(GPUrank[g], top_limb_buffer_handle[i]));
            if (top_limb_buffer2_handle[i])
                NCCLCHECK(ncclCommDeregister(GPUrank[g], top_limb_buffer2_handle[i]));
            NCCLCHECK(ncclMemFree(top_limb_buffer[i]));
            NCCLCHECK(ncclMemFree(top_limb_buffer2[i]));
#else

            cudaFree(top_limb_buffer[i]);
            cudaFree(top_limb_buffer2[i]);
#endif
        } else {
            cudaFree(top_limb_buffer[i]);
            cudaFree(top_limb_buffer2[i]);
        }
        top_limbptr[i].free(top_limb_stream[g]);
        top_limbptr2[i].free(top_limb_stream2[g]);
    }
    top_limbptr.clear();
    top_limbptr2.clear();
    CudaCheckErrorMod;
#ifdef NCCL
    NCCLCHECK(ncclGroupStart());
    for (auto rank : precom.dev_to_communicator) {
        if (rank.second) {
            cudaSetDevice(rank.first);
            NCCLCHECK(ncclCommFinalize(*rank.second));
            // CudaCheckErrorMod;
            NCCLCHECK(ncclCommDestroy(*rank.second));
            // CudaCheckErrorMod;
        }
    }
    NCCLCHECK(ncclGroupEnd());
#endif
    Ciphertext::clearOpRecord();
    CudaCheckErrorMod;
}

bool ContextData::hasAuxilarPoly() const {
    return precom.auxPoly.empty();
}

RNSPoly ContextData::getAuxilarPoly() {

    if (precom.auxPoly.empty()) {
        return RNSPoly(*this);
    } else {
        RNSPoly res(std::move(precom.auxPoly.back()));
        precom.auxPoly.pop_back();
        return res;
    }
}

void ContextData::returnAuxilarPoly(RNSPoly&& c) {
    precom.auxPoly.emplace_back(std::move(c));
}

void ContextData::trimAuxilarPoly(size_t size) {
    while (precom.auxPoly.size() > size)
        precom.auxPoly.pop_back();
    //precom.auxPoly.erase(precom.auxPoly.begin() + std::min(size, precom.auxPoly.size()), precom.auxPoly.end());
}

void ContextData::clearAuxilarPoly() {
    precom.auxPoly.clear();
}

void ContextData::clearAutomorphismKeys(const KeyHash& KeyID) {
    for (auto& i : precom.keys) {
        if (KeyID.empty() || KeyID == i.first) {
            i.second.rot_keys.clear();
        }
    }
}
void ContextData::clearEvalMultKeys(const KeyHash& KeyID) {
    for (auto& i : precom.keys) {
        if (KeyID.empty() || KeyID == i.first) {
            i.second.eval_key.reset();
        }
    }
}
void ContextData::clearBootPrecomputation(const int slots) {
    if (slots == -1)
        precom.boot.clear();
    else {
        if (precom.boot.contains(slots)) {
            precom.boot.erase(slots);
        }
    }
}

void ContextData::clearParamSwitchKeys(const KeyHash& KeyID) {

    if (KeyID.empty())
        map_param_switch.clear();
    else {
        for (auto& i : map_param_switch) {
            if ((i.first.first == this->param || i.first.second == this->param)) {
                if (i.second) {
                    if (i.second->contains(KeyID)) {
                        i.second->erase(KeyID);
                    }
                }
            }
        }
    }
}

Context GenCryptoContextGPU(const Parameters& param, const std::vector<int>& devs) {

    ContextData* data = new ContextData(param, devs);
    if (OK) {
        //Context cc();
        Context cc(data);  //= std::make_shared<ContextData>(param, devs);
        //Context cc;

        map_param_context[data->param] = cc;

        //if (!currentContext)
        SetCurrentContext(cc);
    } else {
        SetCurrentContext(map_param_context[data->param]);
        delete data;
    }
    Context res = GetCurrentContext();
    return res;
}

void DeregisterCryptoContextGPU(const Parameters& param) {
    map_param_context.erase(param);
    if (currentContext && currentContext->param == param) {
        currentContext.reset();
    }
}
void DeregisterCryptoContextGPU(Context cc) {
    DeregisterCryptoContextGPU(cc->param);
}

Context GetCurrentContext() {
    return currentContext;
}

__global__ void dummy() {}

void SetCurrentContext(Context& cc) {
    if (cc != currentContext) {
        CudaNvtxRange r(std::string{sc::current().function_name()}.substr());
        currentContext = cc;
        if (currentContext) {

            parallel_for(0, currentContext->GPUid.size(), 1, [&](int i) {
                //for (size_t i = 0; i < currentContext->GPUid.size(); ++i) {
                cudaSetDevice(currentContext->GPUid[i]);
                //cudaDeviceSynchronize();
                cudaMemcpyToSymbolAsync(FIDESlib::constants, &(currentContext->precom.constants[i]),
                                        sizeof(FIDESlib::Constants), 0, cudaMemcpyHostToDevice, 0);
                //cudaDeviceSynchronize();
            });
        }
    }
}

void AddSecretSwitchingKey(KeySwitchingKey&& ksk_a, KeySwitchingKey&& ksk_b) {

    KeyHash id_a = ksk_a.keyID;
    KeyHash id_b = ksk_b.keyID;
    Parameters& param_a = ksk_a.cc->param;
    Parameters& param_b = ksk_b.cc->param;

    {
        std::pair<std::pair<Parameters, Parameters>, std::unique_ptr<std::map<KeyHash, KeySwitchingKey>>>* entry =
            nullptr;
        for (auto& i : map_param_switch) {
            if (i.first.first == param_a && i.first.second == param_b) {
                entry = &i;
            }
        }

        if (!entry) {
            map_param_switch.emplace_back(std::pair<Parameters, Parameters>{param_a, param_b},
                                          std::make_unique<std::map<KeyHash, KeySwitchingKey>>());
            entry = &map_param_switch.back();
        }
        entry->second->emplace(id_b, std::move(ksk_b));
    }

    {
        std::pair<std::pair<Parameters, Parameters>, std::unique_ptr<std::map<KeyHash, KeySwitchingKey>>>* entry =
            nullptr;
        for (auto& i : map_param_switch) {
            if (i.first.first == param_b && i.first.second == param_a) {
                entry = &i;
            }
        }

        if (!entry) {
            map_param_switch.emplace_back(std::pair<Parameters, Parameters>{param_b, param_a},
                                          std::make_unique<std::map<KeyHash, KeySwitchingKey>>());
            entry = &map_param_switch.back();
        }
        entry->second->emplace(id_a, std::move(ksk_a));
    }

    /*
    if (!map_param_switch[{param_a, param_b}])
        map_param_switch[{param_a, param_b}] = std::make_shared<std::map<KeyHash, KeySwitchingKey>>();
    map_param_switch[{param_a, param_b}]->emplace(id_b, std::move(ksk_b));

    if (!map_param_switch[{param_b, param_a}])
        map_param_switch[{param_b, param_a}] = std::make_shared<std::map<KeyHash, KeySwitchingKey>>();
    map_param_switch[{param_b, param_a}]->emplace(id_a, std::move(ksk_a));
    */
}

bool HasSecretSwitchingKey(const Context& a, const Context& b, const KeyHash& key_b) {
    /*
    if (map_param_switch[{a->param, b->param}] != nullptr) {
        if (map_param_switch[{a->param, b->param}]->contains(key_b)) {
            return true;
        }
    }
    */
    for (auto& i : map_param_switch) {
        if (i.first.first == a->param && i.first.second == b->param) {
            return i.second->contains(key_b);
        }
    }
    return false;
}

KeySwitchingKey& GetSecretSwitchingKey(const Context& a, const Context& b, const KeyHash& key_b) {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());
    /*
    std::pair<Parameters, Parameters> param_key{a->param, b->param};
    if (map_param_switch.contains(param_key)) {
        auto entry = map_param_switch.find(param_key);
        if (entry != map_param_switch.end() && entry->second) {
            auto & key = entry->second->find(key_b);
            if (key != entry->second->end()) {
                return key->second;
            }
        }
    }
    */
    for (auto& i : map_param_switch) {
        if (i.first.first == a->param && i.first.second == b->param) {
            auto key = i.second->find(key_b);
            if (key != i.second->end()) {
                return key->second;
            }
        }
    }

    assert("No key present");
    exit(-1);
}

void DeregisterAllContexts() {
    map_param_switch.clear();
    map_param_context.clear();
    currentContext.reset();
}

}  // namespace FIDESlib::CKKS
