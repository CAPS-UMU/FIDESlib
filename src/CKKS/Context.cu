//
// Created by carlosad on 2/05/24.
//
#include "CKKS/Context.cuh"

#include <source_location>
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/KeySwitchingKey.cuh"

namespace FIDESlib::CKKS {

std::vector<dim3> Context::generateLimbGPUid(const std::vector<std::vector<LimbRecord>>& meta, const int L) {
    std::vector<dim3> res(L + 1, 0);
    for (int i = 0; i < static_cast<int>(meta.size()); ++i) {
        for (size_t j = 0; j < meta.at(i).size(); ++j) {
            res.at(meta[i][j].id) = {static_cast<uint32_t>(i), static_cast<uint32_t>(j)};
        }
    }
    return res;
}

std::vector<std::vector<std::vector<LimbRecord>>> Context::generateDigitMeta(
    const std::vector<std::vector<LimbRecord>>& meta, const std::vector<LimbRecord>& specialMeta,
    const std::vector<std::vector<int>> digitGPUid) {
    std::vector<std::vector<std::vector<LimbRecord>>> digitMeta(meta.size());

    for (size_t i = 0; i < digitGPUid.size(); ++i) {
        for (int d : digitGPUid.at(i)) {
            digitMeta[i].emplace_back();
            for (auto& l : specialMeta) {
                digitMeta[i].back().emplace_back(LimbRecord{.id = l.id, .type = l.type, .digit = l.digit});
                digitMeta[i].back().back().stream.init();
            }

            for (auto& l : meta.at(i)) {
                if (l.digit != d) {
                    digitMeta[i].back().emplace_back(LimbRecord{.id = l.id, .type = l.type, .digit = l.digit});
                    digitMeta[i].back().back().stream.init();
                }
            }
        }
    }
    return digitMeta;
}

std::vector<std::vector<std::vector<LimbRecord>>> Context::generateDecompMeta(
    const std::vector<std::vector<LimbRecord>>& meta, const std::vector<std::vector<int>> digitGPUid) {
    std::vector<std::vector<std::vector<LimbRecord>>> decompMeta(meta.size());

    for (size_t i = 0; i < digitGPUid.size(); ++i) {
        for (int d : digitGPUid.at(i)) {
            decompMeta[i].emplace_back();
            for (auto& l : meta.at(i)) {
                if (l.digit == d) {
                    decompMeta[i].back().push_back(LimbRecord{.id = l.id, .type = l.type, .digit = l.digit});
                    decompMeta[i].back().back().stream.init();
                }
            }
        }
    }

    return decompMeta;
}

bool Context::isValidPrimeId(const int i) const {
    return (i >= 0 && i < L + 1 + K);
}

Context::Context(Parameters param, const std::vector<int>& devs, const int secBits)
    : my_range(loc, LIFETIME),
      param((CudaNvtxStart(std::string{std::source_location::current().function_name()}.substr(18 + strlen(loc))),
             param)),
      logN(param.logN),
      N(1 << logN),
      slots(1 << (logN - 1)),
      rescaleTechnique(translateRescalingTechnique(param.scalingTechnique)),
      L(param.L),
      logQ(computeLogQ(L, param.primes)),
      batch(param.batch),
      GPUid(devs),
      dnum(validateDnum(GPUid, param.dnum)),
      digitGPUid(generateDigitGPUid(dnum, devs)),
      prime(param.primes.begin(), param.primes.begin() + L + 1),
      meta{generateMeta(GPUid, dnum, digitGPUid, prime, param)},
      logQ_d(computeLogQ_d(dnum, meta, prime)),
      K(computeK(logQ_d, param.Sprimes, param)),
      logP(computeLogQ(K - 1, param.Sprimes)),
      specialPrime(param.Sprimes.begin(), param.Sprimes.begin() + K),
      specialMeta(generateSpecialMeta(meta, specialPrime, L + 1)),
      decompMeta(generateDecompMeta(meta, digitGPUid)),
      digitMeta(generateDigitMeta(meta, specialMeta, digitGPUid)),
      limbGPUid(generateLimbGPUid(meta, L)),
      GPUrank(GPUid.size(), -1) {

    SetupConstants<Parameters>(prime, meta, specialPrime, specialMeta, decompMeta, digitMeta, digitGPUid, GPUid, N,
                               param);
    // PrepareNCCLCommunication();
    // CheckBitSecurity();
    for (int dev : GPUid) {
        cudaMemPool_t mp;
        cudaDeviceGetDefaultMemPool(&mp, dev);
        uint64_t threshold = UINT64_MAX;  //5l * 1024l * 1024l * 1024l;  // One Gigabyte of memory
        cudaMemPoolSetAttribute(mp, cudaMemPoolAttrReleaseThreshold, &threshold);
        CudaCheckErrorModNoSync;
    }

    CudaNvtxStop();
}

int Context::computeLogQ(const int L, std::vector<PrimeRecord>& primes) {
    int res = 0;
    assert(L <= (int)primes.size());
    for (int i = 0; i <= L; ++i) {
        res += (primes[i].bits == -1) ? (primes[i].bits = (int)std::bit_width(primes[i].p)) : primes[i].bits;
    }
    return res;
}

int Context::validateDnum(const std::vector<int>& GPUid, const int dnum) {
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

std::vector<std::vector<LimbRecord>> Context::generateMeta(const std::vector<int>& GPUid, const int dnum,
                                                           const std::vector<std::vector<int>> digitGPUid,
                                                           const std::vector<PrimeRecord>& prime,
                                                           const Parameters& param) {
    int devs = GPUid.size();
    std::vector<std::vector<LimbRecord>> meta(devs);

    //for (int i = 0; i < devs; ++i) {
    // cudaSetDevice(GPUid.at(i));
    // meta.at(i).resize((prime.size() + devs - i - 1) / devs);
    //}

    for (int i = 0; i < (int)prime.size(); ++i) {
        int digit = param.raw == nullptr ? i % dnum : findDigitOnParam(param, prime.at(i).p);

        int dev = -1;
        for (size_t j = 0; j < digitGPUid.size(); ++j) {
            for (auto& k : digitGPUid.at(j))
                if (k == digit)
                    dev = j;
        }

        meta[dev].push_back(LimbRecord{
            .id = i, .type = (prime[i].type ? *(prime[i].type) : (prime[i].bits <= 30 ? U32 : U64)), .digit = digit});
        meta[dev].back().stream.init();
    }

    return meta;
}

std::vector<int> Context::computeLogQ_d(const int dnum, const std::vector<std::vector<LimbRecord>>& meta,
                                        const std::vector<PrimeRecord>& prime) {
    std::vector<int> logQ_d(dnum, 0);

    for (auto& i : meta)
        for (auto& j : i)
            logQ_d.at(j.digit) += prime.at(j.id).bits;

    return logQ_d;
}

int Context::computeK(const std::vector<int>& logQ_d, std::vector<PrimeRecord>& Sprimes, const Parameters& param) {

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
    return res;
}

std::vector<LimbRecord> Context::generateSpecialMeta(const std::vector<std::vector<LimbRecord>>& meta,
                                                     const std::vector<PrimeRecord>& specialPrime, const int ID0) {
    std::vector<LimbRecord> specialMeta(specialPrime.size());

    for (int i = 0; i < (int)specialPrime.size(); ++i) {
        specialMeta.at(i).id = ID0 + i;
        specialMeta.at(i).type =
            (specialPrime[i].type ? *(specialPrime[i].type) : (specialPrime[i].bits <= 30 ? U32 : U64));
        specialMeta.at(i).stream.init();
    }

    return specialMeta;
}

std::vector<std::vector<int>> Context::generateDigitGPUid(const int dnum, const std::vector<int>& devs) {
    std::vector<std::vector<int>> res(devs.size());
    for (int d = 0; d < dnum; ++d) {
        res[d % res.size()].push_back(d);
    }
    return res;
}
RNSPoly& Context::getKeySwitchAux() {
    if (key_switch_aux == nullptr)
        key_switch_aux = std::make_unique<RNSPoly>(*this, L, true);
    key_switch_aux->generateDecompAndDigit();
    return *key_switch_aux;
}

RNSPoly& Context::getKeySwitchAux2() {
    if (key_switch_aux2 == nullptr)
        key_switch_aux2 = std::make_unique<RNSPoly>(*this, L, true);
    key_switch_aux2->generateDecompAndDigit();
    return *key_switch_aux2;
}

RNSPoly& Context::getModdownAux() {
    if (moddown_aux == nullptr)
        moddown_aux = std::make_unique<RNSPoly>(*this, L, true);
    return *moddown_aux;
}
std::vector<uint64_t> Context::ElemForEvalMult(int level, const double operand) {

    uint32_t numTowers = level + 1;
    std::vector<lbcrypto::DCRTPoly::Integer> moduli(numTowers);
    for (usint i = 0; i < numTowers; i++) {
        moduli[i] = prime[i].p;
    }

    double scFactor = param.ScalingFactorReal[level];

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

std::vector<uint64_t> Context::ElemForEvalAddOrSub(const int level, const double operand, const int noise_deg) {
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
std::vector<double>& Context::GetCoeffsChebyshev() {
    assert(param.raw != nullptr);
    return param.raw->coefficientsCheby;
}
int Context::GetDoubleAngleIts() {
    assert(param.raw != nullptr);
    return param.raw ? param.raw->doubleAngleIts : 3;
}

int Context::GetBootK() {
    assert(param.raw != nullptr);
    return param.raw ? param.raw->bootK : 1.0;
}

std::map<int, BootstrapPrecomputation> boot_precomps;

BootstrapPrecomputation& Context::GetBootPrecomputation(int slots) {
    if (!boot_precomps.contains(slots))
        assert("No precomputation." == nullptr);
    return boot_precomps[slots];
}

std::map<int, KeySwitchingKey> rot_keys;

KeySwitchingKey& Context::GetRotationKey(int index) {
    //index = index % (cc.N / 2);
    if (index < 0)
        index += this->N / 2;
    return rot_keys.at(index);
}
void Context::AddRotationKey(int index, KeySwitchingKey&& ksk) {
    //index = index % (cc.N / 2);
    if (index < 0)
        index += this->N / 2;
    rot_keys.emplace(index, std::move(ksk));
}
bool Context::HasRotationKey(int index) {
    //index = index % (cc.N / 2);
    if (index < 0)
        index += this->N / 2;
    return rot_keys.contains(index);
}

std::optional<KeySwitchingKey> eval_key;

void Context::AddEvalKey(KeySwitchingKey&& ksk) {
    eval_key.emplace(std::move(ksk));
}
KeySwitchingKey& Context::GetEvalKey() {
    return eval_key.value();
}
Context::~Context() {
    eval_key.reset();
    rot_keys.clear();
    boot_precomps.clear();
}

void Context::AddBootPrecomputation(int slots, BootstrapPrecomputation&& precomp) const {
    {
        std::cout << "Adding bootstrap precomputation to GPU for " << slots << " slots.\n"
                  << "Rotation keys loaded: " << rot_keys.size() << " ~ "
                  << 2 * ((long long)rot_keys.size() * dnum * (L + K + 1) * N * 8 / (1 << 20)) << "MB\n"
                  << "Plaintexts loaded: "
                  << (precomp.CtS.size() == 0 ? (precomp.LT.A.size() + precomp.LT.invA.size())
                                              : (precomp.StC.size() * precomp.StC.at(0).A.size() +
                                                 precomp.CtS.size() * precomp.CtS.at(0).A.size()))
                  << " ~ "
                  << (precomp.CtS.size() == 0
                          ? (precomp.LT.A.size() * precomp.LT.A.at(0).c0.getLevel() +
                             precomp.LT.invA.size() * precomp.LT.invA.at(0).c0.getLevel())
                          : (precomp.StC.size() * precomp.StC.at(0).A.size() * precomp.StC.at(0).A.at(0).c0.getLevel() +
                             precomp.CtS.size() * precomp.CtS.at(0).A.size() *
                                 precomp.CtS.at(0).A.at(0).c0.getLevel())) *
                         N * 8 / (1 << 20)
                  << "MB\n";
    }

    boot_precomps.emplace(slots, std::move(precomp));
}

Context::RESCALE_TECHNIQUE Context::translateRescalingTechnique(lbcrypto::ScalingTechnique technique) {
    return technique == lbcrypto::ScalingTechnique::FIXEDAUTO         ? Context::FIXEDAUTO
           : technique == lbcrypto::ScalingTechnique::FIXEDMANUAL     ? Context::FIXEDMANUAL
           : technique == lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT ? Context::FLEXIBLEAUTOEXT
           : technique == lbcrypto::ScalingTechnique::FLEXIBLEAUTO    ? Context::FLEXIBLEAUTO
                                                                      : Context::NO_RESCALE;
}

}  // namespace FIDESlib::CKKS
