//
// Created by oscar on 21/10/24.
//
#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

#include <benchmark/benchmark.h>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/Context.cuh>
#include <CKKS/Parameters.cuh>
#include <CKKS/Plaintext.cuh>
#include <string>

namespace FIDESlib::Benchmarks {

extern std::vector<FIDESlib::PrimeRecord> p32;
extern std::vector<FIDESlib::PrimeRecord> p64;
extern std::vector<FIDESlib::PrimeRecord> sp64;

#define BATCH_CONFIG {1, 2, 3, 4, 6, 8, 12 /*, 16, 100*/}

#define LEVEL_CONFIG \
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}

struct GeneralBenchParams {
    uint64_t multDepth;
    uint64_t scaleModSize;
    uint64_t batchSize;
    uint64_t ringDim;
    uint64_t dnum;
    std::vector<int> GPUs;
    lbcrypto::ScalingTechnique tech = lbcrypto::FIXEDMANUAL;
};

inline std::string general_bench_params_to_string(GeneralBenchParams& par) {
    // Build a string wit general bench params
    return "multDepth: " + std::to_string(par.multDepth) + ", scaleModSize: " + std::to_string(par.scaleModSize) +
           ", batchSize: " + std::to_string(par.batchSize) + ", ringDim: " + std::to_string(par.ringDim) +
           ", dnum: " + std::to_string(par.dnum);
}

inline std::string fideslib_bench_params_to_string(FIDESlib::CKKS::Parameters& par) {
    return "logN: " + std::to_string(par.logN) + ", L: " + std::to_string(par.L) +
           ", dnum: " + std::to_string(par.dnum);
}

/** Following: https://bu-icsg.github.io/publications/2024/fhe_parallelized_bootstrapping_isca_2024.pdf
 * C. Parameter Set for HEAP
 */
extern GeneralBenchParams gparams64_13;

/** Following this: https://eprint.iacr.org/2024/463.pdf
 * Table 5.7 Col.1
 */
extern GeneralBenchParams gparams64_14;

/** Following this: https://eprint.iacr.org/2024/463.pdf
 * Table 5.7 Col.2
 */
extern GeneralBenchParams gparams64_15;

/** Following this: https://eprint.iacr.org/2024/463.pdf
 * Table 5.8 Col.1
 */
extern GeneralBenchParams gparams64_16;

extern GeneralBenchParams gparams32_15;

extern GeneralBenchParams gparams64_17;

extern std::array<GeneralBenchParams, 28> general_bench_params;
extern std::array<FIDESlib::CKKS::Parameters, 9> fideslib_bench_params;

extern std::map<int, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>> context_map;
extern std::map<int, lbcrypto::KeyPair<lbcrypto::DCRTPoly>> key_map;

class GeneralFixture : public benchmark::Fixture {
   public:
    GeneralBenchParams generalTestParams{};
    FIDESlib::CKKS::Parameters fideslibParams{};

    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = nullptr;
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys;
    lbcrypto::Plaintext pt;
    GeneralFixture() = default;

    static void SetContext() {
        for (int i = 0; i < static_cast<int>(general_bench_params.size()); ++i) {
            benchmark::AddCustomContext("GeneralFixture:" + std::to_string(i),
                                        general_bench_params_to_string(general_bench_params[i]));
        }
    }

    void SetUp(const benchmark::State& state) override {
        fideslibParams = fideslib_bench_params[state.range(1)];
        generalTestParams = general_bench_params[state.range(0)];

        // Caché OpenFHE context.
        if (!context_map.contains(state.range(0))) {
            lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
            parameters.SetMultiplicativeDepth(generalTestParams.multDepth);
            parameters.SetScalingModSize(generalTestParams.scaleModSize);
            parameters.SetBatchSize(generalTestParams.batchSize);
            parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
            parameters.SetRingDim(generalTestParams.ringDim);
            parameters.SetNumLargeDigits(generalTestParams.dnum);
            parameters.SetScalingTechnique(generalTestParams.tech);
            context_map[state.range(0)] = GenCryptoContext(parameters);
            context_map[state.range(0)]->Enable(lbcrypto::PKE);
            context_map[state.range(0)]->Enable(lbcrypto::KEYSWITCH);
            context_map[state.range(0)]->Enable(lbcrypto::LEVELEDSHE);
            context_map[state.range(0)]->Enable(lbcrypto::ADVANCEDSHE);
            context_map[state.range(0)]->Enable(lbcrypto::FHE);
            key_map[state.range(0)] = context_map[state.range(0)]->KeyGen();
            context_map[state.range(0)]->EvalMultKeyGen(key_map[state.range(0)].secretKey);
            context_map[state.range(0)]->EvalRotateKeyGen(key_map[state.range(0)].secretKey, {1, 2, 3, 4, 1});
        }
        keys = key_map[state.range(0)];
        cc = context_map[state.range(0)];
    }
    void SetUp(benchmark::State& state) override { SetUp(const_cast<const benchmark::State&>(state)); }

    void TearDown(benchmark::State& state) override {
        //lbcrypto::CryptoContextImpl<bigintdyn::DCRTPoly>::s_evalAutomorphismKeyMap.clear();
        //lbcrypto::CryptoContextImpl<bigintdyn::DCRTPoly>::s_evalMultKeyMap.clear();
    }
    void TearDown(const benchmark::State& state) override { TearDown(const_cast<const benchmark::State&>(state)); }
};

class FIDESlibFixture : public benchmark::Fixture {
   public:
    FIDESlib::CKKS::Parameters fideslibParams{};

    FIDESlibFixture() {
        //Iterations(3);
    }

    static void SetContext() {
        for (int i = 0; i < static_cast<int>(FIDESlib::Benchmarks::fideslib_bench_params.size()); ++i) {
            benchmark::AddCustomContext(
                "FIDESlibFixture:" + std::to_string(i),
                FIDESlib::Benchmarks::fideslib_bench_params_to_string(FIDESlib::Benchmarks::fideslib_bench_params[i]));
        }
    }

    void SetUp(benchmark::State& state) override { fideslibParams = fideslib_bench_params[state.range(0)]; }
    void SetUp(const benchmark::State& state) override { fideslibParams = fideslib_bench_params[state.range(0)]; }

    void TearDown(benchmark::State& state) override {}
    void TearDown(const benchmark::State&) override {}
};
}  // namespace FIDESlib::Benchmarks
#endif  // BENCHMARK_CUH
