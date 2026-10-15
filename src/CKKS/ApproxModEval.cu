//
// Created by carlosad on 12/11/24.
//

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CudaUtils.cuh"
#include "ElemenwiseBatchKernels.cuh"
#include <complex>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

constexpr bool PRINT = false;

using namespace FIDESlib::CKKS;

void evalChebyshevSeries(Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, std::vector<double>& coefficients, double lower_bound, double upper_bound);
void applyDoubleAngleIterations(Ciphertext& ctxt, int its, const KeySwitchingKey& kskEval);

void FIDESlib::CKKS::approxModReduction(Ciphertext& ctxtEnc, Ciphertext& ctxtEncI, const KeySwitchingKey& keySwitchingKey, uint64_t post) {
    CudaNvtxRange r(std::string{ sc::current().function_name() });

    // cudaDeviceSynchronize();
    if constexpr (PRINT)
        std::cout << "Approx mod red start " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;

    bool constexpr COMPLEX = true;
    ContextData& cc = ctxtEnc.cc;

    if constexpr (COMPLEX)
        evalChebyshevSeries(ctxtEncI, cc.GetCoeffsChebyshev(), -1.0, 1.0);
    evalChebyshevSeries(ctxtEnc, cc.GetCoeffsChebyshev(), -1.0, 1.0);
    if constexpr (PRINT) {
        std::cout << "ctxtEnc res " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
        for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
            cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;

        std::cout << "ctxtEncI res " << ctxtEncI.getLevel() << " " << ctxtEncI.NoiseLevel << std::endl;
        for (auto& i : ctxtEncI.c0.GPU.at(0).limb) {
            cudaSetDevice(ctxtEncI.c0.GPU.at(0).device);
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }

    applyDoubleAngleIterations(ctxtEnc, cc.GetDoubleAngleIts(), keySwitchingKey);
    if constexpr (COMPLEX)
        applyDoubleAngleIterations(ctxtEncI, cc.GetDoubleAngleIts(), keySwitchingKey);
    if constexpr (PRINT) {
        std::cout << "ctxtEnc DA res " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
        for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
            cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;

        std::cout << "ctxtEncI DA res " << ctxtEncI.getLevel() << " " << ctxtEncI.NoiseLevel << std::endl;
        for (auto& i : ctxtEncI.c0.GPU.at(0).limb) {
            cudaSetDevice(ctxtEncI.c0.GPU.at(0).device);
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    // cudaDeviceSynchronize();
    if constexpr (COMPLEX)
        ctxtEncI.multMonomial(cc.N / 2);
    // cudaDeviceSynchronize();
    if constexpr (COMPLEX)
        ctxtEnc.add(ctxtEncI);
    if constexpr (!COMPLEX)
        ctxtEnc.add(ctxtEnc);
    // cudaDeviceSynchronize();
    multIntScalar(ctxtEnc, post);
    // cudaDeviceSynchronize();
    if constexpr (PRINT) {
        std::cout << "ctxtEnc final res " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
        for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
            cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
}

void FIDESlib::CKKS::approxModReductionSparse(Ciphertext& ctxtEnc, uint64_t post) {
    CudaNvtxRange r(std::string{ sc::current().function_name() });
    ContextData& cc = ctxtEnc.cc;

    KeySwitchingKey& keySwitchingKey = cc.GetEvalKey(ctxtEnc.keyID);

    evalChebyshevSeries(ctxtEnc, cc.GetCoeffsChebyshev(), (double)-1.0, (double)1.0);

    if constexpr (PRINT) {
        std::cout << "ctxtEnc res " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
        for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
            cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    applyDoubleAngleIterations(ctxtEnc, cc.GetDoubleAngleIts(), keySwitchingKey);
    if constexpr (PRINT) {
        std::cout << "ctxtEnc DA " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
        for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
            cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    multIntScalar(ctxtEnc, post);
    if constexpr (PRINT) {
        std::cout << "ctxtEnc final " << ctxtEnc.getLevel() << " " << ctxtEnc.NoiseLevel << std::endl;
        for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
            cudaSetDevice(ctxtEnc.c0.GPU.at(0).device);
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    // cudaDeviceSynchronize();
}

void FIDESlib::CKKS::multIntScalar(Ciphertext& ctxt, uint64_t op) {
    CudaNvtxRange r(std::string{ sc::current().function_name() });
    std::vector<uint64_t> op_(ctxt.getLevel() + 1, op);
    ctxt.c0.multScalar(op_);
    ctxt.c1.multScalar(op_);
}

namespace FIDESlib::CKKS {

enum PS_MODE { FULL, RECORD, REMAINING };

// Templated record for the RECORD/REMAINING two-phase evaluation — shared by the Chebyshev
// and the fused power-basis PS paths. Per terminal (non-recursive) WSum node: the WithBias
// weight vector ([0] = bias, [1..] = power/Chebyshev terms) and its target level, plus the
// reversed stack of precomputed raw (un-rescaled) WSum results the REMAINING pass consumes.
template <typename T>
struct PSRecordFused {
    std::vector<std::vector<T>> constants;
    std::vector<int> target_level;
    std::vector<Ciphertext> rev_result_ciphertexts;
};

// FIDESlib bit-compat: direct transcription of OpenFHE InnerEvalChebyshevPS, templated on
// the Chebyshev coefficient scalar type (double / std::complex<double>). Both
// instantiations run the FULL/RECORD/REMAINING modes: the terminal WSum nodes are recorded
// (pure CPU splits + target levels) and replayed from a reversed result stack, exactly as
// the fused power-basis PS does (PSRecordFused). Both the double and complex paths evaluate the
// recorded nodes with the batched evalMultiplePartialLinearWSumWithBias kernels (the complex
// overload builds the per-term BuildConstant2 factors and runs the position-varying
// eval_batched_linear_w_sum_complex_with_bias_ kernel).
template <typename Scalar>
void innerEvalChebyshevPS(const Ciphertext& x,
    Ciphertext& out,
    const std::vector<Scalar>& coefficients,
    const uint32_t k,
    uint32_t m,
    const std::vector<Ciphertext*>& T,
    const std::vector<Ciphertext*>& T2,
    int level_offset = 0,
    PS_MODE mode = FULL,
    PSRecordFused<Scalar>* psRecord = nullptr, const std::vector<Ciphertext*>&T2_norescale = {}) {
    FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
    const FIDESlib::CKKS::Context& cc_ = x.cc_;

    // Compute k*2^{m-1}-k because we use it a lot
    uint32_t k2m2k = k * (1 << (m - 1)) - k;

    // Divide coefficients by T^{k*2^{m-1}}
    std::vector<Scalar> Tkm(k2m2k + k + 1);
    Tkm.back() = 1;
    auto divqr = lbcrypto::LongDivisionChebyshev(coefficients, Tkm);

    // Subtract x^{k(2^{m-1} - 1)} from r
    auto& r2 = divqr->r;
    if (uint32_t n = lbcrypto::Degree(r2); static_cast<int32_t>(k2m2k - n) <= 0) {
        r2.resize(n + 1);
        r2[k2m2k] -= 1;
    } else {
        r2.resize(k2m2k + 1);
        r2.back() = -1;
    }

    auto divcs = lbcrypto::LongDivisionChebyshev(r2, divqr->q);

    Ciphertext su_(cc_), qu_(cc_);
    std::unique_ptr<Ciphertext> su_tmp, qu_tmp;
    Ciphertext *su = nullptr, *qu = nullptr;
    Ciphertext& cu = out;

    // Evaluate q and s2 at u.
    // If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
    if (lbcrypto::Degree(divqr->q) > k) {
        innerEvalChebyshevPS(x, qu_, divqr->q, k, m - 1, T, T2, level_offset, mode, psRecord, T2_norescale);
        if (mode != RECORD)
            qu_.rescale();
        qu = &qu_;
    } else {
        if (mode == RECORD) {
            auto copy = divqr->q;
            copy[0] /= 2.0;
            psRecord->constants.emplace_back(std::move(copy));
            psRecord->target_level.emplace_back((int)T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1) - level_offset);
        } else if (mode == REMAINING) {
            qu_tmp = std::make_unique<Ciphertext>(std::move(psRecord->rev_result_ciphertexts.back()));
            psRecord->rev_result_ciphertexts.pop_back();
            qu_tmp->rescale();
            qu = qu_tmp.get();
        } else {
            qu_.evalPartialLinearWSumWithBias(T, divqr->q, divqr->q.front() / 2.0, divqr->q.size() - 1, (int)T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1) - level_offset);
            qu = &qu_;
        }
    }

    // Add x^{k(2^{m-1} - 1)} to s
    auto& s2 = divcs->r;
    s2.resize(k2m2k + 1);
    s2.back() = 1;

    if (lbcrypto::Degree(s2) > k) {
        innerEvalChebyshevPS(x, su_, s2, k, m - 1, T, T2, level_offset + 1, mode, psRecord, T2_norescale);
        su = &su_;
    } else {
        if (mode == RECORD) {
            auto copy = s2;
            copy[0] /= 2.0;
            psRecord->constants.emplace_back(std::move(copy));
            psRecord->target_level.emplace_back((int)T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1) - level_offset - 1);
        } else if (mode == REMAINING) {
            su_tmp = std::make_unique<Ciphertext>(std::move(psRecord->rev_result_ciphertexts.back()));
            psRecord->rev_result_ciphertexts.pop_back();
            su = su_tmp.get();
        } else {
            su_.evalPartialLinearWSumWithBias(
                T, s2, s2.front() / 2.0, (uint32_t)s2.size() - 1, (int)T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1) - level_offset - 1, false);
            su = &su_;
        }
    }

    if (uint32_t n = lbcrypto::Degree(divcs->q); n >= 1) {
        if (mode == RECORD) {
            auto copy = divcs->q;
            copy[0] /= 2.0;
            psRecord->constants.emplace_back(std::move(copy));
            psRecord->target_level.emplace_back((int)T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1) - level_offset);
        } else if (mode == REMAINING) {
            if (psRecord->rev_result_ciphertexts.back().getLevel() == T2_norescale[m-1]->getLevel()) {
                cu.addMutable(psRecord->rev_result_ciphertexts.back(), (const Ciphertext&) *T2_norescale[m-1]);
                cu.rescale();
            } else {
                psRecord->rev_result_ciphertexts.back().rescale();
                cu.addMutable(psRecord->rev_result_ciphertexts.back(), (const Ciphertext&) *T2[m - 1]);
            }
            psRecord->rev_result_ciphertexts.pop_back();
        } else {
            cu.evalPartialLinearWSumWithBias(
                T, divcs->q, divcs->q.front() / 2.0, divcs->q.size() - 1, (int)T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1) - level_offset, false);
            if (cu.getLevel() == T2_norescale[m-1]->getLevel()) {
                cu.add(*T2_norescale[m -1]);
                cu.rescale();
            } else {
                cu.rescale();
                cu.add(*T2[m - 1]);
            }
        }
    } else {
        if (mode != RECORD)
            cu.addScalar(*T2[m - 1], divcs->q.front() / 2.0);
    }

    if (mode != RECORD) {
        cu.multMutable(*qu);
        cu.addMutable(*su);
    }
}

constexpr bool SEQUENTIAL_PS = false;

//------------------------------------------------------------------------------
// EVAL CHEBYSHEV SERIES — degree < 5 linear path. Bit-exact transcription of OpenFHE's
// internalEvalChebyPolysLinear + internalEvalChebyshevSeriesLinearWithPrecomp
// (ckksrns-advancedshe.cpp L639-714), selected by EvalChebyshevSeries's `Degree < 5`
// dispatch (L951-965). It differs from the PS prelude/recurrence in the operation order:
// the reference reduces *before* the additive term (range: ModReduce before EvalAdd;
// even recurrence: ModReduce before EvalAdd -1), the basis is level-reduced only under
// FIXEDMANUAL (LevelReduceInPlace is a no-op otherwise), and the series uses the c0/2
// constant convention.
//------------------------------------------------------------------------------
template <typename Scalar>
static std::vector<Ciphertext> evalChebyPolysLinear(Ciphertext& x, const std::vector<Scalar>& coefficients,
                                                    double a, double b) {
    const uint32_t k = static_cast<uint32_t>(coefficients.size()) - 1; // T[0..k-1] = T_1..T_k
    auto cc = x.cc_;
    std::vector<Ciphertext> T;
    T.reserve(k);
    T.emplace_back(cc);

    // computes linear transformation y = -1 + 2 (x-a)/(b-a)
    // consumes one level when a <> -1 && b <> 1
    if (!lbcrypto::IsNotEqualNegOne(a) && !lbcrypto::IsNotEqualOne(b)) {
        T[0].copy(x);
    } else {
        const double alpha = 2 / (b - a);
        const double beta = a * alpha;
        T[0].multScalar(x, alpha); // EvalMult(x, alpha)
        T[0].rescale();            // ModReduceInPlace (FIXEDMANUAL-gated)
        if (lbcrypto::IsNotEqualZero(-1.0 - beta))
            T[0].addScalar(-1.0 - beta); // EvalAddInPlace(T[0], -1 - beta)
    }

    // Computes Chebyshev polynomials up to degree k: T_1(y), T_2(y), ... , T_k(y)
    for (uint32_t i = 2; i <= k; ++i) {
        T.emplace_back(cc);
        if (i & 0x1) {
            // T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
            T[i - 1].mult(T[i / 2 - 1], T[i / 2]);
            T[i - 1].add(T[i - 1]); // EvalAddInPlaceNoCheck(T[i-1], T[i-1])
            T[i - 1].rescale();     // ModReduceInPlace
            T[i - 1].sub(T[0]);
        } else {
            // T_{2i}(y) = 2*T_i(y)^2 - 1
            T[i - 1].square(T[i / 2 - 1]);
            T[i - 1].add(T[i - 1]);
            T[i - 1].rescale(); // ModReduceInPlace
            T[i - 1].addScalar(-1.0);
        }
    }

    // LevelReduceInPlace(T[i-1], (T[k-1].level - T[i-1].level)/compositeDegree): a no-op
    // outside FIXEDMANUAL (compositeDegree == 1 for every non-composite RNS parameter set).
    if (x.cc.rescaleTechnique == FIXEDMANUAL) {
        const int target = T[k - 1].getLevel();
        for (uint32_t i = 1; i < k; ++i)
            if (T[i - 1].getLevel() > target)
                T[i - 1].dropToLevel(target, /*skip_adjust=*/true);
    }
    return T;
}

template <typename Scalar>
static void evalChebyshevSeriesLinear(Ciphertext& ctxt, const std::vector<Scalar>& coefficients, double a, double b) {
    std::vector<Ciphertext> T = evalChebyPolysLinear(ctxt, coefficients, a, b);
    const uint32_t k = static_cast<uint32_t>(coefficients.size()) - 2; // highest T index
    auto cc = ctxt.cc_;

    // perform scalar multiplication for the highest-order term
    Ciphertext result(cc);
    result.multScalar(T[k], coefficients[k + 1]); // EvalMult(T[k], c[k+1])

    // perform scalar multiplication for all other nonzero terms and sum them up
    for (uint32_t i = 0; i < k; ++i) {
        if (lbcrypto::IsNotEqualZero(coefficients[i + 1])) {
            Ciphertext term(cc);
            term.multScalar(T[i], coefficients[i + 1]); // EvalMult(T[i], c[i+1])
            result.add(term);                           // EvalAddInPlace(result, term)
        }
    }

    result.rescale(); // ModReduceInPlace(result)

    // adds the free term (at T_0): the reference uses the c0/2 Chebyshev convention
    result.addScalar(coefficients[0] / 2.0); // EvalAddInPlace(result, c[0]/2)
    ctxt.copy(result);
}

// Templated Chebyshev series (double: the real coeff_cos tables / standard bootstrap;
// std::complex<double>: the complex coeff_exp tables). Both scalar types share the
// FULL/RECORD/REMAINING batching and the Chebyshev-power construction (T/T2/T2km1); only the
// coefficient-driven weighted sums differ (batched kernel specialization on Scalar).
template <typename Scalar>
void evalChebyshevSeries(Ciphertext& ctxt, std::vector<Scalar>& coefficients, double lower_bound, double upper_bound) {
    FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });

    auto degree = lbcrypto::Degree(coefficients);

    // EvalChebyshevSeries dispatch (ckksrns-advancedshe.cpp L951-965): degree < 5 selects the
    // linear basis + series evaluator instead of Paterson-Stockmeyer.
    if (degree < 5) {
        evalChebyshevSeriesLinear(ctxt, coefficients, lower_bound, upper_bound);
        return;
    }
    auto degs = lbcrypto::ComputeDegreesPS(degree);
    uint32_t k = degs[0];
    uint32_t m = degs[1];

    // Both scalar types now ride the same RECORD/REMAINING pipeline (one templated
    // PSRecordFused<Scalar>, batched evalMultiplePartialLinearWSumWithBias for double AND
    // std::complex<double>), so the old complex-only `k*2^{m-1} >= 128` guard is gone. The
    // OpenFHE `k*2^{m-1} < 128` test (ckksrns-advancedshe.cpp) only chooses whether to wrap
    // InnerEvalChebyshevPS in an OMP parallel region; the inner function is identical, so the
    // GPU's fixed serial order transcribes both branches.

    // computes linear transformation y = -1 + 2 (x-a)/(b-a)
    // consumes one level when a <> -1 && b <> 1
    Ciphertext& x = ctxt;
    // The evalPartialLinearWSumWithBias machinery (and the T recurrence that
    // feeds it) requires every input at NoiseLevel == 1: OpenFHE's own
    // EvalChebyshevSeries states the same contract ("Ensure all T[i] ciphertexts
    // are kept at noiseScaleDegree == 1 so that evalPartialLinearWSumWithBias
    // works correctly"). Normalize here so a caller that hands over a
    // still-unrescaled ciphertext (NoiseLevel 2 — e.g. a freshly loaded one)
    // does not leave T[0] at NoiseLevel 2 and trip the WSum bookkeeping.

    double& a = lower_bound;
    double& b = upper_bound;
    auto cc = ctxt.cc_;
    std::vector<Ciphertext> T, T_norescale;
    T.emplace_back(cc);
    if (cc->rescaleTechnique == FIXEDMANUAL)
        T_norescale.emplace_back(cc);
    T[0].copy(x);
    if (lbcrypto::IsNotEqualNegOne(a) || lbcrypto::IsNotEqualOne(b)) {
        // linear transformation is needed
        double alpha = 2 / (b - a);
        double beta = a * alpha;

        if (lbcrypto::IsNotEqualOne(alpha)) {
            if (x.NoiseLevel == 2) {
                x.rescaleInternal();
            }
            T[0].multScalar(x, alpha);
        }
        if (lbcrypto::IsNotEqualZero(-1.0 - beta))
            T[0].addScalar(-1.0 - beta);

        if (cc->rescaleTechnique == FIXEDMANUAL)
            T_norescale[0].copy(T[0]);
        if (T[0].NoiseLevel == 2) {
            T[0].rescaleInternal();
        }

    } else {
        if (cc->rescaleTechnique == FIXEDMANUAL)
            T_norescale[0].copy(T[0]);

        if (T[0].NoiseLevel == 2) {
            T[0].rescaleInternal();
        }
    }
    if (cc->rescaleTechnique == FIXEDMANUAL) {
        if (T_norescale[0].NoiseLevel == 1) {
            T_norescale[0].multScalar(1.0);
        }
    }

    assert(T[0].NoiseLevel == 1);
    if (cc->rescaleTechnique == FIXEDMANUAL) assert(T_norescale[0].NoiseLevel == 2);

    // Computes Chebyshev polynomials up to degree k
    // for y: T_1(y) = y, T_2(y), ... , T_k(y)
    // uses binary tree multiplication
    for (uint32_t i = 2; i <= k; ++i) {
        T.emplace_back(cc);
        if (i & 0x1) {
            // if i is odd
            // compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
            T[i - 1].mult(T[i / 2 - 1], T[i / 2]);
            T[i - 1].add(T[i - 1]);
            if (cc->rescaleTechnique == FIXEDMANUAL) {
                T[i - 1].sub(T_norescale[0]);
            } else {
                T[i - 1].sub(T[0]); // T[0] multiplied by scalar to match scale degree adn foctor at once. T_3 sits 2 levels lower than T_1, so this is the only case
            }
            T[i - 1].rescaleInternal();
        } else {
            // compute T_{2i}(y) = 2*T_i(y)^2 - 1
            T[i - 1].square(T[i / 2 - 1]);

            T[i - 1].add(T[i - 1]);
            T[i - 1].addScalar(-1.0);
            T[i - 1].rescaleInternal();
        }
    }
    std::vector<Ciphertext> T2, T2_norescale;
    // T2[0] is used as a placeholder
    T2.emplace_back(cc);
    T2_norescale.emplace_back(cc);

    // computes T_{k(2*m - 1)}(y)
    Ciphertext T2km1(cc);
    T2km1.copy(T.back());

    for (uint32_t i = 1; i < m; ++i) {
        T2.emplace_back(cc);
        T2_norescale.emplace_back(cc);
        //  Compute the Chebyshev polynomials T_k(y), T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
        T2[i].square(i - 1 == 0 ? T.back() : T2[i - 1]);
        T2[i].add(T2[i]);
        T2[i].addScalar(-1.0);
        T2_norescale[i].copy(T2[i]);
        T2[i].rescale();
        if (T2[i].NoiseLevel == 2)
            T2[i].rescaleInternal();
        // compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
        T2km1.mult(T2[i]);
        T2km1.add(T2km1);
        T2km1.rescale();
        T2km1.sub(T.back());
    }

    std::vector<Ciphertext*> T_ptr, T2_ptr, T2_ptr_norescale;
    for (auto& i : T)
        T_ptr.push_back(&i);
    for (auto& i : T2)
        T2_ptr.push_back(&i);
    for (auto& i : T2_norescale)
        T2_ptr_norescale.push_back(&i);

    // Compute k*2^{m-1}-k because we use it a lot
    uint32_t k2m2k = k * (1 << (m - 1)) - k;

    // Add T^{k(2^m - 1)}(y) to the polynomial that has to be evaluated
    auto f2 = coefficients;
    f2.resize(lbcrypto::Degree(f2) + 1);
    f2.resize(2 * k2m2k + k + 1);
    f2.back() = 1;

    if (SEQUENTIAL_PS) {
        PSRecordFused<Scalar>* noRecord = nullptr;
        innerEvalChebyshevPS(T[0], ctxt, f2, k, m, T_ptr, T2_ptr, 0, FULL, noRecord, T2_ptr_norescale);
    } else {
        // RECORD/REMAINING for both scalar types: record all terminal WSum nodes (the
        // splits are pure CPU work), evaluate them all in ONE batched
        // evalMultiplePartialLinearWSumWithBias pass (the complex overload builds the per-term
        // BuildConstant2 factors), then replay the composition tree from the reversed stack.
        PSRecordFused<Scalar> record;
        innerEvalChebyshevPS(T[0], ctxt, f2, k, m, T_ptr, T2_ptr, 0, RECORD, &record, T2_ptr_norescale);
        std::vector<Ciphertext> ws;
        ws.reserve(record.constants.size());
        ws = Ciphertext::evalMultiplePartialLinearWSumWithBias(T_ptr, record.constants, record.target_level,
                                                               /*rescale=*/false);
        record.rev_result_ciphertexts.reserve(ws.size());
        for (size_t i = 0; i < ws.size(); ++i)
            record.rev_result_ciphertexts.emplace_back(std::move(ws[ws.size() - 1 - i]));
        innerEvalChebyshevPS(T[0], ctxt, f2, k, m, T_ptr, T2_ptr, 0, REMAINING, &record, T2_ptr_norescale);
    }
    ctxt.rescale();
    ctxt.subMutable(T2km1);
}

//------------------------------------------------------------------------------
// EVAL POWERS — bit-exact GPU transcription of OpenFHE's internalEvalPowersLinear /
// internalEvalPowersPS + the EvalPowers dispatch (ckksrns-advancedshe.cpp L399-566).
//------------------------------------------------------------------------------

// 1-based index of the most significant bit (OpenFHE GetMSB, nbtheory.h); the linear
// power-index decomposition below relies on it.
static inline uint32_t msbOf(uint64_t v) {
    return 64 - __builtin_clzll(v);
}

// OpenFHE internalEvalPowersLinear (L399-473): builds only the powers of x that the
// coefficient sparsity map (`indices`) demands — squares for power-of-2 exponents, one mult
// per composite exponent. Returns `powers` only; the PS extension set is unused (left empty).
// NOTE: like the reference, the linear path leaves the unneeded powers[i-1] entries empty.
// `fused`: every power is rescaleInternal'd (NoiseLevel 1 under every technique) and the
// final alignment is skipped — the fused evalPartialLinearWSumWithBias folds the level drop
// into its target_level constant multiply (evalPowers(..., fused = true) lives inside the
// SeriesPowers the fused evalPolyWithPrecomp consumes).
template <typename T>
SeriesPowers evalPowersLinear(Ciphertext& x, const std::vector<T>& coefficients, bool fused = false) {
    FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
    const uint32_t k = static_cast<uint32_t>(coefficients.size()) - 1;
    std::vector<bool> indices(k, false);

    // find indices for powers of x that need to be computed
    for (uint32_t i = k; i > 0; --i) {
        if (0 == (i & (i - 1))) {
            // if i is a power of 2
            indices[i - 1] = true;
        } else {
            // non-power of 2
            if (lbcrypto::IsNotEqualZero(coefficients[i])) {
                uint32_t rem = i;

                // while rem is not a power of 2, mark the indices required to compute rem
                while (0 != (rem & (rem - 1))) {
                    indices[rem - 1] = true;
                    rem &= (uint64_t(1) << (msbOf(rem) - 1)) - 1;
                }
            }
        }
    }

    // GPU pre-condition: composite degree == 1 (prime-per-level chains; see Bootstrap.cu);
    // OpenFHE divides the level drop by GetCompositeDegree(), a no-op here.
    constexpr uint32_t compositeDegree = 1;

    SeriesPowers sp(x.cc_);
    sp.powers.reserve(k);
    for (uint32_t i = 0; i < k; ++i)
        sp.powers.emplace_back(x.cc_);
    sp.powers[0].copy(x);

    // Computes the needed powers of x: powers[p]*powers[r] -> x^{i}.
    for (uint32_t i = 2; i <= k; ++i) {
        if (0 == (i & (i - 1))) {
            sp.powers[i - 1].square(sp.powers[i / 2 - 1]); // EvalSquare
            // ModReduceInPlace: FIXEDMANUAL-gated (rns-leveledshe.cpp L406-411); the fused
            // mode always rescales so every power is NoiseLevel 1 for the fused WSum.
            fused ? sp.powers[i - 1].rescaleInternal() : sp.powers[i - 1].rescale();
        } else if (indices[i - 1]) {
            uint64_t p = (uint64_t(1) << (msbOf(i) - 1)) - 1;
            uint64_t r = (i & p) - 1;
            uint32_t diff = sp.powers[r].getLevel() - sp.powers[p].getLevel();
            // LevelReduceInPlace(powers[r], nullptr, diff / compositeDegree) — the RNS
            // implementation (rns-leveledshe.cpp L428-436) only drops towers for FIXEDMANUAL.
            if (x.cc.rescaleTechnique == FIXEDMANUAL)
                sp.powers[r].dropToLevel(sp.powers[r].getLevel() - diff / compositeDegree);
            sp.powers[i - 1].mult(sp.powers[p], sp.powers[r]); // EvalMult
            fused ? sp.powers[i - 1].rescaleInternal() : sp.powers[i - 1].rescale();
        }
    }

    // brings the needed powers of x to the same level (powers[k-1] is the reference);
    // skipped in the fused mode (the WSum target_level folds the drop).
    if (!fused) {
        for (uint32_t i = 1; i < k; ++i) {
            if (indices[i - 1]) {
                uint32_t diff = sp.powers[i - 1].getLevel() - sp.powers[k - 1].getLevel();
                if (x.cc.rescaleTechnique == FIXEDMANUAL)
                    sp.powers[i - 1].dropToLevel(sp.powers[i - 1].getLevel() - diff / compositeDegree);
            }
        }
    }
    return sp;
}

// OpenFHE internalEvalPowersPS (L475-543): the full power set x^1..x^k via the powerOf2/rem
// recurrence, then the PS extension set — powers2 (x^{k*2^i}) and power2km1 (x^{k(2^m - 1)})
// — consumed by the recursive PS polynomial evaluator (M5). The coefficients only matter
// through their degree (the dispatch), so this is scalar-type independent.
// `fused`: like evalPowersLinear — ungated rescaleInternal (all powers NoiseLevel 1) and no
// final level alignment (the fused WSum's target_level folds the drop). powers2 is also stored
// rescaled (NoiseLevel 1) under every scaling technique, since the fused PS evaluator anchors
// its target levels on powers2[m-1]'s rescaled level (see innerEvalPolyPSFused).
template <typename T>
SeriesPowers evalPowersPS(Ciphertext& x, uint32_t degree, bool fused = false) {
    FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
    auto degs = lbcrypto::ComputeDegreesPS(degree);
    const uint32_t k = degs[0];
    const uint32_t m = degs[1];

    // GPU pre-condition: composite degree == 1 (see evalPowersLinear).
    constexpr uint32_t compositeDegree = 1;

    SeriesPowers sp(x.cc_);
    sp.k = k;
    sp.m = m;
    sp.powers.reserve(k);
    for (uint32_t i = 0; i < k; ++i)
        sp.powers.emplace_back(x.cc_);
    sp.powers[0].copy(x);

    // computes all powers up to k for x
    uint32_t powerOf2 = 2;
    uint32_t rem = 0;
    for (uint32_t i = 2; i <= k; ++i) {
        if (rem == 0) {
            sp.powers[i - 1].square(sp.powers[(powerOf2 >> 1) - 1]); // EvalSquare
        } else {
            uint32_t diff = sp.powers[rem - 1].getLevel() - sp.powers[powerOf2 - 1].getLevel();
            // LevelReduceInPlace(powers[rem - 1], nullptr, diff) — OpenFHE's LevelReduceInPlace is a
            // real tower drop ONLY under FIXEDMANUAL (rns-leveledshe.cpp L371-378: it calls
            // LevelReduceInternalInPlace solely when the technique is FIXEDMANUAL). FIXEDAUTO falls
            // through to a no-op, exactly like FLEXIBLEAUTO. The drop shifts powers[rem-1]'s level,
            // and the fused branch below multiplies with powers[i/2-1] (== powers[rem-1] for the
            // odd-i decompositions), so dropping it under FIXEDAUTO fed the fused product a
            // differently-scaled operand than the reference's fused powers, which disturbed the
            // constant/DC term and shifted the decoded LUT by P/2. The non-fused path is unaffected
            // because its final AdjustLevelsAndDepthInPlace (adjustScaleAndLevel) re-aligns the set.
            if (x.cc.rescaleTechnique == FIXEDMANUAL)
                sp.powers[rem - 1].dropToLevel(sp.powers[rem - 1].getLevel() - diff / compositeDegree);
            if (!fused)
                sp.powers[i - 1].mult(sp.powers[powerOf2 - 1], sp.powers[rem - 1]); // EvalMult
            else {
                if (i & 1 == 0) {
                    sp.powers[i - 1].square(sp.powers[i/2 - 1]);
                } else {
                    sp.powers[i - 1].mult(sp.powers[i/2 - 1], sp.powers[i/2 - 1 + (i&1)]);
                }
            }
        }

        if (++rem == powerOf2) {
            powerOf2 <<= 1;
            rem = 0;
        }
        // ModReduceInPlace: FIXEDMANUAL-gated; the fused mode always rescales so every
        // power is NoiseLevel 1 for the fused WSum.
        fused ? sp.powers[i - 1].rescaleInternal() : sp.powers[i - 1].rescale();
    }

    // brings all powers of x to the same level (powers[k-1] defines the reference level);
    // skipped in the fused mode (the WSum target_level folds the drop).
    if (!fused) {
        uint32_t levelk = sp.powers[k - 1].getLevel();
        if (x.cc.rescaleTechnique == FIXEDMANUAL) {
            for (uint32_t i = 1; i < k; ++i)
                sp.powers[i - 1].dropToLevel(levelk); // LevelReduceInPlace(powers[i-1], nullptr, ...)
        } else {
            // OpenFHE: AdjustLevelsAndDepthInPlace(powers[i-1], powers[k-1]) via the
            // library-native Ciphertext::adjustScaleAndLevel (Ciphertext.cpp L1900-1991).
            // Under FLEXIBLEAUTO the loop products already auto-rescale to the reference
            // level/NL2/SFRealBig, so only powers above powers[k-1] (typically the clone
            // powers[0]) move; equal-level entries no-op.
            for (uint32_t i = 1; i < k; ++i) {
                const Ciphertext& c2 = sp.powers[k - 1];
                sp.powers[i - 1].adjustScaleAndLevel(c2.NoiseLevel, c2.getLevel(), c2.NoiseFactor);
            }
        }
    }

    // computes powers of form k*2^i for x and the product of the powers in powers2 that
    // yields x^{k(2*m - 1)}
    sp.powers2.reserve(m);
    sp.powers2_norescale.reserve(m);
    for (uint32_t i = 0; i < m; ++i) {
        sp.powers2.emplace_back(x.cc_);
        sp.powers2_norescale.emplace_back(x.cc_);
    }
    //sp.powers2[0].copy(sp.powers[k - 1]);
    sp.power2km1.copy(sp.powers[k - 1]);

    for (uint32_t i = 1; i < m; ++i) {
        sp.powers2[i].square(i == 1 ? sp.powers.back() : sp.powers2[i - 1]); // EvalSquare
        // The fused consumer (innerEvalPolyPSFused) requires powers2 at NoiseLevel 1, so the
        // fused path stores every powers2 result rescaled regardless of the scaling technique
        // (mirrors evalChebyshevSeries's forced T2[i] rescale: rescale() then rescaleInternal()
        // while still NL2). The non-fused path keeps the reference's gated rescale() — a real
        // ModReduce only for FIXEDMANUAL — so the bit-exact transcription is untouched.
        if (fused) {
            sp.powers2_norescale[i].copy(sp.powers2[i]);
            if (sp.powers2[i].NoiseLevel == 2)
                sp.powers2[i].rescaleInternal();
        } else {
            sp.powers2[i].rescale(); // ModReduceInPlace
        }

        sp.power2km1.mult(sp.powers2[i]);
        sp.power2km1.rescale();
    }
    return sp;
}

// AdvancedSHECKKSRNS::EvalPowers dispatch (L545-566): degree < 5 -> linear, else PS.
// `fused` selects the optimized generation (see the header doc).
template <typename T>
SeriesPowers evalPowers(Ciphertext& ctxt, const std::vector<T>& coefficients, bool fused) {
    FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
    uint32_t d = lbcrypto::Degree(coefficients);
    if (d < 5)
        return evalPowersLinear<T>(ctxt, coefficients, fused);
    return evalPowersPS<T>(ctxt, d, fused);
}

//------------------------------------------------------------------------------
// EVAL POLY (power basis) — bit-exact GPU transcription of internalEvalPolyLinearWithPrecomp,
// InnerEvalPolyPS, internalEvalPolyPSWithPrecomp and the EvalPolyWithPrecomp dispatch
// (ckksrns-advancedshe.cpp L568-816).
//------------------------------------------------------------------------------

// EvalPartialLinearWSum with target_level == -1 (L205-250) → evalStreamedLinearWSum
// (L55-106): the no-bias weighted sum sum_{i=1..limit} weights[i]*ctxs[i-1] (weights[0] is
// the constant term and is skipped — the caller adds it separately). FIXEDMANUAL: acc =
// EvalMult(ctxs[0], weights[1]) + NoCheck adds of EvalMult(ctxs[i], weights[i+1]), then
// ModReduceInPlace. Other techniques: per-term copies are brought to the max-level input
// (same level for the aligned power sets → no-op), all reduced one tower when the max is
// NL2, scalar-multiplied and accumulated; the trailing ModReduceInPlace is a FIXEDMANUAL-gated
// no-op. The reference operates on clones — this uses copies too (never mutates `ctxs`).
template <typename T>
void evalPartialLinearWSum(Ciphertext& out, const std::vector<Ciphertext>& ctxs, const std::vector<T>& weights,
                           uint32_t limit) {
    FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
    ContextData& cc = ctxs[0].cc;
    if (cc.rescaleTechnique == FIXEDMANUAL) {
        out.multScalar(ctxs[0], weights[1]); // acc = EvalMult(ctxs[0], weights[1])
        for (uint32_t i = 1; i < limit; ++i) {
            Ciphertext term(ctxs[0].cc_);
            term.copy(ctxs[i]);
            term.multScalar(weights[i + 1]); // EvalMult(ctxs[i], weights[i+1])
            out.add(term);                  // EvalAddInPlaceNoCheck (same level/NL → plain add)
        }
        out.rescale(); // ModReduceInPlace
    } else {
        // non-FIXEDMANUAL: max-level scan, then per-term AdjustLevelsAndDepthInPlace(term, ctm).
        uint32_t maxLevel = ctxs[0].getLevel();
        uint32_t maxIdx = 0;
        for (uint32_t i = 1; i < limit; ++i) {
            if ((ctxs[i].getLevel() > maxLevel) ||
                ((ctxs[i].getLevel() == maxLevel) && (ctxs[i].NoiseLevel == 2))) {
                maxLevel = ctxs[i].getLevel();
                maxIdx = i;
            }
        }
        const Ciphertext& ctm = ctxs[maxIdx];
        const bool reduceAll = (ctm.NoiseLevel == 2);
        for (uint32_t i = 0; i < limit; ++i) {
            Ciphertext term(ctxs[0].cc_);
            term.copy(ctxs[i]);
            // OpenFHE: AdjustLevelsAndDepthInPlace(term, ctm). The fork's ct1-fewer-towers
            // branch mutates ct2 (the shared max) — unreachable for the level-aligned
            // power sets (all powers sit at powers[k-1]'s level), assert to surface regressions.
            if (term.getLevel() > ctm.getLevel()) {
                term.adjustScaleAndLevel(ctm.NoiseLevel, ctm.getLevel(), ctm.NoiseFactor);
            } else if (term.getLevel() < ctm.getLevel()) {
                assert(false);
            }
            if (reduceAll)
                term.rescaleInternal(); // ModReduceInternalInPlace(term, compositeDegree=1)
            term.multScalar(weights[i + 1]); // EvalMultInPlace(term, weights[i+1])
            if (i == 0)
                out.copy(term);
            else
                out.add(term); // EvalAddInPlaceNoCheck
        }
        // ModReduceInPlace(out) — FIXEDMANUAL-gated no-op for the flexible techniques
    }
}

// InnerEvalPolyPS (L604-703): the recursive PS power-basis evaluation reusing the precomputed
// power set. Serial on the GPU (the reference omp tasks are independent so any order is
// bit-identical); the recursion descends on m for the q/s branches with degree > k.
template <typename T>
void innerEvalPolyPS(const Ciphertext& x,
    Ciphertext& out,
    const std::vector<T>& coefficients,
    const uint32_t k,
    uint32_t m,
    const std::vector<Ciphertext>& powers,
    const std::vector<Ciphertext>& powers2) {
    FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
    FIDESlib::CKKS::Context cc_ = x.cc_;

    // Compute k*2^{m-1}-k because we use it a lot
    uint32_t k2m2k = k * (1 << (m - 1)) - k;

    // Divide coefficients by x^{k*2^{m-1}}
    std::vector<T> xkm(k2m2k + k + 1);
    xkm.back() = 1;
    auto divqr = lbcrypto::LongDivisionPoly(coefficients, xkm);

    // Subtract x^{k(2^{m-1} - 1)} from r
    auto& r2 = divqr->r;
    if (uint32_t n = lbcrypto::Degree(r2); static_cast<int32_t>(k2m2k - n) <= 0) {
        r2.resize(n + 1);
        r2[k2m2k] -= 1;
    } else {
        r2.resize(k2m2k + 1);
        r2.back() = -1;
    }

    auto divcs = lbcrypto::LongDivisionPoly(r2, divqr->q);

    Ciphertext qu(cc_), su(cc_), cu(cc_);

    // Evaluate q at u. If its degree is larger than k, recursively apply the
    // Paterson-Stockmeyer algorithm.
    if (lbcrypto::Degree(divqr->q) > k) {
        innerEvalPolyPS<T>(x, qu, divqr->q, k, m - 1, powers, powers2);
    } else {
        qu.copy(powers[k - 1]);
        qu.addScalar(divqr->q.front()); // EvalAdd(powers[k-1], q.front())
        divqr->q.resize(k);
        if (uint32_t n = lbcrypto::Degree(divqr->q); n > 0) {
            Ciphertext tmp(cc_);
            evalPartialLinearWSum<T>(tmp, powers, divqr->q, n);
            qu.add(tmp); // EvalAddInPlace(qu, WSum)
        }
    }

    // Add x^{k(2^{m-1} - 1)} to s, then evaluate s at u (mirror of the q branch).
    auto& s2 = divcs->r;
    s2.resize(k2m2k + 1);
    s2.back() = 1;
    if (lbcrypto::Degree(s2) > k) {
        innerEvalPolyPS<T>(x, su, s2, k, m - 1, powers, powers2);
    } else {
        su.copy(powers[k - 1]);
        su.addScalar(s2.front());
        s2.resize(k);
        if (uint32_t n = lbcrypto::Degree(s2); n > 0) {
            Ciphertext tmp(cc_);
            evalPartialLinearWSum<T>(tmp, powers, s2, n);
            su.add(tmp);
        }
    }

    // Evaluate c = divcs->q combined with powers2[m-1] (x^{k*2^{m-1}}).
    if (uint32_t n = lbcrypto::Degree(divcs->q); n == 0) {
        cu.copy(powers2[m - 1]);
        cu.addScalar(divcs->q.front()); // EvalAdd(powers2[m-1], q.front())
    } else if (n == 1) {
        if (lbcrypto::IsNotEqualOne(divcs->q[1])) {
            cu.multScalar(powers.front(), divcs->q[1]); // EvalMult(powers[0], q[1])
            cu.rescale();                               // ModReduceInPlace
            cu.add(powers2[m - 1]);                     // EvalAddInPlace(cu, powers2[m-1])
        } else {
            cu.copy(powers2[m - 1]);
            cu.add(powers.front()); // EvalAdd(powers2[m-1], powers[0])
        }
        cu.addScalar(divcs->q.front());
    } else {
        Ciphertext tmp(cc_);
        evalPartialLinearWSum<T>(tmp, powers, divcs->q, n);
        cu.copy(powers2[m - 1]);
        cu.add(tmp); // EvalAdd(powers2[m-1], WSum)
        cu.addScalar(divcs->q.front());
    }

    // result = EvalMult(cu, qu); ModReduceInPlace(result); EvalAddInPlace(result, su)
    out.mult(cu, qu);
    out.rescale();
    out.add(su);
}

// internalEvalPolyLinearWithPrecomp (L568-602) for degree < 5. The reference destroys the
// precomputed powers (EvalMultInPlace powers[i-1]) — a temp copy is value- and
// metadata-identical, and leaves the power set reusable.
template <typename T>
Ciphertext evalPolyLinearWithPrecomp(const std::vector<Ciphertext>& powers, const std::vector<T>& coefficients) {
    if (coefficients.size() < 2)
        throw std::invalid_argument("The coefficients vector should contain at least 2 elements");
    const uint32_t k = static_cast<uint32_t>(coefficients.size()) - 1;
    if (!lbcrypto::IsNotEqualZero(coefficients[k]))
        throw std::invalid_argument("The highest-order coefficient cannot be set to 0");

    FIDESlib::CKKS::Context cc_ = powers[0].cc_;
    Ciphertext result(cc_);

    // perform scalar multiplication for the highest-order term
    result.multScalar(powers[k - 1], coefficients[k]);

    // perform scalar multiplication for all other nonzero terms and sum them up
    for (uint32_t i = 1; i < k; ++i) {
        if (lbcrypto::IsNotEqualZero(coefficients[i])) {
            Ciphertext tmp(cc_);
            tmp.copy(powers[i - 1]);
            tmp.multScalar(coefficients[i]); // EvalMultInPlace(powers[i-1], coefficients[i])
            result.add(tmp);                 // EvalAddInPlace(result, powers[i-1])
        }
    }

    result.rescale(); // ModReduceInPlace(result)

    // adds the free term (at x^0)
    result.addScalar(coefficients[0]); // EvalAddInPlace(result, coefficients[0])

    return result;
}

// internalEvalPolyPSWithPrecomp (L705-738): pad with +1 at degree 2*k2m2k+k, run InnerEvalPolyPS
// at (k, m) on the shared power set, subtract power2km1. The k*2^{m-1} < 128 vs parallel
// swap is irrelevant on the serial GPU — single path, bit-identical either way.
template <typename T>
Ciphertext evalPolyPSWithPrecomp(const SeriesPowers& sp, const std::vector<T>& coefficients) {
    FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
    const auto& powers = sp.powers;
    const auto& powers2 = sp.powers2;
    const Ciphertext& power2km1 = sp.power2km1;
    const uint32_t k = sp.k;
    const uint32_t m = sp.m;
    FIDESlib::CKKS::Context cc_ = sp.cc_;

    // Add x^{k(2^m - 1)} to the polynomial that has to be evaluated
    const uint32_t k2m2k = k * (1 << (m - 1)) - k;
    auto f2 = coefficients;
    f2.resize(lbcrypto::Degree(f2) + 1);
    f2.resize(2 * k2m2k + k + 1);
    f2.back() = 1;

    Ciphertext result(cc_);
    innerEvalPolyPS<T>(powers[0], result, f2, k, m, powers, powers2);
    result.sub(power2km1); // EvalSub(InnerEvalPolyPS(...), power2km1)
    return result;
}

// -------------------------------
// FUSED (optimized) PS EVALUATION
// -------------------------------
// Same Patterson-Stockmeyer algebra as innerEvalPolyPS / evalPolyPSWithPrecomp, but every
// weighted sum is a single evalPartialLinearWSumWithBias call (the shared fused kernels):
//   * the constant term is folded as the WSum's bias — added BEFORE the final rescale (the
//     reordered addScalar/rescale of the fused design; in the reference it was added at the
//     un-rescaled scale via EvalAdd(powers[k-1], q.front()));
//   * the x^k = powers[k-1] contribution is appended as an extra weight (w[k] = 1) instead
//     of a separate EvalAdd;
//   * target_level = powers[k-1]'s level (q/s branches) or powers2[m-1]'s level (c branch) —
//     the natural "aligned" levels — and since the fused WSum folds each input's tower drop
//     into the constant multiply, evalPowers(..., fused = true) needs no dropToLevel anymore.
// The result decrypts to the same polynomial value with small rescale-ordering error, but is
// NOT bit-exact vs OpenFHE. Dense power sets only (the fused WSum assumes every power exists).

// RECORD/REMAINING for the fused PS path — mirrors the Chebyshev evaluator's two-phase
// structure (PSRecordFused defined above): the RECORD pass walks the PS recursion computing
// the splits and recording, per terminal (non-recursive) WSum node, the WithBias weight
// vector ([0] = bias, [1..] = power terms) and its target level — CPU-only, no ciphertext
// work. The recorded nodes are then all evaluated in one batched
// evalMultiplePartialLinearWSumWithBias pass (fused for both double and complex), and the
// REMAINING pass rewalks the recursion popping the precomputed raw (un-rescaled) sums and
// replaying only the composition tree.
enum PS_FUSED_MODE { FUSED_FULL, FUSED_RECORD, FUSED_REMAINING };

template <typename T>
void innerEvalPolyPSFused(Ciphertext& out,
    const std::vector<Ciphertext*>& powers,
    const std::vector<Ciphertext>& powers2,
    const std::vector<T>& coefficients,
    const uint32_t k,
    uint32_t m,
    int level_offset = 0,
    PS_FUSED_MODE mode = FUSED_FULL,
    PSRecordFused<T>* record = nullptr,const std::vector<Ciphertext>&powers2_norescale = {}) {
    FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
    FIDESlib::CKKS::Context cc_ = out.cc_;

    // The fused path multiplies with the single-operand multMutable (this *= qu), which looks the
    // relin key up by THIS ciphertext's keyID — unlike the non-fused 3-arg mult, which first
    // copy()s an operated operand (and hence inherits its tag). The recursion carrier `out` and
    // the fresh qu_/su_ scratch start tag-less, so inherit the power set's tag.
    if (out.keyID.empty() && !powers.empty())
        out.keyID = powers.front()->keyID;

    // Compute k*2^{m-1}-k because we use it a lot
    uint32_t k2m2k = k * (1 << (m - 1)) - k;

    // Level discipline (mirrors the working Chebyshev evaluator, innerEvalChebyshevPS):
    // every branch's WSum target is anchored to the x^{k*2^{m-1}} power
    // (powers2[m-1]) at the current recursion depth, plus one when that power is already
    // NoiseLevel 1, minus the recursion's level_offset — not to the shallower powers[k-1]
    // level. The fused evalPowersPS stores powers2 rescaled (NoiseLevel 1) under every
    // scaling technique — like the Chebyshev T2 set — so powers2[m-1].getLevel() is its
    // post-ModReduce level and the +1 recovers the pre-rescale (product) level the WSum must
    // land on; the add of the rescaled powers2 into cu then rides the usual NN-adjust. That
    // keeps qu/cu at the SAME (deep) level so the final EvalMult(cu, qu) is a uniform-level
    // mult (no mixed-level adjust), and su one level deeper, exactly like the reference/cheb
    // pipeline. Recursion offsets: the q branch keeps the offset, the s branch (a mirror
    // image at one deeper level) shifts by +1.

    // Divide coefficients by x^{k*2^{m-1}}
    std::vector<T> xkm(k2m2k + k + 1);
    xkm.back() = 1;
    auto divqr = lbcrypto::LongDivisionPoly(coefficients, xkm);

    // Subtract x^{k(2^{m-1} - 1)} from r
    auto& r2 = divqr->r;
    if (uint32_t n = lbcrypto::Degree(r2); static_cast<int32_t>(k2m2k - n) <= 0) {
        r2.resize(n + 1);
        r2[k2m2k] -= 1;
    } else {
        r2.resize(k2m2k + 1);
        r2.back() = -1;
    }

    auto divcs = lbcrypto::LongDivisionPoly(r2, divqr->q);

    Ciphertext qu_(cc_), su_(cc_);
    qu_.keyID = su_.keyID = out.keyID;
    std::unique_ptr<Ciphertext> qu_tmp, su_tmp;
    Ciphertext* qu = nullptr;
    Ciphertext* su = nullptr;
    Ciphertext& cu = out; // the c branch is built straight into the result (no copy at the mult)
    const int targetBase = powers2[m - 1].getLevel() + (powers2[m - 1].NoiseLevel == 1 ? 1 : 0);

    // REMAINING helper: pop the precomputed raw (un-rescaled) WSum result for the current
    // terminal node (the driver pushed them in reverse record order, so back() == next).
    auto popRaw = [&]() -> std::unique_ptr<Ciphertext> {
        auto r = std::make_unique<Ciphertext>(std::move(record->rev_result_ciphertexts.back()));
        record->rev_result_ciphertexts.pop_back();
        return r;
    };

    // q branch: qu = q[0] + x^k + sum_{i=1..k-1} q[i]*x^i as one fused WSum (weights
    // {q[0], q[1], ..., q[k-1], 1}, bias = q[0], limit = k). Single-use WSum target: write
    // straight into qu, no intermediate copy.
    if (lbcrypto::Degree(divqr->q) > k) {
        innerEvalPolyPSFused<T>(qu_, powers, powers2, divqr->q, k, m - 1, level_offset, mode, record, powers2_norescale);
        // The recursion returns the un-rescaled composition (mult -> add, no ModReduce, below);
        // qu feeds the parent's EvalMult, so the caller applies the rescale here — exactly like
        // innerEvalChebyshevPS.
        if (mode != FUSED_RECORD)
            qu_.rescale();
    } else {
        auto v = divqr->q;
        v.resize(k + 1);
        v[k] = (T)1; // the x^k = powers[k-1] term
        if (mode == FUSED_RECORD) {
            record->constants.emplace_back(std::move(v));
            record->target_level.emplace_back(targetBase - level_offset);
        } else if (mode == FUSED_REMAINING) {
            qu_tmp = popRaw();
            qu_tmp->rescale(); // the fused qu's trailing with-bias rescale (rescale=1)
            qu = qu_tmp.get();
        } else {
            qu_.evalPartialLinearWSumWithBias(powers, v, v.front(), static_cast<int>(k), targetBase - level_offset,
                                              /*rescale=*/1);
            qu = &qu_;
        }
    }
    if (qu == nullptr)
        qu = &qu_;

    // s branch: mirror of the q branch on s2 (s2.back() = 1 added first, like the reference).
    // The WithBias runs WITHOUT the trailing rescale (rescale=false, like the Chebyshev
    // evaluator): the su contribution is added to the un-rescaled product and the whole
    // composition is reduced once by the caller, saving one ModReduce per recursion level.
    auto& s2 = divcs->r;
    s2.resize(k2m2k + 1);
    s2.back() = 1;
    if (lbcrypto::Degree(s2) > k) {
        innerEvalPolyPSFused<T>(su_, powers, powers2, s2, k, m - 1, level_offset + 1, mode, record, powers2_norescale);
    } else {
        auto v = s2;
        v.resize(k + 1);
        v[k] = (T)1;
        if (mode == FUSED_RECORD) {
            record->constants.emplace_back(std::move(v));
            record->target_level.emplace_back(targetBase - level_offset - 1);
        } else if (mode == FUSED_REMAINING) {
            su_tmp = popRaw();
            su = su_tmp.get();
        } else {
            su_.evalPartialLinearWSumWithBias(powers, v, v.front(), static_cast<int>(k),
                                              targetBase - level_offset - 1, /*rescale=*/0);
            su = &su_;
        }
    }
    if (su == nullptr)
        su = &su_;

    // c branch: cu = powers2[m-1] + sum_{i=1..n} q[i]*x^i + q[0] (n >= 1); n == 0 keeps the
    // trivial powers2[m-1] + q[0]. The fused WSum covers the degree-1 case (q[1]*x) too.
    if (uint32_t n = lbcrypto::Degree(divcs->q); n == 0) {
        cu.copy(powers2[m - 1]);
        cu.addScalar(divcs->q.front());
    } else {
        if (mode == FUSED_RECORD) {
            record->constants.emplace_back(divcs->q);
            record->target_level.emplace_back(targetBase - level_offset);
        } else if (mode == FUSED_REMAINING) {
            // Mirror OpenFHE's InnerEvalPolyPSFused c-branch (and the FUSED_FULL arm below): when the
            // popped WSum sits at the pre-ModReduce powers2 level, add the UN-reduced powers2 and then
            // rescale; otherwise rescale first, then add the reduced powers2. The batch itself always
            // evaluates with rescale=false, so this reproduces the reference's ordering bit-for-bit.
            auto w = popRaw();
            if (w->getLevel() == powers2_norescale[m - 1].getLevel()) {
                cu.copy(*w);
                cu.add(powers2_norescale[m - 1]);
                cu.rescale();
            } else {
                w->rescale();
                cu.copy(*w);
                cu.add(powers2[m - 1]);
            }
        } else {
            cu.evalPartialLinearWSumWithBias(powers, divcs->q, divcs->q.front(), static_cast<int>(n),
                                             targetBase - level_offset, /*rescale=*/0);
            if (cu.getLevel() == powers2_norescale[m - 1].getLevel()) {
                cu.add(powers2_norescale[m - 1]);
                cu.rescale();
            } else {
                cu.rescale();
                cu.add(powers2[m - 1]);
            }
        }
    }

    if (mode == FUSED_RECORD)
        return;

    // result = EvalMult(cu, qu) + su, WITHOUT the trailing ModReduce (mirrors
    // innerEvalChebyshevPS): the caller owns the rescale — the recursion's parent rescales the
    // q-child it consumes (above) and evalPolyPSWithPrecompFused rescales the root, so each
    // s-branch subtree rides un-rescaled through this add and skips one ModReduce per internal
    // node. *Mutable: cu/qu/su are single-use — the mult/addition operate in place, saving the
    // lower-operand copy of `mult(cu, qu)`.
    out.multMutable(*qu);
    out.addMutable(*su);
}

template <typename T>
Ciphertext evalPolyPSWithPrecompFused(SeriesPowers& sp, const std::vector<T>& coefficients) {
    FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
    const uint32_t k = sp.k;
    const uint32_t m = sp.m;
    FIDESlib::CKKS::Context cc_ = sp.cc_;

    const uint32_t k2m2k = k * (1 << (m - 1)) - k;

    // Add x^{k(2^m - 1)} to the polynomial that has to be evaluated
    auto f2 = coefficients;
    f2.resize(lbcrypto::Degree(f2) + 1);
    f2.resize(2 * k2m2k + k + 1);
    f2.back() = 1;

    std::vector<Ciphertext*> powers_ptrs;
    powers_ptrs.reserve(sp.powers.size());
    for (auto& p : sp.powers)
        powers_ptrs.push_back(&p);

    Ciphertext result(cc_);
    // RECORD/REMAINING (Chebyshev-style): record the terminal WSum nodes (splits are pure
    // CPU work), evaluate all of them at once with the batched complex/double kernels, then
    // replay the composition tree from the reversed result stack.
    {
        PSRecordFused<T> record;
        innerEvalPolyPSFused<T>(result, powers_ptrs, sp.powers2, f2, k, m, 0, FUSED_RECORD, &record, sp.powers2_norescale);
        std::vector<Ciphertext> ws;
        ws.reserve(record.constants.size());
        // Both scalar types ride the batched fused WSum kernels: the double overload packs the
        // uniform per-limb factors, the complex overload builds the BuildConstant2 term factors
        // and evaluates every recorded node in one evalLinearWSumComplexMultiple pass.
        ws = Ciphertext::evalMultiplePartialLinearWSumWithBias(powers_ptrs, record.constants, record.target_level,
                                                               /*rescale=*/false);
        // Reversed stack: the RECORD/REMAINING walk visits the terminals in the same order in
        // both passes, so back() must yield the first-recorded result.
        record.rev_result_ciphertexts.reserve(ws.size());
        for (size_t i = 0; i < ws.size(); ++i)
            record.rev_result_ciphertexts.emplace_back(std::move(ws[ws.size() - 1 - i]));
        innerEvalPolyPSFused<T>(result, powers_ptrs, sp.powers2, f2, k, m, 0, FUSED_REMAINING, &record, sp.powers2_norescale);
    }
    // The composition tree no longer rescales internally (mult -> add); the root owns the single
    // ModReduce, exactly like the Chebyshev driver.
    result.rescale();
    result.sub(sp.power2km1); // EvalSub(InnerEvalPolyPS(...), power2km1)
    return result;
}

// AdvancedSHECKKSRNS::EvalPolyWithPrecomp dispatch (L758-780): degree < 5 -> linear, else PS.
// `fused` selects the optimized variant that rides the shared evalPartialLinearWSumWithBias
// kernels (value-correct, NOT bit-exact; see the header doc). The fused linear path needs a
// dense coefficient table (the sparse linear power gaps would feed empty ciphertexts into
// the fused WSum).
template <typename T>
Ciphertext evalPolyWithPrecomp(SeriesPowers& sp, const std::vector<T>& coefficients, bool fused) {
    FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
    uint32_t d = lbcrypto::Degree(coefficients);
    if (d < 5) {
        // The fused single-WSum linear path rounds differently from the reference, and for the
        // shallow coefficient tables that residue profile is incompatible with the FLEXIBLEAUTO
        // homomorphic decode (its centered-lift boundary flips the value across a full modulus).
        // The reference linear consumer is canonical — verified bit-identical to the exact path
        // even when fed the fused power set — so use it. The fused power build is still fine.
        (void)fused;
        return evalPolyLinearWithPrecomp<T>(sp.powers, coefficients);
    }
    if (!fused)
        return evalPolyPSWithPrecomp<T>(sp, coefficients);
    return evalPolyPSWithPrecompFused<T>(sp, coefficients);
}

} // namespace FIDESlib::CKKS

// Explicit instantiations — the only two scalar types the Chebyshev coefficients take:
// double (real coeff_cos tables / standard bootstrap) and std::complex<double> (the
// coeff_exp tables).
template void FIDESlib::CKKS::innerEvalChebyshevPS<double>(
    const FIDESlib::CKKS::Ciphertext&, FIDESlib::CKKS::Ciphertext&, const std::vector<double>&, const uint32_t, uint32_t,
    const std::vector<FIDESlib::CKKS::Ciphertext*>&, const std::vector<FIDESlib::CKKS::Ciphertext*>&, int,
    FIDESlib::CKKS::PS_MODE, FIDESlib::CKKS::PSRecordFused<double>*,const std::vector<Ciphertext*>&T2_norescale);
template void FIDESlib::CKKS::innerEvalChebyshevPS<std::complex<double>>(
    const FIDESlib::CKKS::Ciphertext&, FIDESlib::CKKS::Ciphertext&, const std::vector<std::complex<double>>&,
    const uint32_t, uint32_t, const std::vector<FIDESlib::CKKS::Ciphertext*>&,
    const std::vector<FIDESlib::CKKS::Ciphertext*>&, int, FIDESlib::CKKS::PS_MODE,
    FIDESlib::CKKS::PSRecordFused<std::complex<double>>*,const std::vector<Ciphertext*>&T2_norescale);
template void FIDESlib::CKKS::evalChebyshevSeries<double>(FIDESlib::CKKS::Ciphertext&, std::vector<double>&, double, double);
template void FIDESlib::CKKS::evalChebyshevSeries<std::complex<double>>(FIDESlib::CKKS::Ciphertext&, std::vector<std::complex<double>>&, double, double);
template FIDESlib::CKKS::SeriesPowers FIDESlib::CKKS::evalPowers<double>(FIDESlib::CKKS::Ciphertext&, const std::vector<double>&, bool);
template FIDESlib::CKKS::SeriesPowers FIDESlib::CKKS::evalPowers<std::complex<double>>(FIDESlib::CKKS::Ciphertext&, const std::vector<std::complex<double>>&, bool);
template void FIDESlib::CKKS::evalPartialLinearWSum<double>(FIDESlib::CKKS::Ciphertext&, const std::vector<FIDESlib::CKKS::Ciphertext>&, const std::vector<double>&, uint32_t);
template void FIDESlib::CKKS::evalPartialLinearWSum<std::complex<double>>(FIDESlib::CKKS::Ciphertext&, const std::vector<FIDESlib::CKKS::Ciphertext>&, const std::vector<std::complex<double>>&, uint32_t);
template void FIDESlib::CKKS::innerEvalPolyPS<double>(const FIDESlib::CKKS::Ciphertext&, FIDESlib::CKKS::Ciphertext&, const std::vector<double>&, const uint32_t, uint32_t, const std::vector<FIDESlib::CKKS::Ciphertext>&, const std::vector<FIDESlib::CKKS::Ciphertext>&);
template void FIDESlib::CKKS::innerEvalPolyPS<std::complex<double>>(const FIDESlib::CKKS::Ciphertext&, FIDESlib::CKKS::Ciphertext&, const std::vector<std::complex<double>>&, const uint32_t, uint32_t, const std::vector<FIDESlib::CKKS::Ciphertext>&, const std::vector<FIDESlib::CKKS::Ciphertext>&);
template FIDESlib::CKKS::Ciphertext FIDESlib::CKKS::evalPolyLinearWithPrecomp<double>(const std::vector<FIDESlib::CKKS::Ciphertext>&, const std::vector<double>&);
template FIDESlib::CKKS::Ciphertext FIDESlib::CKKS::evalPolyLinearWithPrecomp<std::complex<double>>(const std::vector<FIDESlib::CKKS::Ciphertext>&, const std::vector<std::complex<double>>&);
template FIDESlib::CKKS::Ciphertext FIDESlib::CKKS::evalPolyPSWithPrecomp<double>(const FIDESlib::CKKS::SeriesPowers&, const std::vector<double>&);
template FIDESlib::CKKS::Ciphertext FIDESlib::CKKS::evalPolyPSWithPrecomp<std::complex<double>>(const FIDESlib::CKKS::SeriesPowers&, const std::vector<std::complex<double>>&);
template FIDESlib::CKKS::Ciphertext FIDESlib::CKKS::evalPolyWithPrecomp<double>(FIDESlib::CKKS::SeriesPowers&, const std::vector<double>&, bool);
template FIDESlib::CKKS::Ciphertext FIDESlib::CKKS::evalPolyWithPrecomp<std::complex<double>>(FIDESlib::CKKS::SeriesPowers&, const std::vector<std::complex<double>>&, bool);

void applyDoubleAngleIterations(Ciphertext& ctxt, int its, const KeySwitchingKey& kskEval) {
    FIDESlib::CudaNvtxRange r_(std::string{ sc::current().function_name() });
    ContextData& cc = ctxt.cc;
    int32_t r = its;
    // std::cout << "Its: " << its << std::endl;
    for (int32_t j = 1; j < r + 1; j++) {
        ctxt.square(false);
        ctxt.add(ctxt);
        double scalar = -std::pow(2.0 * M_PI, -std::pow(2.0, j - r));
        ctxt.addScalar(scalar);
        if (cc.rescaleTechnique == FIDESlib::CKKS::FIXEDMANUAL)
            ctxt.rescaleInternal();
    }
}
