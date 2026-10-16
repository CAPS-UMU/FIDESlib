//
// Created by carlosad on 12/11/24.
//

#ifndef GPUCKKS_APPROXMODEVAL_CUH
#define GPUCKKS_APPROXMODEVAL_CUH

#include "CKKS/Ciphertext.cuh"
#include "CKKS/forwardDefs.cuh"
#include <cinttypes>
#include <vector>

namespace FIDESlib::CKKS {
/**
 * Mirrors lbcrypto::seriesPowers<DCRTPoly> (ciphertext-fwd.h L56-90): the power set produced
 * by evalPowers for the polynomial evaluators:
 *   - powers:    x, x^2, ..., x^k            (powers[k-1] == x^k)
 *   - powers2:   x^k, x^{2k}, ..., x^{2^{m-1}k}  (PS path only, m >= 1)
 *   - power2km1: x^{k(2^m - 1)}                   (PS path only)
 * The linear (degree < 5) path only fills `powers`. NOTE: like OpenFHE's seriesPowers, the
 * linear path leaves the *unneeded* powers[i-1] entries empty (only power-of-2 indices and
 * indices reachable from a nonzero coefficient are computed); evalPoly (M5) must only touch
 * entries it weights.
 */
struct SeriesPowers {
    Context cc_;                      // owning context (Ciphertext members are context-bound)
    std::vector<Ciphertext> powers;   // x^1 .. x^k
    std::vector<Ciphertext> powers2;
    std::vector<Ciphertext> powers2_norescale; // x^{k*2^i}, i = 0..m-1  (PS only)
    Ciphertext power2km1;             // x^{k(2^m - 1)}          (PS only)
    uint32_t k = 0;
    uint32_t m = 0;

    explicit SeriesPowers(const Context& cc) : cc_(cc), power2km1(cc) {}
};

/**
 * Bit-exact GPU transcription of AdvancedSHECKKSRNS::EvalPowers (ckksrns-advancedshe.cpp
 * L545-566), which dispatches on Degree(coefficients): internalEvalPowersPS for degree >= 5,
 * internalEvalPowersLinear otherwise (L399-543).
 *
 * Templated on the coefficient scalar type (double for the real coeff_cos tables,
 * std::complex<double> for the coeff_exp tables): the PS path is coefficient-type-independent
 * (the powers depend only on the degree); the linear path uses IsNotEqualZero(T) for the
 * power-index sparsity. Explicitly instantiated for both types in ApproxModEval.cu.
 *
 * GPU pre-conditions: composite degree == 1 (prime-per-level chains, as everywhere in
 * FIDESlib) and the input ciphertext at NoiseLevel == 1.
 *
 * `fused = true` selects the optimized (value-correct, NOT bit-exact) variant used by the
 * fused evalPolyWithPrecomp: every power is rescaleInternal'd (all NoiseLevel == 1 under
 * every technique) and the final level alignment (LevelReduceInPlace / AdjustLevelsAndDepth)
 * is skipped — the downstream evalPartialLinearWSumWithBias folds each term's tower drop into
 * its target_level constant multiply, so the powers stay at their natural levels.
 */
template <typename T>
SeriesPowers evalPowers(Ciphertext& ctxt, const std::vector<T>& coefficients, bool fused = false);

/**
 * Streamed no-bias partial linear weighted sum — EvalPartialLinearWSum with target_level == -1
 * (ckksrns-advancedshe.cpp L205-250 → evalStreamedLinearWSum L55-106): computes
 * sum_{i=1..limit} weights[i] * ctxs[i-1], where weights[0] is the constant term and is
 * SKIPPED (the caller adds it separately). Templated on the weight scalar type.
 *
 * FIXEDMANUAL: acc = EvalMult(ctxs[0], weights[1]) + EvalAddInPlaceNoCheck of
 * EvalMult(ctxs[i], weights[i+1]), then ModReduceInPlace (rescale()). For the other
 * techniques: per-term clones are adjusted to the max-level input, all terms are
 * ModReduceInternalInPlace'd (one tower, NL2->1) when the max is NL2, then scalar-multiplied
 * and accumulated; the trailing ModReduceInPlace is a FIXEDMANUAL-gated no-op. Mirrors the
 * reference exactly (non-destructive: operates on copies).
 */
template <typename T>
void evalPartialLinearWSum(Ciphertext& out, const std::vector<Ciphertext>& ctxs, const std::vector<T>& weights,
                           uint32_t limit);

/**
 * Bit-exact GPU transcription of InnerEvalPolyPS (ckksrns-advancedshe.cpp L604-703): the
 * recursive Patterson-Stockmeyer power-basis evaluation of `coefficients` at x using the
 * precomputed power set (powers = x^1..x^k, powers2 = x^k, x^{2k}, ...). The q/s branches
 * build the quotient evaluations (powers[k-1] + q.front() + the no-bias
 * evalPartialLinearWSum with weights[1..limit]); the c branch folds divcs->q against
 * powers2[m-1]; out = EvalMult(cu, qu) -> ModReduceInPlace -> EvalAddInPlace(su) ->
 * EvalSub(power2km1) is left to the caller (evalPolyPSWithPrecomp). Recursion on m (degrees
 * > k recurse with m-1, reusing the same power set). Templated on the coefficient scalar
 * type (double / std::complex<double>); explicitly instantiated in ApproxModEval.cu.
 */
template <typename T>
void innerEvalPolyPS(const Ciphertext& x, Ciphertext& out, const std::vector<T>& coefficients, uint32_t k, uint32_t m,
                     const std::vector<Ciphertext>& powers, const std::vector<Ciphertext>& powers2);

/**
 * internalEvalPolyLinearWithPrecomp (ckksrns-advancedshe.cpp L568-602) for degree < 5: the
 * leading EvalMult(powers[k-1], c_k) plus per-nonzero-coefficient
 * EvalMultInPlace(powers[i-1], c_i) + EvalAddInPlace (transcribed with temporaries — the
 * reference destroys the precomputed powers; the results and metadata are identical),
 * then ModReduceInPlace + the constant term. Templated; explicitly instantiated.
 */
template <typename T>
Ciphertext evalPolyLinearWithPrecomp(const std::vector<Ciphertext>& powers, const std::vector<T>& coefficients);

/**
 * internalEvalPolyPSWithPrecomp (ckksrns-advancedshe.cpp L705-738): pad the coefficients
 * with +1 at degree 2*k2m2k+k, run InnerEvalPolyPS at (k, m), subtract power2km1
 * (x^{k(2^m-1)}). The k*2^{m-1} < 128 vs parallel split is irrelevant here (serial GPU).
 */
template <typename T>
Ciphertext evalPolyPSWithPrecomp(const SeriesPowers& sp, const std::vector<T>& coefficients);

/**
 * EvalPolyWithPrecomp dispatch (ckksrns-advancedshe.cpp L758-780): Degree < 5 -> the linear
 * path, else the PS path. Returns a new ciphertext (never mutates the power set).
 *
 * `fused = true` selects the optimized (value-correct, NOT bit-exact) variant that rides the
 * shared evalPartialLinearWSumWithBias kernels: every weighted sum is a single fused call
 * with the constant term folded as the bias (added before the final rescale — the reordered
 * addScalar/rescale of the fused design) and the x^k term appended as an extra weight; it
 * expects the power set from evalPowers(..., fused = true) (NoiseLevel 1, natural levels, no
 * alignment). The result decrypts to the same polynomial value with small rescale-ordering
 * error, but the ciphertexts are not bit-exact. Dense linear tables only (the sparse linear
 * power gaps would feed empty ciphertexts into the fused WSum).
 */
template <typename T>
Ciphertext evalPolyWithPrecomp(SeriesPowers& sp, const std::vector<T>& coefficients, bool fused = false);

/**
 * EvalChebyshevSeries dispatch (ckksrns-advancedshe.cpp L951-965): Degree < 5 uses the
 * linear basis + series evaluator (internalEvalChebyPolysLinear +
 * internalEvalChebyshevSeriesLinearWithPrecomp), else Paterson-Stockmeyer. Both paths are
 * bit-exact GPU transcriptions.
 *
 * Templated on the Chebyshev coefficient scalar type (double for the real coeff_cos tables,
 * std::complex<double> for the coeff_exp tables, e.g. the complex exp path). Both instantiations
 * run the same RECORD/REMAINING batching (one templated PSRecordFused<Scalar>, batched
 * evalMultiplePartialLinearWSumWithBias for double and complex alike) over the serial
 * InnerEvalChebyshevPS transcription. Explicitly instantiated for both types in
 * ApproxModEval.cu.
 */
template <typename T>
void evalChebyshevSeries(Ciphertext& ctxt, std::vector<T>& coefficients, double lower_bound = -1.0, double upper_bound = 1.0);

void approxModReduction(Ciphertext& ctxtEnc, Ciphertext& ctxtEncI, const KeySwitchingKey& keySwitchingKey, uint64_t post);

void multIntScalar(Ciphertext& ctxt, uint64_t op);

void approxModReductionSparse(Ciphertext& ctxtEnc, uint64_t post);
} // namespace FIDESlib::CKKS

#endif // GPUCKKS_APPROXMODEVAL_CUH