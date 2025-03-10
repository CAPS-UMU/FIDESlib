//
// Created by carlosad on 12/11/24.
//

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Context.cuh"
#include "CudaUtils.cuh"

constexpr bool PRINT = false;

using namespace FIDESlib::CKKS;

void evalChebyshevSeries(Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, std::vector<double>& coefficients,
                         double lower_bound, double upper_bound);
void applyDoubleAngleIterations(Ciphertext& ctxt, int its, const KeySwitchingKey& kskEval);
void multMonomial(Ciphertext& ctxt, int power);
void multIntScalar(Ciphertext& ctxt, uint64_t op);

void FIDESlib::CKKS::approxModReduction(Ciphertext& ctxtEnc, Ciphertext& ctxtEncI,
                                        const KeySwitchingKey& keySwitchingKey, uint64_t post) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});

    cudaDeviceSynchronize();
    Context& cc = ctxtEnc.cc;
    evalChebyshevSeries(ctxtEncI, keySwitchingKey, cc.GetCoeffsChebyshev(), -1.0, 1.0);
    evalChebyshevSeries(ctxtEnc, keySwitchingKey, cc.GetCoeffsChebyshev(), -1.0, 1.0);
    if constexpr (PRINT) {
        std::cout << "ctxtEnc res " << ctxtEnc.getLevel() << std::endl;
        for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;

        std::cout << "ctxtEncI res " << ctxtEncI.getLevel() << std::endl;
        for (auto& i : ctxtEncI.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }

    applyDoubleAngleIterations(ctxtEnc, cc.GetDoubleAngleIts(), keySwitchingKey);
    applyDoubleAngleIterations(ctxtEncI, cc.GetDoubleAngleIts(), keySwitchingKey);
    if constexpr (PRINT) {
        std::cout << "ctxtEnc DA res " << ctxtEnc.getLevel() << std::endl;
        for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;

        std::cout << "ctxtEncI DA res " << ctxtEncI.getLevel() << std::endl;
        for (auto& i : ctxtEncI.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();
    multMonomial(ctxtEncI, cc.N / 2);
    cudaDeviceSynchronize();
    ctxtEnc.add(ctxtEncI);
    cudaDeviceSynchronize();
    multIntScalar(ctxtEnc, post);
    cudaDeviceSynchronize();
    return;
    if constexpr (PRINT) {
        std::cout << "ctxtEnc final res " << ctxtEnc.getLevel() << std::endl;
        for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
}

void FIDESlib::CKKS::approxModReductionSparse(Ciphertext& ctxtEnc, const KeySwitchingKey& keySwitchingKey,
                                              uint64_t post) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    Context& cc = ctxtEnc.cc;
    evalChebyshevSeries(ctxtEnc, keySwitchingKey, cc.GetCoeffsChebyshev(), -1.0, 1.0);

    if constexpr (PRINT) {
        std::cout << "ctxtEnc res " << ctxtEnc.getLevel() << std::endl;
        for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }

    applyDoubleAngleIterations(ctxtEnc, cc.GetDoubleAngleIts(), keySwitchingKey);
    if constexpr (PRINT) {
        std::cout << "ctxtEnc DA " << ctxtEnc.getLevel() << std::endl;
        for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();

    multIntScalar(ctxtEnc, post);
    if constexpr (PRINT) {
        std::cout << "ctxtEnc final " << ctxtEnc.getLevel() << std::endl;
        for (auto& i : ctxtEnc.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    cudaDeviceSynchronize();
}

void FIDESlib::CKKS::multIntScalar(Ciphertext& ctxt, uint64_t op) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    std::vector<uint64_t> op_(ctxt.getLevel() + 1, op);
    ctxt.c0.multScalar(op_);
    ctxt.c1.multScalar(op_);
};

void FIDESlib::CKKS::multMonomial(Ciphertext& ctxt, int power) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    //assert(power < ctxt.cc.N);
    // TODO compute fully as a GPU function.
    RNSPoly monomial(ctxt.cc, ctxt.c0.getLevel(), true);
    std::vector<uint64_t> coefs(ctxt.cc.N, 0);

    if (power < ctxt.cc.N) {
        coefs[power] = 1;
        for (auto& g : monomial.GPU) {
            cudaSetDevice(g.device);
            for (auto& l : g.limb) {
                SWITCH(l, load(coefs));
            }
        }
    } else {

        for (auto& g : monomial.GPU) {
            cudaSetDevice(g.device);
            for (auto& l : g.limb) {
                coefs[power % ctxt.cc.N] = (ctxt.cc.prime[PRIMEID(l)].p - 1) /*% ctxt.cc.prime[PRIMEID(l)].p*/;
                SWITCH(l, load(coefs));
                cudaDeviceSynchronize();
            }
        }
    }
    cudaDeviceSynchronize();
    monomial.NTT(ctxt.cc.batch);
    cudaDeviceSynchronize();
    if constexpr (PRINT) {
        std::cout << "Monomial ";
        for (auto& j : monomial.GPU)
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        std::cout << std::endl;
    }
    ctxt.c0.multElement(monomial);
    ctxt.c1.multElement(monomial);

    /* Based on this (OpenFHE):
std::vector<DCRTPoly>& cv = ciphertext->GetElements();
const auto elemParams     = cv[0].GetParams();
auto paramsNative         = elemParams->GetParams()[0];
usint N                   = elemParams->GetRingDimension();
usint M                   = 2 * N;

    NativePoly monomial(paramsNative, Format::COEFFICIENT, true);

    usint powerReduced = power % M;
    usint index        = power % N;
    monomial[index]    = powerReduced < N ? NativeInteger(1) : paramsNative->GetModulus() - NativeInteger(1);

    DCRTPoly monomialDCRT(elemParams, Format::COEFFICIENT, true);
    monomialDCRT = monomial;
    monomialDCRT.SetFormat(Format::EVALUATION);

    for (usint i = 0; i < ciphertext->NumberCiphertextElements(); i++) {
        cv[i] *= monomialDCRT;
    }
    */
}

void innerEvalChebyshevPS(const Ciphertext& ctxt, const KeySwitchingKey& kskEval, Ciphertext& out,
                          const std::vector<double>& coefficients, uint32_t k, uint32_t m, std::vector<Ciphertext>& T,
                          std::vector<Ciphertext>& T2) {
    FIDESlib::CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    /*
Ciphertext<DCRTPoly> AdvancedSHECKKSRNS::InnerEvalChebyshevPS(ConstCiphertext<DCRTPoly> x,
                                                              const std::vector<double>& coefficients, uint32_t k,
                                                              uint32_t m, std::vector<Ciphertext<DCRTPoly>>& T,
                                                              std::vector<Ciphertext<DCRTPoly>>& T2) const {
*/
    Context& cc = ctxt.cc;
    /// Left AS IS ///
    // Compute k*2^{m-1}-k because we use it a lot
    uint32_t k2m2k = k * (1 << (m - 1)) - k;

    // Divide coefficients by T^{k*2^{m-1}}
    std::vector<double> Tkm(int32_t(k2m2k + k) + 1, 0.0);
    Tkm.back() = 1;
    auto divqr = lbcrypto::LongDivisionChebyshev(coefficients, Tkm);

    // Subtract x^{k(2^{m-1} - 1)} from r
    std::vector<double> r2 = divqr->r;
    if (int32_t(k2m2k - lbcrypto::Degree(divqr->r)) <= 0) {
        r2[int32_t(k2m2k)] -= 1;
        r2.resize(lbcrypto::Degree(r2) + 1);
    } else {
        r2.resize(int32_t(k2m2k + 1), 0.0);
        r2.back() = -1;
    }

    // Divide r2 by q
    auto divcs = lbcrypto::LongDivisionChebyshev(r2, divqr->q);

    // Add x^{k(2^{m-1} - 1)} to s
    std::vector<double> s2 = divcs->r;
    s2.resize(int32_t(k2m2k + 1), 0.0);
    s2.back() = 1;

    /// Left AS IS ///

    Ciphertext cu(cc);
    uint32_t dc = lbcrypto::Degree(divcs->q);
    bool flag_c = false;
    if (dc >= 1) {
        if (dc == 1) {
            if (divcs->q[1] != 1) {
                cu.multScalar(T[0], divcs->q[1], true);
            } else {
                cu.copy(T[0]);
            }
        } else {
            std::vector<double> weights(dc);

            for (uint32_t i = 0; i < dc; i++) {
                weights[i] = divcs->q[i + 1];
            }

            cu.evalLinearWSumMutable(dc, T, weights);
            cu.rescale();
        }

        // adds the free term (at x^0)
        cu.addScalar(divcs->q.front() / 2);
        // Need to reduce levels up to the level of T2[m-1].
        cu.dropToLevel(T2[m - 1].getLevel());

        flag_c = true;
    }
    if constexpr (PRINT) {
        std::cout << "cu cheby m=" << m << " " << cu.getLevel() << std::endl;
        for (auto& i : cu.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    /*
    // Evaluate c at u
    Ciphertext<DCRTPoly> cu;
    uint32_t dc = lbcrypto::Degree(divcs->q);
    bool flag_c = false;
    if (dc >= 1) {
        if (dc == 1) {
            if (divcs->q[1] != 1) {
                cu = cc->EvalMult(T.front(), divcs->q[1]);
                cc->ModReduceInPlace(cu);
            } else {
                cu = T.front()->Clone();
            }
        } else {
            std::vector<Ciphertext<DCRTPoly>> ctxs(dc);
            std::vector<double> weights(dc);

            for (uint32_t i = 0; i < dc; i++) {
                ctxs[i] = T[i];
                weights[i] = divcs->q[i + 1];
            }

            cu = cc->EvalLinearWSumMutable(ctxs, weights);
        }

        // adds the free term (at x^0)
        cc->EvalAddInPlace(cu, divcs->q.front() / 2);
        // Need to reduce levels up to the level of T2[m-1].
        usint levelDiff = T2[m - 1]->GetLevel() - cu->GetLevel();
        cc->LevelReduceInPlace(cu, nullptr, levelDiff);

        flag_c = true;
    }
    */
    // Evaluate q and s2 at u. If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
    Ciphertext qu(cc);

    if (lbcrypto::Degree(divqr->q) > k) {
        innerEvalChebyshevPS(ctxt, kskEval, qu, divqr->q, k, m - 1, T, T2);
    } else {
        // dq = k from construction
        // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        auto qcopy = divqr->q;
        qcopy.resize(k);

        if (lbcrypto::Degree(qcopy) > 0) {
            std::vector<double> weights(lbcrypto::Degree(qcopy));

            for (uint32_t i = 0; i < lbcrypto::Degree(qcopy); i++) {
                weights[i] = divqr->q[i + 1];
            }

            qu.evalLinearWSumMutable(lbcrypto::Degree(qcopy), T, weights);
            qu.rescale();
            // the highest order coefficient will always be a power of two up to 2^{m-1} because q is "monic" but the Chebyshev rule adds a factor of 2
            // we don't need to increase the depth by multiplying the highest order coefficient, but instead checking and summing, since we work with m <= 4.
            Ciphertext sum(cc);
            sum.copy(T[k - 1]);
            for (uint32_t i = 0; i < log2(divqr->q.back()); i++) {
                sum.add(sum);
            }
            qu.add(sum);
        } else {
            qu.copy(T[k - 1]);
            for (uint32_t i = 0; i < log2(divqr->q.back()); i++) {
                qu.add(qu);
            }
        }
        // adds the free term (at x^0)
        qu.addScalar(divqr->q.front() / 2);
        // The number of levels of qu is the same as the number of levels of T[k-1] or T[k-1] + 1.
        // No need to reduce it to T2[m-1] because it only reaches here when m = 2.
    }
    if constexpr (PRINT) {
        std::cout << "qu cheby m=" << m << " " << qu.getLevel() << std::endl;
        for (auto& i : qu.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    /*
    // Evaluate q and s2 at u. If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
    Ciphertext<DCRTPoly> qu;

    if (Degree(divqr->q) > k) {
        qu = InnerEvalChebyshevPS(x, divqr->q, k, m - 1, T, T2);
    } else {
        // dq = k from construction
        // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        auto qcopy = divqr->q;
        qcopy.resize(k);
        if (Degree(qcopy) > 0) {
            std::vector<Ciphertext<DCRTPoly>> ctxs(Degree(qcopy));
            std::vector<double> weights(Degree(qcopy));

            for (uint32_t i = 0; i < Degree(qcopy); i++) {
                ctxs[i] = T[i];
                weights[i] = divqr->q[i + 1];
            }

            qu = cc->EvalLinearWSumMutable(ctxs, weights);
            // the highest order coefficient will always be a power of two up to 2^{m-1} because q is "monic" but the Chebyshev rule adds a factor of 2
            // we don't need to increase the depth by multiplying the highest order coefficient, but instead checking and summing, since we work with m <= 4.
            Ciphertext<DCRTPoly> sum = T[k - 1]->Clone();
            for (uint32_t i = 0; i < log2(divqr->q.back()); i++) {
                sum = cc->EvalAdd(sum, sum);
            }
            cc->EvalAddInPlace(qu, sum);
        } else {
            Ciphertext<DCRTPoly> sum = T[k - 1]->Clone();
            for (uint32_t i = 0; i < log2(divqr->q.back()); i++) {
                sum = cc->EvalAdd(sum, sum);
            }
            qu = sum;
        }

        // adds the free term (at x^0)
        cc->EvalAddInPlace(qu, divqr->q.front() / 2);
        // The number of levels of qu is the same as the number of levels of T[k-1] or T[k-1] + 1.
        // No need to reduce it to T2[m-1] because it only reaches here when m = 2.
    }
    */
    Ciphertext su(cc);

    if (lbcrypto::Degree(s2) > k) {
        innerEvalChebyshevPS(ctxt, kskEval, su, s2, k, m - 1, T, T2);
    } else {
        // ds = k from construction
        // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        auto scopy = s2;
        scopy.resize(k);
        if (lbcrypto::Degree(scopy) > 0) {
            std::vector<double> weights(lbcrypto::Degree(scopy));

            for (uint32_t i = 0; i < lbcrypto::Degree(scopy); i++) {
                weights[i] = s2[i + 1];
            }

            su.evalLinearWSumMutable(lbcrypto::Degree(scopy), T, weights);
            su.rescale();
            // the highest order coefficient will always be 1 because s2 is monic.
            su.add(T[k - 1]);
        } else {
            su.copy(T[k - 1]);
        }

        // adds the free term (at x^0)
        su.addScalar(s2.front() / 2);
        // The number of levels of su is the same as the number of levels of T[k-1] or T[k-1] + 1. Need to reduce it to T2[m-1] + 1.
        // su = cc->LevelReduce(su, nullptr, su->GetElements()[0].GetNumOfElements() - Lm + 1) ;
        su.dropToLevel(su.getLevel() - 1);
    }
    if constexpr (PRINT) {
        std::cout << "su cheby m=" << m << " " << su.getLevel() << std::endl;
        for (auto& i : su.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    if (flag_c) {
        out.add(T2[m - 1], cu);
    } else {
        out.addScalar(T2[m - 1], divcs->q.front() / 2);
    }
    cudaDeviceSynchronize();
    out.mult(qu, kskEval, true);
    out.add(su);

    /*
    Ciphertext<DCRTPoly> su;

    if (Degree(s2) > k) {
        su = InnerEvalChebyshevPS(x, s2, k, m - 1, T, T2);
    }
    else {
        // ds = k from construction
        // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        auto scopy = s2;
        scopy.resize(k);
        if (Degree(scopy) > 0) {
            std::vector<Ciphertext<DCRTPoly>> ctxs(Degree(scopy));
            std::vector<double> weights(Degree(scopy));

            for (uint32_t i = 0; i < Degree(scopy); i++) {
                ctxs[i]    = T[i];
                weights[i] = s2[i + 1];
            }

            su = cc->EvalLinearWSumMutable(ctxs, weights);
            // the highest order coefficient will always be 1 because s2 is monic.
            cc->EvalAddInPlace(su, T[k - 1]);
        }
        else {
            su = T[k - 1]->Clone();
        }

        // adds the free term (at x^0)
        cc->EvalAddInPlace(su, s2.front() / 2);
        // The number of levels of su is the same as the number of levels of T[k-1] or T[k-1] + 1. Need to reduce it to T2[m-1] + 1.
        // su = cc->LevelReduce(su, nullptr, su->GetElements()[0].GetNumOfElements() - Lm + 1) ;
        cc->LevelReduceInPlace(su, nullptr);
    }

    Ciphertext<DCRTPoly> result;

    if (flag_c) {
        result = cc->EvalAdd(T2[m - 1], cu);
    }
    else {
        result = cc->EvalAdd(T2[m - 1], divcs->q.front() / 2);
    }

    result = cc->EvalMult(result, qu);
    cc->ModReduceInPlace(result);

    cc->EvalAddInPlace(result, su);

    return result;
    */
    if constexpr (PRINT) {
        std::cout << "out cheby m=" << m << " " << out.getLevel() << std::endl;
        for (auto& i : out.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
}

/**
 * Adaptation of OpenFHE's implementation.
 */
void evalChebyshevSeries(Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, std::vector<double>& coefficients,
                         double lower_bound, double upper_bound) {
    FIDESlib::CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    /*
    Ciphertext<DCRTPoly> AdvancedSHECKKSRNS::EvalChebyshevSeriesPS(ConstCiphertext<DCRTPoly> x,
const std::vector<double>& coefficients, double a, double b) const {
    */

    uint32_t n = lbcrypto::Degree(coefficients);
    std::vector<double> f2 = coefficients;
    f2.resize(n + 1);
    /*
     uint32_t n = Degree(coefficients);
    std::vector<double> f2 = coefficients;
    // Make sure the coefficients do not have the zero dominant terms
    if (coefficients[coefficients.size() - 1] == 0)
        f2.resize(n + 1);
    */

    std::vector<uint32_t> degs = lbcrypto::ComputeDegreesPS(n);
    uint32_t k = degs[0];
    uint32_t m = degs[1];
    /*
    std::vector<uint32_t> degs = ComputeDegreesPS(n);
    uint32_t k                 = degs[0];
    uint32_t m                 = degs[1];
    // std::cerr << "\n Degree: n = " << n << ", k = " << k << ", m = " << m << std::endl;
    */

    Context& cc = ctxt.cc;
    std::vector<Ciphertext> T;
    assert((lower_bound - std::round(lower_bound) < 1e-10) && (upper_bound - std::round(upper_bound) < 1e-10) &&
           (std::round(lower_bound) == -1) && (std::round(upper_bound) == 1));
    T.emplace_back(cc);
    T[0].copy(ctxt);
    for (uint32_t i = 1; i < k; ++i)
        T.emplace_back(cc);
    /*
    // computes linear transformation y = -1 + 2 (x-a)/(b-a)
    // consumes one level when a <> -1 && b <> 1
    auto cc = x->GetCryptoContext();
    std::vector<Ciphertext<DCRTPoly>> T(k);
    if ((a - std::round(a) < 1e-10) && (b - std::round(b) < 1e-10) && (std::round(a) == -1) && (std::round(b) == 1)) {
        // no linear transformation is needed if a = -1, b = 1
        // T_1(y) = y
        T[0] = x->Clone();
    }
    else {
        // linear transformation is needed
        double alpha = 2 / (b - a);
        double beta  = 2 * a / (b - a);

        T[0] = cc->EvalMult(x, alpha);
        cc->ModReduceInPlace(T[0]);
        cc->EvalAddInPlace(T[0], -1.0 - beta);
    }
    */
    //Ciphertext y(cc);
    //y.copy(T[0]);
    for (uint32_t i = 2; i <= k; i++) {
        // if i is a power of two
        cudaDeviceSynchronize();
        if (!(i & (i - 1))) {
            // compute T_{2i}(y) = 2*T_i(y)^2 - 1
            T[i - 1].square(T[i / 2 - 1], keySwitchingKey, false);

            T[i - 1].add(T[i - 1]);  // TODO times two optimized operation
            T[i - 1].rescale();
            T[i - 1].addScalar(-1.0);
        } else {
            // non-power of 2
            if (i % 2 == 1) {
                // if i is odd
                // compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
                T[i - 1].mult(T[i / 2 - 1], T[i / 2], keySwitchingKey, false);
                T[i - 1].add(T[i - 1]);
                T[i - 1].rescale();
                T[i - 1].sub(T[0]);
            } else {
                // i is even but not power of 2
                // compute T_{2i}(y) = 2*T_i(y)^2 - 1
                T[i - 1].square(T[i / 2 - 1], keySwitchingKey, false);
                T[i - 1].add(T[i - 1]);
                T[i - 1].rescale();
                T[i - 1].addScalar(-1.0);
            }
        }
    }

    if constexpr (PRINT) {
        for (size_t j = 0; j < k; ++j) {
            for (auto& i : T[j].c0.GPU.at(0).limb) {
                SWITCH(i, printThisLimb(1));
            }
            std::cout << std::endl;
        }
    }
    //cudaDeviceSynchronize();
    /*
    Ciphertext<DCRTPoly> y = T[0]->Clone();

    // Computes Chebyshev polynomials up to degree k
    // for y: T_1(y) = y, T_2(y), ... , T_k(y)
    // uses binary tree multiplication
    for (uint32_t i = 2; i <= k; i++) {
        // if i is a power of two
        if (!(i & (i - 1))) {
            // compute T_{2i}(y) = 2*T_i(y)^2 - 1
            auto square = cc->EvalSquare(T[i / 2 - 1]);
            T[i - 1] = cc->EvalAdd(square, square);
            cc->ModReduceInPlace(T[i - 1]);
            cc->EvalAddInPlace(T[i - 1], -1.0);
        } else {
            // non-power of 2
            if (i % 2 == 1) {
                // if i is odd
                // compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
                auto prod = cc->EvalMult(T[i / 2 - 1], T[i / 2]);
                T[i - 1] = cc->EvalAdd(prod, prod);

                cc->ModReduceInPlace(T[i - 1]);
                cc->EvalSubInPlace(T[i - 1], y);
            } else {
                // i is even but not power of 2
                // compute T_{2i}(y) = 2*T_i(y)^2 - 1
                auto square = cc->EvalSquare(T[i / 2 - 1]);
                T[i - 1] = cc->EvalAdd(square, square);
                cc->ModReduceInPlace(T[i - 1]);
                cc->EvalAddInPlace(T[i - 1], -1.0);
            }
        }
    }
    */
    if (cc.rescaleTechnique == Context::FIXEDMANUAL) {
        for (size_t i = 1; i < k; i++) {
            T[i - 1].dropToLevel(T[k - 1].getLevel());
        }
    } else {
        for (size_t i = 1; i < k; i++) {
            if (!T[i - 1].adjustForAddOrSub(T[k - 1])) {
                if (!T[k - 1].adjustForAddOrSub(T[i - 1])) {
                    assert("false");
                    std::cerr << "PANIC" << std::endl;
                }
            }
            //algo->AdjustLevelsAndDepthInPlace(T[i - 1], T[k - 1]);
        }
    }
    /*
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(T[k - 1]->GetCryptoParameters());

    auto algo = cc->GetScheme();

    if (cryptoParams->GetScalingTechnique() == FIXEDMANUAL) {
        // brings all powers of x to the same level
        for (size_t i = 1; i < k; i++) {
            usint levelDiff = T[k - 1]->GetLevel() - T[i - 1]->GetLevel();
            cc->LevelReduceInPlace(T[i - 1], nullptr, levelDiff);
        }
    } else {
        for (size_t i = 1; i < k; i++) {
            algo->AdjustLevelsAndDepthInPlace(T[i - 1], T[k - 1]);
        }
    }
     */

    std::vector<Ciphertext> T2;
    T2.emplace_back(cc);
    T2[0].copy(T.back());
    // Compute the Chebyshev polynomials T_k(y), T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
    for (uint32_t i = 1; i < m; i++) {
        T2.emplace_back(cc);
        T2[i].square(T2[i - 1], keySwitchingKey, false);
        T2[i].add(T2[i]);
        T2[i].rescale();
        T2[i].addScalar(-1.0);
    }
    if constexpr (PRINT) {
        for (size_t j = 0; j < m; ++j) {
            for (auto& i : T2[j].c0.GPU.at(0).limb) {
                SWITCH(i, printThisLimb(1));
            }
            std::cout << std::endl;
        }
    }
    cudaDeviceSynchronize();
    /*
    std::vector<Ciphertext<DCRTPoly>> T2(m);
    // Compute the Chebyshev polynomials T_k(y), T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
    // T2[0] is used as a placeholder
    T2.front() = T.back();
    for (uint32_t i = 1; i < m; i++) {
        auto square = cc->EvalSquare(T2[i - 1]);
        T2[i] = cc->EvalAdd(square, square);
        cc->ModReduceInPlace(T2[i]);
        cc->EvalAddInPlace(T2[i], -1.0);
    }
    */

    // computes T_{k(2*m - 1)}(y)
    Ciphertext T2km1(cc);
    if (cc.rescaleTechnique == Context::FIXEDMANUAL) {
        T2[0].dropToLevel(T2[1].getLevel());
    } else {
        if (!T2[0].adjustForAddOrSub(T2[1])) {
            assert(false);
        }
    }
    T2km1.copy(T2[0]);

    for (uint32_t i = 1; i < m; i++) {
        // compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
        T2km1.mult(T2[i], keySwitchingKey, false);
        T2km1.add(T2km1);
        T2km1.rescale();
        T2km1.sub(T2[0]);
        cudaDeviceSynchronize();
    }
    if constexpr (PRINT) {
        std::cout << "T2kmi cheby " << T2km1.getLevel() << std::endl;
        for (auto& i : T2km1.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    /*
    // computes T_{k(2*m - 1)}(y)
    auto T2km1 = T2.front();
    for (uint32_t i = 1; i < m; i++) {
        // compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
        auto prod = cc->EvalMult(T2km1, T2[i]);
        T2km1 = cc->EvalAdd(prod, prod);
        cc->ModReduceInPlace(T2km1);
        cc->EvalSubInPlace(T2km1, T2.front());
    }

    // We also need to reduce the number of levels of T[k-1] and of T2[0] by another level.
    //  cc->LevelReduceInPlace(T[k-1], nullptr);
    //  cc->LevelReduceInPlace(T2.front(), nullptr);
    */

    /// This is left as is ///
    // Compute k*2^{m-1}-k because we use it a lot
    uint32_t k2m2k = k * (1 << (m - 1)) - k;
    // Add T^{k(2^m - 1)}(y) to the polynomial that has to be evaluated
    f2.resize(2 * k2m2k + k + 1, 0.0);
    f2.back() = 1;
    // Divide f2 by T^{k*2^{m-1}}
    std::vector<double> Tkm(int32_t(k2m2k + k) + 1, 0.0);
    Tkm.back() = 1;
    auto divqr = lbcrypto::LongDivisionChebyshev(f2, Tkm);
    // Subtract x^{k(2^{m-1} - 1)} from r
    std::vector<double> r2 = divqr->r;
    if (int32_t(k2m2k - lbcrypto::Degree(divqr->r)) <= 0) {
        r2[int32_t(k2m2k)] -= 1;
        r2.resize(lbcrypto::Degree(r2) + 1);
    } else {
        r2.resize(int32_t(k2m2k + 1), 0.0);
        r2.back() = -1;
    }
    // Divide r2 by q
    auto divcs = lbcrypto::LongDivisionChebyshev(r2, divqr->q);

    // Add x^{k(2^{m-1} - 1)} to s
    std::vector<double> s2 = divcs->r;
    s2.resize(int32_t(k2m2k + 1), 0.0);
    s2.back() = 1;
    /// END This is left as is ///

    // Evaluate c at u
    Ciphertext cu(cc);
    uint32_t dc = lbcrypto::Degree(divcs->q);
    bool flag_c = false;
    if (dc >= 1) {
        if (dc == 1) {
            if (divcs->q[1] != 1) {
                cu.multScalar(T.front(), divcs->q[1], true);
            } else {
                cu.copy(T.front());
            }
        } else {
            std::vector<Ciphertext>& ctxs = T;
            std::vector<double> weights(dc);

            for (uint32_t i = 0; i < dc; i++) {
                weights[i] = divcs->q[i + 1];
            }

            cu.evalLinearWSumMutable(dc, ctxs, weights);
            cu.rescale();
        }

        cu.addScalar(divcs->q.front() / 2);

        flag_c = true;
    }
    if constexpr (PRINT) {
        std::cout << "cu cheby " << cu.getLevel() << std::endl;
        for (auto& i : cu.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    /*
    // Evaluate c at u
    Ciphertext<DCRTPoly> cu;
    uint32_t dc = Degree(divcs->q);
    bool flag_c = false;
    if (dc >= 1) {
        if (dc == 1) {
            if (divcs->q[1] != 1) {
                cu = cc->EvalMult(T.front(), divcs->q[1]);
                cc->ModReduceInPlace(cu);
            } else {
                cu = T.front()->Clone();
            }
        } else {
            std::vector<Ciphertext<DCRTPoly>> ctxs(dc);
            std::vector<double> weights(dc);

            for (uint32_t i = 0; i < dc; i++) {
                ctxs[i] = T[i];
                weights[i] = divcs->q[i + 1];
            }

            cu = cc->EvalLinearWSumMutable(ctxs, weights);
        }

        // adds the free term (at x^0)
        cc->EvalAddInPlace(cu, divcs->q.front() / 2);
        // Need to reduce levels to the level of T2[m-1].
        //    usint levelDiff = y->GetLevel() - cu->GetLevel() + ceil(log2(k)) + m - 1;
        //    cc->LevelReduceInPlace(cu, nullptr, levelDiff);

        flag_c = true;
    }
    */
    Ciphertext qu(cc);
    // Evaluate q and s2 at u. If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
    if (lbcrypto::Degree(divqr->q) > k) {
        innerEvalChebyshevPS(ctxt, keySwitchingKey, qu, divqr->q, k, m - 1, T, T2);
    } else {
        // dq = k from construction
        // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        auto qcopy = divqr->q;
        qcopy.resize(k);
        if (lbcrypto::Degree(qcopy) > 0) {
            std::vector<Ciphertext> ctxs;
            std::vector<double> weights(lbcrypto::Degree(qcopy));

            for (uint32_t i = 0; i < lbcrypto::Degree(qcopy); i++) {
                ctxs.emplace_back(cc);
                ctxs[i].copy(T[i]);
                weights[i] = divqr->q[i + 1];
            }

            qu.evalLinearWSumMutable(ctxs.size(), ctxs, weights);
            // the highest order coefficient will always be 2 after one division because of the Chebyshev division rule
            Ciphertext sum(cc);
            /*
            sum.add(T[k - 1], T[k - 1]);
            qu.add(sum);
             */
            qu.add(T[k - 1]);
            qu.add(T[k - 1]);
        } else {
            qu.copy(T[k - 1]);

            for (uint32_t i = 1; i < divqr->q.back(); i++) {
                qu.add(T[k - 1]);
            }
        }

        // adds the free term (at x^0)
        qu.addScalar(divqr->q.front() / 2);
        // The number of levels of qu is the same as the number of levels of T[k-1] + 1.
        // Will only get here when m = 2, so the number of levels of qu and T2[m-1] will be the same.
    }
    if constexpr (PRINT) {
        std::cout << "qu cheby " << qu.getLevel() << std::endl;
        for (auto& i : qu.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    /*

    // Evaluate q and s2 at u. If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
    Ciphertext<DCRTPoly> qu;

    if (Degree(divqr->q) > k) {
        qu = InnerEvalChebyshevPS(x, divqr->q, k, m - 1, T, T2);
    } else {
        // dq = k from construction
        // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        auto qcopy = divqr->q;
        qcopy.resize(k);
        if (Degree(qcopy) > 0) {
            std::vector<Ciphertext<DCRTPoly>> ctxs(Degree(qcopy));
            std::vector<double> weights(Degree(qcopy));

            for (uint32_t i = 0; i < Degree(qcopy); i++) {
                ctxs[i] = T[i];
                weights[i] = divqr->q[i + 1];
            }

            qu = cc->EvalLinearWSumMutable(ctxs, weights);
            // the highest order coefficient will always be 2 after one division because of the Chebyshev division rule
            Ciphertext<DCRTPoly> sum = cc->EvalAdd(T[k - 1], T[k - 1]);
            cc->EvalAddInPlace(qu, sum);
        } else {
            qu = T[k - 1]->Clone();

            for (uint32_t i = 1; i < divqr->q.back(); i++) {
                cc->EvalAddInPlace(qu, T[k - 1]);
            }
        }

        // adds the free term (at x^0)
        cc->EvalAddInPlace(qu, divqr->q.front() / 2);
        // The number of levels of qu is the same as the number of levels of T[k-1] + 1.
        // Will only get here when m = 2, so the number of levels of qu and T2[m-1] will be the same.
    }
*/
    Ciphertext su(cc);
    if (lbcrypto::Degree(s2) > k) {
        innerEvalChebyshevPS(ctxt, keySwitchingKey, su, s2, k, m - 1, T, T2);
    } else {
        // ds = k from construction
        // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        auto scopy = s2;
        scopy.resize(k);
        if (lbcrypto::Degree(scopy) > 0) {
            std::vector<Ciphertext> ctxs;
            std::vector<double> weights(lbcrypto::Degree(scopy));

            for (uint32_t i = 0; i < lbcrypto::Degree(scopy); i++) {
                ctxs.emplace_back(cc);
                ctxs[i].copy(T[i]);
                weights[i] = s2[i + 1];
            }

            su.evalLinearWSumMutable(ctxs.size(), ctxs, weights);
            // the highest order coefficient will always be 1 because s2 is monic.
            su.add(T[k - 1]);
        } else {
            su.copy(T[k - 1]);
        }

        // adds the free term (at x^0)
        su.addScalar(s2.front() / 2);
        // The number of levels of su is the same as the number of levels of T[k-1] + 1.
        // Will only get here when m = 2, so need to reduce the number of levels by 1.
    }
    if constexpr (PRINT) {
        std::cout << "su cheby " << su.getLevel() << std::endl;
        for (auto& i : su.c0.GPU.at(0).limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
    /*
    Ciphertext<DCRTPoly> su;

    if (Degree(s2) > k) {
        su = InnerEvalChebyshevPS(x, s2, k, m - 1, T, T2);
    } else {
        // ds = k from construction
        // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        auto scopy = s2;
        scopy.resize(k);
        if (Degree(scopy) > 0) {
            std::vector<Ciphertext<DCRTPoly>> ctxs(Degree(scopy));
            std::vector<double> weights(Degree(scopy));

            for (uint32_t i = 0; i < Degree(scopy); i++) {
                ctxs[i] = T[i];
                weights[i] = s2[i + 1];
            }

            su = cc->EvalLinearWSumMutable(ctxs, weights);
            // the highest order coefficient will always be 1 because s2 is monic.
            cc->EvalAddInPlace(su, T[k - 1]);
        } else {
            su = T[k - 1];
        }

        // adds the free term (at x^0)
        cc->EvalAddInPlace(su, s2.front() / 2);
        // The number of levels of su is the same as the number of levels of T[k-1] + 1.
        // Will only get here when m = 2, so need to reduce the number of levels by 1.
    }

    // Reduce number of levels of su to number of levels of T2km1.
    //  cc->LevelReduceInPlace(su, nullptr);
     */

    if (flag_c) {
        ctxt.add(T2[m - 1], cu);
    } else {
        ctxt.addScalar(T2[m - 1], divcs->q.front() / 2);
    }
    if constexpr (PRINT)
        std::cout << "Last steps cheby " << T2[m - 1].getLevel() << " " << su.getLevel() << std::endl;

    cudaDeviceSynchronize();
    ctxt.mult(qu, keySwitchingKey, true);
    ctxt.add(su);
    ctxt.sub(T2km1);
    /*

    Ciphertext<DCRTPoly> result;

    if (flag_c) {
        result = cc->EvalAdd(T2[m - 1], cu);
    } else {
        result = cc->EvalAdd(T2[m - 1], divcs->q.front() / 2);
    }

    result = cc->EvalMult(result, qu);
    cc->ModReduceInPlace(result);

    cc->EvalAddInPlace(result, su);
    cc->EvalSubInPlace(result, T2km1);

    return result;
    */
    cudaDeviceSynchronize();
}

void applyDoubleAngleIterations(Ciphertext& ctxt, int its, const KeySwitchingKey& kskEval) {
    FIDESlib::CudaNvtxRange r_(std::string{std::source_location::current().function_name()});
    Context& cc = ctxt.cc;
    int32_t r = its;
    //std::cout << "Its: " << its << std::endl;
    for (int32_t j = 1; j < r + 1; j++) {
        ctxt.square(kskEval, false);
        ctxt.add(ctxt);
        double scalar = -1.0 / std::pow((2.0 * M_PI), std::pow(2.0, j - r));
        ctxt.addScalar(scalar);
        if (cc.rescaleTechnique == Context::FIXEDMANUAL)
            ctxt.rescale();
        cudaDeviceSynchronize();
    }
}