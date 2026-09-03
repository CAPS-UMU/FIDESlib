//
// Created by carlosad on 12/11/24.
//

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CudaUtils.cuh"
#include "ElemenwiseBatchKernels.cuh"
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
	ContextData& cc        = ctxtEnc.cc;

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

enum PS_MODE { FULL, RECORD, REMAINING };

struct PSRecord {
	std::vector<std::vector<double>> constants;
	std::vector<int> target_level;
	std::vector<Ciphertext> rev_result_ciphertexts;
};

// FIDESlib bit-compat: direct transcription of OpenFHE InnerEvalChebyshevPS.
void innerEvalChebyshevPS(const Ciphertext& x,
                          Ciphertext& out,
                          const std::vector<double>& coefficients,
                          const uint32_t k,
                          uint32_t m,
                          const std::vector<Ciphertext*>& T,
                          const std::vector<Ciphertext*>& T2,
                          const std::vector<Ciphertext*>& T2_norescale,
                          int level_offset   = 0,
                          PS_MODE mode       = FULL,
                          PSRecord* psRecord = nullptr) {
	FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });
	const FIDESlib::CKKS::Context& cc_ = x.cc_;
	//ContextData& cc                    = x.cc;

	// Compute k*2^{m-1}-k because we use it a lot
	uint32_t k2m2k = k * (1 << (m - 1)) - k;

	// Divide coefficients by T^{k*2^{m-1}}
	std::vector<double> Tkm(k2m2k + k + 1);
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
	//auto cc    = x->GetCryptoContext();

	Ciphertext su_(cc_), qu_(cc_);
	std::unique_ptr<Ciphertext> su_tmp, qu_tmp;
	Ciphertext *su = nullptr, *qu = nullptr;
	Ciphertext& cu = out;

	// Evaluate q and s2 at u.
	// If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
	if (lbcrypto::Degree(divqr->q) > k) {
		innerEvalChebyshevPS(x, qu_, divqr->q, k, m - 1, T, T2, T2_norescale, level_offset, mode, psRecord);
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
		} else if (mode == FULL) {
			qu_.evalPartialLinearWSumWithBias(T,
			                                  divqr->q,
			                                  divqr->q.front() / 2.0,
			                                  divqr->q.size() - 1,
			                                  (int)T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1) - level_offset);
			qu = &qu_;
		}
	}

	// Add x^{k(2^{m-1} - 1)} to s
	auto& s2 = divcs->r;
	s2.resize(k2m2k + 1);
	s2.back() = 1;

	if (lbcrypto::Degree(s2) > k) {
		innerEvalChebyshevPS(x, su_, s2, k, m - 1, T, T2, T2_norescale, level_offset + 1, mode, psRecord);
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
		} else if (mode == FULL) {
			su_.evalPartialLinearWSumWithBias(T,
			                                  s2,
			                                  s2.front() / 2.0,
			                                  (uint32_t)s2.size() - 1,
			                                  (int)T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1) - level_offset - 1,
			                                  false);
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
			//if (T2_norescale[m - 1]->NoiseLevel == 2 && psRecord->rev_result_ciphertexts.back().NoiseLevel == 2 && psRecord->rev_result_ciphertexts.back().
			//	getLevel() == T2_norescale[m - 1]->getLevel()) {
			//	cu.addMutable(psRecord->rev_result_ciphertexts.back(), (const Ciphertext&)*T2_norescale[m - 1]);
			//	cu.rescale();
			//} else {
			psRecord->rev_result_ciphertexts.back().rescale();
			cu.addMutable(psRecord->rev_result_ciphertexts.back(), (const Ciphertext&)*T2[m - 1]);
			//}
			psRecord->rev_result_ciphertexts.pop_back();
		} else if (mode == FULL) {
			cu.evalPartialLinearWSumWithBias(T,
			                                 divcs->q,
			                                 divcs->q.front() / 2.0,
			                                 divcs->q.size() - 1,
			                                 (int)T2[m - 1]->getLevel() + (T2[m - 1]->NoiseLevel == 1) - level_offset,
			                                 true);
			cu.add(*T2[m - 1]);
		}
	} else {
		if (mode != RECORD)
			cu.addScalar(*T2[m - 1], divcs->q.front() / 2.0);
	}

	if (mode != RECORD) {
		//std::cout << "cu " << cu.getLevel() << " " << cu.NoiseLevel << std::endl;
		//std::cout << "qu " << qu.getLevel() << " " << qu.NoiseLevel << std::endl;
		//std::cout << "su " << su.getLevel() << " " << su.NoiseLevel << std::endl;
		//return;
		cu.multMutable(*qu);
		cu.addMutable(*su);
	}
}

constexpr bool SEQUENTIAL_PS = false;
/**
 * Adaptation of OpenFHE's implementation.
 */
void FIDESlib::CKKS::evalChebyshevSeries(Ciphertext& ctxt, std::vector<double>& coefficients, double lower_bound, double upper_bound) {
	FIDESlib::CudaNvtxRange r(std::string{ sc::current().function_name() });

	auto degree = lbcrypto::Degree(coefficients);
	auto degs   = lbcrypto::ComputeDegreesPS(degree);
	uint32_t k  = degs[0];
	uint32_t m  = degs[1];

	// computes linear transformation y = -1 + 2 (x-a)/(b-a)
	// consumes one level when a <> -1 && b <> 1
	Ciphertext& x = ctxt;
	double& a     = lower_bound;
	double& b     = upper_bound;
	auto cc       = ctxt.cc_;
	std::vector<Ciphertext> T;
	T.emplace_back(cc);
	T[0].copy(x);
	if (lbcrypto::IsNotEqualNegOne(a) || lbcrypto::IsNotEqualOne(b)) {
		// linear transformation is needed
		double alpha = 2 / (b - a);
		double beta  = a * alpha;

		if (lbcrypto::IsNotEqualOne(alpha)) {
			T[0].multScalar(x, alpha);
		}
		if (lbcrypto::IsNotEqualZero(-1.0 - beta))
			T[0].addScalar(-1.0 - beta);

		if (T[0].NoiseLevel == 2) {
			T[0].rescaleInternal();
		}
	}

	// Computes Chebyshev polynomials up to degree k
	// for y: T_1(y) = y, T_2(y), ... , T_k(y)
	// uses binary tree multiplication
	if (0) {
		// Less copies path (changes bit behaviour) (measured about 1% bootstrap speedup)
		for (uint32_t i = 2; i <= k; ++i) {
			T.emplace_back(cc);
			if (i & 0x1) {
				// if i is odd
				// compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
				T[i - 1].multMutable(T[i / 2 - 1], T[i / 2]);
				T[i - 1].add(T[i - 1]);
				T[i - 1].rescaleInternal();
				T[i - 1].subMutable(T[0]);
			} else {
				// compute T_{2i}(y) = 2*T_i(y)^2 - 1
				T[i - 1].square(T[i / 2 - 1]);
				T[i - 1].addMutable(T[i - 1]);
				T[i - 1].addScalar(-1.0);
				T[i - 1].rescaleInternal();
			}
		}
	} else {
		for (uint32_t i = 2; i <= k; ++i) {
			T.emplace_back(cc);
			if (i & 0x1) {
				// if i is odd
				// compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
				T[i - 1].mult(T[i / 2 - 1], T[i / 2]);
				T[i - 1].add(T[i - 1]);
				T[i - 1].rescaleInternal();
				T[i - 1].sub(T[0]);
			} else {
				// compute T_{2i}(y) = 2*T_i(y)^2 - 1
				T[i - 1].square(T[i / 2 - 1]);

				T[i - 1].add(T[i - 1]);
				T[i - 1].addScalar(-1.0);
				T[i - 1].rescaleInternal();
			}
		}
	}
	std::vector<Ciphertext> T2, T2_norescale;
	// T2[0] is used as a placeholder
	T2.emplace_back(cc);
	//T2_norescale.emplace_back(cc);
	//T2[0].copy(T.back());

	// computes T_{k(2*m - 1)}(y)
	Ciphertext T2km1(cc);
	T2km1.copy(T.back());

	for (uint32_t i = 1; i < m; ++i) {
		T2.emplace_back(cc);
		//T2_norescale.emplace_back(cc);
		// Compute the Chebyshev polynomials T_k(y), T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
		T2[i].square(i - 1 == 0 ? T.back() : T2[i - 1]);
		T2[i].add(T2[i]);
		T2[i].addScalar(-1.0);
		T2[i].rescale();
		//T2_norescale[i].copy(T2[i]);
		if (T2[i].NoiseLevel == 2)
			T2[i].rescaleInternal();
		// compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
		T2km1.mult(T2[i]);
		T2km1.add(T2km1);
		T2km1.rescale();
		T2km1.sub(T.back());
	}

	std::vector<Ciphertext*> T_ptr, T2_ptr, T2_norescale_ptr;
	for (auto& i : T)
		T_ptr.push_back(&i);
	for (auto& i : T2)
		T2_ptr.push_back(&i);
	for (auto& i : T2_norescale)
		T2_norescale_ptr.push_back(&i);

	// Compute k*2^{m-1}-k because we use it a lot
	uint32_t k2m2k = k * (1 << (m - 1)) - k;

	// Add T^{k(2^m - 1)}(y) to the polynomial that has to be evaluated
	auto f2 = coefficients;
	f2.resize(lbcrypto::Degree(f2) + 1);
	f2.resize(2 * k2m2k + k + 1);
	f2.back() = 1;

	if (SEQUENTIAL_PS) {
		innerEvalChebyshevPS(T[0], ctxt, f2, k, m, T_ptr, T2_ptr, T2_norescale_ptr);
	} else {
		PSRecord record;
		innerEvalChebyshevPS(T[0], ctxt, f2, k, m, T_ptr, T2_ptr, T2_norescale_ptr, 0, RECORD, &record);
		auto result_ciphertexts = Ciphertext::evalMultiplePartialLinearWSumWithBias(T_ptr, record.constants, record.target_level, false);
		record.rev_result_ciphertexts.reserve(result_ciphertexts.size());
		for (auto i = 0u; i < result_ciphertexts.size(); ++i) {
			record.rev_result_ciphertexts.emplace_back(std::move(result_ciphertexts[result_ciphertexts.size() - 1 - i]));
		}
		innerEvalChebyshevPS(T[0], ctxt, f2, k, m, T_ptr, T2_ptr, T2_norescale_ptr, 0, REMAINING, &record);
	}
	ctxt.rescale();
	ctxt.subMutable(T2km1);
}

void applyDoubleAngleIterations(Ciphertext& ctxt, int its, const KeySwitchingKey& kskEval) {
	FIDESlib::CudaNvtxRange r_(std::string{ sc::current().function_name() });
	ContextData& cc = ctxt.cc;
	int32_t r       = its;
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
