#include "engine/cuda/CudaEngine.hpp"

#include "CryptoContext.hpp"
#include "engine/EngineCommon.hpp"

// GPU CKKS implementation headers for the migrated device-side ops.
#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Parameters.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/forwardDefs.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "CudaUtils.cuh"

#include <any>
#include <memory>
#include <openfhe.h> // OPENFHE_THROW in shared precondition checks

namespace fideslib {

namespace {
// The neutral value types store the GPU-resident payload as a shared_ptr inside their `device` slot
// (the device object is not copyable). These unwrap it; an empty slot means "not resident", which the
// engine ensures via LoadCiphertext/LoadPlaintext before every use.
std::shared_ptr<FIDESlib::CKKS::Ciphertext> deviceCt(const Ciphertext<DCRTPoly>& ct) {
	return std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct->device);
}

std::shared_ptr<FIDESlib::CKKS::Plaintext> devicePt(const Plaintext& pt) {
	return std::any_cast<std::shared_ptr<FIDESlib::CKKS::Plaintext>>(pt->device);
}
} // namespace

// CKKS prime tables for the GPU context parameters (used by loadContext).
static std::vector<FIDESlib::PrimeRecord> p64{ { .p = 2305843009218281473 },
	{ .p = 2251799661248513 },
	{ .p = 2251799661641729 },
	{ .p = 2251799665180673 },
	{ .p = 2251799682088961 },
	{ .p = 2251799678943233 },
	{ .p = 2251799717609473 },
	{ .p = 2251799710138369 },
	{ .p = 2251799708827649 },
	{ .p = 2251799707385857 },
	{ .p = 2251799713677313 },
	{ .p = 2251799712366593 },
	{ .p = 2251799716691969 },
	{ .p = 2251799714856961 },
	{ .p = 2251799726522369 },
	{ .p = 2251799726129153 },
	{ .p = 2251799747493889 },
	{ .p = 2251799741857793 },
	{ .p = 2251799740416001 },
	{ .p = 2251799746707457 },
	{ .p = 2251799756013569 },
	{ .p = 2251799775805441 },
	{ .p = 2251799763091457 },
	{ .p = 2251799767154689 },
	{ .p = 2251799765975041 },
	{ .p = 2251799770562561 },
	{ .p = 2251799769776129 },
	{ .p = 2251799772266497 },
	{ .p = 2251799775281153 },
	{ .p = 2251799774887937 },
	{ .p = 2251799797432321 },
	{ .p = 2251799787995137 },
	{ .p = 2251799787601921 },
	{ .p = 2251799791403009 },
	{ .p = 2251799789568001 },
	{ .p = 2251799795466241 },
	{ .p = 2251799807131649 },
	{ .p = 2251799806345217 },
	{ .p = 2251799805165569 },
	{ .p = 2251799813554177 },
	{ .p = 2251799809884161 },
	{ .p = 2251799810670593 },
	{ .p = 2251799818928129 },
	{ .p = 2251799816568833 },
	{ .p = 2251799815520257 } };

static std::vector<FIDESlib::PrimeRecord> sp64{ { .p = 2305843009218936833 },
	{ .p = 2305843009220116481 },
	{ .p = 2305843009221820417 },
	{ .p = 2305843009224179713 },
	{ .p = 2305843009225228289 },
	{ .p = 2305843009227980801 },
	{ .p = 2305843009229160449 },
	{ .p = 2305843009229946881 },
	{ .p = 2305843009231650817 },
	{ .p = 2305843009235189761 },
	{ .p = 2305843009240301569 },
	{ .p = 2305843009242923009 },
	{ .p = 2305843009244889089 },
	{ .p = 2305843009245413377 },
	{ .p = 2305843009247641601 } };

Ciphertext<DCRTPoly> CudaEngine::evalNegate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	res_gpu->multScalar(-1.0);
	return result;
}

void CudaEngine::evalNegateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(ct);
	auto ct_gpu = deviceCt(ct);
	ct_gpu->multScalar(-1.0);
}

Ciphertext<DCRTPoly> CudaEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= deviceCt(result);
	auto ct2_gpu				= deviceCt(ct2);
	res_gpu->add(*ct2_gpu);
	return result;
}

void CudaEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(ct1);
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));
	auto res_gpu = deviceCt(ct1);
	auto ct2_gpu = deviceCt(ct2);
	res_gpu->add(*ct2_gpu);
}

Ciphertext<DCRTPoly> CudaEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	ctx.LoadPlaintext(pt);
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	auto pt_gpu					= devicePt(pt);
	res_gpu->addPt(*pt_gpu);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	res_gpu->addScalar(scalar);
	return result;
}

void CudaEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	ctx.LoadCiphertext(ct1);
	ctx.LoadPlaintext(pt);
	auto res_gpu = deviceCt(ct1);
	auto pt_gpu	 = devicePt(pt);
	res_gpu->addPt(*pt_gpu);
}

void CudaEngine::evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) {
	ctx.LoadCiphertext(ct1);
	auto res_gpu = deviceCt(ct1);
	res_gpu->addScalar(scalar);
}

Ciphertext<DCRTPoly> CudaEngine::evalAddMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	if (ciphertexts.empty()) {
		OPENFHE_THROW("EvalAddMany: input ciphertext vector is empty");
	}

	for (const auto& ct : ciphertexts) {
		ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	}

	const size_t inSize = ciphertexts.size();
	const size_t lim	= inSize * 2 - 2;
	std::vector<Ciphertext<DCRTPoly>> ciphertextSumVec;
	ciphertextSumVec.resize(inSize - 1);
	size_t ctrIndex = 0;

	for (size_t i = 0; i < lim; i = i + 2) {
		ciphertextSumVec[ctrIndex++] =
		  ctx.EvalAdd(i < inSize ? ciphertexts[i] : ciphertextSumVec[i - inSize], i + 1 < inSize ? ciphertexts[i + 1] : ciphertextSumVec[i + 1 - inSize]);
	}

	return ciphertextSumVec.back();
}

void CudaEngine::evalAddManyInPlace(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	if (ciphertexts.empty()) {
		OPENFHE_THROW("EvalAddManyInPlace: input ciphertext vector is empty");
	}

	for (const auto& ct : ciphertexts) {
		ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	}

	for (size_t j = 1; j < ciphertexts.size(); j = j * 2) {
		for (size_t i = 0; i < ciphertexts.size(); i = i + 2 * j) {
			if ((i + j) < ciphertexts.size()) {
				if (ciphertexts[i] != nullptr && ciphertexts[i + j] != nullptr) {
					ctx.EvalAddInPlace(ciphertexts[i], ciphertexts[i + j]);
				} else if (ciphertexts[i] == nullptr && ciphertexts[i + j] != nullptr) {
					ciphertexts[i] = ciphertexts[i + j];
				}
			}
		}
	}
}

Ciphertext<DCRTPoly> CudaEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= deviceCt(result);
	auto ct2_gpu				= deviceCt(ct2);
	res_gpu->sub(*ct2_gpu);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	ctx.LoadPlaintext(pt);
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	auto pt_gpu					= devicePt(pt);
	res_gpu->subPt(*pt_gpu);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	ctx.LoadPlaintext(pt);
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	auto pt_gpu					= devicePt(pt);
	res_gpu->multScalar(-1.0);
	res_gpu->addPt(*pt_gpu);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	res_gpu->addScalar(-scalar);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalSub(CryptoContextImpl<DCRTPoly>& ctx, double scalar, const Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	res_gpu->multScalar(-1.0);
	res_gpu->addScalar(scalar);
	res_gpu->multScalar(-1.0);
	return result;
}

void CudaEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(ct1);
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));
	auto res_gpu = deviceCt(ct1);
	auto ct2_gpu = deviceCt(ct2);
	res_gpu->sub(*ct2_gpu);
}

void CudaEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) {
	ctx.LoadCiphertext(ct1);
	auto res_gpu = deviceCt(ct1);
	res_gpu->addScalar(-scalar);
}

void CudaEngine::evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, double scalar, Ciphertext<DCRTPoly>& ct1) {
	ctx.LoadCiphertext(ct1);
	auto res_gpu = deviceCt(ct1);
	res_gpu->multScalar(-1.0);
	res_gpu->addScalar(scalar);
	res_gpu->multScalar(-1.0);
}

Ciphertext<DCRTPoly> CudaEngine::evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= deviceCt(result);
	auto ct2_gpu				= deviceCt(ct2);
	res_gpu->mult(*ct2_gpu);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	ctx.LoadPlaintext(pt);
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= deviceCt(result);
	auto pt_gpu					= devicePt(pt);
	res_gpu->multPt(*pt_gpu);
	return result;
}

Ciphertext<DCRTPoly> CudaEngine::evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, double scalar) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= deviceCt(result);
	res_gpu->multScalar(scalar);
	return result;
}

void CudaEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	ctx.LoadCiphertext(ct1);
	ctx.LoadPlaintext(pt);
	auto res_gpu = deviceCt(ct1);
	auto pt_gpu	 = devicePt(pt);
	res_gpu->multPt(*pt_gpu);
}

void CudaEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) {
	ctx.LoadCiphertext(ct1);
	auto res_gpu = deviceCt(ct1);
	res_gpu->multScalar(scalar);
}

void CudaEngine::evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	ctx.LoadCiphertext(ct1);
	ctx.LoadCiphertext(ct2);
	auto res_gpu = deviceCt(ct1);
	auto ct2_gpu = deviceCt(ct2);
	res_gpu->mult(*ct2_gpu);
}

Ciphertext<DCRTPoly> CudaEngine::evalSquare(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	res_gpu->square();
	return result;
}

void CudaEngine::evalSquareInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	ctx.LoadCiphertext(ct);
	auto ct_gpu = deviceCt(ct);
	ct_gpu->square();
}

Ciphertext<DCRTPoly> CudaEngine::evalRotate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	auto res_gpu				= deviceCt(result);
	res_gpu->rotate(index);
	return result;
}

void CudaEngine::evalRotateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
	ctx.LoadCiphertext(ciphertext);
	auto ct_gpu = deviceCt(ciphertext);
	ct_gpu->rotate(index);
}

Ciphertext<DCRTPoly>
CudaEngine::evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	auto ct_gpu					= deviceCt(ct);
	res_gpu->copy(*ct_gpu);
	res_gpu->rotate((int)index, true);

	return result;
}

Ciphertext<DCRTPoly>
CudaEngine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	auto ct_gpu					= deviceCt(ct);
	// ct_gpu->rotate((int)index, false);
	res_gpu->copy(*ct_gpu);
	res_gpu->rotate((int)index, false);

	return result;
}

std::vector<Ciphertext<DCRTPoly>> CudaEngine::evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx,
  const Ciphertext<DCRTPoly>& ct,
  const std::vector<int32_t>& indices,
  const uint32_t m,
  const std::shared_ptr<void>& precomp) {
	std::vector<Ciphertext<DCRTPoly>> results;

	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	auto ct_gpu = deviceCt(ct);

	// Create result ciphertexts.
	std::vector<FIDESlib::CKKS::Ciphertext*> results_gpu;
	std::vector<int32_t> indices_real;
	for (int indice : indices) {
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);

		if (indice != 0) {
			indices_real.push_back(indice);
			auto res_gpu = deviceCt(result);
			results_gpu.push_back(res_gpu.get());
		}
		results.push_back(result);
	}

	ct_gpu->rotate_hoisted(indices_real, results_gpu, false);
	return results;
}

std::vector<Ciphertext<DCRTPoly>>
CudaEngine::evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst) {
	std::vector<Ciphertext<DCRTPoly>> results;

	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	auto ct_gpu = deviceCt(ct);

	std::vector<FIDESlib::CKKS::Ciphertext*> results_gpu;
	std::vector<int32_t> indices_real;
	for (int indice : indices) {
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);

		if (indice != 0) {
			indices_real.push_back(indice);
			auto res_gpu = deviceCt(result);
			results_gpu.push_back(res_gpu.get());
		}
		results.push_back(result);
	}

	ct_gpu->rotate_hoisted(indices_real, results_gpu, true);
	return results;
}

Ciphertext<DCRTPoly> CudaEngine::evalChebyshevSeries(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	FIDESlib::CKKS::evalChebyshevSeries(*res_gpu, coeffs, a, b);
	return result;
}

void CudaEngine::evalChebyshevSeriesInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {
	ctx.LoadCiphertext(ct);
	auto res_gpu = deviceCt(ct);
	FIDESlib::CKKS::evalChebyshevSeries(*res_gpu, coeffs, a, b);
}

Ciphertext<DCRTPoly> CudaEngine::rescale(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	auto res_gpu				= deviceCt(result);
	res_gpu->rescale();
	return result;
}

void CudaEngine::rescaleInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext) {
	ctx.LoadCiphertext(ciphertext);
	auto res_gpu = deviceCt(ciphertext);
	res_gpu->rescale();
}

Ciphertext<DCRTPoly> CudaEngine::accumulateSum(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= deviceCt(result);
	FIDESlib::CKKS::Accumulate(*res_gpu, 4, stride, slots);
	return result;
}

void CudaEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	ctx.LoadCiphertext(ct);
	auto res_gpu = deviceCt(ct);
	FIDESlib::CKKS::Accumulate(*res_gpu, 4, stride, slots);
}

void CudaEngine::accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride, int start) {
	ctx.LoadCiphertext(ct);
	auto res_gpu = deviceCt(ct);
	FIDESlib::CKKS::Accumulate(*res_gpu, 4, stride, slots, start);
}

BootstrapSetupPolicy CudaEngine::bootstrapSetupPolicy(bool precompute, bool btsfirstboot, int32_t modEvalLevels) const {
	// The GPU path forwards the caller's precompute/firstboot and the mod-eval levels unchanged
	// (matching the previous 7-arg EvalBootstrapSetup(..., precompute, btsfirstboot, modall) call).
	return BootstrapSetupPolicy{ precompute, btsfirstboot, modEvalLevels };
}

void CudaEngine::evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>& ctx, const PrivateKey<DCRTPoly>& secretKey, uint32_t slots) {
	if (isContextLoaded()) {
		OPENFHE_THROW("Context is already loaded");
	}
	auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(secretKey->pimpl);

	ctx.slots_bootstrap.push_back(slots);
	FIDESlib::CKKS::GenBootstrapKeys(skImpl, slots, ctx.keyDist == fideslib::SPARSE_ENCAPSULATED);
}

Ciphertext<DCRTPoly>
CudaEngine::evalBootstrap(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	auto res_gpu				= deviceCt(result);
	FIDESlib::CKKS::Bootstrap(*res_gpu, res_gpu->slots, prescaled);
	return result;
}

void CudaEngine::evalBootstrapInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	ctx.LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));
	auto res_gpu = deviceCt(ciphertext);
	FIDESlib::CKKS::Bootstrap(*res_gpu, res_gpu->slots, prescaled);
}

void CudaEngine::recoverHostCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	if (!ct->device.has_value()) {
		// Already resident on the host; nothing to read back.
		return;
	}
	ct->EnsureLazyHostCopy();
	auto& ct_cpu = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->host);

	auto ct_gpu = deviceCt(ct);
	FIDESlib::CKKS::RawCipherText raw_ct;
	ct_gpu->store(raw_ct);

	// Grow the host shadow if the device result carries more RNS limbs than it currently holds
	// (e.g. after a level-raising op like ModRaise). GetOpenFHECipherText overwrites the limb values
	// and trims down to the device limb count, so a zero ciphertext at the full level is a valid
	// container — built directly from the crypto parameters, no secret key needed.
	size_t cpu_levels = ct_cpu->GetElements()[0].GetAllElements().size();
	size_t gpu_levels = raw_ct.numRes;
	if (cpu_levels < gpu_levels) {
		auto& context		  = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
		const auto elemParams = context->GetCryptoParameters()->GetElementParams();
		lbcrypto::DCRTPoly zero(elemParams, Format::EVALUATION, true);
		auto fresh = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*ct_cpu);
		fresh->SetElements({ zero, zero });
		ct_cpu = fresh;
	}

	// Overwrite the host ciphertext with the device data.
	FIDESlib::CKKS::GetOpenFHECipherText(ct_cpu, raw_ct);
}

void CudaEngine::convolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
  Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  const std::vector<int>& indexes,
  int stride,
  int rowSize) {
	ctx.LoadCiphertext(ct);
	auto ct_gpu = deviceCt(ct);
	std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
	pts_gpu.reserve(pts.size());
	for (const auto& pt : pts) {
		ctx.LoadPlaintext(const_cast<Plaintext&>(pt));
		auto pt_gpu = devicePt(pt);
		pts_gpu.push_back(pt_gpu.get());
	}

	if (rowSize == 0) {
		rowSize = bStep * gStep;
	}

	FIDESlib::CKKS::ConvolutionTransform(*ct_gpu, rowSize, bStep, pts_gpu, stride, indexes, gStep);
}

void CudaEngine::specialConvolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
  Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  Plaintext& mask,
  const std::vector<int>& indexes,
  int stride,
  int maskRotationStride,
  int rowSize) {
	ctx.LoadCiphertext(ct);
	auto ct_gpu = deviceCt(ct);
	std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
	pts_gpu.reserve(pts.size());
	for (const auto& pt : pts) {
		ctx.LoadPlaintext(const_cast<Plaintext&>(pt));
		auto pt_gpu = devicePt(pt);
		pts_gpu.push_back(pt_gpu.get());
	}

	// Load mask
	ctx.LoadPlaintext(mask);
	auto mask_gpu = devicePt(mask);

	if (rowSize == 0) {
		rowSize = bStep * gStep;
	}

	FIDESlib::CKKS::SpecialConvolutionTransform(*ct_gpu, rowSize, bStep, pts_gpu, *mask_gpu, stride, maskRotationStride, indexes, gStep);
}

void CudaEngine::loadContext(CryptoContextImpl<DCRTPoly>& ctx, const PublicKey<DCRTPoly>& publicKey) {
	if (isContextLoaded())
		return;

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
	FIDESlib::CKKS::Parameters params{ .logN = 16, .L = 6, .dnum = 2, .primes = std::vector(p64), .Sprimes = std::vector(sp64), .batch = 100 };

	// Determine the boot configuration based on the secret key distribution.
	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	FIDESlib::BOOT_CONFIG bootConfig;
	switch (ctx.keyDist) {
	case fideslib::UNIFORM_TERNARY: bootConfig = FIDESlib::UNIFORM; break;
	case fideslib::SPARSE_TERNARY: bootConfig = FIDESlib::SPARSE; break;
	case fideslib::SPARSE_ENCAPSULATED: bootConfig = FIDESlib::ENCAPS; break;
	default: bootConfig = FIDESlib::UNIFORM; break;
	}

	FIDESlib::CKKS::RawParams rawParams = FIDESlib::CKKS::GetRawParams(context, bootConfig);
	params								= params.adaptTo(rawParams);
	FIDESlib::CKKS::Context c			= FIDESlib::CKKS::GenCryptoContextGPU(params, devices_);

	auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(publicKey->pimpl);

	// Multiplicative key switching key.
	auto& keyMap = context->GetAllEvalMultKeys(); // lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalMultKeyMap;
	if (keyMap.find(pkImpl->GetKeyTag()) != keyMap.end()) {
		auto raw_eval_ksk = FIDESlib::CKKS::GetEvalKeySwitchKey(pkImpl);
		FIDESlib::CKKS::KeySwitchingKey eval_ksk(c);
		eval_ksk.Initialize(raw_eval_ksk);
		c->AddEvalKey(std::move(eval_ksk));
	}
	// Rotational key switching keys.
	for (const auto& step : ctx.rotation_indexes) {
		auto raw_rot_ksk = FIDESlib::CKKS::GetRotationKeySwitchKey(pkImpl, step);
		FIDESlib::CKKS::KeySwitchingKey rot_ksk(c);
		rot_ksk.Initialize(raw_rot_ksk);
		c->AddRotationKey(step, std::move(rot_ksk));
	}
	// Bootstrapping keys.
	for (const auto& slot : ctx.slots_bootstrap) {
		FIDESlib::CKKS::AddBootstrapPrecomputation(pkImpl, slot, c);
	}

	context_ = std::make_unique<FIDESlib::CKKS::Context>(std::move(c));
}

void CudaEngine::loadPlaintext(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt) {
	if (pt->device.has_value())
		return;

	if (!isContextLoaded()) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}

	auto& context_gpu								  = *context_;
	auto& context									  = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
	const auto& ptImpl								  = std::any_cast<const lbcrypto::Plaintext&>(pt->host);
	FIDESlib::CKKS::RawPlainText raw_pt				  = FIDESlib::CKKS::GetRawPlainText(context, ptImpl);
	std::shared_ptr<FIDESlib::CKKS::Plaintext> gpu_pt = std::make_shared<FIDESlib::CKKS::Plaintext>(context_gpu, raw_pt);
	pt->device										  = std::make_any<std::shared_ptr<FIDESlib::CKKS::Plaintext>>(std::move(gpu_pt));
}

void CudaEngine::loadCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) {
	if (ct->device.has_value())
		return;

	if (!isContextLoaded()) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}

	auto& context_gpu								   = *context_;
	auto& context									   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(ctx.host);
	const auto& ctImpl								   = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->host);
	FIDESlib::CKKS::RawCipherText raw_ct			   = FIDESlib::CKKS::GetRawCipherText(context, ctImpl);
	std::shared_ptr<FIDESlib::CKKS::Ciphertext> gpu_ct = std::make_shared<FIDESlib::CKKS::Ciphertext>(context_gpu, raw_ct);
	ct->device										   = std::make_any<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(std::move(gpu_ct));
}

std::shared_ptr<void> CudaEngine::evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>&, const Ciphertext<DCRTPoly>&) {
	// GPU key switching does not use a precompute handle.
	return nullptr;
}

// ---- Ciphertext backend hooks ----

std::any CudaEngine::cloneCiphertextBackend(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>& src) {
	// Not device-resident → nothing to clone (the host value is copied by the value type itself).
	if (!src.device.has_value()) {
		return std::any{};
	}
	// Deep-copy the device payload into a fresh device ciphertext.
	auto src_gpu = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(src.device);
	auto new_ct	 = std::make_shared<FIDESlib::CKKS::Ciphertext>(*context_);
	new_ct->copy(*src_gpu);
	return std::make_any<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(std::move(new_ct));
}

size_t CudaEngine::ciphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		return ct.GetLevelHost();
	}
	// Depth is reversed in FIDESlib, so level = maxDepth - deviceLevel.
	auto ct_gpu = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct.device);
	return ctx.multiplicative_depth - ct_gpu->getLevel();
}

size_t CudaEngine::ciphertextNoiseScaleDeg(CryptoContextImpl<DCRTPoly>&, const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.device.has_value()) {
		return ct.GetNoiseScaleDegHost();
	}
	auto ct_gpu = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct.device);
	return ct_gpu->NoiseLevel;
}

void CudaEngine::setCiphertextSlots(CryptoContextImpl<DCRTPoly>&, CiphertextImpl<DCRTPoly>& ct, size_t slots) {
	if (!ct.device.has_value()) {
		ct.SetSlotsHost(slots);
		return;
	}
	auto ct_gpu	  = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct.device);
	ct_gpu->slots = static_cast<int>(slots);
}

void CudaEngine::setCiphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t level) {
	if (!ct.device.has_value()) {
		ct.SetLevelHost(level);
		return;
	}
	auto ct_gpu = std::any_cast<std::shared_ptr<FIDESlib::CKKS::Ciphertext>>(ct.device);
	ct_gpu->dropToLevel(ctx.multiplicative_depth - level);
}

// ---- Context backend state ----

bool CudaEngine::isContextLoaded() const {
	return context_ != nullptr;
}

void CudaEngine::synchronize() const {
	if (!context_) {
		return;
	}
	for (const auto& device : devices_) {
		cudaSetDevice(device);
		cudaDeviceSynchronize();
		CudaCheckErrorModNoSync;
	}
}

void CudaEngine::teardown() {
	if (!context_) {
		return;
	}
	FIDESlib::CKKS::DeregisterCryptoContextGPU(*context_);
	context_.reset();
}

void CudaEngine::setDevices(const std::vector<int>& devices) {
	devices_ = devices;
}

std::vector<int> CudaEngine::devices() const {
	return devices_;
}

CudaEngine::~CudaEngine() {
	teardown();
}

} // namespace fideslib
