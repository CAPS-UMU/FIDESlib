#include "CryptoContext.hpp"

#include "HostMath.hpp"

#include "Definitions.hpp"
#include "PublicKey.hpp"
#include "Serialize.hpp"
#include "ciphertext-fwd.h"
#include "cryptocontext-fwd.h"
#include "engine/Engine.hpp"
#include "engine/EngineCommon.hpp"
#include "lattice/hal/lat-backend.h"

#include <any>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <openfhe.h>
// Serialization headers - required for cereal type registration.
#include <ciphertext-ser.h>
#include <cryptocontext-ser.h>
#include <key/key-ser.h>
#include <scheme/ckksrns/ckksrns-ser.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

template <> std::map<std::string, std::vector<lbcrypto::EvalKey<lbcrypto::DCRTPoly>>> lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalMultKeyMap;
template <>
std::map<std::string, std::shared_ptr<std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>>> lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalAutomorphismKeyMap;

namespace fideslib {

static std::unordered_map<PKESchemeFeature, lbcrypto::PKESchemeFeature> PKESchemeFeatureMap = {
	{ PKESchemeFeature::PKE, lbcrypto::PKE },
	{ PKESchemeFeature::KEYSWITCH, lbcrypto::KEYSWITCH },
	{ PKESchemeFeature::PRE, lbcrypto::PRE },
	{ PKESchemeFeature::LEVELEDSHE, lbcrypto::LEVELEDSHE },
	{ PKESchemeFeature::ADVANCEDSHE, lbcrypto::ADVANCEDSHE },
	{ PKESchemeFeature::MULTIPARTY, lbcrypto::MULTIPARTY },
	{ PKESchemeFeature::FHE, lbcrypto::FHE },
	{ PKESchemeFeature::SCHEMESWITCH, lbcrypto::SCHEMESWITCH },
};

CryptoContextImpl<DCRTPoly>::~CryptoContextImpl() {
	// Tear down backend state (e.g. GPU context) before clearing the OpenFHE eval keys.
	// engine_ is null on a moved-from context.
	if (engine_) {
		engine_->teardown();
	}
	lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::ClearEvalMultKeys();
	lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::ClearEvalAutomorphismKeys();
}

// ---- Enable features ----

void CryptoContextImpl<DCRTPoly>::Enable(PKESchemeFeature feature) {
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	context->Enable(PKESchemeFeatureMap[feature]);
}

void CryptoContextImpl<DCRTPoly>::Enable(uint32_t featureMask) {
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	context->Enable(featureMask);
}

// ---- Getters ----

uint32_t CryptoContextImpl<DCRTPoly>::GetCyclotomicOrder() const {
	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	return context->GetCyclotomicOrder();
}

uint32_t CryptoContextImpl<DCRTPoly>::GetRingDimension() const {
	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	return context->GetRingDimension();
}

double CryptoContextImpl<DCRTPoly>::GetPreScaleFactor(uint32_t /*slots*/) {
	// Pure host computation (a single scalar derived from the crypto parameters + the bootstrap
	// correction factor); it touches no device, so it lives here rather than on the engine. This
	// reproduces the GPU GetPreScaleFactor exactly, reading the OpenFHE host parameters in place of
	// that function's device-mirrored copies (see engine-host-audit.md). `slots` is unused:
	// OpenFHE keeps a single m_correctionFactor (the last EvalBootstrapSetup wins), which is what the
	// device copy mirrors — callers set up bootstrap for one slot count.
	auto& context			 = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	const auto cryptoParams	 = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	const auto& elementParam = cryptoParams->GetElementParams()->GetParams();
	const size_t L			 = elementParam.size() - 1;

	const double qDouble = elementParam[0]->GetModulus().ConvertToDouble(); // q0 (FIDESlib prime[0])
	const double powP	 = std::pow(2.0, static_cast<double>(cryptoParams->GetPlaintextModulus()));
	const int32_t deg	 = static_cast<int32_t>(std::round(std::log2(qDouble / powP)));

	const uint32_t correctionFactor = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(context->GetScheme()->m_FHE)->GetCKKSBootCorrectionFactor();
	const uint32_t correction		= correctionFactor - deg;

	const auto st = cryptoParams->GetScalingTechnique();
	if (st == lbcrypto::FLEXIBLEAUTO || st == lbcrypto::FLEXIBLEAUTOEXT) {
		const uint32_t lvl		= (st == lbcrypto::FLEXIBLEAUTOEXT) ? 1U : 0U;
		const double targetSF	= cryptoParams->GetScalingFactorReal(lvl);		   // FIDESlib SFReal[L-lvl]
		const double sourceSF	= cryptoParams->GetScalingFactorReal(L - 1);	   // FIDESlib SFReal[1]
		const double modToDrop	= elementParam[1]->GetModulus().ConvertToDouble(); // FIDESlib prime[1]
		double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF);
		adjustmentFactor *= std::pow(2.0, -1.0 * static_cast<double>(correction));
		return adjustmentFactor;
	}
	return std::pow(2.0, -1.0 * static_cast<double>(correction));
}

// ---- Setters ----

void CryptoContextImpl<DCRTPoly>::SetAutoLoadPlaintexts(bool autoload) {
	this->auto_load_plaintexts = autoload;
}

void CryptoContextImpl<DCRTPoly>::SetAutoLoadCiphertexts(bool autoload) {
	this->auto_load_ciphertexts = autoload;
}

void CryptoContextImpl<DCRTPoly>::SetCudaDevices(const std::vector<int>& devices) {
	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("SetCudaDevices must be called before LoadContext");
	}
	engine_->setDevices(devices);
}

std::vector<int> CryptoContextImpl<DCRTPoly>::GetCudaDevices() const {
	return engine_->devices();
}

// ---- Load to devices ----

void CryptoContextImpl<DCRTPoly>::LoadContext(const PublicKey<DCRTPoly>& publicKey) {
	engine_->loadContext(*this, publicKey);
}

void CryptoContextImpl<DCRTPoly>::LoadPlaintext(Plaintext& pt) {
	engine_->loadPlaintext(*this, pt);
}

void CryptoContextImpl<DCRTPoly>::LoadCiphertext(Ciphertext<DCRTPoly>& ct) {
	engine_->loadCiphertext(*this, ct);
}

// ---- Key Generation ----

KeyPair<DCRTPoly> CryptoContextImpl<DCRTPoly>::KeyGen() {
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	auto keys	  = context->KeyGen();

	KeyPair<DCRTPoly> keypair;
	keypair.publicKey = std::make_shared<PublicKeyImpl<DCRTPoly>>();
	keypair.secretKey = std::make_shared<PrivateKeyImpl<DCRTPoly>>();

	keypair.publicKey->pimpl = std::make_any<lbcrypto::PublicKey<lbcrypto::DCRTPoly>>(keys.publicKey);
	keypair.secretKey->pimpl = std::make_any<lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>(keys.secretKey);

	return keypair;
}

void CryptoContextImpl<DCRTPoly>::EvalMultKeyGen(const PrivateKey<DCRTPoly>& sk) {

	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("EvalMultKeyGen must be called before LoadContext");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	context->EvalMultKeyGen(skImpl);
}

void CryptoContextImpl<DCRTPoly>::EvalRotateKeyGen(const PrivateKey<DCRTPoly>& sk, const std::vector<int32_t>& steps) {

	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("EvalRotateKeyGen must be called before LoadContext");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	context->EvalRotateKeyGen(skImpl, steps);
	this->rotation_indexes.insert(this->rotation_indexes.end(), steps.begin(), steps.end());
}

// ---- Bootstrapping ----

void CryptoContextImpl<DCRTPoly>::EvalBootstrapSetup(const std::vector<uint32_t>& levelBudget, std::vector<uint32_t> dim1, uint32_t slots, uint32_t correctionFactor, bool precompute, bool btsfirstboot) {
	// Bootstrap setup is a host (OpenFHE) computation; the facade performs it and the engine only
	// supplies the per-backend argument policy.
	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("EvalBootstrapSetup must be called before LoadContext");
	}
	auto& context		   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	int32_t modall		   = bootstrapModEvalLevels(this->keyDist);
	BootstrapSetupPolicy p = engine_->bootstrapSetupPolicy(precompute, btsfirstboot, modall);
	context->EvalBootstrapSetup(levelBudget, std::move(dim1), slots, correctionFactor, p.precompute, p.btSlotsEncoding, p.modEvalLevels);
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapKeyGen(const PrivateKey<DCRTPoly>& secretKey, uint32_t slots) {
	engine_->evalBootstrapKeyGen(*this, secretKey, slots);
}

// ---- Serialization ----

bool CryptoContextImpl<DCRTPoly>::SerializeEvalMultKey(std::ostream& ser, const fideslib::SerType& sertype, const std::string& keyTag) {
	bool res;
	switch (sertype) {
	case fideslib::SerType::BINARY:
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::SerializeEvalMultKey(
		  ser, lbcrypto::SerType::BINARY, keyTag);
		break;
	case fideslib::SerType::JSON:
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::SerializeEvalMultKey(
		  ser, lbcrypto::SerType::JSON, keyTag);
		break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

bool CryptoContextImpl<DCRTPoly>::SerializeEvalAutomorphismKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag) {
	bool res;
	switch (sertype) {
	case SerType::BINARY:
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::SerializeEvalAutomorphismKey(
		  ser, lbcrypto::SerType::BINARY, keyTag);
		break;
	case SerType::JSON:
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::SerializeEvalAutomorphismKey(
		  ser, lbcrypto::SerType::JSON, keyTag);
		break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

// ---- Deserialization ----

bool CryptoContextImpl<DCRTPoly>::DeserializeEvalMultKey(std::istream& ser, const SerType& sertype) const {

	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("DeserializeEvalMultKey must be called before LoadContext");
	}

	bool res;
	switch (sertype) {
	case SerType::BINARY:
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::DeserializeEvalMultKey(
		  ser, lbcrypto::SerType::BINARY);
		break;
	case SerType::JSON:
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::DeserializeEvalMultKey(
		  ser, lbcrypto::SerType::JSON);
		break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

bool CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(std::istream& ser, const SerType& sertype) const {

	if (engine_->isContextLoaded()) {
		OPENFHE_THROW("DeserializeEvalAutomorphismKey must be called before LoadContext");
	}

	bool res;
	switch (sertype) {
	case SerType::BINARY:
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::DeserializeEvalAutomorphismKey(
		  ser, lbcrypto::SerType::BINARY);
		break;
	case SerType::JSON:
		res = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::DeserializeEvalAutomorphismKey(
		  ser, lbcrypto::SerType::JSON);
		break;
	default: OPENFHE_THROW("Unsupported serialization type");
	}

	return res;
}

// ---- Encoding ----

Plaintext CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value,
  size_t noiseScaleDeg,
  uint32_t level,
  const std::shared_ptr<void> params,
  uint32_t slots) {

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	auto pt		  = context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->host		= std::make_any<lbcrypto::Plaintext>(pt);

	if (!this->auto_load_plaintexts) {
		return plaintext;
	}

	this->LoadPlaintext(plaintext);

	return plaintext;
}

Plaintext
CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg, uint32_t level, const std::shared_ptr<void> params, uint32_t slots) {

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	auto pt		  = context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->host		= std::make_any<lbcrypto::Plaintext>(pt);

	if (!this->auto_load_plaintexts) {
		return plaintext;
	}

	this->LoadPlaintext(plaintext);

	return plaintext;
}

// ---- Encryption ----

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(Plaintext& pt, const PublicKey<DCRTPoly>& pk) {

	auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	const auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(pk->pimpl);
	const auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->host);

	auto ct							= context->Encrypt(pkImpl, ptImpl);
	Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
	ciphertext->host				= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);

	if (!this->auto_load_ciphertexts) {
		return ciphertext;
	}

	this->LoadCiphertext(ciphertext);

	return ciphertext;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(const PublicKey<DCRTPoly>& pk, Plaintext& pt) {
	return Encrypt(pt, pk);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(Plaintext& pt, const PrivateKey<DCRTPoly>& sk) {

	auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	const auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	const auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->host);

	auto ct							= context->Encrypt(skImpl, ptImpl);
	Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
	ciphertext->host				= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);

	if (!this->auto_load_ciphertexts) {
		return ciphertext;
	}

	this->LoadCiphertext(ciphertext);

	return ciphertext;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(const PrivateKey<DCRTPoly>& sk, Plaintext& pt) {
	return Encrypt(pt, sk);
}

DecryptResult CryptoContextImpl<DCRTPoly>::Decrypt(Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, Plaintext* pt) {
	if (pt == nullptr) {
		OPENFHE_THROW("Plaintext pointer is null");
	}
	// The only backend-specific step is the readback (device->host); it is its own engine hook
	// (no-op on CPU). The decrypt itself is a host OpenFHE computation, so it lives here.
	engine_->recoverHostCiphertext(*this, ct);

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->host);
	auto& ct_host = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->host);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	lbcrypto::Plaintext ptImpl;
	auto res = context->Decrypt(skImpl, ct_host, &ptImpl);

	if (pt->get() != nullptr) {
		(*pt)->host = std::make_any<lbcrypto::Plaintext>(std::move(ptImpl));
		// Drop any stale backend payload a reused output plaintext may still hold (empty `any` ==
		// not resident; RAII frees a device copy). Clearing the value type's own slot — no backend
		// dispatch, and no inspection of device-residency state in this neutral facade.
		(*pt)->device.reset();
	} else {
		*pt			= std::make_shared<PlaintextImpl>();
		(*pt)->host = std::make_any<lbcrypto::Plaintext>(std::move(ptImpl));
	}

	DecryptResult result{};
	result.isValid		 = res.isValid;
	result.messageLength = res.messageLength;
	return result;
}

DecryptResult CryptoContextImpl<DCRTPoly>::Decrypt(const PrivateKey<DCRTPoly>& sk, Ciphertext<DCRTPoly>& ct, Plaintext* pt) {
	return Decrypt(ct, sk, pt);
}

void CryptoContextImpl<DCRTPoly>::RecoverHostCiphertext(Ciphertext<DCRTPoly>& ct) {
	engine_->recoverHostCiphertext(*this, ct);
}

// ---- Operations ----

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalNegate(const Ciphertext<DCRTPoly>& ct) {
	return engine_->evalNegate(*this, ct);
}

void CryptoContextImpl<DCRTPoly>::EvalNegateInPlace(Ciphertext<DCRTPoly>& ct) {
	engine_->evalNegateInPlace(*this, ct);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	return engine_->evalAdd(*this, ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return engine_->evalAdd(*this, ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {
	return EvalAdd(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct, double scalar) {
	return engine_->evalAdd(*this, ct, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(double scalar, const Ciphertext<DCRTPoly>& ct) {
	return EvalAdd(ct, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	engine_->evalAddInPlace(*this, ct1, ct2);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	engine_->evalAddInPlace(*this, ct1, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Plaintext& pt, Ciphertext<DCRTPoly>& ct1) {
	EvalAddInPlace(ct1, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {
	engine_->evalAddInPlace(*this, ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
	EvalAddInPlace(ct1, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	return EvalAdd(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return EvalAdd(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct) {
	return EvalAdd(ct, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalAddMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	EvalAddInPlace(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAddMany(const std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	return engine_->evalAddMany(*this, ciphertexts);
}

void CryptoContextImpl<DCRTPoly>::EvalAddManyInPlace(std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {
	engine_->evalAddManyInPlace(*this, ciphertexts);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	return engine_->evalSub(*this, ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return engine_->evalSub(*this, ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {
	return engine_->evalSub(*this, pt, ct);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct, double scalar) {
	return engine_->evalSub(*this, ct, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(double scalar, const Ciphertext<DCRTPoly>& ct) {
	return engine_->evalSub(*this, scalar, ct);
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	engine_->evalSubInPlace(*this, ct1, ct2);
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {
	engine_->evalSubInPlace(*this, ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
	engine_->evalSubInPlace(*this, scalar, ct1);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSubMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	return EvalSub(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSubMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return EvalSub(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSubMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct) {
	return EvalSub(pt, ct);
}

void CryptoContextImpl<DCRTPoly>::EvalSubMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	EvalSubInPlace(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {
	return engine_->evalMult(*this, ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	return engine_->evalMult(*this, ct1, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(Plaintext& pt, const Ciphertext<DCRTPoly>& ct1) {
	return EvalMult(ct1, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, double scalar) {
	return engine_->evalMult(*this, ct1, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(double scalar, const Ciphertext<DCRTPoly>& ct1) {
	return EvalMult(ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {
	engine_->evalMultInPlace(*this, ct1, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {
	engine_->evalMultInPlace(*this, ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
	EvalMultInPlace(ct1, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	return EvalMult(ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt) {
	return EvalMult(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMultMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct) {
	return EvalMult(ct, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalMultMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) {
	engine_->evalMultInPlace(*this, ct1, ct2);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSquare(const Ciphertext<DCRTPoly>& ct) {
	return engine_->evalSquare(*this, ct);
}

void CryptoContextImpl<DCRTPoly>::EvalSquareInPlace(Ciphertext<DCRTPoly>& ct) {
	engine_->evalSquareInPlace(*this, ct);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSquareMutable(Ciphertext<DCRTPoly>& ct) {
	return EvalSquare(ct);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalRotate(const Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
	return engine_->evalRotate(*this, ciphertext, index);
}

void CryptoContextImpl<DCRTPoly>::EvalRotateInPlace(Ciphertext<DCRTPoly>& ciphertext, int32_t index) {
	engine_->evalRotateInPlace(*this, ciphertext, index);
}

std::shared_ptr<void> CryptoContextImpl<DCRTPoly>::EvalFastRotationPrecompute(const Ciphertext<DCRTPoly>& ct) {
	return engine_->evalFastRotationPrecompute(*this, ct);
}

Ciphertext<DCRTPoly>
CryptoContextImpl<DCRTPoly>::EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp) {
	return engine_->evalFastRotation(*this, ct, index, m, precomp);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) {
	return engine_->evalFastRotationExt(*this, ct, index, digits, addFirst);
}

std::vector<Ciphertext<DCRTPoly>>
CryptoContextImpl<DCRTPoly>::EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const uint32_t m, const std::shared_ptr<void>& precomp) {
	return engine_->evalFastRotation(*this, ct, indices, m, precomp);
}

std::vector<Ciphertext<DCRTPoly>>
CryptoContextImpl<DCRTPoly>::EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst) {
	return engine_->evalFastRotationExt(*this, ct, indices, digits, addFirst);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalChebyshevSeries(const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {
	return engine_->evalChebyshevSeries(*this, ct, coeffs, a, b);
}

void CryptoContextImpl<DCRTPoly>::EvalChebyshevSeriesInPlace(Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {
	engine_->evalChebyshevSeriesInPlace(*this, ct, coeffs, a, b);
}

std::vector<double> CryptoContextImpl<DCRTPoly>::GetChebyshevCoefficients(std::function<double(double)>& func, double a, double b, size_t degree) {
	return fideslib::get_chebyshev_coefficients(func, a, b, degree);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Rescale(const Ciphertext<DCRTPoly>& ciphertext) {
	return engine_->rescale(*this, ciphertext);
}

void CryptoContextImpl<DCRTPoly>::RescaleInPlace(Ciphertext<DCRTPoly>& ciphertext) {
	engine_->rescaleInPlace(*this, ciphertext);
}

void CryptoContextImpl<DCRTPoly>::SetLevel(Ciphertext<DCRTPoly>& ct, size_t level) {
	ct->SetLevel(level);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalBootstrap(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	return engine_->evalBootstrap(*this, ciphertext, numIterations, precision, prescaled);
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	engine_->evalBootstrapInPlace(*this, ciphertext, numIterations, precision, prescaled);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::AccumulateSum(const Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	return engine_->accumulateSum(*this, ct, slots, stride);
}

void CryptoContextImpl<DCRTPoly>::AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride) {
	engine_->accumulateSumInPlace(*this, ct, slots, stride);
}

void CryptoContextImpl<DCRTPoly>::AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride, int start) {
	engine_->accumulateSumInPlace(*this, ct, slots, stride, start);
}

void CryptoContextImpl<DCRTPoly>::ConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  const std::vector<int>& indexes,
  int stride,
  int rowSize) {
	engine_->convolutionTransformInPlace(*this, ct, gStep, bStep, pts, indexes, stride, rowSize);
}

void CryptoContextImpl<DCRTPoly>::SpecialConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  Plaintext& mask,
  const std::vector<int>& indexes,
  int stride,
  int maskRotationStride,
  int rowSize) {
	engine_->specialConvolutionTransformInPlace(*this, ct, gStep, bStep, pts, mask, indexes, stride, maskRotationStride, rowSize);
}

// ---- Ciphertext backend hooks ----

std::any CryptoContextImpl<DCRTPoly>::CloneCiphertextBackend(const CiphertextImpl<DCRTPoly>& src) {
	return engine_->cloneCiphertextBackend(*this, src);
}

size_t CryptoContextImpl<DCRTPoly>::CiphertextLevel(const CiphertextImpl<DCRTPoly>& ct) {
	return engine_->ciphertextLevel(*this, ct);
}

size_t CryptoContextImpl<DCRTPoly>::CiphertextNoiseScaleDeg(const CiphertextImpl<DCRTPoly>& ct) {
	return engine_->ciphertextNoiseScaleDeg(*this, ct);
}

void CryptoContextImpl<DCRTPoly>::SetCiphertextSlots(CiphertextImpl<DCRTPoly>& ct, size_t slots) {
	engine_->setCiphertextSlots(*this, ct, slots);
}

void CryptoContextImpl<DCRTPoly>::SetCiphertextLevel(CiphertextImpl<DCRTPoly>& ct, size_t level) {
	engine_->setCiphertextLevel(*this, ct, level);
}

void CryptoContextImpl<DCRTPoly>::Synchronize() const {
	engine_->synchronize();
}

std::vector<int> CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep) {
	return fideslib::GetConvolutionTransformRotationIndices(rowSize, bStep, stride, gStep);
}

} // namespace fideslib