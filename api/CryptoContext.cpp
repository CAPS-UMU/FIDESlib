#include "CryptoContext.hpp"
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
#include "Definitions.hpp"
#include "PolyApprox.cuh"
#include "PublicKey.hpp"
#include "Serialize.hpp"
#include "ciphertext-fwd.h"
#include "cryptocontext-fwd.h"
#include "lattice/hal/lat-backend.h"

#include <any>
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
	lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::ClearEvalMultKeys();
	lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::ClearEvalAutomorphismKeys();
}

// ---- Enable features ----

void CryptoContextImpl<DCRTPoly>::Enable(PKESchemeFeature feature) {
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	context->Enable(PKESchemeFeatureMap[feature]);
}

void CryptoContextImpl<DCRTPoly>::Enable(uint32_t featureMask) {
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	context->Enable(featureMask);
}

// ---- Getters ----

uint32_t CryptoContextImpl<DCRTPoly>::GetCyclotomicOrder() const {
	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	return context->GetCyclotomicOrder();
}

uint32_t CryptoContextImpl<DCRTPoly>::GetRingDimension() const {
	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	return context->GetRingDimension();
}

double CryptoContextImpl<DCRTPoly>::GetPreScaleFactor(uint32_t slots) {
	if (!this->loaded) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}
	auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	return FIDESlib::CKKS::GetPreScaleFactor(context_gpu, static_cast<int32_t>(slots));
}

// ---- Setters ----

void CryptoContextImpl<DCRTPoly>::SetAutoLoadPlaintexts(bool autoload) {
	this->auto_load_plaintexts = autoload;
}

void CryptoContextImpl<DCRTPoly>::SetAutoLoadCiphertexts(bool autoload) {
	this->auto_load_ciphertexts = autoload;
}

void CryptoContextImpl<DCRTPoly>::SetDevices(const std::vector<int>& devices) {
	if (this->loaded) {
		OPENFHE_THROW("SetDevices must be called before LoadContext");
	}

	this->devices = devices;
}

// ---- Load to devices ----

void CryptoContextImpl<DCRTPoly>::LoadContext(const PublicKey<DCRTPoly>& publicKey) {
	if (this->loaded || this->devices.empty())
		return;

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	FIDESlib::CKKS::Parameters params{ .logN = 16, .L = 6, .dnum = 2, .primes = std::vector(p64), .Sprimes = std::vector(sp64), .batch = 100 };

	// Determine the boot configuration based on the secret key distribution.
	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
	FIDESlib::BOOT_CONFIG bootConfig;
	switch (this->keyDist) {
	case fideslib::UNIFORM_TERNARY: bootConfig = FIDESlib::UNIFORM; break;
	case fideslib::SPARSE_TERNARY: bootConfig = FIDESlib::SPARSE; break;
	case fideslib::SPARSE_ENCAPSULATED: bootConfig = FIDESlib::ENCAPS; break;
	default: bootConfig = FIDESlib::UNIFORM; break;
	}

	FIDESlib::CKKS::RawParams rawParams = FIDESlib::CKKS::GetRawParams(context, bootConfig);
	params								= params.adaptTo(rawParams);
	FIDESlib::CKKS::Context c			= FIDESlib::CKKS::GenCryptoContextGPU(params, this->devices);

	auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(publicKey->pimpl);

	// Multiplicative key switching key.
	auto& keyMap = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::s_evalMultKeyMap;
	if (keyMap.find(pkImpl->GetKeyTag()) != keyMap.end()) {
		auto raw_eval_ksk = FIDESlib::CKKS::GetEvalKeySwitchKey(pkImpl);
		FIDESlib::CKKS::KeySwitchingKey eval_ksk(c);
		eval_ksk.Initialize(raw_eval_ksk);
		c->AddEvalKey(std::move(eval_ksk));
	}
	// Rotational key switching keys.
	for (const auto& step : this->rotation_indexes) {
		auto raw_rot_ksk = FIDESlib::CKKS::GetRotationKeySwitchKey(pkImpl, step);
		FIDESlib::CKKS::KeySwitchingKey rot_ksk(c);
		rot_ksk.Initialize(raw_rot_ksk);
		c->AddRotationKey(step, std::move(rot_ksk));
	}

	// Bootstrapping precomputations.
	auto fhe = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(context->GetScheme()->m_FHE);
	if (fhe) {
		auto precom = fhe->m_bootPrecomMap;
		if (!precom.empty()) {
			for (const auto& [slots, _] : precom) {
				FIDESlib::CKKS::AddBootstrapPrecomputation(pkImpl, static_cast<int32_t>(slots), c);
			}
		}
	}

	this->gpu	 = std::make_any<FIDESlib::CKKS::Context>(std::move(c));
	this->loaded = true;
}

void CryptoContextImpl<DCRTPoly>::LoadPlaintext(Plaintext& pt) {
	if (pt->loaded || this->devices.empty())
		return;

	if (!this->loaded) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}

	auto& context_gpu								  = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	auto& context									  = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto& ptImpl								  = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
	FIDESlib::CKKS::RawPlainText raw_pt				  = FIDESlib::CKKS::GetRawPlainText(context, ptImpl);
	std::shared_ptr<FIDESlib::CKKS::Plaintext> gpu_pt = std::make_shared<FIDESlib::CKKS::Plaintext>(context_gpu, raw_pt);
	uint32_t handle									  = this->RegisterDevicePlaintext(std::move(gpu_pt));
	pt->gpu											  = handle;
	pt->loaded										  = true;
}

void CryptoContextImpl<DCRTPoly>::LoadCiphertext(Ciphertext<DCRTPoly>& ct) {
	if (ct->loaded || this->devices.empty())
		return;

	if (!this->loaded) {
		OPENFHE_THROW("CryptoContext not loaded to any device");
	}

	auto& context_gpu								   = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	auto& context									   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto& ctImpl								   = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
	FIDESlib::CKKS::RawCipherText raw_ct			   = FIDESlib::CKKS::GetRawCipherText(context, ctImpl);
	std::shared_ptr<FIDESlib::CKKS::Ciphertext> gpu_ct = std::make_shared<FIDESlib::CKKS::Ciphertext>(context_gpu, raw_ct);
	uint32_t handle									   = this->RegisterDeviceCiphertext(std::move(gpu_ct));
	ct->gpu											   = handle;
	ct->loaded										   = true;
	ct->original_level								   = this->multiplicative_depth - ct->GetLevel();
}

// ---- Key Generation ----

KeyPair<DCRTPoly> CryptoContextImpl<DCRTPoly>::KeyGen() {
	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto keys	  = context->KeyGen();

	KeyPair<DCRTPoly> keypair;
	keypair.publicKey = std::make_shared<PublicKeyImpl<DCRTPoly>>();
	keypair.secretKey = std::make_shared<PrivateKeyImpl<DCRTPoly>>();

	keypair.publicKey->pimpl = std::make_any<lbcrypto::PublicKey<lbcrypto::DCRTPoly>>(keys.publicKey);
	keypair.secretKey->pimpl = std::make_any<lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>(keys.secretKey);

	return keypair;
}

void CryptoContextImpl<DCRTPoly>::EvalMultKeyGen(const PrivateKey<DCRTPoly>& sk) {

	if (!this->devices.empty() && this->loaded) {
		OPENFHE_THROW("EvalMultKeyGen must be called before LoadContext");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	context->EvalMultKeyGen(skImpl);
}

void CryptoContextImpl<DCRTPoly>::EvalRotateKeyGen(const PrivateKey<DCRTPoly>& sk, const std::vector<int32_t>& steps) {

	if (!this->devices.empty() && this->loaded) {
		OPENFHE_THROW("EvalRotateKeyGen must be called before LoadContext");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	context->EvalRotateKeyGen(skImpl, steps);
	this->rotation_indexes.insert(this->rotation_indexes.end(), steps.begin(), steps.end());
}

// ---- Bootstrapping ----

void CryptoContextImpl<DCRTPoly>::EvalBootstrapSetup(const std::vector<uint32_t>& levelBudget, std::vector<uint32_t> dim1, uint32_t slots, uint32_t correctionFactor) {

	// Only before loading one must compute the bootstrapping auxiliary data.
	if (!this->devices.empty() && this->loaded) {
		OPENFHE_THROW("EvalBootstrapSetup must be called before LoadContext");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);

	std::vector<double> coeffchebyshev;
	int doubleAngleIts = 3;

	if (this->keyDist == fideslib::SPARSE_ENCAPSULATED) {
		coeffchebyshev = { 0.24554573401685137,
			-0.047919064883347899,
			0.28388702040840819,
			-0.029944538735513584,
			0.35576522619036460,
			0.015106561885073030,
			0.29532946674499999,
			0.071203602333739374,
			-0.10347347339668074,
			0.044997590512555294,
			-0.42750712431925747,
			-0.090342129729094875,
			0.36762876269324946,
			0.049318066039335348,
			-0.14535986272411980,
			-0.015106938483063579,
			0.035951935499240355,
			0.0031036582188686437,
			-0.0062644606607068463,
			-0.00046609430477154916,
			0.00082128798852385086,
			0.000053910533892372678,
			-0.000084551549768927401,
			-4.9773801787288514e-6,
			7.0466620439083618e-6,
			3.7659807574103204e-7,
			-4.8648510153626034e-7,
			-2.3830267651437146e-8,
			2.8329709716159918e-8,
			1.2817720050334158e-9,
			-1.4122220430105397e-9,
			-5.9306213139085216e-11,
			6.3298928388417848e-11 };
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
	} else if (this->keyDist == fideslib::SPARSE_TERNARY) {
		coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsSparse;
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_SPARSE;
	} else if (this->keyDist == fideslib::UNIFORM_TERNARY) {
		coeffchebyshev = lbcrypto::FHECKKSRNS::g_coefficientsUniform;
		doubleAngleIts = lbcrypto::FHECKKSRNS::R_UNIFORM;
	} else {
		OPENFHE_THROW("Unsupported key distribution");
	}

	int32_t modall = static_cast<int>(lbcrypto::GetMultiplicativeDepthByCoeffVector(coeffchebyshev, false)) + doubleAngleIts;

	if (this->devices.empty()) {
		context->EvalBootstrapSetup(levelBudget, std::move(dim1), slots, correctionFactor, true);
		return;
	}

	context->EvalBootstrapSetup(levelBudget, std::move(dim1), slots, correctionFactor, true, modall);
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapKeyGen(const PrivateKey<DCRTPoly>& sk, uint32_t slots) {

	if (!this->devices.empty() && this->loaded) {
		OPENFHE_THROW("EvalBootstrapKeyGen must be called before LoadContext");
	}

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);

	if (this->devices.empty()) {
		context->EvalBootstrapKeyGen(skImpl, slots);
	} else {
		FIDESlib::CKKS::GenBootstrapKeys(skImpl, static_cast<int>(slots));
	}
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

	if (!this->devices.empty() && this->loaded) {
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

	if (!this->devices.empty() && this->loaded) {
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

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto pt		  = context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->cpu		= std::make_any<lbcrypto::Plaintext>(pt);
	plaintext->loaded	= false;

	if (this->devices.empty() || !this->auto_load_plaintexts) {
		return plaintext;
	}

	this->LoadPlaintext(plaintext);

	return plaintext;
}

Plaintext
CryptoContextImpl<DCRTPoly>::MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg, uint32_t level, const std::shared_ptr<void> params, uint32_t slots) {

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto pt		  = context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);

	Plaintext plaintext = std::make_shared<PlaintextImpl>(this->self_reference.lock());
	plaintext->cpu		= std::make_any<lbcrypto::Plaintext>(pt);
	plaintext->loaded	= false;

	if (this->devices.empty() || !this->auto_load_plaintexts) {
		return plaintext;
	}

	this->LoadPlaintext(plaintext);

	return plaintext;
}

// ---- Encryption ----

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(Plaintext& pt, const PublicKey<DCRTPoly>& pk) {

	auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto& pkImpl = std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(pk->pimpl);
	const auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);

	auto ct							= context->Encrypt(pkImpl, ptImpl);
	Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
	ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);

	if (this->devices.empty() || !this->auto_load_ciphertexts) {
		return ciphertext;
	}

	this->LoadCiphertext(ciphertext);

	return ciphertext;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(const PublicKey<DCRTPoly>& pk, Plaintext& pt) {
	return Encrypt(pt, pk);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Encrypt(Plaintext& pt, const PrivateKey<DCRTPoly>& sk) {

	auto& context	   = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	const auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	const auto& ptImpl = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);

	auto ct							= context->Encrypt(skImpl, ptImpl);
	Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
	ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);

	if (this->devices.empty() || !this->auto_load_ciphertexts) {
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

	auto& context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	auto& ct_cpu  = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);

	// Copy ciphertext to CPU if needed.
	if (ct->loaded) {
		auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
		FIDESlib::CKKS::RawCipherText raw_ct;
		ct_gpu->store(raw_ct);

		// Check if CPU ciphertext has enough levels to hold GPU ciphertext.
		size_t cpu_levels = ct_cpu->GetElements()[0].GetAllElements().size();
		size_t gpu_levels = raw_ct.numRes;

		if (cpu_levels < gpu_levels) {
			// Create a fresh ciphertext at the top level with enough space
			std::vector<double> dummy(1, 0.0);
			auto pt_dummy = context->MakeCKKSPackedPlaintext(dummy, 1, this->multiplicative_depth - ct_gpu->getLevel());
			auto& skImpl  = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
			ct_cpu		  = context->Encrypt(skImpl, pt_dummy);
		}

		// Overwrite cpu ct with the data from GPU.
		FIDESlib::CKKS::GetOpenFHECipherText(ct_cpu, raw_ct);
	}

	auto& skImpl = std::any_cast<const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>&>(sk->pimpl);
	lbcrypto::Plaintext ptImpl;
	auto res = context->Decrypt(skImpl, ct_cpu, &ptImpl);

	if (pt->get() != nullptr) {

		if ((*pt)->loaded && !this->devices.empty()) {
			OPENFHE_THROW("Inconsistent state: Plaintext is marked as loaded but no devices are available");
		}
		if ((*pt)->loaded && !this->EvictDevicePlaintext((*pt)->gpu)) {
			OPENFHE_THROW("Plaintext eviction error: could not evict Plaintext from device");
		}

		(*pt)->cpu	  = std::make_any<lbcrypto::Plaintext>(std::move(ptImpl));
		(*pt)->loaded = false;
		(*pt)->gpu	  = 0;
	} else {
		*pt			  = std::make_shared<PlaintextImpl>();
		(*pt)->cpu	  = std::make_any<lbcrypto::Plaintext>(std::move(ptImpl));
		(*pt)->loaded = false;
		(*pt)->gpu	  = 0;
	}

	DecryptResult result{};
	result.isValid		 = res.isValid;
	result.messageLength = res.messageLength;
	return result;
}

DecryptResult CryptoContextImpl<DCRTPoly>::Decrypt(const PrivateKey<DCRTPoly>& sk, Ciphertext<DCRTPoly>& ct, Plaintext* pt) {
	return Decrypt(ct, sk, pt);
}

// ---- Operations ----

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalNegate(const Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct							= context->EvalNegate(ctImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->multScalar(-1.0);
	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalNegateInPlace(Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		context->EvalNegateInPlace(ctImpl);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct);

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	ct_gpu->multScalar(-1.0);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ct2Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
		auto ct							= context->EvalAdd(ct1Impl, ct2Impl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto ct2_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
	res_gpu->add(*ct2_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto& ptImpl					= std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
		auto ct							= context->EvalAdd(ctImpl, ptImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	this->LoadPlaintext(pt);

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto pt_gpu					= std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
	res_gpu->addPt(*pt_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {
	return EvalAdd(ct, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct, double scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct							= context->EvalAdd(ctImpl, scalar);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->addScalar(scalar);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(double scalar, const Ciphertext<DCRTPoly>& ct) {
	return EvalAdd(ct, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(const Ciphertext<DCRTPoly>& ct, std::complex<double> scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct							= context->EvalAdd(ctImpl, scalar);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->addScalar(scalar);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalAdd(std::complex<double> scalar, const Ciphertext<DCRTPoly>& ct) {
	return EvalAdd(ct, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ct2Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
		context->EvalAddInPlace(ct1Impl, ct2Impl);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
	res_gpu->add(*ct2_gpu);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ptImpl  = std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
		context->EvalAddInPlace(ct1Impl, ptImpl);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);
	this->LoadPlaintext(pt);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	auto pt_gpu	 = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
	res_gpu->addPt(*pt_gpu);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Plaintext& pt, Ciphertext<DCRTPoly>& ct1) {
	EvalAddInPlace(ct1, pt);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		context->EvalAddInPlace(ct1Impl, scalar);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	res_gpu->addScalar(scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
	EvalAddInPlace(ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, std::complex<double> scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		context->EvalAddInPlace(ct1Impl, scalar);
		return;
	}

	// GPU path. 
	this->LoadCiphertext(ct1);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	res_gpu->addScalar(scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalAddInPlace(std::complex<double> scalar, Ciphertext<DCRTPoly>& ct1) {
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

	if (ciphertexts.empty()) {
		OPENFHE_THROW("EvalAddMany: input ciphertext vector is empty");
	}

	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctImpls;
		ctImpls.reserve(ciphertexts.size());
		for (const auto& ct : ciphertexts) {
			ctImpls.push_back(std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu));
		}
		auto ct							= context->EvalAddMany(ctImpls);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.

	for (const auto& ct : ciphertexts) {
		this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	}

	// Initialize result with the first ciphertext.
	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertexts[0]);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));

	const size_t inSize = ciphertexts.size();
	const size_t lim	= inSize * 2 - 2;
	std::vector<Ciphertext<DCRTPoly>> ciphertextSumVec;
	ciphertextSumVec.resize(inSize - 1);
	size_t ctrIndex = 0;

	for (size_t i = 0; i < lim; i = i + 2) {
		ciphertextSumVec[ctrIndex++] =
		  this->EvalAdd(i < inSize ? ciphertexts[i] : ciphertextSumVec[i - inSize], i + 1 < inSize ? ciphertexts[i + 1] : ciphertextSumVec[i + 1 - inSize]);
	}

	return ciphertextSumVec.back();
}

void CryptoContextImpl<DCRTPoly>::EvalAddManyInPlace(std::vector<Ciphertext<DCRTPoly>>& ciphertexts) {

	if (ciphertexts.empty()) {
		OPENFHE_THROW("EvalAddManyInPlace: input ciphertext vector is empty");
	}

	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctImpls;
		ctImpls.reserve(ciphertexts.size());
		for (const auto& ct : ciphertexts) {
			ctImpls.push_back(std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu));
		}
		context->EvalAddManyInPlace(ctImpls);
		ciphertexts[0]->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ctImpls[0]);
		return;
	}

	// GPU path.

	for (const auto& ct : ciphertexts) {
		this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	}

	for (size_t j = 1; j < ciphertexts.size(); j = j * 2) {
		for (size_t i = 0; i < ciphertexts.size(); i = i + 2 * j) {
			if ((i + j) < ciphertexts.size()) {
				if (ciphertexts[i] != nullptr && ciphertexts[i + j] != nullptr) {
					this->EvalAddInPlace(ciphertexts[i], ciphertexts[i + j]);
				} else if (ciphertexts[i] == nullptr && ciphertexts[i + j] != nullptr) {
					ciphertexts[i] = ciphertexts[i + j];
				}
			}
		}
	}
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ct2Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
		auto ct							= context->EvalSub(ct1Impl, ct2Impl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto ct2_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
	res_gpu->sub(*ct2_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct, Plaintext& pt) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto& ptImpl					= std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
		auto ct							= context->EvalSub(ctImpl, ptImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	this->LoadPlaintext(pt);

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto pt_gpu					= std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
	res_gpu->subPt(*pt_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(Plaintext& pt, const Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto& ptImpl					= std::any_cast<lbcrypto::Plaintext&>(pt->cpu);
		auto ct							= context->EvalSub(ptImpl, ctImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));
	this->LoadPlaintext(pt);

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto pt_gpu					= std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
	res_gpu->multScalar(-1.0);
	res_gpu->addPt(*pt_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(const Ciphertext<DCRTPoly>& ct, double scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct							= context->EvalSub(ctImpl, scalar);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->addScalar(-scalar);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSub(double scalar, const Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct							= context->EvalSub(scalar, ctImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->multScalar(-1.0);
	res_gpu->addScalar(scalar);
	res_gpu->multScalar(-1.0);

	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ct2Impl = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
		context->EvalSubInPlace(ct1Impl, ct2Impl);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
	res_gpu->sub(*ct2_gpu);
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		context->EvalSubInPlace(ct1Impl, scalar);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	res_gpu->addScalar(-scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalSubInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		context->EvalSubInPlace(scalar, ct1Impl);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	res_gpu->multScalar(-1.0);
	res_gpu->addScalar(scalar);
	res_gpu->multScalar(-1.0);
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

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ct2Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
		auto ct							= context->EvalMult(ct1Impl, ct2Impl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct2));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto ct2_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
	res_gpu->mult(*ct2_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ptImpl					= std::any_cast<const lbcrypto::ConstPlaintext&>(pt->cpu);
		auto ct							= context->EvalMult(ct1Impl, ptImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));
	this->LoadPlaintext(pt);

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto pt_gpu					= std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
	res_gpu->multPt(*pt_gpu);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(Plaintext& pt, const Ciphertext<DCRTPoly>& ct1) {
	return EvalMult(ct1, pt);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, double scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto ct							= context->EvalMult(ct1Impl, scalar);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->multScalar(scalar);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(double scalar, const Ciphertext<DCRTPoly>& ct1) {
	return EvalMult(ct1, scalar);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(const Ciphertext<DCRTPoly>& ct1, std::complex<double> scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto ct							= context->EvalMult(ct1Impl, scalar);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct1));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct1);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->multScalar(scalar);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalMult(std::complex<double> scalar, const Ciphertext<DCRTPoly>& ct1) {
	return EvalMult(ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt) {

	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ptImpl  = std::any_cast<const lbcrypto::ConstPlaintext&>(pt->cpu);
		auto res	  = context->EvalMult(ct1Impl, ptImpl);
		ct1->cpu	  = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(res);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);
	this->LoadPlaintext(pt);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	auto pt_gpu	 = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
	res_gpu->multPt(*pt_gpu);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, double scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		context->EvalMultInPlace(ct1Impl, scalar);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	res_gpu->multScalar(scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(double scalar, Ciphertext<DCRTPoly>& ct1) {
	EvalMultInPlace(ct1, scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, std::complex<double> scalar) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		context->EvalMultInPlace(ct1Impl, scalar);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct1);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	res_gpu->multScalar(scalar);
}

void CryptoContextImpl<DCRTPoly>::EvalMultInPlace(std::complex<double> scalar, Ciphertext<DCRTPoly>& ct1) {
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

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ct1Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct1->cpu);
		auto& ct2Impl = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct2->cpu);
		context->EvalMultMutableInPlace(ct1Impl, ct2Impl);
		return;
	}

	this->LoadCiphertext(ct1);
	this->LoadCiphertext(ct2);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct1->gpu));
	auto ct2_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct2->gpu));
	res_gpu->mult(*ct2_gpu);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSquare(const Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context					= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl					= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct							= context->EvalSquare(ctImpl);
		Ciphertext<DCRTPoly> ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		ciphertext->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return ciphertext;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->square();

	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalSquareInPlace(Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		context->EvalSquareInPlace(ctImpl);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct);

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	ct_gpu->square();
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalSquareMutable(Ciphertext<DCRTPoly>& ct) {
	return EvalSquare(ct);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalRotate(const Ciphertext<DCRTPoly>& ciphertext, int32_t index) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context				= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl				= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		auto ct						= context->EvalRotate(ctImpl, index);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return result;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->rotate(index);

	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalRotateInPlace(Ciphertext<DCRTPoly>& ciphertext, int32_t index) {

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context	= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl	= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		auto ct			= context->EvalRotate(ctImpl, index);
		ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ciphertext);

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ciphertext->gpu));
	ct_gpu->rotate(index);
}

std::shared_ptr<void> CryptoContextImpl<DCRTPoly>::EvalFastRotationPrecompute(const Ciphertext<DCRTPoly>& ct) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		return context->EvalFastRotationPrecompute(ctImpl);
	}

	// GPU path not needed.

	return nullptr;
}

#include <core/math/hal/bigintdyn/ubintdyn.h>

Ciphertext<DCRTPoly>
CryptoContextImpl<DCRTPoly>::EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(precomp);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(context->EvalFastRotation(ctImpl, index, m, casted));
		return result;
	}

	// GPU path. Inefficient for only one rotation.

	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto ct_gpu					= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	ct_gpu->rotate_hoisted({ (int)index }, { res_gpu.get() }, false);

	return result;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(digits);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(context->EvalFastRotationExt(ctImpl, index, casted, addFirst));
		return result;
	}

	// GPU path. Inefficient for only one rotation.

	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	auto ct_gpu					= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	ct_gpu->rotate_hoisted({ (int)index }, { res_gpu.get() }, true);

	return result;
}

std::vector<Ciphertext<DCRTPoly>>
CryptoContextImpl<DCRTPoly>::EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const uint32_t m, const std::shared_ptr<void>& precomp) {

	std::vector<Ciphertext<DCRTPoly>> results;

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(precomp);

		for (const auto& index : indices) {
			Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
			result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(context->EvalFastRotation(ctImpl, index, m, casted));
			results.push_back(result);
		}
		return results;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));

	// Create result ciphertexts.
	std::vector<FIDESlib::CKKS::Ciphertext*> results_gpu;
	std::vector<int32_t> indices_real;
	for (int indice : indices) {
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);

		if (indice != 0) {
			indices_real.push_back(indice);
			auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
			results_gpu.push_back(res_gpu.get());
		}
		results.push_back(result);
	}

	ct_gpu->rotate_hoisted(indices_real, results_gpu, false);
	return results;
}

std::vector<Ciphertext<DCRTPoly>>
CryptoContextImpl<DCRTPoly>::EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst) {

	std::vector<Ciphertext<DCRTPoly>> results;

	// Fall back to CPU.
	if (this->devices.empty()) {
		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto casted	  = std::static_pointer_cast<std::vector<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>>(digits);

		for (const auto& index : indices) {
			Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
			result->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(context->EvalFastRotationExt(ctImpl, index, casted, addFirst));
			results.push_back(result);
		}
		return results;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));

	std::vector<FIDESlib::CKKS::Ciphertext*> results_gpu;
	std::vector<int32_t> indices_real;
	for (int indice : indices) {
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);

		if (indice != 0) {
			indices_real.push_back(indice);
			auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
			results_gpu.push_back(res_gpu.get());
		}
		results.push_back(result);
	}

	ct_gpu->rotate_hoisted(indices_real, results_gpu, true);
	return results;
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalChebyshevSeries(const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context				= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl				= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto ct						= context->EvalChebyshevSeries(ctImpl, coeffs, a, b);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return result;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	FIDESlib::CKKS::evalChebyshevSeries(*res_gpu, coeffs, a, b);

	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalChebyshevSeriesInPlace(Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);
		auto res	  = context->EvalChebyshevSeries(ctImpl, coeffs, a, b);
		ct->cpu		  = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(res);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ct);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	FIDESlib::CKKS::evalChebyshevSeries(*res_gpu, coeffs, a, b);
}

std::vector<double> CryptoContextImpl<DCRTPoly>::GetChebyshevCoefficients(std::function<double(double)>& func, double a, double b, size_t degree) {
	return FIDESlib::CKKS::get_chebyshev_coefficients(func, a, b, degree);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::Rescale(const Ciphertext<DCRTPoly>& ciphertext) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context				= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl				= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		auto ct						= context->Rescale(ctImpl);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return result;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));
	res_gpu->rescale();

	return result;
}

void CryptoContextImpl<DCRTPoly>::RescaleInPlace(Ciphertext<DCRTPoly>& ciphertext) {

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& context	= std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl	= std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		auto ct			= context->Rescale(ctImpl);
		ciphertext->cpu = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return;
	}

	// GPU path.
	this->LoadCiphertext(ciphertext);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ciphertext->gpu));
	res_gpu->rescale();
}

void CryptoContextImpl<DCRTPoly>::SetLevel(Ciphertext<DCRTPoly>& ct, size_t level) {
	ct->SetLevel(level);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalBootstrap(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {

	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& ctImpl				= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		auto ct						= context->EvalBootstrap(ctImpl, numIterations, precision);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		return result;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ciphertext);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));

	FIDESlib::CKKS::Bootstrap(*res_gpu, res_gpu->slots, prescaled);

	return result;
}

void CryptoContextImpl<DCRTPoly>::EvalBootstrapInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) {
	auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);

	// Fall back to CPU.
	if (this->devices.empty()) {

		auto& ctImpl				= std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
		auto ct						= context->EvalBootstrap(ctImpl, numIterations, precision);
		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(ct);
		ciphertext					= result;
		return;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ciphertext));

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ciphertext->gpu));

	FIDESlib::CKKS::Bootstrap(*res_gpu, res_gpu->slots, prescaled);
}

Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::AccumulateSum(const Ciphertext<DCRTPoly>& ct, int slots, int stride) {

	if (this->devices.empty()) {
		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result_ct = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(ctImpl);

		for (int i = 0; i < log2(slots); i++) {
			int rot_idx = stride * (1 << i);
			auto tmp	= context->EvalRotate(result_ct, rot_idx);
			context->EvalAddInPlace(result_ct, tmp);
		}

		Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(this->self_reference.lock());
		result->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(result_ct);
		return result;
	}

	// GPU path.
	this->LoadCiphertext(const_cast<Ciphertext<DCRTPoly>&>(ct));

	Ciphertext<DCRTPoly> result = std::make_shared<CiphertextImpl<DCRTPoly>>(*ct);
	auto res_gpu				= std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(result->gpu));

	FIDESlib::CKKS::Accumulate(*res_gpu, 4, stride, slots);

	return result;
}

void CryptoContextImpl<DCRTPoly>::AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride) {

	if (this->devices.empty()) {
		auto& context = std::any_cast<const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(this->cpu);
		auto& ctImpl  = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ct->cpu);

		for (int i = 0; i < log2(slots); i++) {
			int rot_idx = stride * (1 << i);
			auto tmp	= context->EvalRotate(ctImpl, rot_idx);
			context->EvalAddInPlace(ctImpl, tmp);
		}

		return;
	}

	// GPU path.
	this->LoadCiphertext(ct);

	auto res_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));

	FIDESlib::CKKS::Accumulate(*res_gpu, 4, stride, slots);
}

void CryptoContextImpl<DCRTPoly>::ConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
  int gStep,
  int bStep,
  const std::vector<Plaintext>& pts,
  const std::vector<int>& indexes,
  int stride,
  int rowSize) {

	if (this->devices.empty()) {
		OPENFHE_THROW("Not implemented for CPU path");
	}

	// GPU path.
	this->LoadCiphertext(ct);
	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
	pts_gpu.reserve(pts.size());
	for (const auto& pt : pts) {
		this->LoadPlaintext(const_cast<Plaintext&>(pt));
		auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
		pts_gpu.push_back(pt_gpu.get());
	}

	if (rowSize == 0) {
		rowSize = bStep * gStep;
	}

	FIDESlib::CKKS::ConvolutionTransform(*ct_gpu, rowSize, bStep, pts_gpu, stride, indexes, gStep);
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

	if (this->devices.empty()) {
		OPENFHE_THROW("Not implemented for CPU path");
	}

	// GPU path.
	this->LoadCiphertext(ct);
	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct->gpu));
	std::vector<FIDESlib::CKKS::Plaintext*> pts_gpu;
	pts_gpu.reserve(pts.size());
	for (const auto& pt : pts) {
		this->LoadPlaintext(const_cast<Plaintext&>(pt));
		auto pt_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(pt->gpu));
		pts_gpu.push_back(pt_gpu.get());
	}

	// Load mask
	this->LoadPlaintext(mask);
	auto mask_gpu = std::static_pointer_cast<FIDESlib::CKKS::Plaintext>(this->GetDevicePlaintext(mask->gpu));

	if (rowSize == 0) {
		rowSize = bStep * gStep;
	}

	FIDESlib::CKKS::SpecialConvolutionTransform(*ct_gpu, rowSize, bStep, pts_gpu, *mask_gpu, stride, maskRotationStride, indexes, gStep);
}

// ---- Copy helpers ----

uint32_t CryptoContextImpl<DCRTPoly>::CopyDeviceCiphertext(const CiphertextImpl<DCRTPoly>& ct) {
	if (!ct.loaded) {
		OPENFHE_THROW("Ciphertext not loaded to any device");
	}

	auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
	auto ct_gpu		  = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->GetDeviceCiphertext(ct.gpu));
	auto new_ct		  = std::make_shared<FIDESlib::CKKS::Ciphertext>(context_gpu);
	new_ct->copy(*ct_gpu);
	uint32_t handle = this->RegisterDeviceCiphertext(std::move(new_ct));
	return handle;
}

// ---- Map Handling ----

uint32_t CryptoContextImpl<DCRTPoly>::RegisterDevicePlaintext(std::shared_ptr<void>&& p) {
	if (this->devices.empty()) {
		OPENFHE_THROW("No devices available to register plaintext");
	}
	device_plaintexts_mutex->lock();
	uint32_t handle = next_gpu_handle++;
	device_plaintexts.emplace(handle, std::move(p));
	device_plaintexts_mutex->unlock();
	return handle;
}

uint32_t CryptoContextImpl<DCRTPoly>::RegisterDeviceCiphertext(std::shared_ptr<void>&& c) {
	if (this->devices.empty()) {
		OPENFHE_THROW("No devices available to register ciphertext");
	}
	device_ciphertexts_mutex->lock();
	uint32_t handle = next_gpu_handle++;
	device_ciphertexts.emplace(handle, std::move(c));
	device_ciphertexts_mutex->unlock();
	return handle;
}

std::shared_ptr<void>& CryptoContextImpl<DCRTPoly>::GetDevicePlaintext(uint32_t handle) {
	device_plaintexts_mutex->lock_shared();
	auto& it = device_plaintexts.at(handle);
	device_plaintexts_mutex->unlock_shared();
	return it;
}

std::shared_ptr<void>& CryptoContextImpl<DCRTPoly>::GetDeviceCiphertext(uint32_t handle) {
	device_ciphertexts_mutex->lock_shared();
	auto& it = device_ciphertexts.at(handle);
	device_ciphertexts_mutex->unlock_shared();
	return it;
}

bool CryptoContextImpl<DCRTPoly>::EvictDevicePlaintext(uint32_t handle) {
	device_plaintexts_mutex->lock();
	auto result = device_plaintexts.erase(handle) > 0;
	device_plaintexts_mutex->unlock();
	return result;
}

bool CryptoContextImpl<DCRTPoly>::EvictDeviceCiphertext(uint32_t handle) {
	device_ciphertexts_mutex->lock();
	auto result = device_ciphertexts.erase(handle) > 0;
	device_ciphertexts_mutex->unlock();
	return result;
}

void CryptoContextImpl<DCRTPoly>::Synchronize() const {
	if (this->devices.empty() || !this->loaded) {
		return;
	}
	for (const auto& device : this->devices) {
		cudaSetDevice(device);
		cudaDeviceSynchronize();
		CudaCheckErrorModNoSync;
	}
}

std::vector<int> CryptoContextImpl<DCRTPoly>::GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep) {
	return FIDESlib::CKKS::GetConvolutionTransformRotationIndices(rowSize, bStep, stride, gStep);
}

} // namespace fideslib