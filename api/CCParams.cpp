#include "CCParams.hpp"
#include "lattice/constants-lattice.h"

#include <openfhe.h>

#include <cassert>
#include <iostream>
#include <string>

namespace fideslib {

CCParams<CryptoContextCKKSRNS>::CCParams() {
	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> params;
	this->host = std::make_any<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>>(std::move(params));
}

// ---- CKKS Parameters ----

void CCParams<CryptoContextCKKSRNS>::SetMultiplicativeDepth(uint32_t depth) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetMultiplicativeDepth(depth);
}

void CCParams<CryptoContextCKKSRNS>::SetScalingModSize(uint32_t size) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetScalingModSize(size);
}

void CCParams<CryptoContextCKKSRNS>::SetBatchSize(uint32_t size) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetBatchSize(size);
}

void CCParams<CryptoContextCKKSRNS>::SetRingDim(uint32_t dim) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetRingDim(dim);
}

void CCParams<CryptoContextCKKSRNS>::SetScalingTechnique(ScalingTechnique tech) {
	auto& params	   = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	auto scale_openfhe = static_cast<lbcrypto::ScalingTechnique>(tech);
	assert((int)scale_openfhe == (int)tech);
	params.SetScalingTechnique(scale_openfhe);
}

void CCParams<CryptoContextCKKSRNS>::SetNumLargeDigits(uint32_t numDigits) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetNumLargeDigits(numDigits);
}

void CCParams<CryptoContextCKKSRNS>::SetFirstModSize(uint32_t size) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetFirstModSize(size);
}

void CCParams<CryptoContextCKKSRNS>::SetDigitSize(uint32_t size) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	params.SetDigitSize(size);
}

void CCParams<CryptoContextCKKSRNS>::SetKeySwitchTechnique(KeySwitchTechnique tech) {
	auto& params	= std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	auto ks_openfhe = static_cast<lbcrypto::KeySwitchTechnique>(tech);
	assert((int)ks_openfhe == (int)tech);
	params.SetKeySwitchTechnique(ks_openfhe);
}

void CCParams<CryptoContextCKKSRNS>::SetSecretKeyDist(SecretKeyDist dist) {
	auto& params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);

	// Record the requested distribution as-is. The CUDA backend's restriction against sparse-ternary keys
	// is enforced at context construction (GenCryptoContext), so an unsupported choice fails loudly
	// there rather than being silently downgraded here.
	if (dist == SecretKeyDist::SPARSE_TERNARY) {
		params.SetSecretKeyDist(lbcrypto::SPARSE_TERNARY);
	} else {
		params.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
	}
	keyDist = dist;
}

void CCParams<CryptoContextCKKSRNS>::SetSecurityLevel(SecurityLevel level) {
	auto& params	= std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	auto sl_openfhe = static_cast<lbcrypto::SecurityLevel>(level);
	assert((int)sl_openfhe == (int)level);
	params.SetSecurityLevel(sl_openfhe);
}

// ---- Getters ----

SecretKeyDist CCParams<CryptoContextCKKSRNS>::GetSecretKeyDist() const {
	auto& params	 = std::any_cast<const lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	auto skd_openfhe = params.GetSecretKeyDist();
	return static_cast<SecretKeyDist>(skd_openfhe);
}

uint32_t CCParams<CryptoContextCKKSRNS>::GetMultiplicativeDepth() const {
	auto& params = std::any_cast<const lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	return params.GetMultiplicativeDepth();
}

uint32_t CCParams<CryptoContextCKKSRNS>::GetBatchSize() const {
	auto& params = std::any_cast<const lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(host);
	return params.GetBatchSize();
}

// ---- Backend Parameters ----

void CCParams<CryptoContextCKKSRNS>::SetBackend(Backend backend) {
	if (!IsBackendAvailable(backend)) {
		// Build a descriptive name for the unavailable backend.
		const char* name = "unknown";
		switch (backend) {
		case Backend::CPU:  name = "CPU";  break;
		case Backend::CUDA: name = "CUDA"; break;
		}
		OPENFHE_THROW(std::string(name) + " backend requested but not available (FIDESlib was not built with support for this backend)");
	}
	this->backend = backend;
}

void CCParams<CryptoContextCKKSRNS>::SetPlaintextAutoload(bool autoload) {
	this->plaintextAutoload = autoload;
}

void CCParams<CryptoContextCKKSRNS>::SetCiphertextAutoload(bool autoload) {
	this->ciphertextAutoload = autoload;
}

} // namespace fideslib