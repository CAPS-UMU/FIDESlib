#include "Ciphertext.hpp"
#include "CKKS/Ciphertext.cuh"
#include "Definitions.hpp"

#include <iostream>
#include <openfhe.h>

namespace fideslib {

CiphertextImpl<DCRTPoly>::~CiphertextImpl() {
	if (this->loaded && this->gpu != 0 && this->parent_context) {
		this->parent_context->EvictDeviceCiphertext(this->gpu);
		this->gpu = 0;
	}
}

CiphertextImpl<DCRTPoly>::CiphertextImpl(const CryptoContext<DCRTPoly>&& context) : parent_context(context) {
	if (!context) {
		OPENFHE_THROW("Cannot create Ciphertext with null CryptoContext");
	}
}

// ---- Copy ----

CiphertextImpl<DCRTPoly>::CiphertextImpl(const CiphertextImpl<DCRTPoly>& other) {
	// Share CPU ciphertext and detach only on first CPU mutation.
	auto const& other_cpu = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(other.cpu);
	this->cpu					= std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(other_cpu);
	this->need_lazy_copy = true;

	// Copy underlying GPU ciphertext if loaded.
	this->loaded = other.loaded;
	if (this->loaded) {
		this->gpu = other.parent_context->CopyDeviceCiphertext(other);
	} else {
		this->gpu = 0;
	}

	// Copy parent context.
	this->parent_context = other.parent_context;
	this->original_level = other.original_level;
}

CiphertextImpl<DCRTPoly>::CiphertextImpl(const Ciphertext<DCRTPoly>& other) : CiphertextImpl<DCRTPoly>(static_cast<const CiphertextImpl<DCRTPoly>&>(other)) {
}

// ---- Clone ----

Ciphertext<DCRTPoly> CiphertextImpl<DCRTPoly>::Clone() const {
	Ciphertext<DCRTPoly> clone = std::make_shared<CiphertextImpl<DCRTPoly>>(*this);
	return clone;
}

// ---- Getters ----

size_t CiphertextImpl<DCRTPoly>::GetLevel() const {

	if (!this->loaded) {
		// Fall back to CPU.
		auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->cpu);
		return ct->GetLevel();
	}

	// GPU path. Depth is reversed in FIDESlib, must do depth = maxDepth - depth
	auto ct_gpu	  = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->parent_context->GetDeviceCiphertext(this->gpu));
	auto maxDepth = this->parent_context->multiplicative_depth;
	return maxDepth - ct_gpu->getLevel();
}

size_t CiphertextImpl<DCRTPoly>::GetNoiseScaleDeg() const {

	if (!this->loaded) {
		// Fall back to CPU.
		auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->cpu);
		return ct->GetNoiseScaleDeg();
	}

	// GPU path.
	auto ct_gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->parent_context->GetDeviceCiphertext(this->gpu));
	return ct_gpu->NoiseLevel;
}

// ---- Setters ----

void CiphertextImpl<DCRTPoly>::SetSlots(size_t slots) {

	if (!this->loaded) {
		// Fall back to CPU.
		this->EnsureLazyCPUCopy();
		auto& ct = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->cpu);
		ct->SetSlots(slots);
		return;
	}
	// GPU path.
	auto ct_gpu	  = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->parent_context->GetDeviceCiphertext(this->gpu));
	ct_gpu->slots = static_cast<int>(slots);
}

void CiphertextImpl<DCRTPoly>::SetLevel(size_t level) {

	if (!this->loaded) {
		// Fall back to CPU.
		this->EnsureLazyCPUCopy();
		auto& ct = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->cpu);

		size_t currentTowers = ct->GetElements()[0].GetNumOfElements();
		size_t currentLevel	 = ct->GetLevel();

		size_t totalPrimes	= currentTowers + currentLevel;
		size_t targetTowers = totalPrimes - level;

		if (currentTowers > targetTowers) {
			// Need to drop towers
			size_t towersToDrop = currentTowers - targetTowers;

			auto& elements = ct->GetElements();
			for (auto& elem : elements) {
				elem.DropLastElements(towersToDrop);
			}
		}

		ct->SetLevel(level);

		return;
	}

	// GPU path.
	auto ct_gpu	  = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(this->parent_context->GetDeviceCiphertext(this->gpu));
	auto maxDepth = this->parent_context->multiplicative_depth;
	ct_gpu->dropToLevel(maxDepth - level);
}

void CiphertextImpl<DCRTPoly>::EnsureLazyCPUCopy() {
	if (!this->need_lazy_copy) {
		return;
	}

	auto const& ct_cpu = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->cpu);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cpu_copy = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*ct_cpu);
	this->cpu											   = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(std::move(cpu_copy));
	this->need_lazy_copy = false;
}

// ---- Operators ----

Ciphertext<DCRTPoly> operator+(const Ciphertext<DCRTPoly>& lhs, const Ciphertext<DCRTPoly>& rhs) {
	if (lhs->parent_context.get() != rhs->parent_context.get()) {
		OPENFHE_THROW("Cannot add ciphertexts from different contexts");
	}

	return lhs->parent_context->EvalAdd(lhs, rhs);
}

} // namespace fideslib
