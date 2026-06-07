#include "Ciphertext.hpp"
#include "Definitions.hpp"

#include <iostream>
#include <openfhe.h>

namespace fideslib {

CiphertextImpl<DCRTPoly>::CiphertextImpl(const CryptoContext<DCRTPoly>&& context) : parent_context(context) {
	if (!context) {
		OPENFHE_THROW("Cannot create Ciphertext with null CryptoContext");
	}
}

// ---- Copy ----

CiphertextImpl<DCRTPoly>::CiphertextImpl(const CiphertextImpl<DCRTPoly>& other) {
	// Share host ciphertext and detach only on first host mutation.
	auto const& other_host = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(other.host);
	this->host			   = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(other_host);
	this->need_lazy_copy   = true;

	// Copy parent context.
	this->parent_context = other.parent_context;

	// Cloning any backend-resident payload is the engine's decision, not ours: it returns an empty
	// slot when `other` is not resident (always so on the CPU backend).
	this->device = other.parent_context->CloneCiphertextBackend(other);
}

CiphertextImpl<DCRTPoly>::CiphertextImpl(const Ciphertext<DCRTPoly>& other) : CiphertextImpl<DCRTPoly>(static_cast<const CiphertextImpl<DCRTPoly>&>(other)) {
}

// ---- Clone ----

Ciphertext<DCRTPoly> CiphertextImpl<DCRTPoly>::Clone() const {
	Ciphertext<DCRTPoly> clone = std::make_shared<CiphertextImpl<DCRTPoly>>(*this);
	return clone;
}

// ---- Getters / setters ----
//
// These delegate unconditionally to the backend; the engine decides whether to read the host value or
// the device-resident copy. The host primitives below hold the host computation the engines reuse.

size_t CiphertextImpl<DCRTPoly>::GetLevel() const {
	return this->parent_context->CiphertextLevel(*this);
}

size_t CiphertextImpl<DCRTPoly>::GetNoiseScaleDeg() const {
	return this->parent_context->CiphertextNoiseScaleDeg(*this);
}

void CiphertextImpl<DCRTPoly>::SetSlots(size_t slots) {
	this->parent_context->SetCiphertextSlots(*this, slots);
}

void CiphertextImpl<DCRTPoly>::SetLevel(size_t level) {
	this->parent_context->SetCiphertextLevel(*this, level);
}

// ---- Host primitives (invoked by the engine backends; no backend logic of their own) ----

size_t CiphertextImpl<DCRTPoly>::GetLevelHost() const {
	auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	return ct->GetLevel();
}

size_t CiphertextImpl<DCRTPoly>::GetNoiseScaleDegHost() const {
	auto& ct = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	return ct->GetNoiseScaleDeg();
}

void CiphertextImpl<DCRTPoly>::SetSlotsHost(size_t slots) {
	this->EnsureLazyHostCopy();
	auto& ct = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	ct->SetSlots(slots);
}

void CiphertextImpl<DCRTPoly>::SetLevelHost(size_t level) {
	this->EnsureLazyHostCopy();
	auto& ct = std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);

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
}

void CiphertextImpl<DCRTPoly>::EnsureLazyHostCopy() {
	auto const& ct_host = std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(this->host);
	// Detach before in-place mutation if still lazily shared or the underlying
	// OpenFHE ciphertext has other owners (e.g. a Clone): copy-on-write.
	if (!this->need_lazy_copy && ct_host.use_count() <= 1) {
		return;
	}
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> host_copy = std::make_shared<lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>>(*ct_host);
	this->host										   = std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(std::move(host_copy));
	this->need_lazy_copy							   = false;
}

// ---- Operators ----

Ciphertext<DCRTPoly> operator+(const Ciphertext<DCRTPoly>& lhs, const Ciphertext<DCRTPoly>& rhs) {
	if (lhs->parent_context.get() != rhs->parent_context.get()) {
		OPENFHE_THROW("Cannot add ciphertexts from different contexts");
	}

	return lhs->parent_context->EvalAdd(lhs, rhs);
}

} // namespace fideslib
