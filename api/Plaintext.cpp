#include "Plaintext.hpp"
#include "CryptoContext.hpp"

#include <openfhe.h>

namespace fideslib {

PlaintextImpl::PlaintextImpl(const CryptoContext<DCRTPoly>&& context)
  : parent_context(context) {
	if (!context) {
		OPENFHE_THROW("Cannot create Ciphertext with null CryptoContext");
	}
}

// ---- Functions ----

void PlaintextImpl::SetLength(size_t length) {
	if (this->host.has_value()) {
		auto& impl = std::any_cast<lbcrypto::Plaintext&>(this->host);
		impl->SetLength(length);
	}
}

void PlaintextImpl::SetSlots(uint32_t slots) {
	if (this->host.has_value()) {
		auto& impl = std::any_cast<lbcrypto::Plaintext&>(this->host);
		impl->SetSlots(slots);
	}
}

double PlaintextImpl::GetLogPrecision() const {
	if (this->host.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->host);
		return impl->GetLogPrecision();
	}
	return 0.0;
}

uint32_t PlaintextImpl::GetLevel() const {
	if (this->host.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->host);
		return impl->GetLevel();
	}
	return 0;
}

std::vector<std::complex<double>> PlaintextImpl::GetCKKSPackedValue() const {
	if (this->host.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->host);
		return impl->GetCKKSPackedValue();
	}
	return {};
}

std::vector<double> PlaintextImpl::GetRealPackedValue() const {
	if (this->host.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->host);
		return impl->GetRealPackedValue();
	}
	return {};
}

// ---- Friend Operators ----

std::ostream& operator<<(std::ostream& os, const PlaintextImpl& pt) {
	if (pt.host.has_value()) {
		const auto& impl = std::any_cast<const lbcrypto::Plaintext&>(pt.host);
		os << impl;
	} else {
		os << "Empty Plaintext";
	}
	return os;
}

std::ostream& operator<<(std::ostream& os, const Plaintext& pt) {
	if (pt && pt->host.has_value()) {
		const auto& impl = std::any_cast<const lbcrypto::Plaintext&>(pt->host);
		os << impl;
	} else {
		os << "Empty Plaintext";
	}
	return os;
}

} // namespace fideslib
