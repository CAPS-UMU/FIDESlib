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

PlaintextImpl::~PlaintextImpl() {
	if (this->loaded && this->gpu != 0 && this->parent_context) {
		this->parent_context->EvictDevicePlaintext(this->gpu);
		this->gpu = 0;
	}
}

// ---- Functions ----

void PlaintextImpl::SetLength(size_t length) {
	if (this->cpu.has_value()) {
		auto& impl = std::any_cast<lbcrypto::Plaintext&>(this->cpu);
		impl->SetLength(length);
	}
}

void PlaintextImpl::SetSlots(uint32_t slots) {
	if (this->cpu.has_value()) {
		auto& impl = std::any_cast<lbcrypto::Plaintext&>(this->cpu);
		impl->SetSlots(slots);
	}
}

double PlaintextImpl::GetLogPrecision() const {
	if (this->cpu.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->cpu);
		return impl->GetLogPrecision();
	}
	return 0.0;
}

uint32_t PlaintextImpl::GetLevel() const {
	if (this->cpu.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->cpu);
		return impl->GetLevel();
	}
	return 0;
}

uint32_t PlaintextImpl::GetSlots() const {
	if (this->cpu.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->cpu);
		return impl->GetSlots();
	}
	return 0;
}

std::vector<std::complex<double>> PlaintextImpl::GetCKKSPackedValue() const {
	if (this->cpu.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->cpu);
		return impl->GetCKKSPackedValue();
	}
	return {};
}

std::vector<double> PlaintextImpl::GetRealPackedValue() const {
	if (this->cpu.has_value()) {
		auto& impl = std::any_cast<const lbcrypto::Plaintext&>(this->cpu);
		return impl->GetRealPackedValue();
	}
	return {};
}

// ---- Friend Operators ----

std::ostream& operator<<(std::ostream& os, const PlaintextImpl& pt) {
	if (pt.cpu.has_value()) {
		const auto& impl = std::any_cast<const lbcrypto::Plaintext&>(pt.cpu);
		os << impl;
	} else {
		os << "Empty Plaintext";
	}
	return os;
}

std::ostream& operator<<(std::ostream& os, const Plaintext& pt) {
	if (pt && pt->cpu.has_value()) {
		const auto& impl = std::any_cast<const lbcrypto::Plaintext&>(pt->cpu);
		os << impl;
	} else {
		os << "Empty Plaintext";
	}
	return os;
}

bool operator==(const PlaintextImpl& lhs, const PlaintextImpl& rhs) {
	if (!lhs.cpu.has_value() || !rhs.cpu.has_value())
		return lhs.cpu.has_value() == rhs.cpu.has_value();
	const auto& l = std::any_cast<const lbcrypto::Plaintext&>(lhs.cpu);
	const auto& r = std::any_cast<const lbcrypto::Plaintext&>(rhs.cpu);
	if (!l || !r)
		return l == r;
	return *l == *r;
}

bool operator!=(const PlaintextImpl& lhs, const PlaintextImpl& rhs) {
	return !(lhs == rhs);
}

} // namespace fideslib
