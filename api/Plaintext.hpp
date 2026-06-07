#ifndef API_PLAINTEXT_HPP
#define API_PLAINTEXT_HPP

#include <any>
#include <complex>
#include <iostream>
#include <memory>
#include <vector>

#include "Definitions.hpp"

namespace fideslib {

/// @brief Plaintext representation for the CKKS-RNS scheme.
class PlaintextImpl {
  public:
	PlaintextImpl()	 = default;
	~PlaintextImpl() = default; // device slot frees its backend payload via RAII

	PlaintextImpl(const CryptoContext<DCRTPoly>&& context);

	// ---- Copy ----

	PlaintextImpl(const PlaintextImpl&);
	PlaintextImpl(const Plaintext&);
	PlaintextImpl& operator=(const PlaintextImpl&) = delete;
	PlaintextImpl& operator=(const Plaintext&)	   = delete;

	// ---- Move ----

	PlaintextImpl(PlaintextImpl&&)			  = delete;
	PlaintextImpl(Plaintext&&)				  = delete;
	PlaintextImpl& operator=(PlaintextImpl&&) = delete;
	PlaintextImpl& operator=(Plaintext&&)	  = delete;

	// ---- Functions ----

	void SetLength(size_t length);
	void SetSlots(uint32_t slots);

	double GetLogPrecision() const;
	uint32_t GetLevel() const;
	std::vector<std::complex<double>> GetCKKSPackedValue() const;
	std::vector<double> GetRealPackedValue() const;

	// ---- Friend Operators ----

	friend std::ostream& operator<<(std::ostream& os, const PlaintextImpl& pt);

	// ---- Internal State ----

	/// @brief Canonical host value (lbcrypto::Plaintext).
	std::any host;
	/// @brief Backend-resident payload (the CUDA backend stores a shared_ptr to its device plaintext); empty == not resident.
	std::any device;
	/// @brief Parent context.
	CryptoContext<DCRTPoly> parent_context;
};

// ---- Override Operators ----

std::ostream& operator<<(std::ostream& os, const Plaintext& pt);
} // namespace fideslib

#endif // API_PLAINTEXT_HPP