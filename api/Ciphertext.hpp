#ifndef API_CIPHERTEXT_HPP
#define API_CIPHERTEXT_HPP

#include <any>
#include <memory>

#include "Definitions.hpp"
#include "CryptoContext.hpp"

namespace fideslib {

/// @brief Specialization of Ciphertext for the DCRTPoly representation.
template <> class CiphertextImpl<DCRTPoly> {
  public:
	CiphertextImpl()  = delete;
	~CiphertextImpl();

	CiphertextImpl(const CryptoContext<DCRTPoly>&& context);

	// ---- Copy ----

	CiphertextImpl(const CiphertextImpl<DCRTPoly>&);
	CiphertextImpl(const Ciphertext<DCRTPoly>&);
	CiphertextImpl& operator=(const CiphertextImpl<DCRTPoly>&) = delete;
	CiphertextImpl& operator=(const Ciphertext<DCRTPoly>&)	   = delete;

	// ---- Move ----

	CiphertextImpl(CiphertextImpl<DCRTPoly>&&)			  = delete;
	CiphertextImpl(Ciphertext<DCRTPoly>&&)				  = delete;
	CiphertextImpl& operator=(CiphertextImpl<DCRTPoly>&&) = delete;
	CiphertextImpl& operator=(Ciphertext<DCRTPoly>&&)	  = delete;

	// ---- Clone ----
	Ciphertext<DCRTPoly> Clone() const;

	// ---- Getters ----

    size_t GetLevel() const;
	size_t GetNoiseScaleDeg() const;

	// ---- Setters ----
	void SetSlots(size_t slots);
	void SetLevel(size_t level);
	void EnsureLazyCPUCopy();

	// ---- Internal State ----

	bool need_lazy_copy = false;
	std::any cpu;
	uint32_t gpu = 0;
	/// @brief Flag indicating whether the ciphertext is loaded to the devices.
	bool loaded = false;
	/// @brief Parent context.
	CryptoContext<DCRTPoly> parent_context;
	/// @brief Original level of the ciphertext when loaded.
	size_t original_level = 0;
};

// ---- Override Operators ----
Ciphertext<DCRTPoly> operator+(const Ciphertext<DCRTPoly>& lhs, const Ciphertext<DCRTPoly>& rhs);

} // namespace fideslib

#endif // API_CIPHERTEXT_HPP