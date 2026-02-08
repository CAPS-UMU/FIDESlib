#ifndef API_CCPARAMS_HPP
#define API_CCPARAMS_HPP

#include <any>
#include <cinttypes>
#include <sys/types.h>
#include <vector>

#include "Definitions.hpp"

namespace fideslib {

/// @brief Class for managing general parameter sets.
/// @tparam T Scheme type.
template <typename T> class CCParams;

/// @brief Specialization of CCParams for the CKKS-RNS scheme.
template <> class CCParams<CryptoContextCKKSRNS> {
  public:
	CCParams();
	~CCParams() = default;

	// ---- Copy ----

	CCParams(const CCParams&)			 = delete;
	CCParams& operator=(const CCParams&) = delete;

	// ---- Move ----

	CCParams(CCParams&&)			= delete;
	CCParams& operator=(CCParams&&) = delete;

	// ---- CKKS Parameters ----

	void SetMultiplicativeDepth(uint32_t depth);
	void SetScalingModSize(uint32_t size);
	void SetBatchSize(uint32_t size);
	void SetRingDim(uint32_t dim);
	void SetScalingTechnique(ScalingTechnique tech);
	void SetNumLargeDigits(uint32_t numDigits);
	void SetFirstModSize(uint32_t size);
	void SetDigitSize(uint32_t size);
	void SetKeySwitchTechnique(KeySwitchTechnique tech);
	void SetSecretKeyDist(SecretKeyDist dist);
	void SetSecurityLevel(SecurityLevel level);

	// ---- Device Parameters ----

	void SetDevices(std::vector<int>&& devices);
	void SetPlaintextAutoload(bool autoload);
	void SetCiphertextAutoload(bool autoload);
	void SetCKKSDataType(CKKSDataType ckksdt);

	// ---- Getters ----
	SecretKeyDist GetSecretKeyDist() const;
	uint32_t GetMultiplicativeDepth() const;
	uint32_t GetBatchSize() const;

	// ---- Internal State ----

	std::any cpu;
	std::vector<int> devices = { 0 };
	SecretKeyDist keyDist	 = UNIFORM_TERNARY;
	bool plaintextAutoload	 = false;
	bool ciphertextAutoload	 = true;
};

} // namespace fideslib

#endif // API_CCPARAMS_HPP