#include "GenCryptoContext.hpp"

#include <openfhe.h>

#include <shared_mutex>
#include <vector>

namespace fideslib {

CryptoContext<DCRTPoly> GenCryptoContext(CCParams<CryptoContextCKKSRNS>& params) {
	auto& impl_params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(params.cpu);
	auto cc			  = lbcrypto::GenCryptoContext(impl_params);

	if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE))
		std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.clear();

	CryptoContextImpl<DCRTPoly> context;
	context.cpu						 = std::make_any<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(cc);
	context.devices					 = std::vector(params.devices);
	context.auto_load_plaintexts	 = params.plaintextAutoload;
	context.auto_load_ciphertexts	 = params.ciphertextAutoload;
	context.multiplicative_depth	 = impl_params.GetMultiplicativeDepth();
	context.keyDist					 = params.keyDist;
	auto ptr						 = std::make_shared<CryptoContextImpl<DCRTPoly>>(std::move(context));
	ptr->self_reference				 = std::weak_ptr<CryptoContextImpl<DCRTPoly>>(ptr);

	return ptr;
}

} // namespace fideslib