#include "GenCryptoContext.hpp"

#include "engine/Backend.hpp"
#include "engine/Engine.hpp"

#include <openfhe.h>

#include <shared_mutex>
#include <vector>

namespace fideslib {

CryptoContext<DCRTPoly> GenCryptoContext(CCParams<CryptoContextCKKSRNS>& params) {
	// CUDA needs a uniform main key, so a sparse-ternary secret key is unsupported there; reject it
	// loudly here, now that the backend and key distribution are both final, rather than silently
	// downgrading it (which would also desync params from the keyDist the bootstrap path reads).
	if (params.backend == Backend::CUDA && params.keyDist == SPARSE_TERNARY) {
		OPENFHE_THROW("CUDA backend does not support SPARSE_TERNARY (a sparse main key); "
					  "use UNIFORM_TERNARY, SPARSE_ENCAPSULATED for sparse bootstrapping, or the CPU backend");
	}
	auto& impl_params = std::any_cast<lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>&>(params.host);
	auto cc			  = lbcrypto::GenCryptoContext(impl_params);

	if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE))
		std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.clear();

	CryptoContextImpl<DCRTPoly> context;
	context.host				  = std::make_any<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(cc);
	context.engine_				  = MakeEngine(params.backend);
	context.auto_load_plaintexts  = params.plaintextAutoload;
	context.auto_load_ciphertexts = params.ciphertextAutoload;
	context.multiplicative_depth  = impl_params.GetMultiplicativeDepth();
	context.keyDist				  = params.keyDist;
	auto ptr					  = std::make_shared<CryptoContextImpl<DCRTPoly>>(std::move(context));
	ptr->self_reference			  = std::weak_ptr<CryptoContextImpl<DCRTPoly>>(ptr);

	return ptr;
}

} // namespace fideslib