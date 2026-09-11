//
// Created by carlosad on 26/09/24.
//

#ifndef GPUCKKS_KEYSWITCHINGKEY_CUH
#define GPUCKKS_KEYSWITCHINGKEY_CUH

#include <cassert>
#include <cinttypes>
#include <memory>
#include <vector>

#include "RNSPoly.cuh"
#include "openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib {
namespace CKKS {

class KeySwitchingKey {
	static constexpr const char* loc{ "KeySwitchingKey" };
	CudaNvtxRange my_range;

  public:
	KeyHash keyID;
	/**
	 * Non-owning, and deliberately neither a reference nor a shared_ptr.
	 *
	 * A `Context&` member dangles: CryptoContextImpl::LoadContext moves its local Context into a
	 * std::any, so the referent dies when LoadContext returns. Holding a `Context` by value fixes
	 * that but closes a reference cycle, because keys live inside ContextData::precom.keys: the
	 * ContextData can then never reach zero references, and every context dropped through the API
	 * keeps all of its device memory - eval and rotation keys, bootstrap plaintexts, auxiliary
	 * buffers - for the life of the process.
	 *
	 * A weak_ptr fixes the dangling reference without owning the object this key is a member of.
	 * It cannot expire while the key is alive, for exactly that reason, and context() asserts it.
	 */
	std::weak_ptr<ContextData> cc_weak;

	/** The context this key belongs to. Every use is a read of one of its fields. */
	[[nodiscard]] Context context() const {
		Context cc = cc_weak.lock();
		assert(cc != nullptr);
		return cc;
	}
	RNSPoly a;
	RNSPoly b;
	// std::vector<RNSPoly> mgpu_a;
	// std::vector<RNSPoly> mgpu_b;

	explicit KeySwitchingKey(Context& cc);

	void Initialize(RawKeySwitchKey& rkk);
};

} // namespace CKKS
} // namespace FIDESlib

#endif // GPUCKKS_KEYSWITCHINGKEY_CUH
