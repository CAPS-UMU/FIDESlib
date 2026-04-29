//
// Created by carlosad on 26/09/24.
//

#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"
#include <source_location>
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
// constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc = std::source_location;
// constexpr int PREFIX_SIZE = 23;
#endif

namespace FIDESlib::CKKS {
void KeySwitchingKey::Initialize(RawKeySwitchKey& rkk) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc);
	keyID = rkk.keyid;

	a.generateDecompAndDigit(true);
	b.generateDecompAndDigit(true);
	if (cc->GPUid.size() > 1) {
		a.grow(cc->L, false, true);
		b.grow(cc->L, false, true);
	}
	a.loadDecompDigit(rkk.r_key[0], rkk.r_key_moduli[0]);
	b.loadDecompDigit(rkk.r_key[1], rkk.r_key_moduli[1]);

	cudaDeviceSynchronize();
}

KeySwitchingKey::KeySwitchingKey(Context& cc)
: my_range(loc, LIFETIME), keyID(""), cc((assert(cc != nullptr), CudaNvtxStart(std::string{ sc::current().function_name() }.substr()), cc)),
  a(*cc, -1, false, true), b(*cc, -1, false, true) {
	CudaNvtxStop();
	/*
	if (cc.GPUid.size() > 1) {
		for (int j = 0; j < cc.dnum; ++j) {
			mgpu_a.emplace_back(cc, -1);
			mgpu_b.emplace_back(cc, -1);
		}
	}
	 */
}
} // namespace FIDESlib::CKKS
