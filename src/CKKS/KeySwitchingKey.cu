//
// Created by carlosad on 26/09/24.
//

#include <source_location>
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"

namespace FIDESlib::CKKS {
void KeySwitchingKey::Initialize(Context& cc, RawKeySwitchKey& rkk) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    for (int j = 0; j < cc.dnum; ++j) {
        a.generateDecompAndDigit();
        b.generateDecompAndDigit();
    }

    a.loadDecompDigit(rkk.r_key[0], rkk.r_key_moduli[0]);
    b.loadDecompDigit(rkk.r_key[1], rkk.r_key_moduli[1]);
    cudaDeviceSynchronize();
}

KeySwitchingKey::KeySwitchingKey(Context& cc)
    : my_range(loc, LIFETIME),
      cc((CudaNvtxStart(std::string{std::source_location::current().function_name()}.substr(18 + strlen(loc))), cc)),
      a(cc, -1),
      b(cc, -1) {
    CudaNvtxStop();
}
}  // namespace FIDESlib::CKKS
