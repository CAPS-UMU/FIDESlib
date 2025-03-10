//
// Created by carlosad on 26/09/24.
//

#ifndef GPUCKKS_KEYSWITCHINGKEY_CUH
#define GPUCKKS_KEYSWITCHINGKEY_CUH

#include <cinttypes>
#include <vector>

#include "RNSPoly.cuh"
#include "openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib {
namespace CKKS {
class Context;

class KeySwitchingKey {
    static constexpr const char* loc{"KeySwitchingKey"};
    CudaNvtxRange my_range;

   public:
    Context& cc;
    RNSPoly a;
    RNSPoly b;

    explicit KeySwitchingKey(Context& cc);

    void Initialize(Context& cc, RawKeySwitchKey& rkk);
};

}  // namespace CKKS
}  // namespace FIDESlib

#endif  //GPUCKKS_KEYSWITCHINGKEY_CUH
