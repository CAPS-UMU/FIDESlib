//
// Created by carlosad on 16/03/24.
//

#ifndef FIDESLIB_LIMBUTILS_CUH
#define FIDESLIB_LIMBUTILS_CUH

#include <optional>
#include "CudaUtils.cuh"

namespace FIDESlib {

    enum TYPE {
        U32, U64
    };

    struct PrimeRecord {
        uint64_t p = (static_cast<uint64_t>(1) << 31) - 1;
        std::optional<TYPE> type{};
        int bits = -1;
    };

    struct LimbRecord {
        int id;
        TYPE type;
        Stream stream;
        int digit = -1;
        int destDeviceRank = -1;
        /*
        LimbRecord(const int id, const TYPE type, cudaStream_t & stream)
            : id(id), type(type), stream(stream) {
            assert(type == U32 || type == U64);
        }
        */
        /*
        void print(){

        }
         */
    };

    //LimbRecord a{ .id = 0, .type = U32};

#define SWITCH(limb, function) \
do{                 \
    if(limb.index() == FIDESlib::TYPE::U32) {std::get<FIDESlib::TYPE::U32>(limb).function;} \
    if(limb.index() == FIDESlib::TYPE::U64) {std::get<FIDESlib::TYPE::U64>(limb).function;} \
} while(0)

#define STREAM(limb) \
    ( (limb).index() == U32 ? std::get<U32>(limb).stream : std::get<U64>(limb).stream )

#define PRIMEID(limb) \
    ( (limb).index() == U32 ? std::get<U32>(limb).primeid : std::get<U64>(limb).primeid )

#define SWITCH_RET(limb, function, ret) \
do{                 \
    if(limb.index() == U32) {ret = std::get<U32>(limb).function;} \
    if(limb.index() == U64) {ret = std::get<U64>(limb).function;} \
} while(0)


}
#endif //FIDESLIB_LIMBUTILS_CUH
