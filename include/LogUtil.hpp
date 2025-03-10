//
// Created by carlosad on 14/03/24.
//

#ifndef FIDESLIB_LOGUTIL_HPP
#define FIDESLIB_LOGUTIL_HPP

#include <iostream>
#include <cassert>

namespace FIDESlib {
    constexpr uint64_t ALL(1 << 0);
    constexpr uint64_t DEBUG(1 << 1);
    constexpr uint64_t CPU(1 << 2);
    constexpr uint64_t MULT(1 << 3);
    constexpr uint64_t MODUP(1 << 4);
    constexpr uint64_t SWAP(1 << 5);
    constexpr uint64_t MODDOWN(1 << 6);
    constexpr uint64_t MEMORY(1 << 7);
    constexpr uint64_t PARAMS(1 << 8);
    constexpr uint64_t NTT(1 << 9);
    constexpr uint64_t KEYSWITCH(1 << 10);

    constexpr uint64_t ENABLED_FLAGS = (0ll
                                        | DEBUG
                                        | CPU
                                        | MULT
                                        | MODUP
                                        | SWAP
                                        | MODDOWN
                                        //| MEMORY
                                        | PARAMS
                                        | NTT
            //| KEYSWITCH
    );

    template<uint64_t flag = ALL, typename T>
    void out(const T &s) {
        if constexpr (flag & ALL) {
            std::cout << s << "\n";
        } else if constexpr (flag & ENABLED_FLAGS) {
            if constexpr (flag & DEBUG) std::cout << "[DEBUG] ";
            if constexpr (flag & CPU) std::cout << "[CPU] ";
            if constexpr (flag & MULT) std::cout << "[MULT] ";
            if constexpr (flag & MODUP) std::cout << "[MODUP] ";
            if constexpr (flag & SWAP) std::cout << "[SWAP] ";
            if constexpr (flag & MODDOWN) std::cout << "[MODDOWN] ";
            if constexpr (flag & MEMORY) std::cout << "[MEMORY] ";
            if constexpr (flag & PARAMS) std::cout << "[PARAMS] ";
            if constexpr (flag & NTT) std::cout << "[NTT] ";
            if constexpr (flag & KEYSWITCH) std::cout << "[KEYSWITCH] ";
            std::cout << s;
        }
    }

#define Assert(flag, value) \
    do { \
        if constexpr(flag & ENABLED_FLAGS){ \
            assert(value); \
        } \
    } while(0)

#define Out(flag, value) \
    do { \
        if constexpr(flag & ENABLED_FLAGS || flag & ALL){ \
            out<flag>(value);                             \
            std::cout <<  "\t@" << __FILE__ << ":" << __LINE__ << std::endl;     \
        } \
    } while(0)

/*
template<uint64_t flag, typename T>
void Assert(flag,x) {
  if constexpr(flag & ENABLED_FLAGS)
    assert(x);
}
*/
}
#endif //FIDESLIB_LOGUTIL_HPP
