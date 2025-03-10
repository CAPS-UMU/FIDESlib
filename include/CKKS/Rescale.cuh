//
// Created by carlosad on 10/09/24.
//

#ifndef GPUCKKS_RESCALE_CUH
#define GPUCKKS_RESCALE_CUH

#include "ConstantsGPU.cuh"
#include "ModMult.cuh"

namespace FIDESlib::CKKS {
    /**
    template <class IntegerType>
    void NativeVectorT<IntegerType>::SwitchModulus(const IntegerType& modulus) {
        // TODO: #ifdef NATIVEINT_BARRET_MOD
        auto size{m_data.size()};
        auto halfQ{m_modulus.m_value >> 1};
        auto om{m_modulus.m_value};
        this->NativeVectorT::SetModulus(modulus);
        auto nm{modulus.m_value};
        if (nm > om) {
            auto diff{nm - om};
            for (size_t i = 0; i < size; ++i) {
                auto& v = m_data[i].m_value;
                if (v > halfQ)
                    v = v + diff;
            }
        }
        else {
            auto diff{nm - (om % nm)};
            for (size_t i = 0; i < size; ++i) {
                auto& v = m_data[i].m_value;
                if (v > halfQ)
                    v = v + diff;
                if (v >= nm)
                    v = v % nm;
            }
        }
    }
*/
/**
 *  Adapted from OpenFHE: mubintvecnat.cpp:109
 */
    template<typename T>
    __device__ __forceinline__  void SwitchModulus(T &a, const int om_pid, const int nm_pid) {

        const T om = C_.primes[om_pid];
        const T nm = C_.primes[nm_pid];
        const T halfQ{om >> 1};

        if (nm > om) {
            T diff{nm - om};
            if (a > halfQ)
                a = a + diff;
        } else {
//auto diff{nm - (om % nm)};
            T diff{nm - modmult<ALGO_BARRETT>(om, 1, nm_pid)}; // TODO: change to modular reduction routine
            if (a > halfQ) a = a + diff;
            if (a >= nm) {
                //a = a % nm;
                a = modmult<ALGO_BARRETT>(a, 1, nm_pid);
            }
        }
    }

    template<typename T>
    __global__ void SwitchModulus(const T *src, const int __grid_constant__ o_primeid,
                                  T *res, const int __grid_constant__ n_primeid);

}

#endif //GPUCKKS_RESCALE_CUH
