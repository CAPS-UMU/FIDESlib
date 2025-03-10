//
// Created by carlosad on 23/03/24.
//

#ifndef FIDESLIB_MATH_CUH
#define FIDESLIB_MATH_CUH

#include <cstdint>
#include <optional>
#include <vector>
//#include <bit>
#include "LimbUtils.cuh"

namespace FIDESlib {

    uint64_t modadd(uint64_t a, uint64_t b, uint64_t p);
    uint64_t modsub(uint64_t a, uint64_t b, uint64_t p);
    uint64_t modprod(uint64_t a, uint64_t b, uint64_t p);
    uint64_t modpow(uint64_t a, uint64_t e, uint64_t p);
    uint64_t modinv(uint64_t a, uint64_t p);
/**
    Modulo is second dimension.
*/
    std::vector<std::vector<uint64_t>> q_inv(const std::vector<PrimeRecord> &p);

/**
    Second dim p
*/
    std::vector<std::vector<uint64_t>>
    q_inv_mod_p(const std::vector<PrimeRecord> &q, const std::vector<PrimeRecord> &p);

/**
  This is Q_l mod q_j j > l.
*/
    std::vector<std::vector<uint64_t>> big_Q_prefix(const std::vector<PrimeRecord> &p);

    std::vector<std::vector<uint64_t>> big_Q_prefix_mod_p(
            const std::vector<PrimeRecord> &q,
            const std::vector<PrimeRecord> &p
    );

/**
 This is Q'_l mod p_j, j < l;
*/
    std::vector<std::vector<uint64_t>> big_Q_suffix(const std::vector<PrimeRecord> &p);

/**
    First dim is current level.
*/
    std::vector<std::vector<uint64_t>> q_hat_inv(const std::vector<PrimeRecord> &p);

/**
    First dim is current level.
*/
    std::vector<std::vector<uint64_t>> p_hat(const std::vector<PrimeRecord> &q,
                                             const std::vector<PrimeRecord> &p);

/**
    For decomp: structure is that of meta.
*/
    std::vector<std::vector<uint64_t>> big_Q_hat(
            const std::vector<PrimeRecord> &p,
            const std::vector<std::vector<LimbRecord>> &meta);

/**
    For ModDown
*/
    std::vector<uint64_t> big_P_inv_mod_q(const std::vector<PrimeRecord> &p, const std::vector<PrimeRecord> &q);

    int bit_reverse(int a, int w);

    template<typename T>
    void bit_reverse_vector(std::vector<T> &  a);
}
#endif //FIDESLIB_MATH_CUH
