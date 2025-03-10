//
// Created by carlosad on 12/09/24.
//
#include <cinttypes>
#include <iostream>

#include "ConstantsGPU.cuh"
#include "Math.cuh"
#include "cpuNTT.hpp"
#include "cpuNTT_nega.hpp"

/*
uint64_t reverse(uint64_t num, int lg_n) {
    uint64_t res = 0;
    for (int i = 0; i < lg_n; i++) {
        if (num & (1 << i))
            res |= 1 << (lg_n - 1 - i);
    }
    return res;
}
 */

void FIDESlib::Testing::nega_fft(std::vector<uint64_t>& a, bool invert, const uint64_t* psi, const uint64_t* inv_psi,
                                 uint64_t mod, int its) {
    int n = a.size();

    // variacion
    int lg_n = 0;
    while ((1 << lg_n) < n)
        lg_n++;
    ////////////////////////

    {
        uint64_t modulus = mod;
        //IntType mu      = modulus.ComputeMu();

        uint64_t loVal, hiVal, omega, omegaFactor;
        int i, m, j1, j2, indexOmega, indexLo, indexHi;
        const uint64_t* rootOfUnityTable = invert ? inv_psi : psi;
        uint64_t* element = a.data();

        int t = n / 2;
        int logt1 = lg_n;

        for (m = 1; m < n; m <<= 1) {
            for (i = 0; i < m; ++i) {
                j1 = i << logt1;
                j2 = j1 + t;
                indexOmega = m + i;
                omega = rootOfUnityTable[indexOmega];
                for (indexLo = j1; indexLo < j2; ++indexLo) {
                    indexHi = indexLo + t;

                    if (n < 1000)
                        printf("CPU: m: %d, j1: %d, j2: %d, psi: %lu, psi_id: %d, a1: %lu, a2: %lu\n", m, indexLo,
                               indexHi, omega, indexOmega, a[indexLo], a[indexHi]);

                    loVal = (element)[indexLo];
                    omegaFactor = (element)[indexHi];

                    omegaFactor = (__uint128_t)omegaFactor * omega % modulus;

                    hiVal = loVal + omegaFactor;
                    if (hiVal >= modulus) {
                        hiVal -= modulus;
                    }

                    if (loVal < omegaFactor) {
                        loVal += modulus;
                    }
                    loVal -= omegaFactor;

                    (element)[indexLo] = hiVal;
                    (element)[indexHi] = loVal;

                    if (n < 1000)
                        printf("CPU: m: %d, j1: %d, j2: %d, psi: %lu, psi_id: %d, a1: %lu, a2: %lu\n", m, indexLo,
                               indexHi, omega, indexOmega, a[indexLo], a[indexHi]);
                }
            }
            t >>= 1;
            logt1--;
        }
    }

    if (invert) {
        uint64_t n_1 = FIDESlib::modinv((uint64_t)n, mod);
        for (uint64_t& x : a)
            x = ((__uint128_t)x * n_1 % mod);
    }

    for (int i = 0; i < n; i++) {
        if (i < bit_reverse(i, lg_n))
            std::swap(a[i], a[bit_reverse(i, lg_n)]);
    }

    /*
    int n = a.size();
    const uint64_t * w = invert ? inv_psi : psi;

    int lg_n = 0;
    while ((1 << lg_n) < n)
        lg_n++;

    int m = 1;
    int k = n / 2;

    for(;m < n; m <<= 1, k >>= 1){
        for(int i = 0; i < m; ++i){
            int j_init = 2 * i * k;
            uint64_t twiddle = w[m + i];
            for(int j = j_init; j < j_init + k; ++j){
                if(n < 1000) printf("CPU: m: %d, j1: %d, j2: %d, psi: %lu, psi_id: %d, a1: %lu, a2: %lu\n", m, j, j + k, twiddle, m + i,
                        a[j],  a[j+ k]);
                uint64_t u = a[j];
                uint64_t v = (__uint128_t) a[j + k] * twiddle % mod;
                a[j] = (u + v) % mod;
                a[j + k] = (u + mod - v) % mod;
                if(n < 1000) printf("CPU: m: %d, j1: %d, j2: %d, psi: %lu, psi_id: %d, a1: %lu, a2: %lu\n", m, j, j + k, twiddle, m + i,
                                    a[j],  a[j+ k]);
            }
        }
    }

    if (invert) {
        uint64_t n_1 = FIDESlib::modinv((uint64_t) n, mod);
        for (uint64_t &x: a)
            x =  ((__uint128_t ) x * n_1 % mod);
    }

    for (int i = 0; i < n; i++) {
        if (i < bit_reverse(i, lg_n))
            std::swap(a[i], a[bit_reverse(i, lg_n)]);
    }
    */
}

void FIDESlib::Testing::nega_fft2(std::vector<uint64_t>& a, bool invert, const uint64_t* psi, const uint64_t* inv_psi,
                                  uint64_t mod, int its) {
    int n = a.size();

    ////////////////////////
    if (!invert) {
        for (int i = 0; i < n; ++i) {
            a[i] = modprod(a[i], modpow(psi[n / 2], i, mod), mod);
        }
    }
    fft(a, invert, psi[n >> 2], inv_psi[n >> 2], mod, its);
    if (invert) {
        for (int i = 0; i < n; ++i) {
            a[i] = modprod(a[i], modpow(inv_psi[n / 2], i, mod), mod);
        }
    }
    ////////////////////////
    /*
        int lg_n = 0;
        while ((1 << lg_n) < n)
            lg_n++;

        for (int i = 0; i < n; i++) {
            if (i < bit_reverse(i, lg_n)) std::swap(a[i], a[bit_reverse(i, lg_n)]);
        }

        uint64_t modulus = mod;
        //IntType mu      = modulus.ComputeMu();

        uint64_t loVal, hiVal, omega, omegaFactor;
        int i, m, j1, j2, indexOmega, indexLo, indexHi;
        const uint64_t *rootOfUnityInverseTable = invert ? inv_psi : psi;
        uint64_t *element = a.data();

        int t = 1;
        int logt1 = 1;
        for (m = (n >> 1); m >= 1; m >>= 1) {
            for (i = 0; i < m; ++i) {
                j1 = i << logt1;
                j2 = j1 + t;
                indexOmega = m + i;
                omega = rootOfUnityInverseTable[indexOmega];

                for (indexLo = j1; indexLo < j2; ++indexLo) {
                    indexHi = indexLo + t;

                    if(n < 1000) printf("CPU: m: %d, j1: %d, j2: %d, psi: %lu, psi_id: %d, a1: %lu, a2: %lu\n", m, indexLo, indexHi, omega, indexOmega,
                                        a[indexLo],  a[indexHi]);

                    hiVal = (element)[indexHi];
                    loVal = (element)[indexLo];

                    omegaFactor = loVal;
                    if (omegaFactor < hiVal) {
                        omegaFactor += modulus;
                    }

                    omegaFactor -= hiVal;

                    loVal += hiVal;
                    if (loVal >= modulus) {
                        loVal -= modulus;
                    }

                    omegaFactor = (__uint128_t) omegaFactor * omega % modulus;
                    //omegaFactor.ModMulFastEq(omega, modulus, mu);

                    (element)[indexLo] = loVal;
                    (element)[indexHi] = omegaFactor;

                    if(n < 1000) printf("CPU: m: %d, j1: %d, j2: %d, psi: %lu, psi_id: %d, a1: %lu, a2: %lu\n", m, indexLo, indexHi, omega, indexOmega,
                                        a[indexLo],  a[indexHi]);
                }
            }
            t <<= 1;
            logt1++;
        }

        uint64_t n_1 = FIDESlib::modinv((uint64_t) n, mod);
        for (i = 0; i < n; i++) {
            //(element)[i].ModMulFastEq(cycloOrderInv, modulus, mu);
            (element)[i] = (__uint128_t) element[i] * n_1 % modulus;
        }
        */
    /*
    int n = a.size();

    const uint64_t * w = invert ? inv_psi : psi;

    int lg_n = 0;
    while ((1 << lg_n) < n)
        lg_n++;

    for (int i = 0; i < n; i++) {
        if (i < bit_reverse(i, lg_n))
            std::swap(a[i], a[bit_reverse(i, lg_n)]);
    }

    int m = n / 2;
    int k = 1;

    for(;m >= 1; m >>= 1, k <<= 1){
        for(int i = 0; i < m; ++i){
            int j_init = 2 * i * k;
            uint64_t twiddle = w[m + i];
            for(int j = j_init; j < j_init + k; ++j){
                if(n < 1000) printf("CPU: m: %d, j1: %d, j2: %d, psi: %lu, psi_id: %d, a1: %lu, a2: %lu\n", m, j, j + k, twiddle, m + i,
                                    a[j],  a[j+ k]);
                uint64_t u = (a[j] + a[j + k]) % mod;
                uint64_t v = (__uint128_t) (a[j] + mod - a[j + k]) * twiddle % mod;
                a[j] = u;
                a[j + k] = v;
                if(n < 1000) printf("CPU: m: %d, j1: %d, j2: %d, psi: %lu, psi_id: %d, a1: %lu, a2: %lu\n", m, j, j + k, twiddle, m + i,
                                    a[j],  a[j+ k]);
            }
        }
    }

    if (invert) {
        uint64_t n_1 = FIDESlib::modinv((uint64_t) n, mod);
        for (uint64_t &x: a)
            x =  ((__uint128_t ) x * n_1 % mod);
    }
    */
}

void FIDESlib::Testing::nega_fft_forPrime(std::vector<uint64_t>& a, bool invert, int primeid, int its) {
    std::cout << "N: " << a.size() << (invert ? "INTT" : "NTT")
              << " w: " << ((uint64_t*)hG_.psi[primeid])[a.size() >> 1]
              << " w^-1: " << ((uint64_t*)hG_.inv_psi[primeid])[a.size() >> 1] << " p: " << hC_.primes[primeid]
              << std::endl;
    nega_fft(a, invert, ((uint64_t*)hG_.psi[primeid]), ((uint64_t*)hG_.inv_psi[primeid]), hC_.primes[primeid], its);
}

void FIDESlib::Testing::nega_fft2_forPrime(std::vector<uint64_t>& a, bool invert, int primeid, int its) {
    std::cout << "N: " << a.size() << (invert ? "INTT" : "NTT")
              << " w: " << ((uint64_t*)hG_.psi[primeid])[a.size() >> 1]
              << " w^-1: " << ((uint64_t*)hG_.inv_psi[primeid])[a.size() >> 1] << " p: " << hC_.primes[primeid]
              << std::endl;
    nega_fft2(a, invert, ((uint64_t*)hG_.psi[primeid]), ((uint64_t*)hG_.inv_psi[primeid]), hC_.primes[primeid], its);
}
