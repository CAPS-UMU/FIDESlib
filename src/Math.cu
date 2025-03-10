//
// Created by carlosad on 24/03/24.
//
#include "Math.cuh"
#include <cassert>

namespace FIDESlib {
    uint64_t modadd(uint64_t a, uint64_t b, uint64_t p) {
        uint64_t c = a + b;
        return c > p ? c - p : c;
    }

    uint64_t modsub(uint64_t a, uint64_t b, uint64_t p) {
        uint64_t c = a - b;
        return c > p ? c + p : c;
    }

    uint64_t modprod(uint64_t a, uint64_t b, uint64_t p) {
        assert(p != 0);
        return (__uint128_t(a)) * b % p;
    }

    uint64_t modpow(uint64_t a, uint64_t e, uint64_t p) {
        assert(p != 0);
        uint64_t r = 1;
        while (e) {
            if (e & 1) r = (__uint128_t(a)) * r % p;
            e >>= 1;
            a = (__uint128_t(a)) * a % p;
        }
        return r;
    }

    uint64_t modinv(uint64_t a, uint64_t p) {
        assert(p != 0);
        return modpow(a, p - 2, p);
    }

    std::vector<std::vector<uint64_t>> q_inv(const std::vector<PrimeRecord> &p) {
        std::vector<std::vector<uint64_t>> table(p.size(), std::vector<uint64_t>(p.size()));
        for (size_t i = 0; i < p.size(); ++i) {
            for (size_t j = 0; j < p.size(); ++j) {
                table[i][j] = modinv(p[i].p, p[j].p);
            }
        }
        return table;
    }

    std::vector<std::vector<uint64_t>>
    q_inv_mod_p(const std::vector<PrimeRecord> &q, const std::vector<PrimeRecord> &p) {
        std::vector<std::vector<uint64_t>> table(q.size(), std::vector<uint64_t>(p.size(), (uint64_t) -1));
        for (size_t i = 0; i < q.size(); ++i) {
            for (size_t j = 0; j < p.size(); ++j) {
                table[i][j] = modinv(q[i].p, p[j].p);
            }
        }
        return table;
    }

    std::vector<std::vector<uint64_t>> big_Q_prefix(const std::vector<PrimeRecord> &p) {
        std::vector<std::vector<uint64_t>> table(p.size(), std::vector<uint64_t>(p.size(), (uint64_t) -1));
        for (size_t i = 0; i < p.size(); ++i) {
            for (size_t j = 0; j < i; ++j) {
                table[i][j] = modprod(j == 0 ? 1 : table[i][j - 1], p[j].p, p[i].p);
            }
        }
        return table;
    }

    std::vector<std::vector<uint64_t>> big_Q_prefix_mod_p(
            const std::vector<PrimeRecord> &q,
            const std::vector<PrimeRecord> &p
    ) {
        std::vector<std::vector<uint64_t>> table(p.size(), std::vector<uint64_t>(q.size(), (uint64_t) -1));
        for (size_t i = 0; i < p.size(); ++i) {
            for (size_t j = 0; j < q.size(); ++j) {
                table[i][j] = modprod(j == 0 ? 1 : table[i][j - 1], q[j].p, p[i].p);
            }
        }
        return table;
    }

    std::vector<std::vector<uint64_t>> big_Q_suffix(const std::vector<PrimeRecord> &p) {
        std::vector<std::vector<uint64_t>> table(p.size(), std::vector<uint64_t>(p.size(), (uint64_t) -1));
        for (size_t i = 0; i < p.size(); ++i) {
            for (size_t j = i + 1; j < p.size(); ++j) {
                table[i][j] = modprod(j == i + 1 ? 1 : table[i][j - 1], p[j].p, p[i].p);
            }
        }
        return table;
    }

    std::vector<std::vector<uint64_t>> q_hat_inv(const std::vector<PrimeRecord> &p) {
        std::vector<std::vector<uint64_t>> table(p.size(), std::vector<uint64_t>(p.size(), (uint64_t) -1));
        for (size_t i = 0; i < p.size(); ++i) {
            for (size_t j = 0; j <= i; ++j) {
                table[i][j] = 1;
                for (size_t k = 0; k <= i; ++k) {
                    table[i][j] = modprod(k == j ? 1 : p[k].p, table[i][j], p[j].p);
                }
                table[i][j] = modinv(table[i][j], p[j].p);
            }
        }
        return table;
    }

    std::vector<std::vector<uint64_t>> p_hat(const std::vector<PrimeRecord> &q, const std::vector<PrimeRecord> &p) {
        std::vector<std::vector<uint64_t>> table(p.size(), std::vector<uint64_t>(q.size(), (uint64_t) -1));
        for (size_t i = 0; i < p.size(); ++i) {
            for (size_t j = 0; j < q.size(); ++j) {
                table[i][j] = 1;
                for (size_t k = 0; k < p.size(); ++k) {
                    table[i][j] = modprod(k == i ? 1 : p[k].p, table[i][j], q[j].p);
                }

            }
        }
        return table;
    }

    std::vector<std::vector<uint64_t>>
    big_Q_hat(const std::vector<PrimeRecord> &p, const std::vector<std::vector<LimbRecord>> &meta) {
        std::vector<std::vector<uint64_t>> table(meta.size());
        for (size_t i = 0; i < table.size(); ++i) {
            table[i].assign(meta[i].size(), 1);
            for (size_t j = 0; j < meta[i].size(); ++j) {

                for (const auto &k: meta)
                    for (const auto &l: k)
                        if (meta[i][j].digit != l.digit)
                            table[i][j] = modprod(table[i][j], p[l.id].p, p[meta[i][j].id].p);
            }
        }
        return table;
    }

    std::vector<uint64_t> big_P_inv_mod_q(const std::vector<PrimeRecord> &q, const std::vector<PrimeRecord> &p) {
        std::vector<uint64_t> table(q.size(), 1);

        for (size_t i = 0; i < q.size(); ++i) {
            for (size_t j = 0; j < p.size(); ++j) {
                table[i] = modprod(table[i], p[j].p, q[i].p);
            }
            table[i] = modinv(table[i], q[i].p);
        }
        return table;
    }



    int bit_reverse(int a, int w) {
        int res = 0;
        for (int i = 0; i < w; ++i) {
            res |= ((a >> i) & 1) << (w - i - 1);
        }
        return res;
    };

    template<typename T>
    void bit_reverse_vector(std::vector<T> &  a) {
        int w = std::bit_width(a.size()) - 1;

        std::vector<T> b(a);

        for(int i = 0; i < (int)a.size(); ++i){
            b[i] = a[bit_reverse(i, w)];
        }

        a = b;
    };

    template
    void bit_reverse_vector(std::vector<uint64_t> &  a);
    template
    void bit_reverse_vector(std::vector<uint32_t> &  a);
}