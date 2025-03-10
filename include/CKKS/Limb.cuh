//
// Created by carlos on 6/03/24.
//
#ifndef FIDESLIB_CKKS_LIMB_CUH
#define FIDESLIB_CKKS_LIMB_CUH

#include <concepts>
#include <iostream>
#include <variant>
#include "ModMult.cuh"
#include "VectorGPU.cuh"
#include "forwardDefs.cuh"

namespace FIDESlib::CKKS {

template <typename T>
class Limb {
    using LimbImpl = std::variant<Limb<uint32_t>, Limb<uint64_t>>;
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);

    Context& cc;

   public:
    int primeid;
    Stream& stream;
    VectorGPU<T> v;
    VectorGPU<T> aux;

   private:
    const int id;
    const bool raw;
    static constexpr int block = 128;

   public:
    ~Limb() noexcept;

    Limb(Limb<T>&& l) noexcept;

    Limb(Context& context, const int device, Stream& stream, const int primeid = -1);

    Limb<T> clone();

    //    Limb(Context &context, const int device, const int primeid = -1);
    Limb(Context& context, T* data, const int offset, const int device, Stream& stream, const int primeid = -1,
         T* data_aux = nullptr, const int offset_aux = 0);
    //void free();

    void add(const LimbImpl& l);

    void add(const Limb<uint64_t>& l);

    void add(const Limb<uint32_t>& l);

    void sub(const LimbImpl& l);

    void sub(const Limb<uint64_t>& l);

    void sub(const Limb<uint32_t>& l);

    void mult(const LimbImpl& l);

    void mult(const Limb<uint64_t>& l);

    void mult(const Limb<uint32_t>& l);

    void mult(const LimbImpl& _l1, const LimbImpl& _l2, const bool inplace = false);

    template <typename Q>
    void load(const std::vector<Q>& dat_);

    void load(const VectorGPU<T>& dat);

    void store(std::vector<T>& dat) const;

    template <typename Q>
    void load_convert(const std::vector<Q>& dat_raw);

    template <typename Q>
    void store_convert(std::vector<Q>& dat_raw);

    template <ALGO algo = ALGO_SHOUP>
    void INTT();

    template <ALGO algo = ALGO_SHOUP>
    void NTT();

    void NTT_rescale_fused(const LimbImpl& l);

    void NTT_rescale_fused(const Limb<uint32_t>& l);

    void NTT_rescale_fused(const Limb<uint64_t>& l);

    void NTT_moddown_fused(const LimbImpl& l);

    void NTT_multpt_fused(const LimbImpl& _l, const LimbImpl& _pt);

    void copyV(const LimbImpl& l);

    void copyV(const Limb<uint32_t>& l);

    void copyV(const Limb<uint64_t>& l);

    void INTT_from(LimbImpl& l);

    void addMult(const LimbImpl& _l1, const LimbImpl& _l2, const bool inplace = false);

    void printThisLimb(int num = 32) const;

    void INTT_from_mult(LimbImpl& res0_, LimbImpl& res1_, const LimbImpl& c1_, const LimbImpl& c1tilde_,
                        const LimbImpl& c0_, const LimbImpl& c0tilde_, const LimbImpl& kska_, const LimbImpl& kskb_);

    void INTT_from_mult_acc(LimbImpl& res0_, LimbImpl& res1_, const LimbImpl& c1_, const LimbImpl& c1tilde_,
                            const LimbImpl& c0_, const LimbImpl& c0tilde_, const LimbImpl& kska_,
                            const LimbImpl& kskb_);

    void NTT_and_ksk_dot(LimbImpl& res0_, LimbImpl& res1_, const LimbImpl& kska_, const LimbImpl& kskb_);

    void NTT_and_ksk_dot_acc(LimbImpl& res0_, LimbImpl& res1_, const LimbImpl& kska_, const LimbImpl& kskb_);

    void automorph(const int index, const int br);
};

using LimbImpl = std::variant<Limb<uint32_t>, Limb<uint64_t>>;
}  // namespace FIDESlib::CKKS

#endif  //FIDESLIB_LIMB_CUH