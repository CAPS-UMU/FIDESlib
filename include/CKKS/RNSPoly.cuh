//
// Created by carlosad on 16/03/24.
//

#ifndef FIDESLIB_CKKS_RNSPOLY_CUH
#define FIDESLIB_CKKS_RNSPOLY_CUH

#include <vector>
#include "CKKS/LimbPartition.cuh"
#include "CudaUtils.cuh"

namespace FIDESlib::CKKS {
class KeySwitchingKey;

class Context;

class RNSPoly {
    Context& cc;
    int level;

   public:
    std::vector<LimbPartition> GPU;

    explicit RNSPoly(Context& context, int level = -1, bool single_malloc = false);

    void grow(int level, bool single_malloc = false, bool constant = false);

    void load(const std::vector<std::vector<uint64_t>>& data, const std::vector<uint64_t>& moduli);

    void store(std::vector<std::vector<uint64_t>>& data);

    explicit RNSPoly(Context& context, const std::vector<std::vector<uint64_t>>& data);

    RNSPoly(RNSPoly&& src) noexcept;

    int getLevel() const;

    void add(const RNSPoly& p);
    void add(const RNSPoly& a, const RNSPoly& b);

    void sub(const RNSPoly& p);

    void multPt(const RNSPoly& p, bool rescale);

    void modup();

    template <ALGO algo = ALGO_SHOUP>
    void moddown(bool ntt = true, bool free = true);

    void rescale();

    void sync();

    void freeSpecialLimbs();

    template <ALGO algo = ALGO_SHOUP>
    void NTT(int batch);

    template <ALGO algo = ALGO_SHOUP>
    void INTT(int batch);

    std::array<RNSPoly, 2> dotKSK(const KeySwitchingKey& ksk);

    void generateSpecialLimbs();

    void multElement(const RNSPoly& poly);

    RNSPoly clone(bool single_malloc = false) const;

    void generateDecompAndDigit();

    void mult1AddMult23Add4(const RNSPoly& poly1, const RNSPoly& poly2, const RNSPoly& poly3, const RNSPoly& poly4);

    void mult1Add2(const RNSPoly& poly1, const RNSPoly& poly2);

    void loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data,
                         const std::vector<std::vector<uint64_t>>& moduli);

    void dotKSKinto(RNSPoly& acc, const RNSPoly& ksk, int level, const RNSPoly* limbsrc = nullptr);

    void multElement(const RNSPoly& poly1, const RNSPoly& poly2);

    void multModupDotKSK(RNSPoly& c1, const RNSPoly& c1tilde, RNSPoly& c0, const RNSPoly& c0tilde,
                         const KeySwitchingKey& key);

    void automorph(const int idx, const int br = 1);

    void automorph_multi(const int idx, const int br = 1);

    RNSPoly& dotKSKInPlace(const KeySwitchingKey& ksk, int level);

    void dotKSKInPlace(const RNSPoly& key_b, int level);

    /** Change the polynomial level only superficially, be very careful as this should only be used for lower
     * level optimizations.
     */
    void setLevel(const int level);
    void modupInto(RNSPoly& poly);
    RNSPoly& dotKSKInPlaceFrom(RNSPoly& poly, const KeySwitchingKey& ksk, int level, const RNSPoly* limbsrc = nullptr);
    void multScalar(std::vector<uint64_t>& vector1);
    void squareElement(const RNSPoly& poly);
    void binomialSquareFold(RNSPoly& c0_res, const RNSPoly& c2_key_switched_0, const RNSPoly& c2_key_switched_1);
    void addScalar(std::vector<uint64_t>& vector1);
    void subScalar(std::vector<uint64_t>& vector1);
    void copy(const RNSPoly& poly);
    void dropToLevel(int level);
    void addMult(const RNSPoly& poly, const RNSPoly& poly1);
    void broadcastLimb0();
    void evalLinearWSum(uint32_t i, std::vector<const RNSPoly*>& vector1, std::vector<uint64_t>& vector2);
    void loadConstant(const std::vector<std::vector<uint64_t>>& vector1, const std::vector<uint64_t>& vector2);
    void rotateModupDotKSK(RNSPoly& poly, RNSPoly& poly1, const KeySwitchingKey& key);
    void squareModupDotKSK(RNSPoly& c0, RNSPoly& c1, const KeySwitchingKey& key);
};
}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_CKKS_RNSPOLY_CUH
