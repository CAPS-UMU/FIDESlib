//
// Created by carlosad on 16/03/24.
//

#ifndef FIDESLIB_CKKS_RNSPOLY_CUH
#define FIDESLIB_CKKS_RNSPOLY_CUH

#include "CKKS/LimbPartition.cuh"
#include "CudaUtils.cuh"
#include <vector>

namespace FIDESlib::CKKS {
class KeySwitchingKey;

class RNSPoly {
	const uint64_t uid;
	ContextData& cc;
	int level;
	bool modUp = false;

  public:
	std::vector<LimbPartition> GPU;

	explicit RNSPoly(ContextData& context, int level = -1, bool single_malloc = false, bool def_stream = false);
	explicit RNSPoly(ContextData& context, const std::vector<std::vector<uint64_t>>& data);
	RNSPoly(RNSPoly&& src) noexcept;

	void grow(int level, bool single_malloc = false, bool constant = false);

	void load(const std::vector<std::vector<uint64_t>>& data, const std::vector<uint64_t>& moduli);

	void store(std::vector<std::vector<uint64_t>>& data);

	bool isModUp() const;
	void SetModUp(bool newValue);

	void scaleByP();

	int32_t getLevel() const;

	void add(const RNSPoly& p);
	void add(const RNSPoly& a, const RNSPoly& b);

	void sub(const RNSPoly& p);

	void multPt(const RNSPoly& p, bool rescale);

	void modup();

	template <ALGO algo = ALGO_SHOUP> void moddown(bool ntt = true, bool free = true, int aux_num = 0);
	int automorph_index_precomp(int idx) const;

	void rescale();

	void sync();

	void freeSpecialLimbs();

	template <ALGO algo = ALGO_SHOUP> void NTT(int batch, bool sync);

	template <ALGO algo = ALGO_SHOUP> void INTT(int batch, bool sync);

	// std::array<RNSPoly, 2> dotKSK(const KeySwitchingKey& ksk);

	void generateSpecialLimbs(bool zero_out, bool for_communication);

	void multElement(const RNSPoly& poly);

	void generateDecompAndDigit(bool iskey);

	void mult1AddMult23Add4(const RNSPoly& poly1, const RNSPoly& poly2, const RNSPoly& poly3, const RNSPoly& poly4);

	void mult1Add2(const RNSPoly& poly1, const RNSPoly& poly2);

	void loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data, const std::vector<std::vector<uint64_t>>& moduli);

	void dotKSKinto(RNSPoly& acc, const RNSPoly& ksk, const RNSPoly* limbsrc = nullptr);

	void multElement(const RNSPoly& poly1, const RNSPoly& poly2);

	void multModupDotKSK(RNSPoly& c1, const RNSPoly& c1tilde, RNSPoly& c0, const RNSPoly& c0tilde, const KeySwitchingKey& key);

	void automorph(const int idx, const int br = 1, RNSPoly* src = nullptr);

	RNSPoly& dotKSKInPlace(const KeySwitchingKey& ksk, RNSPoly* limb_src);

	void hoistedRotationFused(std::vector<int> indexes,
	  std::vector<RNSPoly*>& c0,
	  std::vector<RNSPoly*>& c1,
	  const std::vector<RNSPoly*>& ksk_a,
	  const std::vector<RNSPoly*>& ksk_b,
	  const RNSPoly& src_c0,
	  const RNSPoly& src_c1);

	/** Change the polynomial level only superficially, be very careful as this should only be used for lower
	 * level optimizations.
	 */
	void setLevel(const int level);
	void modupInto(RNSPoly& poly);
	RNSPoly& dotKSKInPlaceFrom(RNSPoly& poly, const KeySwitchingKey& ksk, const RNSPoly* limbsrc = nullptr);
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
	void generatePartialSpecialLimbs();
	void dotKSKfused(RNSPoly& out2, const RNSPoly& digitSrc, const RNSPoly& ksk_a, const RNSPoly& ksk_b, const RNSPoly* source);
	void dotProductPt(RNSPoly& c1, const std::vector<const RNSPoly*>& c0s, const std::vector<const RNSPoly*>& c1s, const std::vector<const RNSPoly*>& pts, bool ext);
	RNSPoly& dotProduct(RNSPoly& c1,
	  const RNSPoly& kskb,
	  const RNSPoly& kska,
	  const std::vector<const RNSPoly*>& c0in,
	  const std::vector<const RNSPoly*>& c1in,
	  const std::vector<const RNSPoly*>& d0in,
	  const std::vector<const RNSPoly*>& d1in,
	  bool ext_in,
	  bool ext_out);
	void gatherAllLimbs();
	void generateGatherLimbs();
	void copyShallow(const RNSPoly& poly);
	RNSPoly& modup_ksk_moddown_mgpu(const KeySwitchingKey& key, bool moddown);
	void rescaleDouble(RNSPoly& poly);

	void multNoModdownEnd(RNSPoly& c0, const RNSPoly& bc0, const RNSPoly& bc1, const RNSPoly& in, const RNSPoly& aux);

	void binomialMult(RNSPoly& c1, RNSPoly& in, const RNSPoly& d0, const RNSPoly& d1, bool moddown, bool square);

	static void multScalarBatchManyToOne(std::vector<RNSPoly*>& polya,
	  const std::vector<std::vector<unsigned long int>>& vector,
	  const std::vector<std::vector<unsigned long int>>& vectors,
	  int stride,
	  double usage);
	static void addScalarBatchManyToOne(std::vector<RNSPoly*>& polya, const std::vector<std::vector<unsigned long int>>& vector, int stride, double usage);
	static void multPtBatchManyToOne(std::vector<RNSPoly*>& polya, const std::vector<RNSPoly*>& polyb, int stride, double usage);
	static void addBatchManyToOne(std::vector<RNSPoly*>& polya, const std::vector<RNSPoly*>& polyb, int stride, double usage, bool sub, bool exta, bool extb);

	static void
	LTdotProductPtBatch(std::vector<RNSPoly*>& out, const std::vector<RNSPoly*>& in, const std::vector<RNSPoly*>& pt, int bStep, int gStep, int stride, double usage, bool ext);
	static void fusedHoistedRotateBatch(std::vector<RNSPoly*>& out,
	  const std::vector<RNSPoly*>& in,
	  const std::vector<RNSPoly*>& ksk_a,
	  const std::vector<RNSPoly*>& ksk_b,
	  const std::vector<int>& indexes,
	  int stride,
	  double usage,
	  bool c0_modup);
};
} // namespace FIDESlib::CKKS
#endif // FIDESLIB_CKKS_RNSPOLY_CUH
