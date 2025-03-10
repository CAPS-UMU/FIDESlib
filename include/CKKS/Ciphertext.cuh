//
// Created by carlosad on 24/04/24.
//

#ifndef FIDESLIB_CKKS_CIPHERTEXT_CUH
#define FIDESLIB_CKKS_CIPHERTEXT_CUH

#include <source_location>
#include "RNSPoly.cuh"
#include "forwardDefs.cuh"
#include "openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib::CKKS {

class Ciphertext {
    static constexpr const char* loc{"Ciphertext"};
    CudaNvtxRange my_range;

   public:
    Context& cc;
    RNSPoly c0, c1;
    double NoiseFactor = 0;
    int NoiseLevel = {1};

    Ciphertext(Ciphertext&& ct_moved) = default;

    explicit Ciphertext(Context& cc);

    Ciphertext(Context& cc, const RawCipherText& rawct);

    void load(const RawCipherText& rawct);

    void store(const Context& cc, RawCipherText& rawct);

    void add(const Ciphertext& b);
    void sub(const Ciphertext& b);

    void addPt(const Plaintext& b);

    void addScalar(const double c);

    void multPt(const Plaintext& b, bool rescale = false);
    void multPt(const Ciphertext& c, const Plaintext& b, bool rescale = false);
    void addMultPt(const Ciphertext& c, const Plaintext& b, bool rescale = false);

    void mult(const Ciphertext& b, const KeySwitchingKey& kskEval, bool rescale = false);
    void mult(const Ciphertext& b, const Ciphertext& c, const KeySwitchingKey& kskEval, bool rescale = false);

    void multScalarNoPrecheck(const double c, bool rescale = false);
    void multScalar(const double c, bool rescale = false);
    void multScalar(const Ciphertext& b, const double c, bool rescale = false);

    void square(const KeySwitchingKey& kskEval, bool rescale = false);

    void square(const Ciphertext& src, const KeySwitchingKey& kskEval, bool rescale = false);

    void rotate(const int index, const KeySwitchingKey& kskRot);
    void rotate(const Ciphertext& c, const int index, const KeySwitchingKey& kskRot);
    void conjugate(const Ciphertext& c);

    void modDown();

    void modUp();

    void rescale();

    void dropToLevel(int level);

    [[nodiscard]] int getLevel() const;

    void automorph(const int index, const int br);

    void automorph_multi(const int index, const int br);

    void rotate_hoisted(const std::vector<KeySwitchingKey*>& ksk, const std::vector<int>& indexes,
                        std::vector<Ciphertext*> results);
    void evalLinearWSumMutable(uint32_t n, const std::vector<Ciphertext>& ctxs, std::vector<double> weights);
    void addMultScalar(const Ciphertext& ciphertext, double d);
    void addScalar(const Ciphertext& b, double c);
    void add(const Ciphertext& ciphertext, const Ciphertext& ciphertext1);
    void sub(const Ciphertext& ciphertext, const Ciphertext& ciphertext1);
    void copy(const Ciphertext& ciphertext);
    void addPt(const Ciphertext& ciphertext, const Plaintext& plaintext);

    bool adjustForAddOrSub(const Ciphertext& ciphertext);
    bool adjustForMult(const Ciphertext& ciphertext);
    [[nodiscard]] bool hasSameScalingFactor(const Plaintext& b) const;
};

}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_CKKS_CIPHERTEXT_CUH
