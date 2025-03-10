//
// Created by carlosad on 25/04/24.
//

#ifndef FIDESLIB_CKKS_PLAINTEXT_CUH
#define FIDESLIB_CKKS_PLAINTEXT_CUH

#include "RNSPoly.cuh"
#include "openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib::CKKS {

class Plaintext {
    static constexpr const char* loc{"Plaintext"};
    CudaNvtxRange my_range;
    Context& cc;

   public:
    RNSPoly c0;
    double NoiseFactor = 0;
    int NoiseLevel = 0;

    Plaintext(Plaintext&& pt) = default;
    explicit Plaintext(Context& cc);
    Plaintext(Context& cc, const RawPlainText& raw);
    void load(const RawPlainText& raw);
    void store(RawPlainText& raw);
    void copy(const Plaintext& p);

    void rescale();
    void multScalar(double c, bool rescale);

    void moddown();
    bool adjustPlaintextToCiphertext(const Plaintext& p, const Ciphertext& c);
};

}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_CKKS_PLAINTEXT_CUH
