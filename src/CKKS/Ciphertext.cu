//
// Created by carlosad on 24/04/24.
//

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Plaintext.cuh"

namespace FIDESlib::CKKS {

Ciphertext::Ciphertext(Context& cc)
    : my_range(loc, LIFETIME),
      cc((CudaNvtxStart(std::string{std::source_location::current().function_name()}.substr(18 + strlen(loc))), cc)),
      c0(cc),
      c1(cc) {
    CudaNvtxStop();
}

Ciphertext::Ciphertext(Context& cc, const RawCipherText& rawct)
    : my_range(loc, LIFETIME),
      cc((CudaNvtxStart(std::string{std::source_location::current().function_name()}.substr(18 + strlen(loc))), cc)),
      c0(cc, rawct.sub_0),
      c1(cc, rawct.sub_1) {
    NoiseLevel = rawct.NoiseLevel;
    NoiseFactor = rawct.Noise;
    CudaNvtxStop();
}

void Ciphertext::add(const Ciphertext& b) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if (!adjustForAddOrSub(b)) {
            Ciphertext b_(cc);
            b_.copy(b);
            if (b_.adjustForAddOrSub(*this))
                add(b_);
            else
                assert(false);
            return;
        }
    }

    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        assert(this->getLevel() == b.getLevel());
    } else if (getLevel() > b.getLevel()) {
        c0.dropToLevel(b.getLevel());
        c1.dropToLevel(b.getLevel());
    }

    c0.add(b.c0);
    c1.add(b.c1);
}

void Ciphertext::sub(const Ciphertext& b) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if (!adjustForAddOrSub(b)) {
            Ciphertext b_(cc);
            b_.copy(b);
            if (b_.adjustForAddOrSub(*this))
                sub(b_);
            else
                assert(false);
            return;
        }
    }

    c0.sub(b.c0);
    c1.sub(b.c1);
}

void Ciphertext::addPt(const Plaintext& b) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT ||
        cc.rescaleTechnique == Context::FIXEDAUTO) {
        if (b.c0.getLevel() != this->getLevel() ||
            (b.NoiseLevel == 1 && NoiseLevel == 2) /*!hasSameScalingFactor(b)*/) {
            Plaintext b_(cc);
            if (!b_.adjustPlaintextToCiphertext(b, *this)) {
                assert(false);
            } else {
                addPt(b_);
            }
            return;
        }
    }
    assert(NoiseLevel == b.NoiseLevel);

    c0.add(b.c0);
    //NoiseFactor += b.NoiseFactor;
}

void Ciphertext::load(const RawCipherText& rawct) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.load(rawct.sub_0, rawct.moduli);
    c1.load(rawct.sub_1, rawct.moduli);

    NoiseLevel = rawct.NoiseLevel;
    NoiseFactor = rawct.Noise;
}

void Ciphertext::store(const Context& cc, RawCipherText& rawct) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    cudaDeviceSynchronize();
    rawct.numRes = c0.getLevel() + 1;
    rawct.sub_0.resize(rawct.numRes);
    rawct.sub_1.resize(rawct.numRes);
    c0.store(rawct.sub_0);
    c1.store(rawct.sub_1);
    rawct.N = cc.N;
    c0.sync();
    c1.sync();

    rawct.NoiseLevel = NoiseLevel;
    rawct.Noise = NoiseFactor;
    cudaDeviceSynchronize();
}

void Ciphertext::modDown() {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.moddown(true, false);
    c1.moddown(true, false);
    c0.freeSpecialLimbs();
    c1.freeSpecialLimbs();
}

void Ciphertext::modUp() {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.modup();
    //c1.modup();
}

void Ciphertext::multPt(const Plaintext& b, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if (NoiseLevel == 2)
            this->rescale();
    }

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if (b.c0.getLevel() != this->getLevel() || b.NoiseLevel == 2 /*!hasSameScalingFactor(b)*/) {
            Plaintext b_(cc);
            if (!b_.adjustPlaintextToCiphertext(b, *this)) {
                assert(false);
            } else {
                if (NoiseLevel == 2)
                    this->rescale();
                if (b_.NoiseLevel == 2)
                    b_.rescale();
                multPt(b_, rescale);
            }
            return;
        }
    }

    assert(NoiseLevel < 2);
    assert(b.NoiseLevel < 2);
    c0.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL);
    c1.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL);

    // Manage metadata
    NoiseLevel += b.NoiseLevel;
    NoiseFactor *= b.NoiseFactor;
    if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
        NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
        NoiseLevel -= 1;
    }
}

void Ciphertext::rescale() {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    assert(this->NoiseLevel == 2);
    if (cc.rescaleTechnique != Context::FIXEDMANUAL) {
        // this wouldn't do anything in OpenFHE
    }

    c0.rescale();
    c1.rescale();

    // Manage metadata
    NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
    NoiseLevel -= 1;
}

void Ciphertext::mult(const Ciphertext& b, const KeySwitchingKey& kskEval, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if (!adjustForMult(b)) {
            Ciphertext b_(cc);
            b_.copy(b);
            if (b_.adjustForMult(*this))
                mult(b_, kskEval, rescale);
            else
                assert(false);
            return;
        }
    }
    assert(NoiseLevel == 1);
    assert(NoiseLevel == b.NoiseLevel);
    /*
    if (getLevel() > b.getLevel()) {
        this->c0.dropToLevel(b.getLevel());
        this->c1.dropToLevel(b.getLevel());
    }
    */
    assert(c0.getLevel() <= b.c0.getLevel());
    assert(c1.getLevel() <= b.c1.getLevel());
    constexpr bool PRINT = false;
    Out(KEYSWITCH, " start ");

    if constexpr (0) {
        cc.getKeySwitchAux().setLevel(c1.getLevel());
        cc.getKeySwitchAux().multElement(c1, b.c1);
        cc.getKeySwitchAux().modup();

        auto& aux0 = cc.getKeySwitchAux().dotKSKInPlace(kskEval, c0.getLevel());

        cudaDeviceSynchronize();
        /*
        std::vector<uint64_t> p(c1.getLevel() + 1);
        for (int i = 0; i <= c1.getLevel(); ++i)
            p[i] = hC_.P[i];

        cc.getKeySwitchAux().addScalar(p);
        */
        c1.mult1AddMult23Add4(b.c0, c0, b.c1, cc.getKeySwitchAux());  // Read 4 first for better cache locality.
        cudaDeviceSynchronize();
        cc.getKeySwitchAux().moddown(true, false);
        cudaDeviceSynchronize();
        c1.copy(cc.getKeySwitchAux());
        cudaDeviceSynchronize();
        /*
        for (int i = 0; i <= c1.getLevel(); ++i)
            p[i] = 1.0;
        cc.getKeySwitchAux().subScalar(p);
        */
        cudaDeviceSynchronize();
        c0.mult1Add2(b.c0, aux0);
        aux0.moddown(true, false);
        c0.copy(aux0);
        cudaDeviceSynchronize();
        //c1.mult1AddMult23Add4(b.c0, c0, b.c1, cc.getKeySwitchAux());  // Read 4 first for better cache locality.

        if (rescale) {
            c1.rescale();
        }
        if (rescale) {
            c0.rescale();
        }
    } else {
        cc.getKeySwitchAux().multModupDotKSK(c1, b.c1, c0, b.c0, kskEval);
        c1.moddown(true, false);
        if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL)
            c1.rescale();
        c0.moddown(true, false);
        if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL)
            c0.rescale();
    }

    // Manage metadata
    NoiseLevel += b.NoiseLevel;
    NoiseFactor *= b.NoiseFactor;
    if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
        NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
        NoiseLevel -= 1;
    }
    Out(KEYSWITCH, " finish ");
}

void Ciphertext::square(const KeySwitchingKey& kskEval, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    constexpr bool PRINT = false;
    Out(KEYSWITCH, " start ");

    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT ||
        cc.rescaleTechnique == Context::FIXEDAUTO) {
        if (NoiseLevel == 2)
            this->rescale();
    }
    assert(this->NoiseLevel == 1);

    if constexpr (0) {
        cc.getKeySwitchAux().setLevel(c1.getLevel());
        cc.getKeySwitchAux().squareElement(c1);
        cc.getKeySwitchAux().modup();
        auto& aux0 = cc.getKeySwitchAux().dotKSKInPlace(kskEval, c0.getLevel());
        cc.getKeySwitchAux().moddown();
        aux0.moddown(true, false);
        //c1.mult1AddMult23Add4(c0, c0, c1, cc.getKeySwitchAux());
        c1.binomialSquareFold(c0, aux0, cc.getKeySwitchAux());
        if (rescale) {
            c1.rescale();
            c0.rescale();
        }
        //   //
        // Manage metadata
        NoiseLevel += NoiseLevel;
        NoiseFactor *= NoiseFactor;
        if (rescale) {
            NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
            NoiseLevel -= 1;
        }
    } else if constexpr (1) {
        cc.getKeySwitchAux().squareModupDotKSK(c0, c1, kskEval);

        c1.moddown(true, false);
        if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL)
            c1.rescale();
        c0.moddown(true, false);
        if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL)
            c0.rescale();

        NoiseLevel += NoiseLevel;
        NoiseFactor *= NoiseFactor;
        if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
            NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
            NoiseLevel -= 1;
        }
    } else {
        this->mult(*this, kskEval, rescale);
    }
    Out(KEYSWITCH, " finish ");
}

void Ciphertext::multScalarNoPrecheck(const double c, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    auto elem = cc.ElemForEvalMult(c0.getLevel(), c);
    c0.multScalar(elem);
    c1.multScalar(elem);

    // Manage metadata
    NoiseLevel += 1;
    NoiseFactor *= cc.param.ScalingFactorReal.at(c0.getLevel());
    if (rescale) {
        NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel());
        NoiseLevel -= 1;
    }

    if (rescale) {
        c0.rescale();
        c1.rescale();
    }
}

void Ciphertext::multScalar(const double c, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT ||
        cc.rescaleTechnique == Context::FIXEDAUTO) {
        if (NoiseLevel == 2)
            this->rescale();
    }
    assert(this->NoiseLevel == 1);
    multScalarNoPrecheck(c, rescale && cc.rescaleTechnique == Context::FIXEDMANUAL);
}

void Ciphertext::addScalar(const double c) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    auto elem = cc.ElemForEvalAddOrSub(c0.getLevel(), std::abs(c), this->NoiseLevel);

    if (c >= 0.0) {
        c0.addScalar(elem);
    } else {
        c0.subScalar(elem);
    }
}

void Ciphertext::automorph(const int index, const int br) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.automorph(index, br);
    c1.automorph(index, br);
}

void Ciphertext::automorph_multi(const int index, const int br) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.automorph_multi(index, br);
    c1.automorph_multi(index, br);
}

void Ciphertext::rotate(const int index, const KeySwitchingKey& kskRot) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    constexpr bool PRINT = false;

    if constexpr (0) {
        if constexpr (PRINT) {
            std::cout << "Output Automorph 1.";
            for (auto& j : c1.GPU)
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(2));
                }
        }
        c1.modupInto(cc.getKeySwitchAux());
        RNSPoly& aux0 = c1.dotKSKInPlaceFrom(cc.getKeySwitchAux(), kskRot, c1.getLevel());
        c1.moddown();
        if constexpr (PRINT) {
            std::cout << "Output Automorph 1.";
            for (auto& j : c1.GPU)
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(2));
                }
        }
        c1.automorph(index, 1);

        aux0.moddown(true, false);
        if constexpr (PRINT) {
            std::cout << "c0\n";
            for (auto& j : c0.GPU)
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(2));
                }
        }
        c0.add(aux0);
        if constexpr (PRINT) {
            std::cout << "Output KeySwitch 0.";
            for (auto& j : aux0.GPU)
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(2));
                }
        }

        if constexpr (PRINT) {
            std::cout << "Output Add 0.";
            for (auto& j : c0.GPU)
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(2));
                }
        }
        c0.automorph(index, 1);
        if constexpr (PRINT) {
            std::cout << "Output Rot 0.";
            for (auto& j : c0.GPU)
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(2));
                }
        }
    } else if constexpr (1) {

        cc.getKeySwitchAux().rotateModupDotKSK(c0, c1, kskRot);

        c1.moddown(true, false);
        c1.automorph(index, 1);
        c0.moddown(true, false);
        c0.automorph(index, 1);
    } else {

        c1.modupInto(cc.getKeySwitchAux());
        RNSPoly& aux0 = c1.dotKSKInPlaceFrom(cc.getKeySwitchAux(), kskRot, c1.getLevel());
        c1.moddown();
        c1.automorph(index, 1);

        aux0.moddown(true, false);
        c0.add(aux0);
        c0.automorph(index, 1);
    }
}

void Ciphertext::rotate(const Ciphertext& c, const int index, const KeySwitchingKey& kskRot) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    this->copy(c);
    this->rotate(index, kskRot);
}

void Ciphertext::conjugate(const Ciphertext& c) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    this->copy(c);
    //this->rotate(2 * cc.N - 1, cc.GetRotationKey(2 * cc.N - 1));

    int index = 2 * cc.N - 1;
    cc.getKeySwitchAux().setLevel(c1.getLevel());
    c1.modupInto(cc.getKeySwitchAux());
    RNSPoly& aux0 = c1.dotKSKInPlaceFrom(cc.getKeySwitchAux(), cc.GetRotationKey(index), c1.getLevel());
    c1.moddown(true, true);
    //c1.automorph(index, 1);

    for (int i = 0; i < (int)c1.GPU.size(); ++i) {
        c1.GPU.at(i).automorph(index, 1);
    }
    aux0.moddown(true, false);
    c0.add(aux0);
    //c0.automorph(index, 1);
    /*
    for (auto& i : cc.GetRotationKey(index).a.GPU) {
        for (auto& j : i.DIGITlimb) {
            for (auto& k : j) {
                SWITCH(k, printThisLimb(1));
            }
        }
    }
    std::cout << std::endl;
    for (auto& i : cc.GetRotationKey(index).b.GPU) {
        for (auto& j : i.DIGITlimb) {
            for (auto& k : j) {
                SWITCH(k, printThisLimb(1));
            }
        }
    }
    std::cout << std::endl;
*/
    for (int i = 0; i < (int)c0.GPU.size(); ++i) {
        c0.GPU.at(i).automorph(index, 1);
    }
}

void Ciphertext::rotate_hoisted(const std::vector<KeySwitchingKey*>& ksk, const std::vector<int>& indexes,
                                std::vector<Ciphertext*> results) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    constexpr bool PRINT = 0;
    assert(ksk.size() == results.size());
    for (auto& i : results) {
        if (this->c0.getLevel() > i->c0.getLevel()) {
            if (i->c0.getLevel() == -1) {
                i->c0.grow(this->c0.getLevel(), true);
            } else {
                assert("Ciphertext initialized but to the wrong level" == nullptr);
            }
        }
        if (this->c1.getLevel() > i->c1.getLevel()) {
            if (i->c1.getLevel() == -1) {
                i->c1.grow(this->c1.getLevel(), true);
            } else {
                assert("Ciphertext initialized but to the wrong level" == nullptr);
            }
        }
    }
    c1.modupInto(cc.getKeySwitchAux());

    for (int i = 0; i < ksk.size(); ++i) {
        if (indexes[i] == 0) {
            results[i]->copy(*this);
        } else {
            RNSPoly& aux0 = results[i]->c1.dotKSKInPlaceFrom(cc.getKeySwitchAux(), *ksk[i], c1.getLevel(), &c1);
            results[i]->c0.dropToLevel(getLevel());
            results[i]->c1.dropToLevel(getLevel());

            results[i]->c1.moddown();
            results[i]->c1.automorph(indexes[i], 1);
            aux0.moddown(true, false);
            results[i]->c0.add(c0, aux0);
            results[i]->c0.automorph(indexes[i], 1);
            results[i]->NoiseLevel = NoiseLevel;
            results[i]->NoiseFactor = NoiseFactor;
        }
    }
}
void Ciphertext::mult(const Ciphertext& b, const Ciphertext& c, const KeySwitchingKey& kskEval, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (this == &b && this == &c) {
        this->square(kskEval, rescale);
    } else if (this == &b) {
        this->mult(c, kskEval, rescale);
    } else if (this == &c) {
        this->mult(b, kskEval, rescale);
    } else {
        if (b.getLevel() <= c.getLevel()) {
            this->copy(b);
            this->mult(c, kskEval, rescale);
        } else {
            this->copy(c);
            this->mult(b, kskEval, rescale);
        }
    }
}

void Ciphertext::square(const Ciphertext& src, const KeySwitchingKey& kskEval, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (this == &src) {
        this->square(kskEval, rescale);
    } else {
        this->copy(src);
        this->square(kskEval, rescale);
    }
}
void Ciphertext::dropToLevel(int level) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.dropToLevel(level);
    c1.dropToLevel(level);
}
int Ciphertext::getLevel() const {
    assert(c0.getLevel() == c1.getLevel());
    return c0.getLevel();
}
void Ciphertext::multScalar(const Ciphertext& b, const double c, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    this->copy(b);
    this->multScalar(c, rescale);
}
void Ciphertext::evalLinearWSumMutable(uint32_t n, const std::vector<Ciphertext>& ctxs, std::vector<double> weights) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if constexpr (1) {
        this->c0.grow(ctxs[0].getLevel(), true);
        this->c1.grow(ctxs[0].getLevel(), true);
        this->NoiseLevel = 1;

        for (int i = 0; i < n; ++i) {
            if (cc.rescaleTechnique == Context::FIXEDMANUAL) {
                assert(ctxs[i].NoiseLevel == 1);
                assert(getLevel() <= ctxs[i].getLevel());
            } else {
                assert(ctxs[i].NoiseLevel == 1);
                assert(getLevel() == ctxs[i].getLevel());
            }
        }

        std::vector<uint64_t> elem;
        for (int i = 0; i < n; ++i) {
            auto aux = cc.ElemForEvalMult(c0.getLevel(), weights[i]);
            for (auto j : aux)
                elem.push_back(j);
        }

        std::vector<const RNSPoly*> c0s(n), c1s(n);

        for (int i = 0; i < n; ++i) {
            c0s[i] = &ctxs[i].c0;
            c1s[i] = &ctxs[i].c1;
        }
        c0.evalLinearWSum(n, c0s, elem);
        c1.evalLinearWSum(n, c1s, elem);

        this->NoiseLevel = 2;
        this->NoiseFactor = ctxs[0].NoiseFactor;
        NoiseFactor *= cc.param.ScalingFactorReal.at(c0.getLevel());
    } else {
        this->multScalar(ctxs[0], weights[0], false);
        for (int i = 1; i < n; ++i) {
            assert(getLevel() <= ctxs[i].getLevel());
        }
        for (int i = 1; i < n; ++i) {
            this->addMultScalar(ctxs[i], weights[i]);
        }
    }
}
void Ciphertext::addMultScalar(const Ciphertext& b, double d) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    assert(NoiseLevel == 2);
    assert(b.NoiseLevel == 1);
    assert(b.getLevel() >= getLevel());
    auto elem = cc.ElemForEvalMult(c0.getLevel(), d);

    RNSPoly aux0(cc);
    RNSPoly aux1(cc);
    aux0.copy(b.c0);
    aux0.multScalar(elem);
    c0.add(aux0);
    aux1.copy(b.c1);
    aux1.multScalar(elem);
    c1.add(aux1);
}
void Ciphertext::addScalar(const Ciphertext& b, double c) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    this->copy(b);
    this->addScalar(c);
}
void Ciphertext::add(const Ciphertext& b, const Ciphertext& c) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    assert(NoiseLevel <= 2);
    if (this == &b && this == &c) {
        this->add(c);  // TODO improve for less memory reads
    } else if (this == &b) {
        this->add(c);
    } else if (this == &c) {
        this->add(b);
    } else {
        if (b.getLevel() <= c.getLevel()) {
            this->copy(b);
            this->add(c);
        } else {
            this->copy(c);
            this->add(b);
        }
    }
}
void Ciphertext::copy(const Ciphertext& ciphertext) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.copy(ciphertext.c0);
    c1.copy(ciphertext.c1);
    //cudaDeviceSynchronize();
    this->NoiseLevel = ciphertext.NoiseLevel;
    this->NoiseFactor = ciphertext.NoiseFactor;
}
void Ciphertext::multPt(const Ciphertext& c, const Plaintext& b, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    this->copy(c);
    multPt(b, rescale);
}
void Ciphertext::addMultPt(const Ciphertext& c, const Plaintext& b, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    assert(NoiseLevel == 2);
    assert(c.NoiseLevel == 1);
    assert(b.NoiseLevel == 1);

    c0.addMult(c.c0, b.c0);
    c1.addMult(c.c1, b.c0);

    if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
        c0.rescale();
        c1.rescale();
    }

    //NoiseFactor += c.NoiseFactor * b.NoiseFactor;
    if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
        NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
        NoiseLevel -= 1;
    }
}
void Ciphertext::addPt(const Ciphertext& ciphertext, const Plaintext& plaintext) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    this->copy(ciphertext);
    this->addPt(plaintext);
}

void Ciphertext::sub(const Ciphertext& ciphertext, const Ciphertext& ciphertext1) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    assert(ciphertext.getLevel() <= ciphertext1.getLevel());
    this->copy(ciphertext);
    this->sub(ciphertext1);
}
bool Ciphertext::adjustForAddOrSub(const Ciphertext& b) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FIXEDMANUAL) {
        if (b.NoiseLevel > NoiseLevel || (b.getLevel() < getLevel()))
            return false;
        else
            return true;
    } else if (cc.rescaleTechnique == Context::FIXEDAUTO) {
        if (getLevel() - NoiseLevel > b.getLevel() - b.NoiseLevel) {
            if (b.NoiseLevel == 1 && NoiseLevel == 2) {
                this->dropToLevel(b.getLevel() + 1);
                rescale();
            } else {
                this->dropToLevel(b.getLevel());
            }
            return true;
        } else if (b.NoiseLevel == 1 && NoiseLevel == 2) {
            rescale();
            return true;
        } else if (NoiseLevel == 1 && b.NoiseLevel == 2) {
            return false;
        } else {
            return true;
        }
    } else if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        usint c1lvl = getLevel();
        usint c2lvl = b.getLevel();
        usint c1depth = this->NoiseLevel;
        usint c2depth = b.NoiseLevel;
        auto sizeQl1 = c1lvl + 1;
        auto sizeQl2 = c2lvl + 1;

        if (c1lvl > c2lvl) {
            if (c1depth == 2) {
                if (c2depth == 2) {
                    double scf1 = NoiseFactor;
                    double scf2 = b.NoiseFactor;
                    double scf = cc.param.ScalingFactorReal[c1lvl];  //cryptoParams->GetScalingFactorReal(c1lvl);
                    double q1 =
                        cc.param.ModReduceFactor[sizeQl1 - 1];  // cryptoParams->GetModReduceFactor(sizeQl1 - 1);
                    multScalarNoPrecheck(scf2 / scf1 * q1 / scf, true);
                    if (c1lvl - 1 > c2lvl) {
                        this->dropToLevel(c2lvl);
                        //LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 1);
                    }
                    NoiseFactor = b.NoiseFactor;
                } else {
                    if (c1lvl - 1 == c2lvl) {
                        rescale();
                    } else {
                        double scf1 = NoiseFactor;
                        double scf2 =
                            cc.param
                                .ScalingFactorRealBig[c2lvl + 1];  //cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
                        double scf = cc.param.ScalingFactorReal[c1lvl];  //cryptoParams->GetScalingFactorReal(c1lvl);
                        double q1 =
                            cc.param.ModReduceFactor[sizeQl1 - 1];  //cryptoParams->GetModReduceFactor(sizeQl1 - 1);
                        multScalarNoPrecheck(scf2 / scf1 * q1 / scf, true);
                        if (c1lvl - 2 > c2lvl) {
                            this->dropToLevel(c2lvl + 1);
                            //LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 2);
                        }
                        rescale();

                        NoiseFactor = b.NoiseFactor;
                    }
                }
            } else {
                if (c2depth == 2) {
                    double scf1 = NoiseFactor;
                    double scf2 = b.NoiseFactor;
                    double scf = cc.param.ScalingFactorReal[c1lvl];  // cryptoParams->GetScalingFactorReal(c1lvl);
                    multScalarNoPrecheck(scf2 / scf1 / scf);
                    this->dropToLevel(c2lvl);
                    //LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl);
                    NoiseFactor = scf2;
                } else {
                    double scf1 = NoiseFactor;
                    double scf2 =
                        cc.param.ScalingFactorRealBig[c2lvl + 1];    //cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
                    double scf = cc.param.ScalingFactorReal[c1lvl];  //cryptoParams->GetScalingFactorReal(c1lvl);
                    multScalarNoPrecheck(scf2 / scf1 / scf);
                    if (c1lvl - 1 > c2lvl) {
                        this->dropToLevel(c2lvl + 1);
                        //LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 1);
                    }
                    rescale();
                    NoiseFactor = b.NoiseFactor;
                }
            }
            return true;
        } else if (c1lvl < c2lvl) {
            return false;
        } else {
            if (c1depth < c2depth) {
                multScalar(1.0, false);
            } else if (c2depth < c1depth) {
                return false;
            }
            return true;
        }
    }
}

bool Ciphertext::adjustForMult(const Ciphertext& ciphertext) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (adjustForAddOrSub(ciphertext)) {
        if (NoiseLevel == 2)
            rescale();
        if (ciphertext.NoiseLevel == 2)
            return false;
        else
            return true;
    } else {
        if (NoiseLevel == 2)
            rescale();
        return false;
    }
}
bool Ciphertext::hasSameScalingFactor(const Plaintext& b) const {
    return NoiseFactor > b.NoiseFactor * (1 - 1e-9) && NoiseFactor < b.NoiseFactor * (1 + 1e-9);
}

}  // namespace FIDESlib::CKKS
