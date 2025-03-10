//
// Created by carlosad on 25/04/24.
//
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"

namespace FIDESlib::CKKS {

void RNSPoly::grow(int new_level, bool single_malloc, bool constant) {
    if (!single_malloc) {
        for (int i = this->level + 1; i <= new_level; ++i) {
            GPU.at(cc.limbGPUid.at(i).x).generateLimb();
        }
    } else {
        for (size_t i = 0; i < GPU.size(); ++i) {
            size_t sub_levels = 0;
            for (; sub_levels < GPU.at(i).meta.size() && GPU.at(i).meta.at(sub_levels).id <= new_level; ++sub_levels)
                ;
            if (!constant) {
                GPU.at(i).generateLimbSingleMalloc(sub_levels);
            } else {
                GPU.at(i).generateLimbConstant(sub_levels);
            }
        }
    }
    level = new_level;
}

RNSPoly::RNSPoly(Context& context, int level, bool single_malloc) : cc(context), level(-1) {
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        cudaSetDevice(cc.GPUid.at(i));
        GPU.emplace_back(cc, i);
    }
    assert(level >= -1 && level <= cc.L);
    grow(level, single_malloc);
}

int RNSPoly::getLevel() const {
    return level;
}

void RNSPoly::store(std::vector<std::vector<uint64_t>>& data) {
    data.resize(level + 1);
    for (size_t i = 0; i < data.size(); ++i) {
        auto& rec = cc.meta[cc.limbGPUid[i].x][cc.limbGPUid[i].y];
        SWITCH(GPU[cc.limbGPUid[i].x].limb[cc.limbGPUid[i].y], store_convert(data[i]));
    }
}

RNSPoly::RNSPoly(Context& context, const std::vector<std::vector<uint64_t>>& data) : RNSPoly(context, data.size() - 1) {

    assert(data.size() <= cc.prime.size());
    std::vector<uint64_t> moduli(data.size());
    for (int i = 0; i < data.size(); ++i)
        moduli[i] = cc.prime[i].p;

    load(data, moduli);
}

void RNSPoly::add(const RNSPoly& p) {
    assert(level <= p.level);
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).add(p.GPU.at(i));
    }
}

void RNSPoly::sub(const RNSPoly& p) {
    assert(level <= p.level);
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).sub(p.GPU.at(i));
    }
}

void RNSPoly::modup() {
    assert(GPU.size() == 1 || 0 == "ModUp Multi-GPU not implemented.");
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).modup(level + 1, cc.getKeySwitchAux2().GPU.at(i));
    }
}

void RNSPoly::sync() {
    for (auto& i : GPU) {
        for (auto& j : i.limb) {
            assert(STREAM(j).ptr != nullptr);
            cudaStreamSynchronize(STREAM(j).ptr);
        }
        cudaStreamSynchronize(i.s.ptr);
    }
}

void RNSPoly::rescale() {
    assert(GPU.size() == 1 && "Rescale Multi-GPU not implemented.");
    for (auto& i : GPU) {
        i.rescale();
    }
    --level;
}

void RNSPoly::multPt(const RNSPoly& p, bool rescale) {
    if (rescale) {
        assert(GPU.size() == 1 && "Rescale Multi-GPU not implemented.");
        for (size_t i = 0; i < GPU.size(); ++i) {
            GPU.at(i).multPt(p.GPU.at(i));
        }
        --level;
    } else {
        for (size_t i = 0; i < GPU.size(); ++i) {
            GPU.at(i).multElement(p.GPU.at(i));
        }
    }
}

void RNSPoly::freeSpecialLimbs() {
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).freeSpecialLimbs();
    }
}

template <ALGO algo>
void RNSPoly::NTT(int batch) {
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).NTT<algo>(batch);
    }
}

#define YY(algo) template void RNSPoly::NTT<algo>(int batch);

#include "ntt_types.inc"

#undef YY

template <ALGO algo>
void RNSPoly::INTT(int batch) {
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).INTT<algo>(batch);
    }
}

#define YY(algo) template void RNSPoly::INTT<algo>(int batch);

#include "ntt_types.inc"

#undef YY

std::array<RNSPoly, 2> RNSPoly::dotKSK(const KeySwitchingKey& ksk) {
    constexpr bool PRINT = false;
    Out(KEYSWITCH, "dotKSK in");

    std::array<RNSPoly, 2> result{RNSPoly(cc, level, true), RNSPoly(cc, level, true)};
    result[0].generateSpecialLimbs();
    result[1].generateSpecialLimbs();

    if constexpr (PRINT)
        for (auto& i : ksk.b.GPU) {
            for (auto& j : i.DECOMPlimb) {
                for (auto& k : j) {
                    SWITCH(k, printThisLimb(1));
                }
            }

            for (auto& j : i.DIGITlimb) {
                for (auto& k : j) {
                    SWITCH(k, printThisLimb(1));
                }
            }
        }
    for (size_t i = 0; i < GPU.size(); ++i) {
        dotKSKinto(result[0], ksk.b, level);
        dotKSKinto(result[1], ksk.a, level);
    }

    Out(KEYSWITCH, "dotKSK out");
    return result;
}

void RNSPoly::generateSpecialLimbs() {
    for (auto& i : GPU)
        i.generateSpecialLimb();
}

RNSPoly::RNSPoly(RNSPoly&& src) noexcept : cc(src.cc), level(src.level), GPU(std::move(src.GPU)) {}

void RNSPoly::multElement(const RNSPoly& poly) {
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).multElement(poly.GPU.at(i));
    }
}

void RNSPoly::multElement(const RNSPoly& poly1, const RNSPoly& poly2) {
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).multElement(poly1.GPU.at(i), poly2.GPU.at(i));
    }
}

RNSPoly RNSPoly::clone(bool single_malloc) const {
    CudaCheckErrorModNoSync;
    auto res = RNSPoly(cc, this->level, single_malloc);

    for (size_t i = 0; i < GPU.size(); ++i) {
        res.GPU.at(i).copyLimb(GPU.at(i));
    }
    return res;
}

void RNSPoly::generateDecompAndDigit() {
    for (auto& i : GPU) {
        i.generateAllDecompAndDigit();
    }
}

void RNSPoly::mult1AddMult23Add4(const RNSPoly& poly1, const RNSPoly& poly2, const RNSPoly& poly3,
                                 const RNSPoly& poly4) {

    for (size_t i = 0; i < GPU.size(); ++i) {
        GPU.at(i).mult1AddMult23Add4(poly1.GPU.at(i), poly2.GPU.at(i), poly3.GPU.at(i), poly4.GPU.at(i));
    }
}

void RNSPoly::mult1Add2(const RNSPoly& poly1, const RNSPoly& poly2) {
    for (size_t i = 0; i < GPU.size(); ++i) {
        GPU.at(i).mult1Add2(poly1.GPU.at(i), poly2.GPU.at(i));
    }
}

void RNSPoly::loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data,
                              const std::vector<std::vector<uint64_t>>& moduli) {
    for (size_t i = 0; i < GPU.size(); ++i) {
        GPU.at(i).loadDecompDigit(data, moduli);
    }
}

void RNSPoly::dotKSKinto(RNSPoly& acc, const RNSPoly& ksk, int level, const RNSPoly* limbsrc) {
    for (size_t i = 0; i < acc.GPU.size(); ++i) {
        acc.GPU.at(i).dotKSK(GPU.at(i), ksk.GPU.at(i), level, false, limbsrc ? &limbsrc->GPU.at(i) : nullptr);
    }
}

void RNSPoly::multModupDotKSK(RNSPoly& c1, const RNSPoly& c1tilde, RNSPoly& c0, const RNSPoly& c0tilde,
                              const KeySwitchingKey& key) {
    assert(GPU.size() == 1 && "multModupDotKSK Multi-GPU not implemented.");
    assert(c1.level <= c1tilde.level);
    generateDecompAndDigit();
    c0.generateSpecialLimbs();
    c1.generateSpecialLimbs();
    for (size_t i = 0; i < GPU.size(); ++i) {
        GPU.at(i).multModupDotKSK(c1.GPU.at(i), c1tilde.GPU.at(i), c0.GPU.at(i), c0tilde.GPU.at(i), key.a.GPU.at(i),
                                  key.b.GPU.at(i), c1.level + 1);
    }
}

void RNSPoly::rotateModupDotKSK(RNSPoly& c0, RNSPoly& c1, const KeySwitchingKey& key) {
    assert(GPU.size() == 1 && "rotateModupDotKSK Multi-GPU not implemented.");
    generateDecompAndDigit();
    c0.generateSpecialLimbs();
    c1.generateSpecialLimbs();
    for (size_t i = 0; i < GPU.size(); ++i) {
        GPU.at(i).rotateModupDotKSK(c1.GPU.at(i), c0.GPU.at(i), key.a.GPU.at(i), key.b.GPU.at(i), c1.level + 1);
    }
}

template <ALGO algo>
void RNSPoly::moddown(bool ntt, bool free) {
    assert(GPU.size() == 1 && "ModDown Multi-GPU not implemented.");
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).moddown<algo>(cc.getModdownAux().GPU.at(i), ntt, free, level);
    }
}

#define YY(algo) template void RNSPoly::moddown<algo>(bool ntt, bool free);

#include "ntt_types.inc"

#undef YY

void RNSPoly::automorph(const int idx, const int br) {
    int k = modpow(5, idx, cc.N * 2);
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).automorph(k, br);
    }
}

void RNSPoly::automorph_multi(const int idx, const int br) {
    int k = modpow(5, idx, cc.N * 2);
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).automorph_multi(k, br);
    }
}

RNSPoly& RNSPoly::dotKSKInPlace(const KeySwitchingKey& ksk, int level) {
    constexpr bool PRINT = false;
    Out(KEYSWITCH, "dotKSK in");

    //RNSPoly result{RNSPoly(cc, level, true)};
    cc.getKeySwitchAux2().setLevel(level);
    cc.getKeySwitchAux2().generateSpecialLimbs();
    generateSpecialLimbs();
    if constexpr (PRINT)
        for (auto& i : ksk.b.GPU) {
            for (auto& j : i.DECOMPlimb) {
                for (auto& k : j) {
                    SWITCH(k, printThisLimb(1));
                }
            }

            for (auto& j : i.DIGITlimb) {
                for (auto& k : j) {
                    SWITCH(k, printThisLimb(1));
                }
            }
        }
    dotKSKinto(cc.getKeySwitchAux2(), ksk.b, level);
    dotKSKInPlace(ksk.a, level);

    Out(KEYSWITCH, "dotKSK out");
    return cc.getKeySwitchAux2();
}

void RNSPoly::dotKSKInPlace(const RNSPoly& ksk_b, int level) {
    for (size_t i = 0; i < GPU.size(); ++i) {
        GPU.at(i).dotKSK(GPU.at(i), ksk_b.GPU.at(i), level, true);
    }
}
void RNSPoly::setLevel(const int level) {
    assert(level >= -1 && level <= cc.L);
    this->level = level;
}
void RNSPoly::modupInto(RNSPoly& poly) {
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).modupInto(poly.GPU.at(i), level + 1, cc.getKeySwitchAux2().GPU.at(i));
    }
}
RNSPoly& RNSPoly::dotKSKInPlaceFrom(RNSPoly& poly, const KeySwitchingKey& ksk, int level, const RNSPoly* limbsrc) {
    constexpr bool PRINT = false;
    Out(KEYSWITCH, "dotKSK in");

    cc.getKeySwitchAux2().setLevel(level);
    cc.getKeySwitchAux2().generateSpecialLimbs();
    generateSpecialLimbs();
    if constexpr (PRINT)
        for (auto& i : ksk.b.GPU) {
            for (auto& j : i.DECOMPlimb) {
                for (auto& k : j) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            for (auto& j : i.DIGITlimb) {
                for (auto& k : j) {
                    SWITCH(k, printThisLimb(1));
                }
            }
        }
    poly.dotKSKinto(cc.getKeySwitchAux2(), ksk.b, level, limbsrc ? limbsrc : this);
    poly.dotKSKinto(*this, ksk.a, level, limbsrc ? limbsrc : this);

    Out(KEYSWITCH, "dotKSK out");
    return cc.getKeySwitchAux2();
}
void RNSPoly::multScalar(std::vector<uint64_t>& vector1) {
    for (auto& i : GPU)
        i.multScalar(vector1);
}
void RNSPoly::add(const RNSPoly& a, const RNSPoly& b) {
    assert(level <= a.level);
    assert(level <= b.level);
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).add(a.GPU.at(i), b.GPU.at(i));
    }
}
void RNSPoly::squareElement(const RNSPoly& poly) {
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).squareElement(poly.GPU.at(i));
    }
}
void RNSPoly::binomialSquareFold(RNSPoly& c0_res, const RNSPoly& c2_key_switched_0, const RNSPoly& c2_key_switched_1) {
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).binomialSquareFold(c0_res.GPU.at(i), c2_key_switched_0.GPU.at(i), c2_key_switched_1.GPU.at(i));
    }
}
void RNSPoly::addScalar(std::vector<uint64_t>& vector1) {
    for (auto& i : GPU)
        i.addScalar(vector1);
}
void RNSPoly::subScalar(std::vector<uint64_t>& vector1) {
    for (auto& i : GPU)
        i.subScalar(vector1);
}
void RNSPoly::copy(const RNSPoly& poly) {
    for (int i = 0; i < (int)GPU.size(); ++i) {
        if (GPU.at(i).limb.size() == 0 && GPU.at(i).bufferLIMB == nullptr) {
            GPU.at(i).generateLimbSingleMalloc(poly.GPU.at(i).limb.size());
        } else {
            for (int j = GPU.at(i).limb.size(); j < poly.GPU.at(i).limb.size(); ++j) {
                GPU.at(i).generateLimb();
                GPU.at(i).s.wait(STREAM(GPU.at(i).limb.back()));
            }
            for (int j = GPU.at(i).limb.size(); j > poly.GPU.at(i).limb.size(); --j) {
                GPU.at(i).limb.pop_back();
            }
        }
        GPU.at(i).copyLimb(poly.GPU.at(i));
    }
    //cudaDeviceSynchronize();
    this->level = poly.level;
}
void RNSPoly::dropToLevel(int level) {
    for (int i = this->level; i > level; --i) {
        GPU.at(cc.limbGPUid.at(i).x).dropLimb();
        //GPU.at(cc.limbGPUid.at(i).x).limb.pop_back();
        this->level--;
    }
}
void RNSPoly::addMult(const RNSPoly& poly, const RNSPoly& poly1) {
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).addMult(poly.GPU.at(i), poly1.GPU.at(i));
    }
}
void RNSPoly::load(const std::vector<std::vector<uint64_t>>& data, const std::vector<uint64_t>& moduli) {

    int limbsize = 0;
    int Slimbsize = 0;
    for (int i = 0; i < data.size(); ++i) {
        if (i <= cc.L && moduli[i] == cc.prime.at(i).p) {
            limbsize++;
        } else {
            Slimbsize++;
        }
    }

    assert(limbsize - 1 <= cc.L);
    if (level < limbsize - 1)
        grow(limbsize - 1, true);
    assert(level == limbsize - 1);
    for (size_t i = 0; i < limbsize; ++i) {
        assert(moduli[i] == cc.prime.at(i).p);
        SWITCH(GPU[cc.limbGPUid[i].x].limb[cc.limbGPUid[i].y], load_convert(data[i]));
    }

    if (data.size() > limbsize)
        generateSpecialLimbs();
    for (size_t i = limbsize; i < data.size(); ++i) {
        for (auto& j : GPU) {
            assert(moduli[i] == cc.specialPrime.at(i - limbsize).p);
            SWITCH(j.SPECIALlimb[i - limbsize], load_convert(data[i]));
        }
    }
}

void RNSPoly::loadConstant(const std::vector<std::vector<uint64_t>>& data, const std::vector<uint64_t>& moduli) {
    int limbsize = 0;
    int Slimbsize = 0;
    for (int i = 0; i < data.size(); ++i) {
        if (i <= cc.L && moduli[i] == cc.prime.at(i).p) {
            limbsize++;
        } else {
            Slimbsize++;
        }
    }

    assert(limbsize - 1 <= cc.L);
    grow(limbsize - 1, true, true);
    assert(level == limbsize - 1);
    for (size_t i = 0; i < limbsize; ++i) {
        assert(moduli[i] == cc.prime.at(i).p);
        SWITCH(GPU[cc.limbGPUid[i].x].limb[cc.limbGPUid[i].y], load_convert(data[i]));
    }

    if (data.size() > limbsize)
        generateSpecialLimbs();
    for (size_t i = limbsize; i < data.size(); ++i) {
        for (auto& j : GPU) {
            assert(moduli[i] == cc.specialPrime.at(i - limbsize).p);
            SWITCH(j.SPECIALlimb[i - limbsize], load_convert(data[i]));
        }
    }
}

void RNSPoly::broadcastLimb0() {
    assert(GPU.size() == 1);
    for (int i = 0; i < (int)GPU.size(); ++i) {
        GPU.at(i).broadcastLimb0();
    }
}
void RNSPoly::evalLinearWSum(uint32_t n, std::vector<const RNSPoly*>& vec, std::vector<uint64_t>& elem) {
    for (int i = 0; i < GPU.size(); ++i) {
        std::vector<const LimbPartition*> ps(n);
        for (int j = 0; j < n; ++j) {
            ps[j] = &vec[j]->GPU.at(i);
        }
        GPU.at(i).evalLinearWSum(n, ps, elem);
    }
}

void RNSPoly::squareModupDotKSK(RNSPoly& c0, RNSPoly& c1, const KeySwitchingKey& key) {
    assert(GPU.size() == 1 && "squareModupDotKSK Multi-GPU not implemented.");
    generateDecompAndDigit();
    c0.generateSpecialLimbs();
    c1.generateSpecialLimbs();
    for (size_t i = 0; i < GPU.size(); ++i) {
        GPU.at(i).squareModupDotKSK(c1.GPU.at(i), c0.GPU.at(i), key.a.GPU.at(i), key.b.GPU.at(i), c1.level + 1);
    }
}

}  // namespace FIDESlib::CKKS