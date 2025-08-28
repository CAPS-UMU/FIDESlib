#include <source_location>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include "CudaUtils.cuh"

void FIDESlib::CKKS::LinearTransform(Ciphertext& ctxt, int rowSize, int bStep, std::vector<Plaintext*> pts, int stride,
                                     int offset) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    assert(pts.size() >= rowSize);
    for (auto i : pts) {
        assert(i != nullptr);
    }
    Context& cc = ctxt.cc;
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();

    std::vector<Ciphertext> fastRotation;

    for (int i = 0; i < bStep; ++i)
        fastRotation.emplace_back(cc);

    std::vector<Ciphertext*> fastRotationPtr;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 0; i < bStep; ++i) {
        fastRotationPtr.push_back(&fastRotation[i]);
        keys.push_back(i == 0 ? nullptr : &cc.GetRotationKey(i * stride));
        indexes.push_back(i * stride);
    }

    bool ext = true;
    if (bStep == 1)
        ext = false;
    for (auto& i : pts) {
        if (!i->c0.isModUp()) {
            ext = false;
        }
    }

    ctxt.rotate_hoisted(keys, indexes, fastRotationPtr, ext);

    Ciphertext inner(cc);

    for (uint32_t j = gStep - 1; j < gStep; --j) {

        // if (j == gStep - 1) {
        //     std::cout << "PT ct: " << fastRotation[0].getLevel() << ", " << fastRotation[0].NoiseLevel << std::endl;
        //     std::cout << "PT pt: " << (*pts[1]).c0.getLevel() << ", " << (*pts[1]).NoiseLevel << std::endl;
        // }
        int n = 1;
        //inner.multPt(ctxt, A[bStep * j], false);
        for (uint32_t i = 1; i < bStep; i++) {
            if (bStep * j + i < rowSize) {
                n++;
                //inner.addMultPt(fastRotation[i - 1], A[bStep * j + i], false);
            }
        }

        if (fastRotation[0].getLevel() > inner.getLevel()) {
            inner.c0.grow(fastRotation[0].getLevel(), true);
            inner.c1.grow(fastRotation[0].getLevel(), true);
        } else {
            inner.dropToLevel(fastRotation[0].getLevel());
        }

        inner.dotProductPt(fastRotation.data(), (const Plaintext**)pts.data() + j * bStep, n, ext);

        if (ext)
            inner.modDown(false);

        if (j > 0) {
            if (j == gStep - 1) {
                ctxt.copy(inner);
            } else {
                ctxt.add(inner);
            }
            ctxt.rotate((int)stride * bStep, cc.GetRotationKey((int)stride * bStep));
        } else {
            if (gStep == 1) {
                ctxt.copy(inner);
            } else {
                ctxt.add(inner);
            }
        }
    }

    if (offset != 0) {
        ctxt.rotate(offset, cc.GetRotationKey(offset));
    }
}

// Masked CCMM
void FIDESlib::CKKS::LinearTransform(Ciphertext& ctxt, int rowSize, int bStep, std::vector<Plaintext*> pts, int stride,
                                     int offset, Plaintext& mask_pt) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    assert(pts.size() >= rowSize);
    for (auto i : pts) {
        assert(i != nullptr);
    }
    Context& cc = ctxt.cc;
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();

    std::vector<Ciphertext> fastRotation;

    for (int i = 0; i < bStep; ++i) {
        fastRotation.emplace_back(cc);
    }

    std::vector<Ciphertext*> fastRotationPtr;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 0; i < bStep; ++i) {
        fastRotationPtr.push_back(&fastRotation[i]);

        keys.push_back(i == 0 ? nullptr : &cc.GetRotationKey(i * stride));
        indexes.push_back(i * stride);
    }

    bool ext = true;
    if (bStep == 1)
        ext = false;

    // int i = 0;
    for (const Plaintext* p : pts) {
        ext &= p->c0.isModUp();
    }

    ctxt.rotate_hoisted(keys, indexes, fastRotationPtr, ext);
    Ciphertext inner(cc);

    Plaintext mask_pt_rot(cc);
    mask_pt_rot.copy(mask_pt);

    for (uint32_t j = gStep - 1; j < gStep; --j) {
        // std::cout << "# j: " << j << std::endl;

        int n = 1;
        //inner.multPt(ctxt, A[bStep * j], false);
        for (uint32_t i = 1; i < bStep; i++) {
            if (bStep * j + i < rowSize) {
                n++;
                //inner.addMultPt(fastRotation[i - 1], A[bStep * j + i], false);
            }
        }

        // masked pts !
        std::vector<Plaintext> tmp_storage;
        tmp_storage.reserve(n);
        for (int i = 0; i < n; i++) {
            Plaintext tmp(cc);
            tmp.copy(mask_pt_rot);
            tmp.automorph(indexes[i]);
            tmp.multPt(*pts[j * bStep + i], true);

            // tmp.copy(*pts[j * bStep + i]);
            tmp_storage.emplace_back(std::move(tmp));
        }
        std::vector<Plaintext*> tmp_pts;
        tmp_pts.reserve(tmp_storage.size());
        for (auto& x : tmp_storage)
            tmp_pts.push_back(&x);

        if (fastRotation[0].getLevel() > inner.getLevel()) {
            inner.c0.grow(fastRotation[0].getLevel(), true);
            inner.c1.grow(fastRotation[0].getLevel(), true);
        } else {
            inner.dropToLevel(fastRotation[0].getLevel());
        }

        // inner.dotProductPt(fastRotation.data(), (const Plaintext**)pts.data() + j * bStep, n, ext);
        inner.dotProductPt(fastRotation.data(), (const Plaintext**)tmp_pts.data(), n, ext);

        // std::cout << "inner: " << inner.getLevel() << ", " << inner.NoiseLevel << std::endl;

        if (ext)
            inner.modDown(false);

        if (j > 0) {
            if (j == gStep - 1) {
                ctxt.copy(inner);
            } else {
                ctxt.add(inner);
            }
            ctxt.rotate((int)stride * bStep, cc.GetRotationKey((int)stride * bStep));
        } else {
            if (gStep == 1) {
                ctxt.copy(inner);
            } else {
                ctxt.add(inner);
            }
        }
    }

    if (offset != 0) {
        ctxt.rotate(offset, cc.GetRotationKey(offset));
    }
}

void FIDESlib::CKKS::LinearTransformSpecial(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt2,
                                            FIDESlib::CKKS::Ciphertext& ctxt3, int rowSize, int bStep,
                                            std::vector<Plaintext*> pts1, std::vector<Plaintext*> pts2, int stride,
                                            int stride3) {
    constexpr bool PRINT = false;
    if constexpr (PRINT)
        std::cout << std::endl << "LinearTransformSpecial " << std::endl;

    if constexpr (PRINT) {
        std::cout << "ctxt1 ";
        for (auto& j : ctxt1.c0.GPU)
            for (auto& k : j.limb) {
                SWITCH(k, printThisLimb(1));
            }
        std::cout << std::endl;
        for (auto& j : ctxt1.c0.GPU)
            for (auto& k : j.SPECIALlimb) {
                SWITCH(k, printThisLimb(1));
            }
        std::cout << std::endl;
    }

    if constexpr (PRINT) {
        std::cout << "ctxt2 ";
        for (auto& j : ctxt2.c0.GPU)
            for (auto& k : j.limb) {
                SWITCH(k, printThisLimb(1));
            }
        std::cout << std::endl;
        for (auto& j : ctxt2.c0.GPU)
            for (auto& k : j.SPECIALlimb) {
                SWITCH(k, printThisLimb(1));
            }
        std::cout << std::endl;
    }

    if constexpr (PRINT) {
        std::cout << "ctxt3 ";
        for (auto& j : ctxt3.c0.GPU)
            for (auto& k : j.limb) {
                SWITCH(k, printThisLimb(1));
            }
        std::cout << std::endl;
        for (auto& j : ctxt3.c0.GPU)
            for (auto& k : j.SPECIALlimb) {
                SWITCH(k, printThisLimb(1));
            }
        std::cout << std::endl;
    }

    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    assert(pts1.size() >= rowSize);
    for (auto i : pts1) {
        assert(i != nullptr);
    }
    assert(pts2.size() >= rowSize);

    Context& cc = ctxt1.cc;
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    if (ctxt1.NoiseLevel == 2)
        ctxt1.rescale();
    if (ctxt2.NoiseLevel == 2)
        ctxt2.rescale();
    if (ctxt3.NoiseLevel == 2)
        ctxt3.rescale();

    std::vector<Ciphertext> fastRotation2;
    std::vector<Ciphertext> fastRotation1;
    for (int i = 0; i < bStep - 1; ++i) {
        fastRotation2.emplace_back(cc);
        fastRotation1.emplace_back(cc);
    }

    std::vector<Ciphertext*> fastRotationPtr2;
    std::vector<Ciphertext*> fastRotationPtr1;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr1.push_back(&fastRotation1[i - 1]);
        fastRotationPtr2.push_back(&fastRotation2[i - 1]);
        keys.push_back(&cc.GetRotationKey(i * stride));
        indexes.push_back(i * stride);
    }

    ctxt1.rotate_hoisted(keys, indexes, fastRotationPtr1, false);
    ctxt2.rotate_hoisted(keys, indexes, fastRotationPtr2, false);

    std::vector<Ciphertext> fastRotation3;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation3.emplace_back(cc);

    std::vector<Ciphertext*> fastRotationPtr3;
    std::vector<int> indexes3;
    std::vector<KeySwitchingKey*> keys3;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr3.push_back(&fastRotation3[i - 1]);
        keys3.push_back(&cc.GetRotationKey(i * stride3));
        indexes3.push_back(i * stride3);
    }

    Ciphertext result(cc);
    Ciphertext inner(cc);
    Ciphertext aux(cc);

    if ((gStep - 1) * bStep * (rowSize - 1) != 0)
        ctxt3.rotate((gStep - 1) * bStep * (rowSize - 1), cc.GetRotationKey((gStep - 1) * bStep * (rowSize - 1)));

    for (uint32_t j = gStep - 1; j < gStep; --j) {

        inner.multPt(ctxt1, *pts1[bStep * j], false);
        if (bStep * j != 0)
            inner.addMultPt(ctxt2, *pts2[bStep * j], true);
        inner.mult(ctxt3, cc.GetEvalKey(), false);

        if constexpr (PRINT) {
            std::cout << "inner " << bStep * j << " ";
            for (auto& j : inner.c0.GPU)
                for (auto& k : j.limb) {
                    SWITCH(k, printThisLimb(1));
                }
            std::cout << std::endl;
            for (auto& j : inner.c0.GPU)
                for (auto& k : j.SPECIALlimb) {
                    SWITCH(k, printThisLimb(1));
                }
            std::cout << std::endl;
        }

        for (uint32_t i = 1; i < bStep; i++) {
            if (bStep * j + i < rowSize) {
                if (i == 1) {
                    int size = std::min((int)bStep - 1, (int)(rowSize - (bStep * j + i)));
                    if (size < bStep - 1) {
                        auto keys3_ = keys3;
                        auto indexes3_ = indexes3;
                        auto fastRotationPtr3_ = fastRotationPtr3;

                        keys3_.resize(size);
                        indexes3_.resize(size);
                        fastRotationPtr3_.resize(size);

                        ctxt3.rotate_hoisted(keys3_, indexes3_, fastRotationPtr3_, false);
                    } else {
                        ctxt3.rotate_hoisted(keys3, indexes3, fastRotationPtr3, false);
                    }
                }
                aux.multPt(fastRotation1[i - 1], *pts1[bStep * j + i], false);
                aux.addMultPt(fastRotation2[i - 1], *pts2[bStep * j + i], true);
                aux.mult(fastRotation3[i - 1], cc.GetEvalKey(), false);
                if constexpr (PRINT) {
                    std::cout << "inner " << bStep * j + i << " ";
                    for (auto& j : aux.c0.GPU)
                        for (auto& k : j.limb) {
                            SWITCH(k, printThisLimb(1));
                        }
                    std::cout << std::endl;
                    for (auto& j : aux.c0.GPU)
                        for (auto& k : j.SPECIALlimb) {
                            SWITCH(k, printThisLimb(1));
                        }
                    std::cout << std::endl;
                }
                inner.add(aux);
            }
        }

        if (j > 0) {
            if (j == gStep - 1) {
                result.copy(inner);
            } else {
                result.add(inner);
            }
            result.rotate(stride * bStep, cc.GetRotationKey(stride * bStep));
            ctxt3.rotate(-bStep * (rowSize - 1), cc.GetRotationKey(-bStep * (rowSize - 1)));
        } else {
            if (gStep == 1) {
                result.copy(inner);
            } else {
                result.add(inner);
            }
        }
    }

    ctxt1.copy(result);
    CudaCheckErrorModNoSync;
}

// Masked CCMM
void FIDESlib::CKKS::LinearTransformSpecial(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt2,
                                            FIDESlib::CKKS::Ciphertext& ctxt3, int rowSize, int bStep,
                                            std::vector<Plaintext*> pts1, std::vector<Plaintext*> pts2, int stride,
                                            int stride3, Plaintext& mask_pt) {
    constexpr bool PRINT = false;

    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    assert(pts1.size() >= rowSize);
    for (auto i : pts1) {
        assert(i != nullptr);
    }
    assert(pts2.size() >= rowSize);

    Context& cc = ctxt1.cc;
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    if (ctxt1.NoiseLevel == 2)
        ctxt1.rescale();
    if (ctxt2.NoiseLevel == 2)
        ctxt2.rescale();
    if (ctxt3.NoiseLevel == 2)
        ctxt3.rescale();

    std::vector<Ciphertext> fastRotation2;
    std::vector<Ciphertext> fastRotation1;
    for (int i = 0; i < bStep - 1; ++i) {
        fastRotation2.emplace_back(cc);
        fastRotation1.emplace_back(cc);
    }

    std::vector<Ciphertext*> fastRotationPtr2;
    std::vector<Ciphertext*> fastRotationPtr1;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr1.push_back(&fastRotation1[i - 1]);
        fastRotationPtr2.push_back(&fastRotation2[i - 1]);
        keys.push_back(&cc.GetRotationKey(i * stride));
        indexes.push_back(i * stride);
    }

    ctxt1.rotate_hoisted(keys, indexes, fastRotationPtr1, false);
    ctxt2.rotate_hoisted(keys, indexes, fastRotationPtr2, false);

    std::vector<Ciphertext> fastRotation3;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation3.emplace_back(cc);

    std::vector<Ciphertext*> fastRotationPtr3;
    std::vector<int> indexes3;
    std::vector<KeySwitchingKey*> keys3;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr3.push_back(&fastRotation3[i - 1]);
        keys3.push_back(&cc.GetRotationKey(i * stride3));
        indexes3.push_back(i * stride3);
    }

    Ciphertext result(cc);
    Ciphertext inner(cc);
    Ciphertext aux(cc);

    if ((gStep - 1) * bStep * (rowSize - 1) != 0)
        ctxt3.rotate((gStep - 1) * bStep * (rowSize - 1), cc.GetRotationKey((gStep - 1) * bStep * (rowSize - 1)));

    Plaintext mask_temp(cc);
    mask_temp.copy(mask_pt);

    for (uint32_t j = gStep - 1; j < gStep; --j) {
        // if (j != gStep - 1 && j > 0) {
        //     mask_temp.automorph(- stride * bStep);
        // }

        Plaintext mask_temp2(cc);
        mask_temp2.copy(mask_pt);
        mask_temp2.automorph(-j * stride * bStep);

        Plaintext tmp1(cc);
        tmp1.copy(*pts1[bStep * j]);
        tmp1.multPt(tmp1, mask_temp2, true);

        // std::cout << ctxt1.getLevel() << tmp1.c0.getLevel() << std::endl;

        inner.multPt(ctxt1, tmp1, false);

        if (bStep * j != 0) {
            Plaintext tmp2(cc);
            tmp2.copy(*pts2[bStep * j]);
            tmp2.multPt(tmp2, mask_temp2, true);

            // std::cout << ctxt2.getLevel() << tmp2.c0.getLevel() << std::endl;

            inner.addMultPt(ctxt2, tmp2, false);
        }

        inner.mult(ctxt3, cc.GetEvalKey(), false);

        for (uint32_t i = 1; i < bStep; i++) {
            if (bStep * j + i < rowSize) {
                if (i == 1) {
                    int size = std::min((int)bStep - 1, (int)(rowSize - (bStep * j + i)));
                    if (size < bStep - 1) {
                        auto keys3_ = keys3;
                        auto indexes3_ = indexes3;
                        auto fastRotationPtr3_ = fastRotationPtr3;

                        keys3_.resize(size);
                        indexes3_.resize(size);
                        fastRotationPtr3_.resize(size);

                        ctxt3.rotate_hoisted(keys3_, indexes3_, fastRotationPtr3_, false);
                    } else {
                        ctxt3.rotate_hoisted(keys3, indexes3, fastRotationPtr3, false);
                    }
                }

                Plaintext temp1(cc);
                temp1.copy(*pts1[bStep * j + i]);
                temp1.multPt(temp1, mask_temp2, true);
                // std::cout << fastRotation1[i - 1].getLevel() << temp1.c0.getLevel() << std::endl;
                aux.multPt(fastRotation1[i - 1], temp1, false);

                Plaintext temp2(cc);
                temp2.copy(*pts2[bStep * j + i]);
                temp2.multPt(temp2, mask_temp2, true);
                // std::cout << fastRotation2[i - 1].getLevel() << temp2.c0.getLevel() << std::endl;
                aux.addMultPt(fastRotation2[i - 1], temp2);

                aux.mult(fastRotation3[i - 1], cc.GetEvalKey(), false);

                inner.add(aux);
            }
        }

        if (j > 0) {
            if (j == gStep - 1) {
                result.copy(inner);
            } else {
                result.add(inner);
            }
            result.rotate(stride * bStep, cc.GetRotationKey(stride * bStep));
            ctxt3.rotate(-bStep * (rowSize - 1), cc.GetRotationKey(-bStep * (rowSize - 1)));
        } else {
            if (gStep == 1) {
                result.copy(inner);
            } else {
                result.add(inner);
            }
        }
    }

    ctxt1.copy(result);
    CudaCheckErrorModNoSync;
}

std::vector<int> FIDESlib::CKKS::GetLinearTransformRotationIndices(int bStep, int stride, int offset) {
    std::vector<int> res(bStep + (offset != 0));
    for (int i = 1; i <= bStep; ++i)
        res[i - 1] = i * stride;
    if (offset != 0)
        res[bStep] = offset;
    return res;
}

std::vector<int> FIDESlib::CKKS::GetLinearTransformPlaintextRotationIndices(int rowSize, int bStep, int stride,
                                                                            int offset) {
    std::vector<int> res(rowSize);
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    for (int j = 0; j < gStep; j++) {
        for (int i = 0; i < bStep; ++i) {
            if (i + j * bStep < rowSize)
                res[i + j * bStep] = -bStep * j * stride - offset;
        }
    }
    return res;
}

void FIDESlib::CKKS::LinearTransformSpecialPt(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt2,
                                              FIDESlib::CKKS::Plaintext& ptxt, int rowSize, int bStep,
                                              std::vector<Plaintext*> pts1, std::vector<Plaintext*> pts2, int stride,
                                              int stride3) {
    constexpr bool PRINT = true;

    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    assert(pts1.size() >= rowSize);
    for (auto i : pts1) {
        assert(i != nullptr);
    }
    assert(pts2.size() >= rowSize);

    Context& cc = ctxt1.cc;
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    if (ctxt1.NoiseLevel == 2)
        ctxt1.rescale();
    if (ctxt2.NoiseLevel == 2)
        ctxt2.rescale();

    std::vector<Ciphertext> fastRotation1;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation1.emplace_back(cc);
    std::vector<Ciphertext> fastRotation2;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation2.emplace_back(cc);

    std::vector<Ciphertext*> fastRotationPtr1;
    std::vector<Ciphertext*> fastRotationPtr2;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr1.push_back(&fastRotation1[i - 1]);
        fastRotationPtr2.push_back(&fastRotation2[i - 1]);
        keys.push_back(&cc.GetRotationKey(i * stride));
        indexes.push_back(i * stride);
    }

    ctxt1.rotate_hoisted(keys, indexes, fastRotationPtr1, false);
    ctxt2.rotate_hoisted(keys, indexes, fastRotationPtr2, false);

    std::vector<Plaintext> fastRotation3;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation3.emplace_back(cc);

    std::vector<Plaintext*> fastRotationPtr3;
    std::vector<int> indexes3;
    //std::vector<KeySwitchingKey*> keys3;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr3.push_back(&fastRotation3[i - 1]);
        //keys3.push_back(&cc.GetRotationKey(i * stride3));
        indexes3.push_back(i * stride3);
    }

    Ciphertext result(cc);
    Ciphertext inner(cc);
    Ciphertext aux(cc);

    if ((gStep - 1) * bStep * (rowSize - 1) != 0)
        // ctxt2.rotate((gStep - 1) * bStep * (rowSize - 1), cc.GetRotationKey((gStep - 1) * bStep * (rowSize - 1)));
        ptxt.automorph((gStep - 1) * bStep * (rowSize - 1));

    for (uint32_t j = gStep - 1; j < gStep; --j) {

        Plaintext tmp1(cc);
        tmp1.copy(*pts1[bStep * j]);
        tmp1.multPt(tmp1, ptxt, true);
        inner.multPt(ctxt1, tmp1, false);

        if (bStep * j != 0) {
            Plaintext tmp2(cc);
            tmp2.copy(*pts2[bStep * j]);
            tmp2.multPt(tmp2, ptxt, true);

            inner.addMultPt(ctxt2, tmp2, false);
        }
        for (uint32_t i = 1; i < bStep; i++) {

            if (bStep * j + i < rowSize) {

                if (i == 1) {
                    int size = std::min((int)bStep - 1, (int)(rowSize - (bStep * j + i)));
                    if (size < bStep - 1) {
                        //auto keys3_ = keys3;
                        auto indexes3_ = indexes3;

                        //keys3_.resize(size);
                        indexes3_.resize(size);
                        // ctxt2.rotate_hoisted(keys3_, indexes3_, fastRotationPtr3);
                        ptxt.rotate_hoisted(indexes3_, fastRotationPtr3);
                    } else {
                        // ctxt2.rotate_hoisted(keys3, indexes3, fastRotationPtr3);
                        ptxt.rotate_hoisted(indexes3, fastRotationPtr3);
                    }
                }
                // (*pts1[bStep * j + i]).multPt(*pts1[bStep * j + i], fastRotation3[i - 1], true);
                Plaintext temp1(cc);
                temp1.copy(*pts1[bStep * j + i]);
                temp1.multPt(temp1, fastRotation3[i - 1], true);
                aux.multPt(fastRotation1[i - 1], temp1, false);

                // (*pts2[bStep * j + i]).multPt(*pts2[bStep * j + i], fastRotation3[i - 1], true);
                Plaintext temp2(cc);
                temp2.copy(*pts2[bStep * j + i]);
                temp2.multPt(temp2, fastRotation3[i - 1], true);
                aux.addMultPt(fastRotation2[i - 1], temp2);

                inner.add(aux);
            }
        }

        if (j > 0) {

            if (j == gStep - 1) {
                result.copy(inner);
            } else {
                result.add(inner);
            }
            result.rotate(stride * bStep, cc.GetRotationKey(stride * bStep));
            // ctxt2.rotate(-bStep * (rowSize - 1), cc.GetRotationKey(-bStep * (rowSize - 1)));
            ptxt.automorph(-bStep * (rowSize - 1));
        } else {
            if (gStep == 1) {
                result.copy(inner);
            } else {
                result.add(inner);
            }
        }
    }

    ctxt1.copy(result);
    cudaDeviceSynchronize();
}

// for merged PCMM
void FIDESlib::CKKS::LinearTransformSpecialPt(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt2,
                                              FIDESlib::CKKS::Plaintext& ptxt, int rowSize, int bStep,
                                              std::vector<Plaintext*> pts1, std::vector<Plaintext*> pts2, int stride,
                                              int stride3, FIDESlib::CKKS::Ciphertext& cProduct) {
    constexpr bool PRINT = true;

    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    assert(pts1.size() >= rowSize);
    for (auto i : pts1) {
        assert(i != nullptr);
    }
    assert(pts2.size() >= rowSize);

    Context& cc = ctxt1.cc;
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    if (ctxt1.NoiseLevel == 2)
        ctxt1.rescale();
    if (ctxt2.NoiseLevel == 2)
        ctxt2.rescale();

    std::vector<Ciphertext> fastRotation1;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation1.emplace_back(cc);
    std::vector<Ciphertext> fastRotation2;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation2.emplace_back(cc);

    std::vector<Ciphertext*> fastRotationPtr1;
    std::vector<Ciphertext*> fastRotationPtr2;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr1.push_back(&fastRotation1[i - 1]);
        fastRotationPtr2.push_back(&fastRotation2[i - 1]);
        keys.push_back(&cc.GetRotationKey(i * stride));
        indexes.push_back(i * stride);
    }

    ctxt1.rotate_hoisted(keys, indexes, fastRotationPtr1, false);
    ctxt2.rotate_hoisted(keys, indexes, fastRotationPtr2, false);

    std::vector<Plaintext> fastRotation3;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation3.emplace_back(cc);

    std::vector<Plaintext*> fastRotationPtr3;
    std::vector<int> indexes3;
    //std::vector<KeySwitchingKey*> keys3;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr3.push_back(&fastRotation3[i - 1]);
        //keys3.push_back(&cc.GetRotationKey(i * stride3));
        indexes3.push_back(i * stride3);
    }

    Ciphertext result(cc);
    Ciphertext inner(cc);
    Ciphertext aux(cc);

    if ((gStep - 1) * bStep * (rowSize - 1) != 0)
        // ctxt2.rotate((gStep - 1) * bStep * (rowSize - 1), cc.GetRotationKey((gStep - 1) * bStep * (rowSize - 1)));
        ptxt.automorph((gStep - 1) * bStep * (rowSize - 1));

    for (uint32_t j = gStep - 1; j < gStep; --j) {

        Plaintext tmp1(cc);
        tmp1.copy(*pts1[bStep * j]);
        tmp1.multPt(tmp1, ptxt, true);
        inner.multPt(ctxt1, tmp1, false);

        if (bStep * j != 0) {
            Plaintext tmp2(cc);
            tmp2.copy(*pts2[bStep * j]);
            tmp2.multPt(tmp2, ptxt, true);

            inner.addMultPt(ctxt2, tmp2, false);
        }
        for (uint32_t i = 1; i < bStep; i++) {

            if (bStep * j + i < rowSize) {

                if (i == 1) {
                    int size = std::min((int)bStep - 1, (int)(rowSize - (bStep * j + i)));
                    if (size < bStep - 1) {
                        //auto keys3_ = keys3;
                        auto indexes3_ = indexes3;

                        //keys3_.resize(size);
                        indexes3_.resize(size);
                        // ctxt2.rotate_hoisted(keys3_, indexes3_, fastRotationPtr3);
                        ptxt.rotate_hoisted(indexes3_, fastRotationPtr3);
                    } else {
                        // ctxt2.rotate_hoisted(keys3, indexes3, fastRotationPtr3);
                        ptxt.rotate_hoisted(indexes3, fastRotationPtr3);
                    }
                }
                // (*pts1[bStep * j + i]).multPt(*pts1[bStep * j + i], fastRotation3[i - 1], true);
                Plaintext temp1(cc);
                temp1.copy(*pts1[bStep * j + i]);
                temp1.multPt(temp1, fastRotation3[i - 1], true);
                aux.multPt(fastRotation1[i - 1], temp1, false);

                // (*pts2[bStep * j + i]).multPt(*pts2[bStep * j + i], fastRotation3[i - 1], true);
                Plaintext temp2(cc);
                temp2.copy(*pts2[bStep * j + i]);
                temp2.multPt(temp2, fastRotation3[i - 1], true);
                aux.addMultPt(fastRotation2[i - 1], temp2);

                inner.add(aux);
            }
        }

        if (j > 0) {

            if (j == gStep - 1) {
                result.copy(inner);
            } else {
                result.add(inner);
            }
            result.rotate(stride * bStep, cc.GetRotationKey(stride * bStep));
            // ctxt2.rotate(-bStep * (rowSize - 1), cc.GetRotationKey(-bStep * (rowSize - 1)));
            ptxt.automorph(-bStep * (rowSize - 1));
        } else {
            if (gStep == 1) {
                result.copy(inner);
            } else {
                result.add(inner);
            }
        }
    }

    cProduct.copy(result);
    cudaDeviceSynchronize();
}

void FIDESlib::CKKS::LinearTransformSpecialPt_2(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt2,
                                                FIDESlib::CKKS::Plaintext& ptxt, int rowSize, int bStep,
                                                std::vector<Plaintext*> pts1, std::vector<Plaintext*> pts2, int stride,
                                                int stride3) {
    constexpr bool PRINT = true;

    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    assert(pts1.size() >= rowSize);
    for (auto i : pts1) {
        assert(i != nullptr);
    }
    assert(pts2.size() >= rowSize);

    Context& cc = ctxt1.cc;
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    if (ctxt1.NoiseLevel == 2)
        ctxt1.rescale();
    if (ctxt2.NoiseLevel == 2)
        ctxt2.rescale();

    std::vector<Ciphertext> fastRotation1;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation1.emplace_back(cc);
    std::vector<Ciphertext> fastRotation2;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation2.emplace_back(cc);

    std::vector<Ciphertext*> fastRotationPtr1;
    std::vector<Ciphertext*> fastRotationPtr2;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 1; i < bStep; ++i) {
        //std::cout << " Hoisteds: " << i * stride << std::endl;
        fastRotationPtr1.push_back(&fastRotation1[i - 1]);
        fastRotationPtr2.push_back(&fastRotation2[i - 1]);
        keys.push_back(&cc.GetRotationKey(i * stride));
        indexes.push_back(i * stride);
    }

    ctxt1.rotate_hoisted(keys, indexes, fastRotationPtr1, false);
    ctxt2.rotate_hoisted(keys, indexes, fastRotationPtr2, false);

    std::vector<Plaintext> fastRotation3;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation3.emplace_back(cc);

    std::vector<Plaintext*> fastRotationPtr3;
    std::vector<int> indexes3;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr3.push_back(&fastRotation3[i - 1]);
        indexes3.push_back(i * stride3);
    }

    Ciphertext result(cc);
    Ciphertext inner(cc);
    Ciphertext aux(cc);

    if ((gStep - 1) * bStep * (rowSize - 1) != 0)
        ptxt.automorph((gStep - 1) * bStep * (rowSize - 1));

    std::vector<Plaintext> tmp2;
    std::vector<Plaintext> tmp;
    for (int i = 0; i < bStep; ++i) {
        tmp.emplace_back(cc);
        tmp2.emplace_back(cc);
    }

    std::vector<Plaintext*> ptxts;
    for (int i = 0; i < bStep; ++i) {
        ptxts.emplace_back(&tmp[i]);
    }
    for (int i = 0; i < bStep; ++i) {
        ptxts.emplace_back(&tmp2[i]);
    }

    for (uint32_t j = gStep - 1; j < gStep; --j) {

        // mask 1

        tmp2[0].copy(*pts1[bStep * j]);
        tmp2[0].multPt(ptxt, true);
        //BroadcastPtMM(tmp2, cc, rowSize);  // broadcast!!!

        //inner.multPt(ctxt1, tmp2, false);

        if (bStep * j != 0) {

            tmp[0].copy(*pts2[bStep * j]);

            tmp[0].multPt(ptxt, true);
            //BroadcastPtMM(tmp, cc, rowSize);  // broadcast!!!

            //inner.addMultPt(ctxt2, tmp, false);
        }

        for (uint32_t i = 1; i < bStep; i++) {

            if (bStep * j + i < rowSize) {

                if (i == 1) {
                    int size = std::min((int)bStep - 1, (int)(rowSize - (bStep * j + i)));
                    if (size < bStep - 1) {
                        auto indexes3_ = indexes3;
                        indexes3_.resize(size);
                        ptxt.rotate_hoisted(indexes3_, fastRotationPtr3);
                    } else {
                        ptxt.rotate_hoisted(indexes3, fastRotationPtr3);
                    }
                }

                // mask 1
                //Plaintext tmp(cc);

                tmp[i].copy(*pts1[bStep * j + i]);
                tmp[i].multPt(fastRotation3[i - 1],
                              true);  // TODO: Factor rotation out as broadcast applies full packing anyway
                //BroadcastPtMM(tmp, cc, rowSize);  // broadcast!!!

                //inner.addMultPt(fastRotation1[i - 1], tmp, false);

                //Plaintext tmp2(cc);

                tmp2[i].copy(*pts2[bStep * j + i]);
                tmp2[i].multPt(fastRotation3[i - 1], true);
                //BroadcastPtMM(tmp2, cc, rowSize);  // broadcast!!!    // Carlos suggestion:    maybe try reordering to reduce the broadcast call to 1

                //inner.addMultPt(fastRotation2[i - 1], tmp2);
            }
        }

        /// Broadcast all
        BroadcastPtBatch(std::min((int)2 * bStep, 2 * (int)(rowSize - bStep * j)), ptxts, cc, 4, rowSize,
                         rowSize * rowSize);
        /// End Broadcast all

        //tmp2.copy(*pts1[bStep * j]);
        //tmp2.multPt(ptxt, true);
        //BroadcastPtMM(tmp2, cc, rowSize);  // broadcast!!!

        inner.multPt(ctxt1, tmp2[0], false);

        if (bStep * j != 0) {

            //tmp.copy(*pts2[bStep * j]);

            //tmp.multPt(ptxt, true);
            //BroadcastPtMM(tmp, cc, rowSize);  // broadcast!!!

            inner.addMultPt(ctxt2, tmp[0], false);
        }

        for (uint32_t i = 1; i < bStep; i++) {

            if (bStep * j + i < rowSize) {
                // mask 1
                //Plaintext tmp(cc);

                //tmp.copy(*pts1[bStep * j + i]);
                //tmp.multPt(tmp, fastRotation3[i - 1], true);
                //BroadcastPtMM(tmp, cc, rowSize);  // broadcast!!!

                inner.addMultPt(fastRotation1[i - 1], tmp[i], false);

                //Plaintext tmp2(cc);

                //tmp2.copy(*pts2[bStep * j + i]);
                //tmp2.multPt(tmp2, fastRotation3[i - 1], true);
                //BroadcastPtMM(tmp2, cc, rowSize);  // broadcast!!!    // Carlos suggestion:    maybe try reordering to reduce the broadcast call to 1

                inner.addMultPt(fastRotation2[i - 1], tmp2[i], false);
            }
        }

        if (j > 0) {

            if (j == gStep - 1) {
                result.copy(inner);
            } else {
                result.add(inner);
            }
            result.rotate(stride * bStep, cc.GetRotationKey(stride * bStep));
            ptxt.automorph(-bStep * (rowSize - 1));
        } else {
            if (gStep == 1) {
                result.copy(inner);
            } else {
                result.add(inner);
            }
        }
    }
    ctxt1.copy(result);

    //cudaDeviceSynchronize();
}
