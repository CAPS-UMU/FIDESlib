//
// Created by carlosad on 12/06/25.
//

#include "CKKS/AccumulateBroadcast.cuh"

#include "CKKS/Context.cuh"
std::vector<int> FIDESlib::CKKS::GetAccumulateRotationIndices(const int bStep, const int stride, const int size) {
    std::vector<int> indices;
    int logbStep = std::bit_width((uint32_t)bStep) - 1;
    for (int s = stride; s < stride * size; s <<= logbStep) {
        for (int idx = s; idx < s * bStep && idx < stride * size; idx += s) {
            indices.push_back(idx);
        }
    }
    return indices;
}
std::vector<int> FIDESlib::CKKS::GetbroadcastRotationIndices(const int bStep, const int initsize, const int outsize) {
    const int size = outsize / initsize;
    const int stride = initsize;
    std::vector<int> indices;

    int logbStep = std::bit_width((uint32_t)bStep) - 1;
    for (int s = stride; s < stride * size; s <<= logbStep) {
        if (stride * s * bStep >= stride * size) {
            for (int i = 1; i <= bStep; ++i) {
                int idx = -(i * s) + 1;
                if (-idx < outsize) {
                    indices.push_back(idx);
                }
            }
        } else {
            for (int idx = s; idx < s * bStep && idx < stride * size; idx += s) {
                indices.push_back(idx);
            }
        }
    }
    return indices;
}
void FIDESlib::CKKS::Accumulate(Ciphertext& ctxt, const int bStep, const int stride, const int size) {
    Context& cc = ctxt.cc;
    std::vector<Ciphertext> aux;

    for (int i = 0; i < bStep - 1; ++i) {
        aux.emplace_back(cc);
    }

    int logbStep = std::bit_width((uint32_t)bStep) - 1;
    for (int s = 1; s < size; s <<= logbStep) {
        std::vector<int> indexes;
        std::vector<KeySwitchingKey*> keys;
        std::vector<Ciphertext*> auxptr;
        for (int idx = stride * s; idx < stride * size && idx < bStep * stride * s; idx += stride * s) {
            // std::cout << idx << std::endl;
            keys.emplace_back(&cc.GetRotationKey(idx));
            indexes.push_back(idx);
            auxptr.emplace_back(&aux[idx / stride / s - 1]);
        }
        ctxt.rotate_hoisted(keys, indexes, auxptr, true);
        ctxt.extend();
        for (size_t i = 0; i < keys.size(); ++i) {
            ctxt.add(*auxptr[i]);
        }
        ctxt.modDown(false);
    }
}

void FIDESlib::CKKS::Broadcast(Ciphertext& ctxt, const int bStep, const int initsize, const int outsize) {

    const int size = outsize / initsize;
    const int stride = initsize;
    Context& cc = ctxt.cc;
    std::vector<Ciphertext> aux;

    for (int i = 0; i < bStep; ++i) {
        aux.emplace_back(cc);
    }

    int logbStep = std::bit_width((uint32_t)bStep) - 1;
    for (int s = 1; s < size; s <<= logbStep) {
        std::vector<int> indexes;
        std::vector<KeySwitchingKey*> keys;
        std::vector<Ciphertext*> auxptr;
        if (s * bStep >= size) {
            for (int i = 1; i <= bStep; ++i) {
                int idx = -(i * stride * s) + 1;
                if (-idx < outsize) {
                    //  std::cout << idx << std::endl;
                    keys.emplace_back(&cc.GetRotationKey(idx));
                    indexes.push_back(idx);
                    auxptr.emplace_back(&aux[i - 1]);
                }
            }
            ctxt.rotate_hoisted(keys, indexes, auxptr, true);
            ctxt.add(*auxptr[0], *auxptr[1]);
            for (size_t i = 2; i < keys.size(); ++i) {
                ctxt.add(*auxptr[i]);
            }
        } else {
            for (int idx = stride * s; idx < stride * size && idx < bStep * stride * s; idx += stride * s) {
                //  std::cout << idx << std::endl;
                keys.emplace_back(&cc.GetRotationKey(idx));
                indexes.push_back(idx);
                auxptr.emplace_back(&aux[idx / stride / s - 1]);
            }
            ctxt.rotate_hoisted(keys, indexes, auxptr, true);
            ctxt.extend();
            for (size_t i = 0; i < keys.size(); ++i) {
                ctxt.add(*auxptr[i]);
            }
        }
        ctxt.modDown(false);
    }
}