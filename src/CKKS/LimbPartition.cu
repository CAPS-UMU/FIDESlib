//
// Created by carlosad on 27/04/24.
//
#include <algorithm>
#include <array>
#include <variant>
#include <vector>

#include "CKKS/Context.cuh"
#include "CKKS/Conv.cuh"
#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/LimbPartition.cuh"
#include "LimbUtils.cuh"
#include "NTT.cuh"
#include "NTTdynamicParallelism.cuh"
#include "Rotation.cuh"
#include "VectorGPU.cuh"

namespace FIDESlib::CKKS {

LimbPartition::LimbPartition(LimbPartition&& l) noexcept
    : cc(l.cc),
      id(l.id),
      device((cudaSetDevice(l.device), l.device)),
      rank(l.rank),
      s(std::move(l.s)),
      meta(l.meta),
      SPECIALmeta(l.SPECIALmeta),
      digitid(l.digitid),
      DECOMPmeta(l.DECOMPmeta),
      DIGITmeta(l.DECOMPmeta),
      limb(std::move(l.limb)),
      SPECIALlimb(std::move(l.SPECIALlimb)),
      DECOMPlimb(std::move(l.DECOMPlimb)),
      DIGITlimb(std::move(l.DECOMPlimb)),
      bufferAUXptrs(l.bufferAUXptrs),
      limbptr(std::move(l.limbptr)),
      auxptr(std::move(l.auxptr)),
      SPECIALlimbptr(std::move(l.SPECIALlimbptr)),
      SPECIALauxptr(std::move(l.SPECIALauxptr)),
      DECOMPlimbptr(std::move(l.DECOMPlimbptr)),
      DECOMPauxptr(std::move(l.DECOMPlimbptr)),
      DIGITlimbptr(std::move(l.DIGITlimbptr)),
      DIGITauxptr(std::move(l.DIGITlimbptr)),
      bufferDECOMPandDIGIT(l.bufferDECOMPandDIGIT),
      bufferSPECIAL(l.bufferSPECIAL),
      bufferLIMB(l.bufferLIMB) {
    l.bufferSPECIAL = nullptr;
    l.bufferLIMB = nullptr;
    l.bufferDECOMPandDIGIT = nullptr;
    l.bufferAUXptrs = nullptr;
}

std::vector<VectorGPU<void*>> LimbPartition::generateDecompLimbptr(
    void** buffer, const std::vector<std::vector<LimbRecord>>& DECOMPmeta, const int device, int offset) {
    std::vector<VectorGPU<void*>> result;
    for (auto& d : DECOMPmeta) {
        result.emplace_back(buffer, d.size(), device, offset);
        offset += MAXP;
    }
    return result;
}

void** CudaMallocAuxBuffer(const Stream& stream, unsigned long size);

void** CudaMallocAuxBuffer(const Stream& stream, unsigned long size) {
    void** malloc;
    CudaCheckErrorModNoSync;
    cudaMallocAsync(&malloc, MAXP * sizeof(void*) * (4 + 4 * size), stream.ptr);
    CudaCheckErrorModNoSync;
    return malloc;
}

Stream initStream() {
    Stream s;
    s.init();
    return s;
}

LimbPartition::LimbPartition(Context& cc, const int id)
    : cc(cc),
      id(id),
      device((cudaSetDevice(cc.GPUid.at(id)), cc.GPUid.at(id))),
      rank(cc.GPUrank.at(id)),
      s(initStream()),
      meta(cc.meta.at(id)),
      SPECIALmeta(cc.specialMeta),
      digitid(cc.digitGPUid.at(id)),
      DECOMPmeta(cc.decompMeta.at(id)),
      DIGITmeta(cc.digitMeta.at(id)),
      DECOMPlimb(DECOMPmeta.size()),
      DIGITlimb(DIGITmeta.size()),
      bufferAUXptrs(CudaMallocAuxBuffer(s, DECOMPmeta.size())),
      /*
            limbptr(s, meta.size(), device),
            auxptr(s, meta.size(), device),
            SPECIALlimbptr(s, SPECIALmeta.size(), device),
            SPECIALauxptr(s, SPECIALmeta.size(), device),
            */

      limbptr(bufferAUXptrs, meta.size(), device, 0),
      auxptr(bufferAUXptrs, meta.size(), device, MAXP),
      SPECIALlimbptr(bufferAUXptrs, SPECIALmeta.size(), device, 2 * MAXP),
      SPECIALauxptr(bufferAUXptrs, SPECIALmeta.size(), device, 3 * MAXP),

      DECOMPlimbptr(generateDecompLimbptr(bufferAUXptrs, DECOMPmeta, device, 4 * MAXP)),
      DECOMPauxptr(generateDecompLimbptr(bufferAUXptrs, DECOMPmeta, device, (4 + DECOMPmeta.size()) * MAXP)),
      DIGITlimbptr(generateDecompLimbptr(bufferAUXptrs, DIGITmeta, device, (4 + 2 * DECOMPmeta.size()) * MAXP)),
      DIGITauxptr(generateDecompLimbptr(bufferAUXptrs, DIGITmeta, device, (4 + 3 * DECOMPmeta.size()) * MAXP)) {}

LimbPartition::~LimbPartition() {

    cudaSetDevice(device);
    /*
        for (auto &i: limb) s.wait(STREAM(i));
        for (auto &i: SPECIALlimb) s.wait(STREAM(i));
        for (auto &i: DECOMPlimb) for (auto &j: i) s.wait(STREAM(j));
        for (auto &i: DIGITlimb) for (auto &j: i) s.wait(STREAM(j));
*/

    limbptr.free(s);
    auxptr.free(s);
    SPECIALlimbptr.free(s);
    SPECIALauxptr.free(s);
    for (auto& d : DECOMPlimbptr)
        d.free(s);
    for (auto& d : DECOMPauxptr)
        d.free(s);
    for (auto& d : DIGITlimbptr)
        d.free(s);
    for (auto& d : DIGITauxptr)
        d.free(s);

    if (bufferDECOMPandDIGIT)
        cudaFreeAsync(bufferDECOMPandDIGIT, s.ptr);
    if (bufferSPECIAL)
        cudaFreeAsync(bufferSPECIAL, s.ptr);
    if (bufferLIMB)
        cudaFreeAsync(bufferLIMB, s.ptr);
    if (bufferAUXptrs)
        cudaFreeAsync(bufferAUXptrs, s.ptr);
}

void LimbPartition::generate(std::vector<LimbRecord>& records, std::vector<LimbImpl>& limbs, VectorGPU<void*>& ptrs,
                             int pos, VectorGPU<void*>* auxptrs, uint64_t* buffer, size_t offset, uint64_t* buffer_aux,
                             size_t offset_aux) {
    assert(pos < (int)records.size());

    const int limbs_size = limbs.size();
    int size = std::max((int)(pos - limbs_size + 1), (int)0);
    std::vector<void*> cpu_ptr(size, nullptr);
    std::vector<void*> cpu_auxptr(size, nullptr);
    CudaCheckErrorModNoSync;
    for (int i = limbs_size; i <= pos; ++i) {
        const LimbRecord& r = records.at(i);
        if (r.type == U32) {
            if (buffer && buffer_aux) {
                limbs.emplace_back(Limb<uint32_t>(cc, (uint32_t*)buffer, 2 * offset, device, records.at(i).stream, r.id,
                                                  (uint32_t*)buffer_aux, 2 * offset_aux));
                offset += cc.N;
                offset_aux += cc.N;
            } else if (buffer) {
                limbs.emplace_back(
                    Limb<uint32_t>(cc, (uint32_t*)buffer, 2 * offset, device, records.at(i).stream, r.id, nullptr, 0));
                offset += cc.N;
            } else
                limbs.emplace_back(Limb<uint32_t>(cc, device, records.at(i).stream, r.id));
            cpu_ptr[i - limbs_size] = {&(std::get<U32>(limbs.back()).v.data)[0]};
            cpu_auxptr[i - limbs_size] = {&(std::get<U32>(limbs.back()).aux.data)[0]};
        }
        if (r.type == U64) {
            if (buffer && buffer_aux) {
                limbs.emplace_back(
                    Limb<uint64_t>(cc, buffer, offset, device, records.at(i).stream, r.id, buffer_aux, offset_aux));
                offset += cc.N;
                offset_aux += cc.N;
            } else if (buffer) {
                limbs.emplace_back(Limb<uint64_t>(cc, buffer, offset, device, records.at(i).stream, r.id, nullptr, 0));
                offset += cc.N;
            } else
                limbs.emplace_back(Limb<uint64_t>(cc, device, records.at(i).stream, r.id));
            cpu_ptr[i - limbs_size] = {&(std::get<U64>(limbs.back()).v.data)[0]};
            cpu_auxptr[i - limbs_size] = {&(std::get<U64>(limbs.back()).aux.data)[0]};
            //cudaFreeHost(aux);
        }
    }
    cudaMemcpyAsync(ptrs.data + limbs_size, cpu_ptr.data(), size * sizeof(void*), cudaMemcpyHostToDevice, s.ptr);
    if (auxptrs) {
        cudaMemcpyAsync((*auxptrs).data + limbs_size, cpu_auxptr.data(), size * sizeof(void*), cudaMemcpyHostToDevice,
                        s.ptr);
    }
    CudaCheckErrorModNoSync;
}

void LimbPartition::generateLimb() {
    generate(meta, limb, limbptr, (int)limb.size(), &auxptr);
}

void LimbPartition::generateAllDecompLimb(uint64_t* pInt, size_t offset) {
    DECOMPlimb.resize(DECOMPmeta.size());
    for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
        generate(DECOMPmeta[i], DECOMPlimb[i], DECOMPlimbptr[i], (int)DECOMPmeta[i].size() - 1, &DECOMPauxptr[i], pInt,
                 offset, nullptr, 0);
        offset += cc.N * DECOMPmeta.at(i).size();
    }
}

void LimbPartition::generateAllDigitLimb(uint64_t* pInt, size_t offset) {
    DIGITlimb.resize(DIGITmeta.size());
    for (size_t i = 0; i < DIGITmeta.size(); ++i) {
        generate(DIGITmeta[i], DIGITlimb[i], DIGITlimbptr[i], (int)DIGITmeta[i].size() - 1, &DIGITauxptr[i], pInt,
                 offset, nullptr, 0);
        offset += cc.N * DIGITmeta.at(i).size();
    }
}

void LimbPartition::generateSpecialLimb() {
    cudaSetDevice(device);
    if (bufferSPECIAL == nullptr) {
        cudaMallocAsync(&bufferSPECIAL, cc.N * SPECIALmeta.size() * 2 * sizeof(uint64_t), s.ptr);
        generate(SPECIALmeta, SPECIALlimb, SPECIALlimbptr, (int)SPECIALmeta.size() - 1, &SPECIALauxptr, bufferSPECIAL,
                 0, bufferSPECIAL, cc.N * SPECIALmeta.size());

        // for (auto& l : SPECIALlimb)
        //     STREAM(l).wait(s);
    }
}

template <ALGO algo, NTT_MODE mode>
void ApplyNTT(int batch, LimbPartition::NTT_fusion_fields fields, std::vector<LimbImpl>& limb,
              VectorGPU<void*>& limbptr, VectorGPU<void*>& auxptr, Context& cc, const int primeid_init,
              const int limbsize = -1) {
    constexpr int M = 4;

    const dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
    const dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
    const int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
    const int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
    const int size = (limbsize != -1 ? limbsize : limb.size()) - (mode == NTT_RESCALE || mode == NTT_MULTPT);

    for (int i = 0; i < size; i += batch) {
        uint32_t num_limbs = std::min((uint32_t)batch, (uint32_t)(size - i));

        NTT_<false, algo, mode>
            <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst, STREAM(limb.at(i)).ptr>>>(
                (mode == NTT_RESCALE || mode == NTT_MULTPT) ? limbptr.data + size
                : (mode == NTT_MODDOWN)                     ? fields.op2->limbptr.data + i
                                                            : limbptr.data + i,
                primeid_init + i, auxptr.data + i, nullptr,
                (mode == NTT_RESCALE || mode == NTT_MULTPT) ? PRIMEID(limb[size]) : 0, nullptr, nullptr);

        NTT_<true, algo, mode><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                                 STREAM(limb.at(i)).ptr>>>(
            auxptr.data + i, primeid_init + i, limbptr.data + i,
            mode == NTT_MULTPT ? fields.pt->limbptr.data + i : nullptr,
            (mode == NTT_RESCALE || mode == NTT_MULTPT) ? PRIMEID(limb[size]) : 0, nullptr, nullptr);
    }
}

template <ALGO algo, NTT_MODE mode>
void LimbPartition::NTT(int batch, NTT_fusion_fields fields, const int limbsize) {
    cudaSetDevice(device);
    /*
    if (batch == 1) {
        for (auto& l : limb) {
            SWITCH(l, NTT<algo>());
        }
    } else
    */
    if (batch == -1) {
        if (!limb.empty()) {
            int stride = 32;
            for (size_t i = 0; i < limb.size(); i += stride) {

                device_launch_batch_NTT<algo, NTT_MODE::NTT_NONE>
                    <<<std::min(limb.size() - i, (size_t)stride), 1, 48000, s.ptr>>>(cc.logN, i, limbptr.data + i,
                                                                                     auxptr.data + i, limbptr.data + i);
            }
        }
    } else if (batch >= 1) {
        ApplyNTT<algo, mode>(batch, fields, limb, limbptr, auxptr, cc, PARTITION(id, 0), limbsize);
    } else if (batch == -2) {
        // Manual graph

        cudaGraph_t g;
        cudaGraphCreate(&g, 0);
        CudaCheckErrorModNoSync;

        std::vector<cudaGraphNode_t> first(limb.size());
        std::vector<cudaGraphNode_t> second(limb.size());
        std::vector<cudaGraphNode_t> nodes(limb.size());

        for (size_t i = 0; i < limb.size(); ++i) {

            cudaGraph_t sub_g;
            cudaGraphCreate(&sub_g, 0);
            if (limb[i].index() == U64) {
                using T = uint64_t;
                constexpr int M = sizeof(T) == 8 ? 4 : 8;
                const int bytes_per_thread = sizeof(T) * (2 * M + 1);

                Limb<T>& l = std::get<U64>(limb[i]);

                void* func = get_NTT_reference(false);
                int primeid = l.primeid;
                const uint32_t BDx = (1 << ((cc.logN + 1 + (cc.logN > 13 ? 2 : 0)) / 2 - 1));
                void* params[5] = {&l.v.data, &primeid, &l.aux.data, &primeid, &primeid};
                cudaKernelNodeParams nodeParams{.func = func,  //(void *) NTT_<T, false, algo, NTT_NONE>,
                                                .gridDim = cc.N / BDx / 2 / M,
                                                .blockDim = BDx,
                                                .sharedMemBytes = BDx * bytes_per_thread,
                                                .kernelParams = params,
                                                .extra = NULL};

                cudaGraphAddKernelNode(&first[i], sub_g, nullptr, 0, &nodeParams);
            }

            if (limb[i].index() == U64) {
                using T = uint64_t;
                constexpr int M = sizeof(T) == 8 ? 4 : 8;
                const int bytes_per_thread = sizeof(T) * (2 * M + 1);

                Limb<T>& l = std::get<U64>(limb[i]);

                const uint32_t BDx = (1 << ((cc.logN - (cc.logN > 13 ? 2 : 0)) / 2 - 1));
                void* params[3] = {(void*)&l.aux.data, (void*)&l.primeid, (void*)&l.v.data};
                cudaKernelNodeParams nodeParams{.func = get_NTT_reference(true),  //(void *) NTT_<T, true, algo>,
                                                .gridDim = cc.N / BDx / 2 / M,
                                                .blockDim = BDx,
                                                .sharedMemBytes = BDx * bytes_per_thread,
                                                .kernelParams = params};
                //cudaGraphKernelNodeSetAttribute(second[i], cudaKer)

                cudaGraphAddKernelNode(&second[i], sub_g, &first[i], 1, &nodeParams);
            }

            cudaGraphAddChildGraphNode(&nodes[i], g, nullptr, 0, sub_g);
            cudaGraphDestroy(sub_g);
        }
        CudaCheckErrorModNoSync;

        static std::map<int, cudaGraphExec_t> execs;
        cudaGraphExec_t& exec = execs[limb.size()];

        if (!exec) {
            cudaGraphInstantiateWithFlags(&exec, g, 0);
            //cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
            CudaCheckErrorModNoSync;
        } else {
            if (cudaGraphExecUpdate(exec, g, NULL, NULL) != cudaSuccess) {
                CudaCheckErrorModNoSync;
                // only instantiate a new graph if update fails
                cudaGraphExecDestroy(exec);
                cudaGraphInstantiateWithFlags(&exec, g, 0);
                //cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
                CudaCheckErrorModNoSync;
            }
        }
        cudaGraphDestroy(g);
        cudaGraphLaunch(exec, s.ptr);
        CudaCheckErrorModNoSync;
    } else {
        assert("Invalid NTT batch configuration!");
    }
}

#define YYY(algo, mode) \
    template void LimbPartition::NTT<algo, mode>(int batch, NTT_fusion_fields fields, const int limbsize);

#include "ntt_types.inc"
#undef YYY

template <ALGO algo, INTT_MODE mode>
void ApplyINTT(int batch, LimbPartition::INTT_fusion_fields fields, std::vector<LimbImpl>& limb,
               VectorGPU<void*>& limbptr, VectorGPU<void*>& auxptr, Context& cc, const int primeid_init) {
    constexpr int M = 4;

    dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN - (cc.logN > 13 ? 0 : 0)) / 2 - 1))};
    dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1 + (cc.logN > 13 ? 0 : 0)) / 2 - 1))};
    int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
    int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

    for (size_t i = 0; i < limb.size(); i += batch) {
        uint32_t num_limbs = std::min((uint32_t)batch, (uint32_t)(limb.size() - i));

        INTT_<false, algo, INTT_NONE>
            <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst, STREAM(limb.at(i)).ptr>>>(
                limbptr.data + i, primeid_init + i, auxptr.data + i);

        INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                                       STREAM(limb.at(i)).ptr>>>(auxptr.data + i, primeid_init + i, limbptr.data + i);
    }
}

template <ALGO algo, INTT_MODE mode>
void LimbPartition::INTT(int batch, INTT_fusion_fields fields) {
    cudaSetDevice(device);

    /*
    if (batch == 1) {
        for (auto& l : limb) {
            SWITCH(l, INTT<algo>());
        }
    } else */
    if (batch == -1) {
        if (!limb.empty()) {
            constexpr int stride = 32;
            for (size_t i = 0; i < limb.size(); i += stride) {
                device_launch_batch_INTT<algo><<<std::min(limb.size() - i, (size_t)stride), 1, 48000, s.ptr>>>(
                    cc.logN, i, limbptr.data + i, auxptr.data + i, limbptr.data + i);
            }
        }
    } else if (batch >= 1) {
        ApplyINTT<algo, mode>(batch, fields, limb, limbptr, auxptr, cc, PARTITION(id, 0));
    } else {
        assert("Invalid INTT batch configuration!");
    }
}

#define WWW(algo, mode) template void LimbPartition::INTT<algo, mode>(int batch, INTT_fusion_fields fields);

#include "ntt_types.inc"
#undef WWW

/*
    void LimbPartition::fullFree() {
        using metaLimb = std::pair<const std::vector<LimbRecord> &, std::vector<LimbImpl> &>;
        std::vector<metaLimb> v{
                metaLimb{meta, limb}, metaLimb{SPECIALmeta, SPECIALlimb}
        };

        for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
            v.emplace_back(DECOMPmeta.at(i), DECOMPlimb.at(i));
        }
    }
*/
void LimbPartition::add(const LimbPartition& p) {
    cudaSetDevice(device);
    /*
    for (size_t i = 0; i < limb.size(); ++i) {
        SWITCH(limb.at(i), add(p.limb.at(i)));
    }
    */
    s.wait(p.s);
    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        add_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    p.s.wait(s);
}

void LimbPartition::sub(const LimbPartition& p) {
    cudaSetDevice(device);
    /*
    for (size_t i = 0; i < limb.size(); ++i) {
        SWITCH(limb.at(i), sub(p.limb.at(i)));
    }
    */
    s.wait(p.s);
    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        sub_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    p.s.wait(s);
}

void LimbPartition::multElement(const LimbPartition& p) {
    //assert(SPECIALlimb.size() == 0 && p.SPECIALlimb.size() == 0);
    assert(limb.size() <= p.limb.size());
    /*
    s.wait(p.s);
    for (size_t i = 0; i < limb.size(); ++i) {
        SWITCH(limb.at(i), mult(p.limb.at(i)));
    }
*/
    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        Mult_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            (void**)limbptr.data + i, (void**)limbptr.data + i, (void**)p.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    p.s.wait(s);
}

void LimbPartition::multElement(const LimbPartition& partition1, const LimbPartition& partition2) {
    cudaSetDevice(device);
    assert(SPECIALlimb.size() == 0 && partition1.SPECIALlimb.size() == 0);
    assert(SPECIALlimb.size() == 0 && partition2.SPECIALlimb.size() == 0);
    assert(limb.size() >= partition1.limb.size());
    assert(limb.size() >= partition2.limb.size());
    assert(partition1.limb.size() <= partition2.limb.size());
    s.wait(partition1.s);
    s.wait(partition2.s);
    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        Mult_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            (void**)limbptr.data + i, (void**)partition1.limbptr.data + i, (void**)partition2.limbptr.data + i,
            PARTITION(id, i));
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    partition1.s.wait(s);
    partition2.s.wait(s);
}

void LimbPartition::rescale() {
    //assert(SPECIALlimb.size() == 0);
    assert(limb.size() > 1);
    cudaSetDevice(device);

    LimbImpl& top = limb.at(limb.size() - 1);
    STREAM(top).wait(s);
    SWITCH(top, INTT<ALGO_SHOUP>());

    {
        for (size_t i = 0; i < limb.size() - 1; i += cc.batch) {
            STREAM(limb[i]).wait(STREAM(top));
        }
        NTT<ALGO_SHOUP, NTT_RESCALE>(cc.batch);
        for (size_t i = 0; i < limb.size() - 1; i += cc.batch) {
            STREAM(top).wait(STREAM(limb[i]));
        }
    }
    s.wait(STREAM(top));
    limb.pop_back();
}

void LimbPartition::multPt(const LimbPartition& p) {
    // assert(SPECIALlimb.size() == 0 && p.SPECIALlimb.size() == 0);
    assert(limb.size() <= p.limb.size());
    assert(limb.size() > 1);
    cudaSetDevice(device);

    constexpr bool capture = false;
    static std::map<int, cudaGraphExec_t> exec_map;

    {
        LimbImpl& top = limb.back();

        cudaGraphExec_t& exec = exec_map[limb.size()];

        run_in_graph<capture>(exec, s, [&]() {
            STREAM(top).wait(s);
            SWITCH(top, mult(p.limb.back()));
            SWITCH(top, INTT<ALGO_SHOUP>());

            for (size_t i = 0; i < limb.size() - 1; i += cc.batch) {
                STREAM(limb.at(i)).wait(STREAM(top));
            }
            NTT<ALGO_SHOUP, NTT_MULTPT>(cc.batch, NTT_fusion_fields{.pt = &p});
            for (size_t i = 0; i < limb.size() - 1; i += cc.batch) {
                STREAM(top).wait(STREAM(limb.at(i)));
            }

            s.wait(STREAM(top));
        });

        limb.pop_back();
    }
}

void LimbPartition::modup(int level, LimbPartition& aux_partition) {
    if (CPU) {

    } else {
        constexpr ALGO algo = ALGO_SHOUP;
        constexpr bool PRINT = false;
        assert(SPECIALlimb.empty());
        cudaSetDevice(device);

        const int limbsize = level;
        generateAllDecompAndDigit();
        s.wait(aux_partition.s);

        if constexpr (PRINT)
            for (auto& i : limb) {
                SWITCH(i, printThisLimb(2));
            }

        for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

            int start = 0;
            for (int j = 0; j < d; ++j)
                start += DECOMPlimb[j].size();
            int size = std::min((int)DECOMPlimb[d].size(), limbsize - start);
            if (size <= 0)
                break;
            /*
        for (auto& l : DECOMPlimb[d]) {
            for (auto& p : limb) {
                if (PRIMEID(l) == PRIMEID(p)) {
                    STREAM(l).wait(STREAM(p));
                    SWITCH(l, INTT_from(p));
                    s.wait(STREAM(l));
                }
            }
        }
        */

            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            for (int i = 0; i < size; i += cc.batch) {
                STREAM(limb.at(start + i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                INTT_<false, algo, INTT_NONE>
                    <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst,
                       STREAM(limb.at(start + i)).ptr>>>(limbptr.data + start + i, start + i, auxptr.data + start + i);

                INTT_<true, algo, INTT_NONE>
                    <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                       STREAM(limb.at(start + i)).ptr>>>(auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
            }
            for (size_t i = 0; i < size; i += cc.batch) {
                s.wait(STREAM(limb.at(start + i)));
            }

            if constexpr (PRINT) {
                std::cout << hC_.primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
                SWITCH(DECOMPlimb[d][0], printThisLimb());
            }

            {
                dim3 blockSize{64, 2};
                dim3 gridSize{(uint32_t)cc.N / blockSize.x};
                int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x;
                DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr>>>(
                    DECOMPlimbptr[d].data, level, DIGITlimbptr[d].data, digitid[d]);
            }
            if constexpr (PRINT)
                for (auto& i : DIGITlimb[d]) {
                    SWITCH(i, printThisLimb(2));
                }

            const int digitsize = hC_.num_primeid_digit_to[digitid.at(d)][level - 1];
            for (size_t i = 0; i < digitsize; i += cc.batch) {
                STREAM(DIGITlimb.at(d).at(i)).wait(s);
            }
            ApplyNTT<algo, NTT_NONE>(cc.batch, NTT_fusion_fields{}, DIGITlimb.at(d), DIGITlimbptr.at(d),
                                     aux_partition.DIGITlimbptr.at(d), cc, DIGIT(digitid.at(d), 0), digitsize);

            for (size_t i = 0; i < digitsize; i += cc.batch) {
                s.wait(STREAM(DIGITlimb.at(d).at(i)));
            }
            if constexpr (PRINT)
                for (auto& i : DIGITlimb[d]) {
                    SWITCH(i, printThisLimb(2));
                }
        }

        aux_partition.s.wait(s);
    }
}

void LimbPartition::freeSpecialLimbs() {
    cudaSetDevice(device);
    for (size_t i = 0; i < SPECIALlimb.size(); ++i) {
        STREAM(SPECIALlimb.at(i)).wait(s);
    }
    SPECIALlimb.clear();
    if (bufferSPECIAL != nullptr) {
        cudaFreeAsync(bufferSPECIAL, s.ptr);
        bufferSPECIAL = nullptr;
    }
}

void LimbPartition::copyLimb(const LimbPartition& partition) {
    assert(limb.size() <= partition.limb.size());
    cudaSetDevice(device);
    s.wait(partition.s);

    copy_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)limb.size()}, 128, 0, s.ptr>>>(partition.limbptr.data, limbptr.data);
    /*
    for (size_t i = 0; i < partition.limb.size(); ++i) {
        STREAM(limb.at(i)).wait(s);
        SWITCH(limb.at(i), copyV(partition.limb.at(i)));
    }
    for (size_t i = 0; i < partition.limb.size(); ++i) {
        s.wait(STREAM(limb.at(i)));
    }
    */
    partition.s.wait(s);
}

void LimbPartition::generateAllDecompAndDigit() {
    cudaSetDevice(device);
    if (bufferDECOMPandDIGIT == nullptr) {
        int decomp_limbs = 0;
        for (auto& d : DECOMPmeta)
            decomp_limbs += d.size();
        int digit_limbs = 0;
        for (auto& d : DIGITmeta)
            digit_limbs += d.size();

        size_t size = cc.N * (decomp_limbs + digit_limbs);
        cudaMallocAsync(&bufferDECOMPandDIGIT, size * sizeof(uint64_t), s.ptr);
        generateAllDecompLimb(bufferDECOMPandDIGIT, 0);
        generateAllDigitLimb(bufferDECOMPandDIGIT, cc.N * decomp_limbs);
    }
}

void LimbPartition::mult1AddMult23Add4(const LimbPartition& partition1, const LimbPartition& partition2,
                                       const LimbPartition& partition3, const LimbPartition& partition4) {
    cudaSetDevice(device);
    assert(limb.size() <= partition1.limb.size());
    assert(limb.size() <= partition2.limb.size());
    assert(limb.size() <= partition3.limb.size());
    assert(limb.size() <= partition4.limb.size());

    s.wait(partition1.s);
    s.wait(partition2.s);
    s.wait(partition3.s);
    s.wait(partition4.s);

    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        mult1AddMult23Add4_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            PARTITION(id, i), limbptr.data + i, partition1.limbptr.data + i, partition2.limbptr.data + i,
            partition3.limbptr.data + i, partition4.limbptr.data + i);
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }

    partition1.s.wait(s);
    partition2.s.wait(s);
    partition3.s.wait(s);
    partition4.s.wait(s);
}

void LimbPartition::mult1Add2(const LimbPartition& partition1, const LimbPartition& partition2) {
    cudaSetDevice(device);
    assert(limb.size() <= partition1.limb.size());
    assert(limb.size() <= partition2.limb.size());

    s.wait(partition1.s);
    s.wait(partition2.s);

    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        mult1Add2_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            PARTITION(id, i), limbptr.data + i, partition1.limbptr.data + i, partition2.limbptr.data + i);
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }

    partition1.s.wait(s);
    partition2.s.wait(s);
}

void LimbPartition::generateLimbSingleMalloc(int num_limbs) {
    cudaSetDevice(device);
    assert(limb.size() == 0);
    assert(num_limbs <= meta.size());
    if (bufferLIMB == nullptr) {
        cudaMallocAsync(&bufferLIMB, cc.N * num_limbs * 2 * sizeof(uint64_t), s.ptr);

        generate(meta, limb, limbptr, (int)num_limbs - 1, &auxptr, bufferLIMB, 0, bufferLIMB, cc.N * num_limbs);
        /*
        for (auto& l : limb)
            STREAM(l).wait(s);
        */
    }
}

void LimbPartition::generateLimbConstant(int num_limbs) {
    cudaSetDevice(device);
    assert(limb.size() == 0);
    assert(num_limbs <= meta.size());
    if (bufferLIMB == nullptr) {
        cudaMallocAsync(&bufferLIMB, cc.N * num_limbs * sizeof(uint64_t), s.ptr);

        generate(meta, limb, limbptr, (int)num_limbs - 1, &auxptr, bufferLIMB, 0, nullptr, 0);
    }
}

void LimbPartition::loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data,
                                    const std::vector<std::vector<uint64_t>>& moduli) {
    cudaSetDevice(device);

    for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
        for (auto& j : DECOMPlimb.at(i)) {
            for (size_t k = 0; k < data.at(i).size(); ++k) {
                if (hC_.primes[PRIMEID(j)] == moduli.at(i).at(k)) {
                    STREAM(j).wait(s);
                    SWITCH(j, load(data.at(i).at(k)));
                    k = data.at(i).size();
                }
            }
        }

        for (auto& j : DIGITlimb.at(i)) {
            for (size_t k = 0; k < data.at(i).size(); ++k) {
                if (hC_.primes[PRIMEID(j)] == moduli.at(i).at(k)) {
                    STREAM(j).wait(s);
                    SWITCH(j, load(data.at(i).at(k)));
                    k = data.at(i).size();
                }
            }
        }
    }
}

void LimbPartition::dotKSK(const LimbPartition& src, const LimbPartition& ksk, const int level, const bool inplace,
                           const LimbPartition* limbsrc) {
    cudaSetDevice(device);
    constexpr bool PRINT = false;
    s.wait(src.s);
    s.wait(ksk.s);
    const int limbsize = level + 1;
    assert(limbsize <= limb.size());
    assert(limbsize <= src.limb.size());

    if constexpr (0) {
        std::map<int, int> used;
        for (size_t i = 0; i < src.DIGITlimb.size(); ++i) {

            {
                int start = 0;
                for (int j = 0; j < i; ++j)
                    start += src.DECOMPlimb[j].size();
                int size = std::min((int)src.DECOMPlimb[i].size(), (int)limb.size() - start);
                if (size <= 0)
                    break;
            }

            for (size_t j = 0; j < ksk.DECOMPlimb.at(i).size(); ++j) {

                int primeid = PRIMEID(ksk.DECOMPlimb.at(i).at(j));

                for (size_t k = 0; k < src.limb.size(); ++k) {
                    auto& l = src.limb.at(k);
                    if (PRIMEID(l) == primeid) {
                        //STREAM(limb.at(k)).wait(s);
                        //STREAM(limb.at(k)).wait(STREAM(ksk.DECOMPlimb.at(i).at(j)));
                        //STREAM(limb.at(k)).wait(STREAM(l));

                        if (!used[primeid]) {
                            SWITCH(limb.at(k), mult(l, ksk.DECOMPlimb.at(i).at(j), inplace));
                            used[primeid]++;
                            if constexpr (PRINT)
                                std::cout << "Init " << primeid;  //<< std::endl;
                        } else {
                            SWITCH(limb.at(k), addMult(l, ksk.DECOMPlimb.at(i).at(j), inplace));
                            if constexpr (PRINT)
                                std::cout << "Acc " << primeid;  // << std::endl;
                        }
                        if constexpr (PRINT)
                            SWITCH(limb.at(k), printThisLimb(1));
                        if constexpr (PRINT)
                            SWITCH(l, printThisLimb(1));
                    }
                }

                if constexpr (PRINT)
                    SWITCH(ksk.DECOMPlimb.at(i).at(j), printThisLimb(1));
            }

            CudaCheckErrorModNoSync;
            for (size_t j = 0; j < src.DIGITlimb.at(i).size(); ++j) {
                int primeid = PRIMEID(src.DIGITlimb.at(i).at(j));

                if (primeid < hC_.L) {
                    for (auto& l : limb) {
                        if (PRIMEID(l) == primeid) {
                            //STREAM(l).wait(s);
                            //STREAM(l).wait(STREAM(src.DIGITlimb.at(i).at(j)));
                            //STREAM(l).wait(STREAM(ksk.DIGITlimb.at(i).at(j)));

                            if (!used[primeid]) {
                                SWITCH(l, mult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
                                used[primeid]++;
                                if constexpr (PRINT)
                                    std::cout << "Init2 " << primeid;  // << std::endl;
                            } else

                            {
                                SWITCH(l, addMult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
                                if constexpr (PRINT)
                                    std::cout << "Acc2 " << primeid;  // << std::endl;
                            }

                            if constexpr (PRINT)
                                SWITCH(l, printThisLimb(1));
                        }
                    }
                } else {

                    for (auto& l : SPECIALlimb) {
                        if (PRIMEID(l) == primeid) {
                            //STREAM(l).wait(s);
                            //STREAM(l).wait(STREAM(src.DIGITlimb.at(i).at(j)));
                            //STREAM(l).wait(STREAM(ksk.DIGITlimb.at(i).at(j)));
                            if (!used[primeid]) {
                                SWITCH(l, mult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
                                used[primeid]++;
                                if constexpr (PRINT)
                                    std::cout << "Init3 " << primeid;  // << std::endl;
                            } else {
                                SWITCH(l, addMult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
                                if constexpr (PRINT)
                                    std::cout << "Acc3 " << primeid;  // << std::endl;
                            }
                            if constexpr (PRINT)
                                SWITCH(l, printThisLimb(1));
                        }
                    }
                }
                if constexpr (PRINT)
                    SWITCH(src.DIGITlimb.at(i).at(j), printThisLimb(1));
                if constexpr (PRINT)
                    SWITCH(ksk.DIGITlimb.at(i).at(j), printThisLimb(1));
            }

            if constexpr (PRINT) {
                for (auto& i : limb) {
                    if constexpr (PRINT)
                        SWITCH(i, printThisLimb(1));
                }
                for (auto& i : SPECIALlimb) {
                    if constexpr (PRINT)
                        SWITCH(i, printThisLimb(1));
                }
            }
        }

        for (auto& l : limb)
            s.wait(STREAM(l));
        for (auto& l : SPECIALlimb)
            s.wait(STREAM(l));
    } else {

        int start = 0;
        int special = SPECIALmeta.size();

        for (int i = 0; i < DECOMPmeta.size(); ++i) {
            int size = std::min((int)src.DECOMPlimb[i].size(), (int)limbsize - start);
            if (size <= 0) {
                //  std::cout << "Out on " << i << std::endl;
                break;
            }
            Mult_<<<{cc.N / 128, size}, 128, 0, s.ptr>>>(
                inplace ? auxptr.data + start : limbptr.data + start, ksk.DECOMPlimbptr[i].data,
                limbsrc ? limbsrc->limbptr.data + start : src.limbptr.data + start, start);
            start += DECOMPmeta[i].size();
        }

        start = 0;
        for (int i = 0; i < DIGITmeta.size(); ++i) {
            if (start >= limbsize) {
                // std::cout << "Out on " << i << std::endl;
                break;
            }
            if (start > 0) {
                int size = start;
                addMult_<<<{cc.N / 128, size}, 128, 0, s.ptr>>>(inplace ? auxptr.data : limbptr.data,
                                                                ksk.DIGITlimbptr[i].data + special,
                                                                src.DIGITlimbptr[i].data + special, 0);
            }
            start += DECOMPmeta[i].size();
            if (start < limbsize) {
                int size = limbsize - start;
                addMult_<<<{cc.N / 128, size}, 128, 0, s.ptr>>>(
                    inplace ? auxptr.data + start : limbptr.data + start,
                    ksk.DIGITlimbptr[i].data + special + start - DECOMPmeta[i].size(),
                    src.DIGITlimbptr[i].data + special + start - DECOMPmeta[i].size(), start);
            }
        }

        start = 0;
        for (int i = 0; i < DIGITmeta.size(); ++i) {
            if (start >= limbsize)
                break;
            start += DECOMPmeta.at(i).size();
            if (i == 0) {
                Mult_<<<{cc.N / 128, special}, 128, 0, s.ptr>>>(inplace ? SPECIALauxptr.data : SPECIALlimbptr.data,
                                                                ksk.DIGITlimbptr[i].data, src.DIGITlimbptr[i].data,
                                                                SPECIAL(id, 0));
            } else {
                addMult_<<<{cc.N / 128, special}, 128, 0, s.ptr>>>(inplace ? SPECIALauxptr.data : SPECIALlimbptr.data,
                                                                   ksk.DIGITlimbptr[i].data, src.DIGITlimbptr[i].data,
                                                                   SPECIAL(id, 0));
            }
        }
    }

    src.s.wait(s);
    ksk.s.wait(s);

    if (inplace) {
        for (auto& l : limb) {
            if (l.index() == U32) {
                std::swap(std::get<U32>(l).v.data, std::get<U32>(l).aux.data);
            } else {
                std::swap(std::get<U64>(l).v.data, std::get<U64>(l).aux.data);
            }
        }
        std::swap(limbptr.data, auxptr.data);

        for (auto& l : SPECIALlimb) {
            if (l.index() == U32) {
                std::swap(std::get<U32>(l).v.data, std::get<U32>(l).aux.data);
            } else {
                std::swap(std::get<U64>(l).v.data, std::get<U64>(l).aux.data);
            }
        }
        std::swap(SPECIALlimbptr.data, SPECIALauxptr.data);
    }

    if constexpr (PRINT) {
        for (auto& i : limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
        for (auto& i : SPECIALlimb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
}

void LimbPartition::multModupDotKSK(LimbPartition& c1, const LimbPartition& c1tilde, LimbPartition& c0,
                                    const LimbPartition& c0tilde, const LimbPartition& ksk_a,
                                    const LimbPartition& ksk_b, const int level) {

    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    assert(c0.SPECIALlimb.size() == SPECIALmeta.size());
    assert(c1.SPECIALlimb.size() == SPECIALmeta.size());
    cudaSetDevice(device);

    //std::map<int, int> used;
    s.wait(c0.s);
    s.wait(c1.s);
    s.wait(c0tilde.s);
    s.wait(c1tilde.s);
    s.wait(ksk_a.s);
    s.wait(ksk_b.s);

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            if (d == 0) {
                std::cout << hC_.primes[PRIMEID(limb[0])] << ": ";
                SWITCH(limb[0], printThisLimb());
            }

        if constexpr (1) {  // Batched
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            for (int i = 0; i < size; i += cc.batch) {
                STREAM(limb.at(start + i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                INTT_<false, algo, INTT_MULT_AND_SAVE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs},
                                                         blockDimFirst, bytesFirst, STREAM(limb.at(start + i)).ptr>>>(
                    c1.limbptr.data + start + i, start + i, c1.auxptr.data + start + i,
                    c1tilde.limbptr.data + start + i, c0.limbptr.data + start + i, c1.limbptr.data + start + i,
                    ksk_a.DECOMPlimbptr[d].data + i, ksk_b.DECOMPlimbptr[d].data + i, c0.limbptr.data + start + i,
                    c0tilde.limbptr.data + start + i);

                INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                               bytesSecond, STREAM(limb.at(start + i)).ptr>>>(
                    c1.auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
            }
            for (size_t i = 0; i < size; i += cc.batch) {
                s.wait(STREAM(limb.at(start + i)));
            }
        }

        if constexpr (PRINT) {
            std::cout << hC_.primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
            SWITCH(DECOMPlimb[d][0], printThisLimb());
        }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (DECOMPlimb[d].size()) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr>>>(DECOMPlimbptr[d].data, level,
                                                                                   DIGITlimbptr[d].data, digitid[d]);
        }

        if constexpr (1) {  // Batched
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            int size = c0.SPECIALlimb.size();
            for (int i = 0; i < size; i += cc.batch) {
                STREAM(c0.SPECIALlimb.at(i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                              bytesFirst, STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                    DIGITlimbptr[d].data + i, SPECIAL(id, i), c1.SPECIALauxptr.data + i, nullptr, 0, nullptr, nullptr);

                if (d == 0) {
                    NTT_<true, algo, NTT_KSK_DOT><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                                    bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                        c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                        ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i, ksk_b.DIGITlimbptr[d].data + i);
                } else {
                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                            c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                            ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i,
                            ksk_b.DIGITlimbptr[d].data + i);
                }
            }
        }
    }

    //cudaDeviceSynchronize();

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        if constexpr (1)  // batched
        {
            int start = 0;
            for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                if (j == d)
                    continue;

                int Dstart = start + c0.SPECIALlimb.size();
                int Lstart = start + (j > d ? DECOMPlimb[d].size() : 0);

                int size = std::min((int)DECOMPlimb[j].size(), (int)c0.limb.size() - Lstart);
                if (size <= 0)
                    break;

                constexpr int M = 4;

                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                for (int i = 0; i < size; i += cc.batch) {
                    STREAM(c0.limb.at(Lstart + i)).wait(s);
                    uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                    NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                                  bytesFirst, STREAM(c0.limb.at(Lstart + i)).ptr>>>(
                        DIGITlimbptr[d].data + Dstart + i, Lstart + i, c1.auxptr.data + Lstart + i, nullptr, 0, nullptr,
                        nullptr);

                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.limb.at(Lstart + i)).ptr>>>(
                            c1.auxptr.data + Lstart + i, Lstart + i, c0.limbptr.data + Lstart + i,
                            ksk_a.DIGITlimbptr[d].data + Dstart + i, 0, c1.limbptr.data + Lstart + i,
                            ksk_b.DIGITlimbptr[d].data + Dstart + i);
                }

                start += DECOMPlimb[j].size();
            }
        }

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
    }
    for (auto& l : c0.limb)
        s.wait(STREAM(l));
    for (auto& l : c0.SPECIALlimb)
        s.wait(STREAM(l));

    c0.s.wait(s);
    c1.s.wait(s);
    c0tilde.s.wait(s);
    c1tilde.s.wait(s);
    ksk_a.s.wait(s);
    ksk_b.s.wait(s);
}

void LimbPartition::rotateModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a,
                                      const LimbPartition& ksk_b, const int level) {

    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    assert(c0.SPECIALlimb.size() == SPECIALmeta.size());
    assert(c1.SPECIALlimb.size() == SPECIALmeta.size());
    cudaSetDevice(device);

    //std::map<int, int> used;
    s.wait(c0.s);
    s.wait(c1.s);
    s.wait(ksk_a.s);
    s.wait(ksk_b.s);

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            if (d == 0) {
                std::cout << hC_.primes[PRIMEID(limb[0])] << ": ";
                SWITCH(limb[0], printThisLimb());
            }

        if constexpr (1) {  // Batched
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            for (int i = 0; i < size; i += cc.batch) {
                STREAM(limb.at(start + i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                INTT_<false, algo, INTT_ROTATE_AND_SAVE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs},
                                                           blockDimFirst, bytesFirst, STREAM(limb.at(start + i)).ptr>>>(
                    c1.limbptr.data + start + i, start + i, c1.auxptr.data + start + i, nullptr,
                    c0.limbptr.data + start + i, c1.limbptr.data + start + i, ksk_a.DECOMPlimbptr[d].data + i,
                    ksk_b.DECOMPlimbptr[d].data + i, c0.limbptr.data + start + i, nullptr);

                INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                               bytesSecond, STREAM(limb.at(start + i)).ptr>>>(
                    c1.auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
            }
            for (size_t i = 0; i < size; i += cc.batch) {
                s.wait(STREAM(limb.at(start + i)));
            }
        }

        if constexpr (PRINT) {
            std::cout << hC_.primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
            SWITCH(DECOMPlimb[d][0], printThisLimb());
        }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (DECOMPlimb[d].size()) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr>>>(DECOMPlimbptr[d].data, level,
                                                                                   DIGITlimbptr[d].data, digitid[d]);
        }

        if constexpr (1) {  // Batched
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            int size = c0.SPECIALlimb.size();
            for (int i = 0; i < size; i += cc.batch) {
                STREAM(c0.SPECIALlimb.at(i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                              bytesFirst, STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                    DIGITlimbptr[d].data + i, SPECIAL(id, i), c1.SPECIALauxptr.data + i, nullptr, 0, nullptr, nullptr);

                if (d == 0) {
                    NTT_<true, algo, NTT_KSK_DOT><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                                    bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                        c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                        ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i, ksk_b.DIGITlimbptr[d].data + i);
                } else {
                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                            c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                            ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i,
                            ksk_b.DIGITlimbptr[d].data + i);
                }
            }
        }
    }

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        if constexpr (1)  // batched
        {
            int start = 0;
            for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                if (j == d)
                    continue;

                int Dstart = start + c0.SPECIALlimb.size();
                int Lstart = start + (j > d ? DECOMPlimb[d].size() : 0);

                int size = std::min((int)DECOMPlimb[j].size(), (int)c0.limb.size() - Lstart);
                if (size <= 0)
                    break;

                constexpr int M = 4;

                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                for (int i = 0; i < size; i += cc.batch) {
                    STREAM(c0.limb.at(Lstart + i)).wait(s);
                    uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                    NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                                  bytesFirst, STREAM(c0.limb.at(Lstart + i)).ptr>>>(
                        DIGITlimbptr[d].data + Dstart + i, Lstart + i, c1.auxptr.data + Lstart + i, nullptr, 0, nullptr,
                        nullptr);

                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.limb.at(Lstart + i)).ptr>>>(
                            c1.auxptr.data + Lstart + i, Lstart + i, c0.limbptr.data + Lstart + i,
                            ksk_a.DIGITlimbptr[d].data + Dstart + i, 0, c1.limbptr.data + Lstart + i,
                            ksk_b.DIGITlimbptr[d].data + Dstart + i);
                }

                start += DECOMPlimb[j].size();
            }
        }

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
    }

    for (auto& l : c0.limb)
        s.wait(STREAM(l));
    for (auto& l : c0.SPECIALlimb)
        s.wait(STREAM(l));

    c0.s.wait(s);
    c1.s.wait(s);
    ksk_a.s.wait(s);
    ksk_b.s.wait(s);
}

void LimbPartition::squareModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a,
                                      const LimbPartition& ksk_b, const int level) {

    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    assert(c0.SPECIALlimb.size() == SPECIALmeta.size());
    assert(c1.SPECIALlimb.size() == SPECIALmeta.size());
    cudaSetDevice(device);

    //std::map<int, int> used;
    s.wait(c0.s);
    s.wait(c1.s);
    s.wait(ksk_a.s);
    s.wait(ksk_b.s);

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            if (d == 0) {
                std::cout << hC_.primes[PRIMEID(limb[0])] << ": ";
                SWITCH(limb[0], printThisLimb());
            }

        if constexpr (1) {  // Batched
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            for (int i = 0; i < size; i += cc.batch) {
                STREAM(limb.at(start + i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                INTT_<false, algo, INTT_SQUARE_AND_SAVE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs},
                                                           blockDimFirst, bytesFirst, STREAM(limb.at(start + i)).ptr>>>(
                    c1.limbptr.data + start + i, start + i, c1.auxptr.data + start + i, nullptr,
                    c0.limbptr.data + start + i, c1.limbptr.data + start + i, ksk_a.DECOMPlimbptr[d].data + i,
                    ksk_b.DECOMPlimbptr[d].data + i, c0.limbptr.data + start + i, nullptr);

                INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                               bytesSecond, STREAM(limb.at(start + i)).ptr>>>(
                    c1.auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
            }
            for (size_t i = 0; i < size; i += cc.batch) {
                s.wait(STREAM(limb.at(start + i)));
            }
        }

        if constexpr (PRINT) {
            std::cout << hC_.primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
            SWITCH(DECOMPlimb[d][0], printThisLimb());
        }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (DECOMPlimb[d].size()) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr>>>(DECOMPlimbptr[d].data, level,
                                                                                   DIGITlimbptr[d].data, digitid[d]);
        }

        if constexpr (1) {  // Batched
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            int size = c0.SPECIALlimb.size();
            for (int i = 0; i < size; i += cc.batch) {
                STREAM(c0.SPECIALlimb.at(i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                              bytesFirst, STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                    DIGITlimbptr[d].data + i, SPECIAL(id, i), c1.SPECIALauxptr.data + i, nullptr, 0, nullptr, nullptr);

                if (d == 0) {
                    NTT_<true, algo, NTT_KSK_DOT><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                                    bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                        c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                        ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i, ksk_b.DIGITlimbptr[d].data + i);
                } else {
                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                            c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                            ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i,
                            ksk_b.DIGITlimbptr[d].data + i);
                }
            }
        }
    }

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        if constexpr (1)  // batched
        {
            int start = 0;
            for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                if (j == d)
                    continue;

                int Dstart = start + c0.SPECIALlimb.size();
                int Lstart = start + (j > d ? DECOMPlimb[d].size() : 0);

                int size = std::min((int)DECOMPlimb[j].size(), (int)c0.limb.size() - Lstart);
                if (size <= 0)
                    break;

                constexpr int M = 4;

                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                for (int i = 0; i < size; i += cc.batch) {
                    STREAM(c0.limb.at(Lstart + i)).wait(s);
                    uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                    NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                                  bytesFirst, STREAM(c0.limb.at(Lstart + i)).ptr>>>(
                        DIGITlimbptr[d].data + Dstart + i, Lstart + i, c1.auxptr.data + Lstart + i, nullptr, 0, nullptr,
                        nullptr);

                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.limb.at(Lstart + i)).ptr>>>(
                            c1.auxptr.data + Lstart + i, Lstart + i, c0.limbptr.data + Lstart + i,
                            ksk_a.DIGITlimbptr[d].data + Dstart + i, 0, c1.limbptr.data + Lstart + i,
                            ksk_b.DIGITlimbptr[d].data + Dstart + i);
                }

                start += DECOMPlimb[j].size();
            }
        }

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
    }

    for (auto& l : c0.limb)
        s.wait(STREAM(l));
    for (auto& l : c0.SPECIALlimb)
        s.wait(STREAM(l));

    c0.s.wait(s);
    c1.s.wait(s);
    ksk_a.s.wait(s);
    ksk_b.s.wait(s);
}

template <ALGO algo>
void LimbPartition::moddown(LimbPartition& auxLimbs, bool ntt, bool free_special_limbs, const int level) {
    assert(SPECIALlimb.size() == SPECIALmeta.size());
    const int limbsize = level + 1;
    cudaSetDevice(device);
    constexpr bool PRINT = false;

    {
        if constexpr (PRINT) {
            std::cout << "pre INTT Special GPU ";
            for (auto& i : SPECIALlimb) {
                SWITCH(i, printThisLimb(2));
            }
        }

        if (ntt) {
            for (size_t i = 0; i < SPECIALlimb.size(); i += cc.batch) {
                STREAM(SPECIALlimb[i]).wait(s);
            }
            ApplyINTT<algo, INTT_NONE>(cc.batch, INTT_fusion_fields{}, SPECIALlimb, SPECIALlimbptr, SPECIALauxptr, cc,
                                       SPECIAL(id, 0));
            for (size_t i = 0; i < SPECIALlimb.size(); i += cc.batch) {
                s.wait(STREAM(SPECIALlimb[i]));
            }
        }

        s.wait(auxLimbs.s);

        {
            dim3 blockSize{64, 2};  // blockSize.x * blockSize.y * blockSize.z <= 1024, blockSize.x a multiple of 32

            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (SPECIALlimb.size()) * blockSize.x;

            ModDown2<algo>
                <<<gridSize, blockSize, shared_bytes, s.ptr>>>(auxLimbs.limbptr.data, limbsize, SPECIALlimbptr.data);
        }

        if constexpr (PRINT) {
            std::cout << "Output ModDown ";
            for (auto& i : auxLimbs.limb) {
                SWITCH(i, printThisLimb(2));
            }
        }

        for (int i = 0; i < limbsize; i += cc.batch) {
            STREAM(limb.at(i)).wait(s);
        }
        NTT<algo, NTT_MODDOWN>(cc.batch, NTT_fusion_fields{.op2 = &auxLimbs}, limbsize);

        if constexpr (PRINT) {
            std::cout << "Output ModDown after sub mult.";
            for (auto& i : limb) {
                SWITCH(i, printThisLimb(2));
            }
        }

        for (int i = 0; i < limbsize; i += cc.batch) {
            s.wait(STREAM(limb.at(i)));
        }
    }
    auxLimbs.s.wait(s);

    if (free_special_limbs) {
        freeSpecialLimbs();
    }
}

#define YY(algo)                                                                                            \
    template void LimbPartition::moddown<algo>(LimbPartition & auxLimbs, bool ntt, bool free_special_limbs, \
                                               const int level);
#include "ntt_types.inc"

#undef YY

//////// SEYDA /////////

void LimbPartition::automorph(const int index, const int br) {
    cudaSetDevice(device);
    /*
    for (auto& i : limb) {
        SWITCH(i, automorph(index, br));
    }
    std::swap(limbptr.data, auxptr.data);
    */

    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        automorph_multi_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, auxptr.data + i, index, br);
    }
    for (auto& i : limb) {
        if (i.index() == U32) {
            std::swap(std::get<U32>(i).v.data, std::get<U32>(i).aux.data);
        } else {
            std::swap(std::get<U64>(i).v.data, std::get<U64>(i).aux.data);
        }
    }
    std::swap(limbptr.data, auxptr.data);
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
}

void LimbPartition::automorph_multi(const int index, const int br) {

    dim3 blockDim{128};
    dim3 gridDim{(cc.N) / blockDim.x, (uint32_t)limb.size()};

    automorph_multi_<<<gridDim, blockDim, 0, s.ptr>>>(limbptr.data, auxptr.data, index, br);

    for (auto& i : limb) {
        if (i.index() == U32) {
            std::swap(std::get<U32>(i).v.data, std::get<U32>(i).aux.data);
        } else {
            std::swap(std::get<U64>(i).v.data, std::get<U64>(i).aux.data);
        }
    }
    std::swap(limbptr.data, auxptr.data);
}
void LimbPartition::modupInto(LimbPartition& partition, int level, LimbPartition& aux_partition) {
    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    //assert(SPECIALlimb.empty());
    cudaSetDevice(device);

    const int limbsize = level;

    s.wait(partition.s);
    s.wait(aux_partition.s);

    if constexpr (PRINT)
        for (auto& i : limb) {
            SWITCH(i, printThisLimb(2));
        }

    for (size_t d = 0; d < DECOMPmeta.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPmeta[j].size();
        int size = std::min((int)DECOMPmeta[d].size(), limbsize - start);
        if (size <= 0)
            break;
        /*
        for (auto& l : DECOMPlimb[d]) {
            for (auto& p : limb) {
                if (PRIMEID(l) == PRIMEID(p)) {
                    STREAM(l).wait(STREAM(p));
                    SWITCH(l, INTT_from(p));
                    s.wait(STREAM(l));
                }
            }
        }
        */

        constexpr int M = 4;

        dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
        dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
        int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
        int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

        for (int i = 0; i < size; i += cc.batch) {
            STREAM(limb.at(start + i)).wait(s);
            uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

            INTT_<false, algo, INTT_NONE>
                <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst,
                   STREAM(limb.at(start + i)).ptr>>>(limbptr.data + start + i, start + i, auxptr.data + start + i);

            INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                           bytesSecond, STREAM(limb.at(start + i)).ptr>>>(
                auxptr.data + start + i, start + i, partition.DECOMPlimbptr[d].data + i);
        }
        for (size_t i = 0; i < size; i += cc.batch) {
            s.wait(STREAM(limb.at(start + i)));
        }
        if constexpr (PRINT)
            for (auto& i : partition.DECOMPlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr>>>(
                partition.DECOMPlimbptr[d].data, level, partition.DIGITlimbptr[d].data, digitid[d]);
        }

        if constexpr (PRINT)
            for (auto& i : partition.DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        const int digitsize = hC_.num_primeid_digit_to[digitid.at(d)][level - 1];

        for (size_t i = 0; i < digitsize; i += cc.batch) {
            STREAM(partition.DIGITlimb.at(d).at(i)).wait(s);
        }

        ApplyNTT<algo, NTT_NONE>(cc.batch, NTT_fusion_fields{}, partition.DIGITlimb.at(d), partition.DIGITlimbptr.at(d),
                                 aux_partition.DIGITlimbptr.at(d), cc, DIGIT(digitid.at(d), 0), digitsize);

        for (size_t i = 0; i < digitsize; i += cc.batch) {
            s.wait(STREAM(partition.DIGITlimb.at(d).at(i)));
        }

        if constexpr (PRINT)
            for (auto& i : partition.DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
    }

    aux_partition.s.wait(s);
    partition.s.wait(s);
}

void LimbPartition::multScalar(std::vector<uint64_t>& vector) {
    /*
    cudaDeviceSynchronize();
    for (auto& l : limb) {
        if (l.index() == U64) {
            scalar_mult_<uint64_t, ALGO_BARRETT>
                <<<cc.N / 128, 128, 0, STREAM(l).ptr>>>(std::get<U64>(l).v.data, vector[PRIMEID(l)], PRIMEID(l));
        } else {
            scalar_mult_<uint32_t, ALGO_BARRETT>
                <<<cc.N / 128, 128, 0, STREAM(l).ptr>>>(std::get<U32>(l).v.data, vector[PRIMEID(l)], PRIMEID(l));
        }
    }
    cudaDeviceSynchronize();
     */

    uint64_t* elems;
    cudaMallocAsync(&elems, vector.size() * sizeof(uint64_t), s.ptr);
    cudaMemcpyAsync(elems, vector.data(), vector.size() * sizeof(uint64_t), cudaMemcpyDefault, s.ptr);
    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        Scalar_mult_<ALGO_BARRETT><<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, elems + i, PARTITION(id, i), nullptr);
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    cudaFreeAsync(elems, s.ptr);
}

void LimbPartition::addScalar(std::vector<uint64_t>& vector) {
    uint64_t* elems;
    cudaMallocAsync(&elems, vector.size() * sizeof(uint64_t), s.ptr);
    cudaMemcpyAsync(elems, vector.data(), vector.size() * sizeof(uint64_t), cudaMemcpyDefault, s.ptr);
    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        scalar_add_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(limbptr.data + i, elems + i,
                                                                                            PARTITION(id, i));
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    cudaFreeAsync(elems, s.ptr);
}
void LimbPartition::subScalar(std::vector<uint64_t>& vector) {
    uint64_t* elems;
    cudaMallocAsync(&elems, vector.size() * sizeof(uint64_t), s.ptr);
    cudaMemcpyAsync(elems, vector.data(), vector.size() * sizeof(uint64_t), cudaMemcpyDefault, s.ptr);
    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        scalar_sub_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(limbptr.data + i, elems + i,
                                                                                            PARTITION(id, i));
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    cudaFreeAsync(elems, s.ptr);
}

void LimbPartition::add(const LimbPartition& a, const LimbPartition& b) {
    s.wait(a.s);
    s.wait(b.s);

    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        add_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, a.limbptr.data + i, b.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    a.s.wait(s);
    b.s.wait(s);
}
void LimbPartition::squareElement(const LimbPartition& p) {
    s.wait(p.s);
    int size = std::min(limb.size(), p.limb.size());
    for (int i = 0; i < size; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)size - i, cc.batch);
        square_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < size; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    p.s.wait(s);
}
void LimbPartition::binomialSquareFold(LimbPartition& c0_res, const LimbPartition& c2_key_switched_0,
                                       const LimbPartition& c2_key_switched_1) {
    s.wait(c0_res.s);
    s.wait(c2_key_switched_0.s);
    s.wait(c2_key_switched_1.s);
    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        binomial_square_fold_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            c0_res.limbptr.data + i, c2_key_switched_0.limbptr.data + i, limbptr.data + i,
            c2_key_switched_1.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    c0_res.s.wait(s);
    c2_key_switched_0.s.wait(s);
    c2_key_switched_1.s.wait(s);
}
void LimbPartition::dropLimb() {
    cudaSetDevice(device);

    STREAM(limb.back()).wait(s);
    limb.pop_back();
}
void LimbPartition::addMult(const LimbPartition& a, const LimbPartition& b) {
    s.wait(a.s);
    s.wait(b.s);
    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);
        addMult_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, a.limbptr.data + i, b.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    a.s.wait(s);
    b.s.wait(s);
}
void LimbPartition::broadcastLimb0() {
    broadcastLimb0_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)limb.size() - 1}, 128, 0, s.ptr>>>(limbptr.data);
}
void LimbPartition::evalLinearWSum(uint32_t n, std::vector<const LimbPartition*> ps, std::vector<uint64_t>& weights) {
    uint64_t* elems;
    cudaMallocAsync(&elems, weights.size() * sizeof(uint64_t), s.ptr);
    cudaMemcpyAsync(elems, weights.data(), weights.size() * sizeof(uint64_t), cudaMemcpyDefault, s.ptr);
    std::vector<void**> psptr(n, nullptr);
    for (int i = 0; i < n; ++i) {
        psptr[i] = ps[i]->limbptr.data;
    }
    void*** d_psptr;
    cudaMallocAsync(&d_psptr, psptr.size() * sizeof(void**), s.ptr);
    cudaMemcpyAsync(d_psptr, psptr.data(), psptr.size() * sizeof(void**), cudaMemcpyDefault, s.ptr);

    for (int i = 0; i < n; ++i) {
        s.wait(ps[i]->s);
    }

    eval_linear_w_sum_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)limb.size()}, 128, 0, s.ptr>>>(n, limbptr.data, d_psptr,
                                                                                             elems);
    /*
    for (int i = 0; i < limb.size(); i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limb.size() - i, cc.batch);

        scalar_mult_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, elems + i, PARTITION(id, i));
        for (int j = 1; j < n; ++j) {
            scalar_sub_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
                limbptr.data + i, elems + i, PARTITION(id, i));
        }
    }
    for (int i = 0; i < limb.size(); i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    */
    for (int i = 0; i < n; ++i) {
        ps[i]->s.wait(s);
    }
    cudaFreeAsync(elems, s.ptr);
    cudaFreeAsync(d_psptr, s.ptr);
}

}  // namespace FIDESlib::CKKS
