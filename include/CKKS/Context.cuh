//
// Created by carlos on 6/03/24.
//

#ifndef FIDESLIB_CKKS_CONTEXT_CUH
#define FIDESLIB_CKKS_CONTEXT_CUH

#include "ConstantsGPU.cuh"
#include "LimbUtils.cuh"
#include "Parameters.cuh"
#include "RNSPoly.cuh"

#include <array>
#include <cassert>
#include <iostream>

#ifdef NCCL
#include "nccl.h"
#endif

namespace FIDESlib::CKKS {

class RNSPoly;

class Context {
    static constexpr const char* loc{"Context"};
    CudaNvtxRange my_range;

   public:
    enum RESCALE_TECHNIQUE { NO_RESCALE, FIXEDMANUAL, FIXEDAUTO, FLEXIBLEAUTO, FLEXIBLEAUTOEXT };

    Parameters param;
    const int logN, N, slots;
    const RESCALE_TECHNIQUE rescaleTechnique;
    const int L, logQ;
    int batch;
    const std::vector<int> GPUid;
    const int dnum;

    std::vector<std::vector<int>> GPUdigits;
    const std::vector<PrimeRecord> prime;

    std::vector<std::vector<LimbRecord>> meta;
    const std::vector<int> logQ_d;

    const int K, logP;
    const std::vector<PrimeRecord> specialPrime;

    std::vector<std::vector<LimbRecord>> specialMeta;  // Make const maybe
    std::vector<std::vector<LimbRecord>> splitSpecialMeta;
    std::vector<std::vector<std::vector<LimbRecord>>> decompMeta;  // Make const maybe
    std::vector<std::vector<std::vector<LimbRecord>>> digitMeta;   // Make const maybe
    std::vector<LimbRecord> gatherMeta;

    const std::vector<dim3> limbGPUid;
    const std::vector<int> digitGPUid;

#ifdef NCCL
    ncclUniqueId communicatorID;
    std::vector<ncclComm_t> GPUrank;
#else
    std::vector<int> GPUrank;
#endif
    std::unique_ptr<RNSPoly> key_switch_aux = nullptr;
    std::unique_ptr<RNSPoly> key_switch_aux2 = nullptr;

    std::array<std::unique_ptr<RNSPoly>, 2> moddown_aux = {nullptr};

    std::vector<Stream> top_limb_stream;
    std::vector<uint64_t*> top_limb_buffer;
    std::vector<void*> top_limb_buffer_handle;
    std::vector<VectorGPU<void*>> top_limbptr;

    std::vector<Stream> top_limb_stream2;
    std::vector<uint64_t*> top_limb_buffer2;
    std::vector<void*> top_limb_buffer2_handle;
    std::vector<VectorGPU<void*>> top_limbptr2;

    std::vector<std::vector<Stream>> gatherStream;
    std::vector<std::vector<Stream>> digitStream;
    std::vector<std::vector<Stream>> digitStream2;

    // std::vector<RNSPoly> key_switch_digits;
    bool canP2P = false;

    //      std::array<Stream, 8> blockingStream;
    //      std::vector<std::vector<Stream>> asyncStream;
    RNSPoly& getKeySwitchAux();
    RNSPoly& getKeySwitchAux2();
    RNSPoly& getModdownAux(const int num);

    bool isValidPrimeId(const int i) const;

    Context(Parameters param, const std::vector<int>& devs, const int secBits = 0);
    ~Context();

   private:
    static int computeLogQ(const int L, std::vector<PrimeRecord>& primes);

    static int validateDnum(const std::vector<int>& GPUid, const int dnum);

    static std::vector<std::vector<LimbRecord>> generateMeta(const std::vector<int>& GPUid, const int dnum,
                                                             const std::vector<std::vector<int>> digitGPUid,
                                                             const std::vector<PrimeRecord>& prime,
                                                             const Parameters& param);

    static std::vector<int> computeLogQ_d(const int dnum, const std::vector<std::vector<LimbRecord>>& meta,
                                          const std::vector<PrimeRecord>& prime);

    static int computeK(const std::vector<int>& logQ_d, std::vector<PrimeRecord>& Sprimes, const Parameters& param);

    static std::vector<std::vector<LimbRecord>> generateSpecialMeta(const std::vector<std::vector<LimbRecord>>& meta,
                                                                    const std::vector<PrimeRecord>& specialPrime,
                                                                    const int ID0, const std::vector<int>& GPUid);

    static std::vector<std::vector<std::vector<LimbRecord>>> generateDecompMeta(
        const std::vector<std::vector<LimbRecord>>& meta, const std::vector<std::vector<int>> dnum,
        const std::vector<int>& vector, int L);

    static std::vector<std::vector<std::vector<LimbRecord>>> generateDigitMeta(
        const std::vector<std::vector<LimbRecord>>& meta, const std::vector<std::vector<LimbRecord>>& splitSpecialMeta,
        const std::vector<LimbRecord>& specialMeta, const std::vector<std::vector<int>>& digitGPUid,
        const std::vector<int>& GPUid);

    static std::vector<dim3> generateLimbGPUid(const std::vector<std::vector<LimbRecord>>& meta, const int L);

    static std::vector<std::vector<int>> generateGPUdigits(const int dnum, const std::vector<int>& devs);

   public:
    std::vector<uint64_t> ElemForEvalMult(int level, const double operand);
    std::vector<uint64_t> ElemForEvalAddOrSub(const int level, const double operand, const int noise_deg);
    std::vector<double>& GetCoeffsChebyshev();
    int GetDoubleAngleIts();
    void AddBootPrecomputation(int slots, BootstrapPrecomputation&& precomp) const;
    bool HasBootPrecomputation(int slots);
    static BootstrapPrecomputation& GetBootPrecomputation(int slots);
    void AddRotationKey(int index, KeySwitchingKey&& ksk);
    KeySwitchingKey& GetRotationKey(int index);
    bool HasRotationKey(int index);
    static void AddEvalKey(KeySwitchingKey&& ksk);
    static KeySwitchingKey& GetEvalKey();
    int GetBootK();
    int GetBootCorrectionFactor();
    static RESCALE_TECHNIQUE translateRescalingTechnique(lbcrypto::ScalingTechnique technique);
    void PrepareNCCLCommunication();
    const std::vector<int> generateDigitGPUid(std::vector<std::vector<LimbRecord>>& meta, const int L, const int dnum);
    static std::vector<std::vector<LimbRecord>> generateSplitSpecialMeta(std::vector<LimbRecord>& specialMeta,
                                                                         const std::vector<int> GPUid);
    static std::vector<LimbRecord> generateGatherMeta(const std::vector<std::vector<LimbRecord>>& meta, int L);
    std::vector<Ciphertext>& getBootstrapAuxilarCiphertexts();
};
}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_CKKS_CONTEXT_CUH