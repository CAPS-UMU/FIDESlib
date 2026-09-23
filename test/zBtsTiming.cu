
#include "CKKS/Bootstrap.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"

using namespace FIDESlib::CKKS;
using namespace std::chrono;

namespace FIDESlib::Testing {

// The parametrized fixture (GeneralParametrizedTest::SetUp) always creates a UNIFORM_TERNARY
// OpenFHE context. The bootstrap timing tests, however, must run the pipeline against a context
// whose secret-key distribution actually matches the distribution under test (SPARSE_TERNARY /
// SPARSE_ENCAPSULATED). Those contexts (and keys) are built and cached here, mirroring the
// fixture's caching scheme (the cache key additionally folds the distribution in).
static std::map<uint64_t, std::pair<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>, lbcrypto::KeyPair<lbcrypto::DCRTPoly>>> cached_bts_cc;

/**
 * Returns (and caches) an OpenFHE context and key pair generated for the requested
 * secret-key distribution, using the same parameter recipe as GeneralParametrizedTest::SetUp.
 */
static std::pair<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>, lbcrypto::KeyPair<lbcrypto::DCRTPoly>>
GetBtsContext(lbcrypto::SecretKeyDist dist, const GeneralTestParams& g, lbcrypto::ScalingTechnique scaling) {
    uint64_t index = g.ringDim + (1ul << 20) * g.multDepth + static_cast<uint64_t>(scaling) + (1ul << 30) * g.dnum + (1ul << 40) * g.scaleModSize +
        (1ul << 48) * static_cast<uint64_t>(dist);
    auto it = cached_bts_cc.find(index);
    if (it != cached_bts_cc.end()) {
        it->second.first->GetEncodingParams()->SetBatchSize(g.batchSize);
        return it->second;
    }

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(g.multDepth);
    parameters.SetFirstModSize(g.firstModSize);
    parameters.SetScalingModSize(g.scaleModSize);
    parameters.SetBatchSize(g.batchSize);
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetRingDim(g.ringDim);
    parameters.SetNumLargeDigits(g.dnum);
    parameters.SetScalingTechnique(scaling);
    parameters.SetSecretKeyDist(dist);
    parameters.SetPREMode(lbcrypto::INDCPA);

    auto cc = lbcrypto::GenCryptoContext(parameters);
    cc->Enable(lbcrypto::PKE);
    auto keys = cc->KeyGen();

    auto res = std::make_pair(cc, keys);
    cached_bts_cc[index] = res;
    return res;
}

/** Wipes the evaluation keys / bootstrap precomputations of every cached OpenFHE context. */
static void ClearBtsEvalKeys() {
    for (auto& i : cached_cc) {
        i.second.first->ClearEvalAutomorphismKeys();
        i.second.first->ClearEvalMultKeys();
        if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
            std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
    }
    for (auto& i : cached_bts_cc) {
        i.second.first->ClearEvalAutomorphismKeys();
        i.second.first->ClearEvalMultKeys();
        if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
            std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
    }
}

class BtsTimingTests : public GeneralParametrizedTest {
protected:
    /**
     * Runs the GPU bootstrap timing loop against an OpenFHE context built for the requested
     * secret-key distribution. UNIFORM_TERNARY reuses the fixture's context, while
     * SPARSE_TERNARY / SPARSE_ENCAPSULATED use the dedicated per-distribution context.
     *
     * @param dist      OpenFHE secret-key distribution to test.
     * @param bootConf  FIDESlib boot configuration to run (UNIFORM / SPARSE / ENCAPS).
     * @param dim1      baby-step/giant-step decomposition for EvalBootstrapSetup.
     * @param dropLevel Level to drop the input ciphertext to before bootstrapping (0 = keep).
     */
    void RunBootstrapTiming(lbcrypto::SecretKeyDist dist, FIDESlib::BOOT_CONFIG bootConf, std::vector<uint32_t> dim1, int dropLevel);
};

void BtsTimingTests::RunBootstrapTiming(lbcrypto::SecretKeyDist dist, FIDESlib::BOOT_CONFIG bootConf, std::vector<uint32_t> dim1, int dropLevel) {
    CKKS::DeregisterAllContexts();
    ClearBtsEvalKeys();

    // The fixture's context is UNIFORM_TERNARY; for the sparse distributions use a dedicated
    // OpenFHE context whose secret-key distribution matches the boot configuration.
    if (dist != lbcrypto::UNIFORM_TERNARY) {
        auto cached = GetBtsContext(dist, generalTestParams, std::get<1>(GetParam()));
        cc = cached.first;
        keys = cached.second;
    }

    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);

    std::cout << "Create context (" << dist << ")" << std::endl;
    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConf);
    FIDESlib::CKKS::Context GPUcc_ = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
    FIDESlib::CKKS::ContextData& GPUcc = *GPUcc_;
    std::cout << "Num large digits " << GPUcc.dnum << ", Chebyshev de: " << lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false)
        << " (degree), double-angle iterations: " << GPUcc.GetDoubleAngleIts() << std::endl;

    // Parameters
    GPUcc.batch = 128;
    int numSlots = cc->GetRingDimension() / 2;

    // Bootstrapping Precomputation
    cc->EvalBootstrapSetup(
        { 3, 3 },
        dim1,
        numSlots,
        0,
        true,
        false,
        lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) + GPUcc.GetDoubleAngleIts());

    // CPU-side bootstrap keygen (mirrors the OpenFHE reference flow; for SPARSE_ENCAPSULATED it
    // authors OpenFHE's own sparse-switch keys at automorphism slots 2N-2/2N-4 and needs the map
    // index 2N-4/2N-2 intact for the CPU EvalBootstrap). FIDESlib keeps its switching keys in the
    // disjoint 2N-6/2N-8 slots (BTS_KSPARSE_*_SLOT in RawCiphertext.cu), so both key sets coexist:
    // the CPU reference reads 2N-2/2N-4, the GPU pipeline reads 2N-6/2N-8.
    cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

    std::cout << "Add bootstrap precomputation" << std::endl;
    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc_);

    std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, numSlots);
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, c1);

    std::cout << "Create ciphertext" << std::endl;
    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc_, raw);

    if (dropLevel > 0)
        GPUct1.dropToLevel(dropLevel);

    int N = 10;

    std::cout << "Begin boot" << std::endl;
    auto start_gpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        Bootstrap(GPUct1, numSlots, false);
        cudaDeviceSynchronize();
    }
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) / N << " ms." << std::endl;

    std::cout << "Level after bootstrap: " << GPUct1.getLevel() << std::endl;

    cudaDeviceSynchronize();

    FIDESlib::CKKS::RawCipherText raw_res;
    GPUct1.store(raw_res);
    auto result(c1);
    GetOpenFHECipherText(result, raw_res);

    lbcrypto::Plaintext result_pt;
    cc->Decrypt(keys.secretKey, result, &result_pt);
    std::cout << "Precision: " << result_pt->GetLogPrecision() << " bits" << std::endl;
    for (int i = 0; i < 8; ++i) {
        std::cout << result_pt->GetRealPackedValue().at(i) << " ";
    }
    std::cout << std::endl;
}

// UNIFORM_TERNARY + UNIFORM: the actual, default secret-key distribution and boot configuration.
TEST_P(BtsTimingTests, Regular) {
    RunBootstrapTiming(lbcrypto::UNIFORM_TERNARY, FIDESlib::UNIFORM, { 16, 16 }, 0);
}

// UNIFORM_TERNARY + ENCAPS: a uniform key run through the (sparse-)encapsulated pipeline.
TEST_P(BtsTimingTests, UniformEncaps) {
    RunBootstrapTiming(lbcrypto::UNIFORM_TERNARY, FIDESlib::ENCAPS, { 16, 16 }, 2);
}

// SPARSE_TERNARY + SPARSE: sparse secret-key distribution, sparse boot configuration.
TEST_P(BtsTimingTests, SparseTernary) {
    RunBootstrapTiming(lbcrypto::SPARSE_TERNARY, FIDESlib::SPARSE, { 0, 0 }, 2);
}

// SPARSE_TERNARY + ENCAPS: a sparse key run through the (sparse-)encapsulated pipeline.
TEST_P(BtsTimingTests, SparseTernaryEncaps) {
    RunBootstrapTiming(lbcrypto::SPARSE_TERNARY, FIDESlib::ENCAPS, { 16, 16 }, 2);
}

// SPARSE_ENCAPSULATED + ENCAPS: the native sparse-encapsulated secret-key distribution.
TEST_P(BtsTimingTests, SparseEncapsulated) {
    RunBootstrapTiming(lbcrypto::SPARSE_ENCAPSULATED, FIDESlib::ENCAPS, { 16, 16 }, 2);
}

INSTANTIATE_TEST_SUITE_P(LLMTests, BtsTimingTests, testing::Values(TTALL64BOOT));
} // namespace FIDESlib::Testing
