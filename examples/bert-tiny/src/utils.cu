// utils.cu
#include "utils.cuh"

#include <CKKS/AccumulateBroadcast.cuh>
#include <CKKS/Bootstrap.cuh>
#include <CKKS/BootstrapPrecomputation.cuh>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/Context.cuh>
#include <CKKS/KeySwitchingKey.cuh>
#include <CKKS/Plaintext.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>

using namespace FIDESlib::CKKS;

lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = nullptr;

std::vector<FIDESlib::PrimeRecord> p64{
    {.p = 2305843009218281473}, {.p = 2251799661248513}, {.p = 2251799661641729}, {.p = 2251799665180673},
    {.p = 2251799682088961},    {.p = 2251799678943233}, {.p = 2251799717609473}, {.p = 2251799710138369},
    {.p = 2251799708827649},    {.p = 2251799707385857}, {.p = 2251799713677313}, {.p = 2251799712366593},
    {.p = 2251799716691969},    {.p = 2251799714856961}, {.p = 2251799726522369}, {.p = 2251799726129153},
    {.p = 2251799747493889},    {.p = 2251799741857793}, {.p = 2251799740416001}, {.p = 2251799746707457},
    {.p = 2251799756013569},    {.p = 2251799775805441}, {.p = 2251799763091457}, {.p = 2251799767154689},
    {.p = 2251799765975041},    {.p = 2251799770562561}, {.p = 2251799769776129}, {.p = 2251799772266497},
    {.p = 2251799775281153},    {.p = 2251799774887937}, {.p = 2251799797432321}, {.p = 2251799787995137},
    {.p = 2251799787601921},    {.p = 2251799791403009}, {.p = 2251799789568001}, {.p = 2251799795466241},
    {.p = 2251799807131649},    {.p = 2251799806345217}, {.p = 2251799805165569}, {.p = 2251799813554177},
    {.p = 2251799809884161},    {.p = 2251799810670593}, {.p = 2251799818928129}, {.p = 2251799816568833},
    {.p = 2251799815520257}};

std::vector<FIDESlib::PrimeRecord> sp64{
    {.p = 2305843009218936833}, {.p = 2305843009220116481}, {.p = 2305843009221820417}, {.p = 2305843009224179713},
    {.p = 2305843009225228289}, {.p = 2305843009227980801}, {.p = 2305843009229160449}, {.p = 2305843009229946881},
    {.p = 2305843009231650817}, {.p = 2305843009235189761}, {.p = 2305843009240301569}, {.p = 2305843009242923009},
    {.p = 2305843009244889089}, {.p = 2305843009245413377}, {.p = 2305843009247641601}};

FIDESlib::CKKS::Parameters params{.logN = 13, .L = 26, .dnum = 5, .primes = p64, .Sprimes = sp64, .batch = 100};

void prepare_gpu_context_bert(FIDESlib::CKKS::Context& cc_gpu, const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys,
                              EncoderConfiguration& conf) {
    if (conf.blockSize * conf.blockSize != conf.numSlots) {
        std::cout << "blockSize: " << conf.blockSize << "; num_slots: " << conf.numSlots << std::endl;
        std::cerr << "Matrix size is different from number of slots" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // EvalMult key (CKKS relinearization)
    auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(cc_gpu);
    eval_key_gpu.Initialize(cc_gpu, eval_key);
    FIDESlib::CKKS::Context::AddEvalKey(std::move(eval_key_gpu));

    // Rotation keys
    std::vector<int32_t> rotation_indices =
        GenerateRotationIndices_GPU(conf.blockSize, conf.bStep, conf.bStepAcc, conf.num_heads);
    cc->EvalAtIndexKeyGen(keys.secretKey, rotation_indices);

    for (int i : rotation_indices) {
        auto rot_k = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, i, cc);
        FIDESlib::CKKS::KeySwitchingKey rot_k_gpu(cc_gpu);
        rot_k_gpu.Initialize(cc_gpu, rot_k);
        cc_gpu.AddRotationKey(i, std::move(rot_k_gpu));
    }

    // Bootstrap precomputation bridging CPU<->GPU
    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, static_cast<int>(conf.numSlots), cc_gpu);
}

void create_cpu_context() {
    // Parameter selection (replace ring_dim/num_slots with actual values you use)
    constexpr uint32_t scale_mod_size = 55;
    constexpr uint32_t first_mod = 60;
    constexpr uint32_t num_large_digits = 5;
    constexpr uint32_t depth = 26;

    // You must provide these two from your configuration, or compute from logN:
    const uint32_t ring_dim = 1u << 16;  // logN = 16 -> N = 65536
    const uint32_t num_slots = 1 << 14;

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetScalingModSize(scale_mod_size);
    parameters.SetFirstModSize(first_mod);
    parameters.SetRingDim(ring_dim);
    parameters.SetBatchSize(num_slots);
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetScalingTechnique(lbcrypto::FLEXIBLEAUTO);
    parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);
    parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
    parameters.SetNumLargeDigits(num_large_digits);
    parameters.SetMultiplicativeDepth(depth);

    // Create/replace global context
    if (cc != nullptr) {
        using Impl =
            lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>;
        Impl::ClearEvalAutomorphismKeys();
        Impl::ClearEvalMultKeys();
        Impl::ClearEvalSumKeys();
    }

    cc = lbcrypto::GenCryptoContext(parameters);
    cc->Enable(lbcrypto::FHE);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::ADVANCEDSHE);
}

void prepare_cpu_context(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, const size_t num_slots,
                         const size_t blockSize) {
    if (blockSize * blockSize != num_slots) {
        std::cout << "blockSize: " << blockSize << "; num_slots: " << num_slots << std::endl;
        std::cerr << "Matrix size is different from number of slots" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<int32_t> rotation_indices = GenerateRotationIndices_GPU(blockSize, 4, 4, 2);
    cc->EvalRotateKeyGen(keys.secretKey, rotation_indices);

    // Bootstrap keys (parameters here must match your pipeline)
    cc->EvalBootstrapSetup({3, 3}, {4, 4}, num_slots);
    cc->EvalBootstrapKeyGen(keys.secretKey, num_slots);
    cc->EvalBootstrapPrecompute(num_slots);
}
