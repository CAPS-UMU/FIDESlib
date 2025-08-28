#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <CKKS/KeySwitchingKey.cuh>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

#include "MatMul.cuh"
#include "PolyApprox.cuh"
#include "Transformer.cuh"
#include "Transpose.cuh"

#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/CoeffsToSlots.cuh"

#include "utils.cuh"

namespace fs = std::filesystem;
using namespace FIDESlib::CKKS;
std::vector<int> devices{0};

int main(const int argc, char** argv) {

    // Context
    create_cpu_context();
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys = cc->KeyGen();
    EncoderConfiguration conf{.numSlots = cc->GetEncodingParams()->GetBatchSize(),

                              .blockSize = int(sqrt(cc->GetEncodingParams()->GetBatchSize())),
                              .token_length = 10,
                              .bStep = 16,
                              .bStepBoot = 16,
                              .bStepAcc = 4};
    prepare_cpu_context(keys, conf.numSlots, conf.blockSize);
    keys_ = keys;
    auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
    auto adapted_params = params.adaptTo(raw_params);
    FIDESlib::CKKS::Context GPUcc(adapted_params, std::vector<int>(devices.begin(), devices.end()));
    prepare_gpu_context_bert(GPUcc, keys, conf);
    GPUcc.batch = 100;

    // Paths
    std::string model_path = std::string(root_dir) + "weights/weights-bert-tiny-sst2";
    std::string path = model_path + "/sst2_validation.csv";
    std::string output_path = "a.txt";

    // Loading weights and biases
    struct PtMasks_GPU masks =
        GetPtMasks_GPU(GPUcc, cc, conf.numSlots, conf.blockSize, conf.token_length, conf.level_matmul + 1);

    struct PtWeights_GPU weights_layer0 =
        GetPtWeightsGPU(GPUcc, keys.publicKey, model_path, 0, conf.numSlots, conf.blockSize, conf.rows, conf.cols,
                        conf.level_matmul + 1, conf.num_heads);
    struct PtWeights_GPU weights_layer1 =
        GetPtWeightsGPU(GPUcc, keys.publicKey, model_path, 1, conf.numSlots, conf.blockSize, conf.rows, conf.cols,
                        conf.level_matmul + 1, conf.num_heads);

    struct MatrixMatrixProductPrecomputations_GPU precomp_gpu =
        getMatrixMatrixProductPrecomputations_GPU(GPUcc, cc, masks, conf.blockSize, conf.bStep, conf.level_matmul + 1,
                                                  conf.level_matmul + 1, conf.prescale, conf.numSlots);

    TransposePrecomputations_GPU Tprecomp_gpu =
        getMatrixTransposePrecomputations_GPU(GPUcc, cc, conf.blockSize, conf.bStep, conf.level_matmul);

    // Loading tokens
    ct_tokens = encryptMatrixtoCPU(std::string(model_path + "/tokens.txt"), keys.publicKey, conf.numSlots,
                                   conf.blockSize, conf.rows, conf.cols);
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu, tokens_gpu2;
    encryptMatrixtoGPU(std::string(model_path + "/tokens.txt"), tokens_gpu, keys.publicKey, GPUcc, conf.numSlots,
                       conf.blockSize, conf.rows, conf.cols, conf.level_matmul);
    encryptMatrixtoGPU(std::string(model_path + "/tokens.txt"), tokens_gpu2, keys.publicKey, GPUcc, conf.numSlots,
                       conf.blockSize, conf.rows, conf.cols, conf.level_matmul);

    if (1) {
        // Baseline
        cudaDeviceSynchronize();
        auto start_gpu = std::chrono::high_resolution_clock::now();
        tokens_gpu = encoder(weights_layer0, precomp_gpu, Tprecomp_gpu, tokens_gpu, masks, conf, 0);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                  << " ms." << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
        tokens_gpu = encoder(weights_layer1, precomp_gpu, Tprecomp_gpu, tokens_gpu, masks, conf, 1);

        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                  << " ms." << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
        uint32_t class_pred = classifier(cc, tokens_gpu, keys.secretKey, ct_tokens, precomp_gpu, weights_layer1, masks,
                                         conf.numSlots, conf.blockSize, conf.token_length, true, output_path);
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "Classifier took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
    }
    // Proposed Method
    for (int test_case : {0, 1, 2}) {
        // 0: Asymmetrical PCMM, 1: Statistical Softmax, 3: LN with delayed norm
        cudaDeviceSynchronize();
        auto start_gpu = std::chrono::high_resolution_clock::now();
        tokens_gpu2 = encoder_helmet(weights_layer0, precomp_gpu, Tprecomp_gpu, tokens_gpu2, masks, conf, 0, test_case);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                  << " ms." << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
        tokens_gpu2 = encoder_helmet(weights_layer1, precomp_gpu, Tprecomp_gpu, tokens_gpu2, masks, conf, 1, test_case);
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                  << " ms." << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
        uint32_t class_pred = classifier(cc, tokens_gpu2, keys.secretKey, ct_tokens, precomp_gpu, weights_layer1, masks,
                                         conf.numSlots, conf.blockSize, conf.token_length, true, output_path);
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "Classifier took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
    }

    return 0;
}