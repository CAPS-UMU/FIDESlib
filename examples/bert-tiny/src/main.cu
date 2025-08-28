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

    std::string model_name = "bert-tiny-mrpc";
    std::string model_path = std::string(root_dir) + "weights/weights-bert-tiny-mrpc";

    int test_case = 1;
    std::string outputBase = ".";
    std::string outputPrefix = "out";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-c" && i + 1 < argc) {
            test_case = std::stoi(argv[++i]);
        } else if (arg == "-w" && i + 1 < argc) {
            outputBase = argv[++i];
        }
    }

    fs::path outDir(outputBase);
    std::error_code ec;
    fs::create_directories(outDir, ec);  // ok if already exists

    int task_id = test_case;
    if (const char* p = std::getenv("SGE_TASK_ID")) {
        try {
            task_id = std::stoi(p);
        } catch (...) { /* keep test_case */
        }
    }

    std::string output_file = "tokens_mrpc" + std::to_string(test_case + 1) + ".txt";

    std::string path = model_path + "/mrpc_validation.csv";

    if (outputBase == "mrpc12_seyda/") {
        path = model_path + "/mrpc_validation12.csv";
    } else {
        output_file = "tokens_mrpc" + std::to_string(test_case + 4) + ".txt";
    }
    std::cout << "input file: " << path << std::endl;
    std::cout << "token file: " << output_file << std::endl;

    fs::path output_path = outDir / ("out_" + std::to_string(test_case) + ".txt");

    std::ofstream outFile(output_path);  // creates the file if it doesn't exist
    if (!outFile.is_open()) {
        std::cerr << "Error: could not create file " << output_path << std::endl;
        return 1;
    }

    outFile << "Experiment: " << test_case << std::endl;
    outFile.close();

    create_cpu_context();
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys = cc->KeyGen();
    EncoderConfiguration conf{.numSlots = cc->GetEncodingParams()->GetBatchSize(),
                              .blockSize = int(sqrt(cc->GetEncodingParams()->GetBatchSize())),
                              .token_length = 63};
    prepare_cpu_context(keys, conf.numSlots, conf.blockSize);
    auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
    auto adapted_params = params.adaptTo(raw_params);
    FIDESlib::CKKS::Context GPUcc(adapted_params, std::vector<int>(devices.begin(), devices.end()));
    prepare_gpu_context_bert(GPUcc, keys, TODO);

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

    ct_tokens = encryptMatrixtoCPU(std::string(model_path + "/" + output_file), keys.publicKey, conf.numSlots,
                                   conf.blockSize, conf.rows, conf.cols);

    std::string output_path_str = output_path.string();
    process_sentences_from_csv_mrpc(path, output_file, model_name, model_path, output_path_str, conf, keys.publicKey,
                                    GPUcc, ct_tokens, weights_layer0, weights_layer1, masks, precomp_gpu, Tprecomp_gpu,
                                    cc, keys.secretKey, test_case);

    return 0;
}