#include "Transformer.cuh"

namespace FIDESlib::CKKS {

std::vector<std::vector<lbcrypto::Ciphertext<DCRTPoly>>> ct_tokens;
lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys_;

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> encoder(
    PtWeights_GPU& weights_layer, MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
    TransposePrecomputations_GPU& Tprecomp_gpu, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& tokens,
    PtMasks_GPU& masks, EncoderConfiguration& conf, int layerNo) {
    constexpr bool PRINT = false;
    constexpr bool TIMING = true;
    std::chrono::time_point<std::chrono::system_clock> start_gpu, end_gpu;

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    Context& cc = tokens[0][0].cc;
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> K, Q, V;
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> GPUResult_QKT, GPUResult_Sm_V, GPUResult_Output, GPUResult_Up,
        GPUResult_Down;

    dropMatrixLevel(tokens, conf.level_matmul);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(tokens, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "tokens",
                    false);

    if constexpr (PRINT)
        std::cout << "# limbs, tokens: " << tokens[0][0].getLevel() << " " << tokens[0][0].NoiseLevel << std::endl;

    PCMM_GPU(tokens, weights_layer.Wk, conf.blockSize, K, precomp_gpu, weights_layer.bk);
    PCMM_GPU(tokens, weights_layer.Wq, conf.blockSize, Q, precomp_gpu, weights_layer.bq);
    PCMM_GPU(tokens, weights_layer.Wv, conf.blockSize, V, precomp_gpu, weights_layer.bv);

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "PCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }

    if constexpr (PRINT)
        std::cout << "# limbs, Q: " << Q[0][0].getLevel() << " " << Q[0][0].NoiseLevel << std::endl;

    ////////////////////////////// Multi Head Attention /////////////////////////////////
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> Sm_V, Sm_V2;

    MatrixBootstrap(Q, conf.numSlots, conf.prescale);
    MatrixBootstrap(K, conf.numSlots, conf.prescale);

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> QKT1, QKT2;

    auto Q1 = MatrixMask(Q, masks.head_masks[0]);
    auto Q2 = MatrixMask(Q, masks.head_masks[1]);

    auto K1 = MatrixMask(K, masks.head_masks[0]);
    auto K2 = MatrixMask(K, masks.head_masks[1]);

    auto K1_T = MatrixTranspose_GPU(std::move(K1), conf.blockSize, Tprecomp_gpu);
    auto K2_T = MatrixTranspose_GPU(std::move(K2), conf.blockSize, Tprecomp_gpu);

    CCMM_GPU(Q1, K1_T, conf.blockSize, QKT1, precomp_gpu);
    CCMM_GPU(Q2, K2_T, conf.blockSize, QKT2, precomp_gpu);

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "CCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(QKT1, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "QKT1: ", false);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(QKT2, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "QKT2: ", false);

    QKT1 = MatrixMask(QKT1, masks.mask_tokens[conf.token_length]);
    QKT2 = MatrixMask(QKT2, masks.mask_tokens[conf.token_length]);

    // if constexpr (PRINT) printMatrix(decryptGPUMatrix(QKT1, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT1: ", false);
    // if constexpr (PRINT) printMatrix(decryptGPUMatrix(QKT2, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT2: ", false);

    MatrixRotate(QKT2, -conf.blockSize / conf.num_heads);

    // auto QKT = MatrixAdd(QKT1, QKT2);

    // if constexpr (PRINT) printMatrix(decryptGPUMatrix(QKT, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT: ", false);

    // if (layerNo == 1){ MatrixAddScalar(QKT, -0.125); }

    MatrixBootstrap(QKT1, conf.numSlots, conf.prescale);
    EvalSoftmax_Matrix(QKT1, ct_tokens[0][0], keys_.secretKey, cc.GetEvalKey(), masks.mask_tokens[conf.token_length],
                       masks.mask_broadcast, masks.mask_layernorm[0], masks.mask_max, conf.numSlots, conf.blockSize,
                       conf.bStepAcc, conf.token_length, true);
    MatrixBootstrap(QKT1, conf.numSlots, conf.prescale);

    MatrixBootstrap(QKT2, conf.numSlots, conf.prescale);
    EvalSoftmax_Matrix(QKT2, ct_tokens[0][0], keys_.secretKey, cc.GetEvalKey(), masks.mask_tokens[conf.token_length],
                       masks.mask_broadcast, masks.mask_layernorm[0], masks.mask_max, conf.numSlots, conf.blockSize,
                       conf.bStepAcc, conf.token_length, true);
    MatrixBootstrap(QKT2, conf.numSlots, conf.prescale);

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "Softmax took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    MatrixRotate(QKT2, conf.blockSize / conf.num_heads);

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(QKT1, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "QKT1: ", false);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(QKT2, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "QKT2: ", false);

    auto V1 = MatrixMask(V, masks.head_masks[0]);
    auto V2 = MatrixMask(V, masks.head_masks[1]);
    MatrixRotate(V2, conf.blockSize / conf.num_heads);

    CCMM_GPU(QKT1, V1, conf.blockSize, Sm_V, precomp_gpu);
    CCMM_GPU(QKT2, V2, conf.blockSize, Sm_V2, precomp_gpu);

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(Sm_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Sm_V1: ", false);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(Sm_V2, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Sm_V2: ", false);
    MatrixRotate(Sm_V2, -conf.blockSize / conf.num_heads);

    Sm_V = MatrixAdd(Sm_V, Sm_V2);

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "CCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }

    //////////////////////////////////////////////////////////////////////////////////////

    // GPUResult_V.clear();
    // GPUResult_QKT.clear();
    if constexpr (PRINT)
        std::cout << "# limbs: " << Sm_V[0][0].getLevel() << " " << Sm_V[0][0].NoiseLevel << std::endl;
    if constexpr (PRINT)
        std::cout << "# ------- bts  ------- " << std::endl;
    MatrixBootstrap(Sm_V, conf.numSlots);
    if constexpr (PRINT)
        std::cout << "# limbs: " << Sm_V[0][0].getLevel() << " " << Sm_V[0][0].NoiseLevel << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(Sm_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Sm_V: ", false);

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "Boot took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    // Output CCMM
    dropMatrixLevel(Sm_V, conf.level_matmul);
    PCMM_GPU(Sm_V, weights_layer.Wo, conf.blockSize, GPUResult_Output, precomp_gpu, weights_layer.bo);
    if constexpr (PRINT)
        std::cout << "# limbs O: " << GPUResult_Output[0][0].getLevel() << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_output: ", false);

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "PCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    // Layer Norm
    // MatrixBootstrap(tokens, conf.numSlots, false);
    GPUResult_Output = MatrixAdd(GPUResult_Output, tokens);
    MatrixBootstrap(GPUResult_Output, conf.numSlots);

    tokens.clear();
    EvalLayerNorm_Matrix(GPUResult_Output, ct_tokens[0][0], keys_.secretKey, cc.GetEvalKey(), masks.mask_layernorm,
                         masks.row_masks[conf.token_length], weights_layer.Wln1, weights_layer.bln1, conf.numSlots,
                         conf.blockSize, conf.bStepAcc, true);
    if constexpr (PRINT)
        std::cout << "# limbs LN: " << GPUResult_Output[0][0].getLevel() << std::endl;

    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_Output, conf.numSlots);
    if constexpr (PRINT)
        std::cout << "# limbs LN: " << GPUResult_Output[0][0].getLevel() << std::endl;

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "LN: ", false);

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "LN took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                  << " ms." << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    // Up PCMM
    // dropMatrixLevel(GPUResult_Output, conf.level_matmul - 2);
    PCMM_GPU(GPUResult_Output, weights_layer.Wu, conf.blockSize, GPUResult_Up, precomp_gpu, weights_layer.bu);

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "PCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    // GPUResult_Output.clear();
    if constexpr (PRINT)
        std::cout << "# limbs U: " << GPUResult_Up[0][0].getLevel() << " " << GPUResult_Up[0][0].NoiseLevel
                  << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Up, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "RELU input: ", false);

    // dropMatrixLevel(GPUResult_Up, conf.level_matmul);

    // ReLU
    EvalGelu_Matrix(GPUResult_Up, cc.GetEvalKey(), conf.numSlots);
    if constexpr (PRINT)
        std::cout << "# limbs r: " << GPUResult_Up[0][0].getLevel() << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Up, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "RELU: ", false);

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "Gelu took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    // Down PCMM
    dropMatrixLevel(GPUResult_Up, conf.level_matmul - 4);
    PCMM_GPU(GPUResult_Up, weights_layer.Wd, conf.blockSize, GPUResult_Down, precomp_gpu, weights_layer.bd);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_Down: ", false);
    if constexpr (PRINT)
        std::cout << "# limbs D: " << GPUResult_Down[0][0].getLevel() << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_Down: ", false);

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "PCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    // Layer Norm
    GPUResult_Down = MatrixAdd(GPUResult_Down, GPUResult_Output);
    MatrixBootstrap(GPUResult_Down, conf.numSlots);

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "LN input: ", false);

    EvalLayerNorm_Matrix(GPUResult_Down, ct_tokens[0][0], keys_.secretKey, cc.GetEvalKey(), masks.mask_layernorm,
                         masks.row_masks[conf.token_length], weights_layer.Wln2, weights_layer.bln2, conf.numSlots,
                         conf.blockSize, conf.bStepAcc, true);
    if constexpr (PRINT)
        std::cout << "# limbs LN: " << GPUResult_Down[0][0].getLevel() << std::endl;
    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_Down, conf.numSlots);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "LN: ", false);

    if constexpr (PRINT)
        std::cout << "# limbs LN: " << GPUResult_Down[0][0].getLevel() << std::endl;

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "LN :q"
                     "took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    return GPUResult_Down;
}

// std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> encoder(
//     PtWeights_GPU& weights_layer, MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
//     TransposePrecomputations_GPU& Tprecomp_gpu, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& tokens,
//     PtMasks_GPU& masks, EncoderConfiguration& conf, int layerNo) {

//     constexpr bool PRINT = false ;

//     Context& cc = tokens[0][0].cc;
//     std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> factor;
//     factor.reserve(tokens.size());
//     for (size_t i = 0; i < tokens.size(); i++) {
//         std::vector<FIDESlib::CKKS::Ciphertext> row;
//         row.reserve(tokens[0].size());
//         for (size_t j = 0; j < tokens[0].size(); j++) {
//             row.emplace_back(cc);
//         }
//         factor.emplace_back(std::move(row));
//     }

//     std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>> K, Q, V, Sm_V_, QKT_heads;
//     std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> GPUResult_Output, GPUResult_Up, GPUResult_Down;
//     std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>> QKV_all;

//     std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>> WeightsQKV, BiasQKV;
//     CombineQKV_2D_to_3(weights_layer, cc, WeightsQKV, BiasQKV);

//     dropMatrixLevel(tokens, conf.level_matmul);
//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(tokens, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "tokens", false);
//     if constexpr (PRINT) std::cout << "# limbs, tokens: " << tokens[0][0].getLevel() << " " << tokens[0][0].NoiseLevel << std::endl;

//     PCMM_GPU_QKV_merged(tokens, WeightsQKV, conf.blockSize, QKV_all, precomp_gpu, BiasQKV, masks.row_masks[conf.token_length]);    // QKV_all: K, Q, V

//     if constexpr (PRINT) std::cout << "# limbs, Q: " << QKV_all[0][0][0].getLevel() << " " << QKV_all[0][0][0].NoiseLevel << std::endl;
//     // if constexpr (PRINT) printMatrix(decryptGPUMatrix(QKV_all[0], keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "K: ", false);
//     // if constexpr (PRINT) printMatrix(decryptGPUMatrix(QKV_all[1], keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Q: ", false);
//     // if constexpr (PRINT) printMatrix(decryptGPUMatrix(QKV_all[2], keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "V: ", false);

//     ////////////////////////////// Multi Head Attention /////////////////////////////////
//     auto K_T = MatrixTranspose_GPU(std::move(QKV_all[0]), conf.blockSize, Tprecomp_gpu);

//     CCMM_GPU_double_mask(QKV_all[1], K_T, conf.blockSize, QKT_heads, precomp_gpu, masks, 0, conf.token_length, true);
//     CCMM_GPU_double_mask(QKV_all[1], K_T, conf.blockSize, QKT_heads, precomp_gpu, masks, 1, conf.token_length, true);

//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(QKT_heads[0], keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT1: ", false);
//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(QKT_heads[1], keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT2: ", false);

//     MatrixRotate(QKT_heads[1], - conf.blockSize / 2);

//     auto QKT = MatrixAdd(QKT_heads[0], QKT_heads[1]);

//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(QKT, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT: ", false);

//     MatrixBootstrap(QKT, conf.numSlots, conf.prescale);
//     EvalSoftmax_Matrix(QKT, ct_tokens[0][0], keys_.secretKey, cc.GetEvalKey(), masks.mask_tokens[conf.token_length], masks.mask_broadcast, masks.mask_layernorm[0], masks.mask_max, conf.numSlots, conf.blockSize, conf.bStepAcc, conf.token_length, true, 0, layerNo);
//     MatrixBootstrap(QKT, conf.numSlots, conf.prescale);

//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(QKT, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT: ", false);
//     if constexpr (PRINT) std::cout << "# limbs: " << QKT[0][0].getLevel() << " " << QKT[0][0].NoiseLevel << std::endl;

//     CCMM_GPU_double_mask(QKT, QKV_all[2], conf.blockSize, Sm_V_, precomp_gpu, masks, 0, conf.token_length, false);
//     MatrixRotate(QKV_all[2], conf.blockSize / 2);
//     MatrixRotate(QKT, conf.blockSize / 2);
//     CCMM_GPU_double_mask(QKT, QKV_all[2], conf.blockSize, Sm_V_, precomp_gpu, masks, 0, conf.token_length, false);

//     if constexpr (PRINT) std::cout << "# limbs: " << Sm_V_[0][0][0].getLevel() << " " << Sm_V_[0][0][0].NoiseLevel << std::endl;
//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(Sm_V_[0], keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Sm_V1: ", false);
//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(Sm_V_[1], keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Sm_V2: ", false);
//     MatrixRotate(Sm_V_[1], - conf.blockSize / conf.num_heads);

//     auto Sm_V = MatrixAdd(Sm_V_[0], Sm_V_[1]);

//     //////////////////////////////////////////////////////////////////////////////////////
//     Sm_V = MatrixMask(Sm_V, masks.row_masks[conf.token_length]);
//     if constexpr (PRINT) std::cout << "# limbs SmV: " << Sm_V[0][0].getLevel() << " "  << Sm_V[0][0].NoiseLevel << std::endl;
//     if constexpr (PRINT)  printMatrix(decryptGPUMatrix(Sm_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "SmV: ", false);

//     MatrixBootstrap(Sm_V, conf.numSlots);

//     // Output CCMM
//     PCMM_GPU(Sm_V, weights_layer.Wo, conf.blockSize, GPUResult_Output, precomp_gpu, weights_layer.bo, masks.row_masks[conf.token_length]);

//     if constexpr (PRINT) std::cout << "# limbs O: " << GPUResult_Output[0][0].getLevel() << " "  << GPUResult_Output[0][0].NoiseLevel << std::endl;
//     if constexpr (PRINT)  printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Result_output: ", false);

//     // Layer Norm
//     GPUResult_Output = MatrixAdd(GPUResult_Output, tokens);

//     if constexpr (PRINT) std::cout << "# ------- bts ------- " << std::endl;
//     MatrixBootstrap(GPUResult_Output, conf.numSlots);

//     if constexpr (PRINT) std::cout << "# limbs LNin: " << GPUResult_Output[0][0].getLevel() << " "  << GPUResult_Output[0][0].NoiseLevel << std::endl;
//     if constexpr (PRINT)  printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "LNin: ", false);
//     tokens.clear();

//     EvalLayerNorm_Matrix(GPUResult_Output, ct_tokens[0][0], keys_.secretKey, cc.GetEvalKey(), masks.mask_layernorm, masks.row_masks[conf.token_length],
//                         weights_layer.Wln1, weights_layer.bln1, conf.numSlots, conf.blockSize, conf.bStepAcc, true);
//     if constexpr (PRINT) std::cout << "# limbs LN: " << GPUResult_Output[0][0].getLevel() << " "  << GPUResult_Output[0][0].NoiseLevel << std::endl;

//     if constexpr (PRINT) std::cout << "# ------- bts ------- " << std::endl;
//     MatrixBootstrap(GPUResult_Output, conf.numSlots);
//     if constexpr (PRINT) std::cout << "# limbs LN: " << GPUResult_Output[0][0].getLevel() << " "  << GPUResult_Output[0][0].NoiseLevel << std::endl;

//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "LNout: ", false);

//     // Up PCMM
//     PCMM_GPU(GPUResult_Output, weights_layer.Wu, conf.blockSize, GPUResult_Up, precomp_gpu, weights_layer.bu, masks.row_masks[conf.token_length]);

//     if constexpr (PRINT) std::cout << "# size U: " << GPUResult_Up.size() << std::endl;
//     if constexpr (PRINT) std::cout << "# limbs U: " << GPUResult_Up[0][0].getLevel() << " " << GPUResult_Up[0][0].NoiseLevel << std::endl;
//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(GPUResult_Up, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Up: ", false);

//     // ReLU
//     EvalGelu_Matrix(GPUResult_Up, cc.GetEvalKey(), conf.numSlots);
//     if constexpr (PRINT) std::cout << "# limbs r: " << GPUResult_Up[0][0].getLevel() << " "  << GPUResult_Up[0][0].NoiseLevel << std::endl;
//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(GPUResult_Up, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "GELU: ", false);

//     // Down PCMM
//     PCMM_GPU(GPUResult_Up, weights_layer.Wd, conf.blockSize, GPUResult_Down, precomp_gpu, weights_layer.bd, masks.row_masks[conf.token_length]);
//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Down: ", false);
//     if constexpr (PRINT) std::cout << "# limbs D: " << GPUResult_Down[0][0].getLevel() << " "  << GPUResult_Down[0][0].NoiseLevel << std::endl;

//     // Layer Norm 2
//     GPUResult_Down = MatrixAdd(GPUResult_Down, GPUResult_Output);
//     MatrixBootstrap(GPUResult_Down, conf.numSlots);

//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "LNin: ", false);

//     EvalLayerNorm_Matrix(GPUResult_Down, ct_tokens[0][0], keys_.secretKey, cc.GetEvalKey(), masks.mask_layernorm, masks.row_masks[conf.token_length],
//                         weights_layer.Wln2, weights_layer.bln2, conf.numSlots, conf.blockSize, conf.bStepAcc, true);
//     if constexpr (PRINT) std::cout << "# limbs LN: " << GPUResult_Down[0][0].getLevel() << " "  << GPUResult_Down[0][0].NoiseLevel << std::endl;
//     if constexpr (PRINT) std::cout << "# ------- bts ------- " << std::endl;
//     MatrixBootstrap(GPUResult_Down, conf.numSlots);
//     if constexpr (PRINT) printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "LNout: ", false);
//     if constexpr (PRINT) std::cout << "# limbs LN: " << GPUResult_Down[0][0].getLevel() << " "  << GPUResult_Down[0][0].NoiseLevel << std::endl;

//     return GPUResult_Down;
// }

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> encoder_helmet(
    PtWeights_GPU& weights_layer, MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
    TransposePrecomputations_GPU& Tprecomp_gpu, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& tokens,
    PtMasks_GPU& masks, EncoderConfiguration& conf, int layerNo, int test_case) {

    constexpr bool PRINT = false;
    constexpr bool TIMING = true;
    std::chrono::time_point<std::chrono::system_clock> start_gpu, end_gpu;

    Context& cc = tokens[0][0].cc;
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> factor;

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> K, Q, V, O, U, D, Sm_V;
    std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>> Sm_V_, QKT_heads;

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    dropMatrixLevel(tokens, conf.level_matmul-2);   // level adjustment

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(tokens, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "tokens",
                    false);
    if constexpr (PRINT)
        std::cout << "# limbs, tokens: " << tokens[0][0].getLevel() << " " << tokens[0][0].NoiseLevel << std::endl;

    PCMM_2(tokens, weights_layer.Wk, conf.blockSize, K, precomp_gpu, weights_layer.bk,
           masks.row_masks[conf.token_length]);

    PCMM_2(tokens, weights_layer.Wq, conf.blockSize, Q, precomp_gpu, weights_layer.bq,
           masks.row_masks[conf.token_length]);

    PCMM_2(tokens, weights_layer.Wv, conf.blockSize, V, precomp_gpu, weights_layer.bv,
           masks.row_masks[conf.token_length]);

    if constexpr (PRINT)
        std::cout << "# limbs, Q: " << Q[0][0].getLevel() << " " << Q[0][0].NoiseLevel << std::endl;

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "PCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    ////////////////////////////// Multi Head Attention /////////////////////////////////
    auto K_T = MatrixTranspose_GPU(std::move(K), conf.blockSize, Tprecomp_gpu);

    CCMM_GPU_double_mask(Q, K_T, conf.blockSize, QKT_heads, precomp_gpu, masks, 0, conf.token_length, true);
    CCMM_GPU_double_mask(Q, K_T, conf.blockSize, QKT_heads, precomp_gpu, masks, 1, conf.token_length, true);

    cudaDeviceSynchronize();

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "CCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    if (conf.token_length < 64) {

        MatrixRotate(QKT_heads[1], -conf.blockSize / 2);

        auto QKT = MatrixAdd(QKT_heads[0], QKT_heads[1]);

        if constexpr (PRINT)
            printMatrix(decryptGPUMatrix(QKT, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                        "QKT: ", false);

        if (test_case == 0) {
            if (layerNo == 0) {
                MatrixAddScalar(QKT, -0.2);
            }
            if (layerNo == 1) {
                MatrixAddScalar(QKT, -0.3);
            }
        }

        if constexpr (PRINT)
            printMatrix(decryptGPUMatrix(QKT, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                        "QKT: ", false);
        // if constexpr (PRINT) printMatrix(decryptGPUMatrix(QKT, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), conf.token_length, conf.token_length, "QKT: ", false, 3, true);

        MatrixBootstrap(QKT, conf.numSlots, conf.prescale);
        EvalSoftmax_Matrix(QKT, ct_tokens[0][0], keys_.secretKey, cc.GetEvalKey(), masks.mask_tokens[conf.token_length],
                           masks.mask_broadcast, masks.mask_layernorm[0], masks.mask_max, conf.numSlots, conf.blockSize,
                           conf.bStepAcc, conf.token_length, true, test_case, layerNo);
        MatrixBootstrap(QKT, conf.numSlots, conf.prescale);

        if constexpr (TIMING) {
            cudaDeviceSynchronize();
            end_gpu = std::chrono::high_resolution_clock::now();
            std::cout << "Softmax took: "
                      << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                      << std::endl;
            start_gpu = std::chrono::high_resolution_clock::now();
        }
        // if constexpr (PRINT) printMatrix(decryptGPUMatrix(QKT, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), conf.token_length, conf.token_length, "QKT: ", false, 3, true);
        if constexpr (PRINT)
            printMatrix(decryptGPUMatrix(QKT, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                        "QKT: ", false);

        if constexpr (PRINT)
            std::cout << "# limbs: " << QKT[0][0].getLevel() << " " << QKT[0][0].NoiseLevel << std::endl;

        dropMatrixLevel(QKT, conf.level_matmul-2);  // level adjustment
        dropMatrixLevel(V, conf.level_matmul-2);    // level adjustment

        CCMM_GPU_double_mask(QKT, V, conf.blockSize, Sm_V_, precomp_gpu, masks, 0, conf.token_length, false);
        MatrixRotate(V, conf.blockSize / 2);
        MatrixRotate(QKT, conf.blockSize / 2);
        CCMM_GPU_double_mask(QKT, V, conf.blockSize, Sm_V_, precomp_gpu, masks, 0, conf.token_length, false);

        if constexpr (PRINT)
            std::cout << "# limbs: " << Sm_V_[0][0][0].getLevel() << " " << Sm_V_[0][0][0].NoiseLevel << std::endl;
        MatrixRotate(Sm_V_[1], -conf.blockSize / conf.num_heads);

        Sm_V = MatrixAdd(Sm_V_[0], Sm_V_[1]);

        if constexpr (TIMING) {
            cudaDeviceSynchronize();
            end_gpu = std::chrono::high_resolution_clock::now();
            std::cout << "CCMM took: "
                      << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                      << std::endl;
            start_gpu = std::chrono::high_resolution_clock::now();
        }
    } else {

        for (int j = 0; j < conf.num_heads; j++) {
            if (test_case == 0) {
                if (layerNo == 0) {
                    MatrixAddScalar(QKT_heads[j], -0.2);
                }
                if (layerNo == 1) {
                    MatrixAddScalar(QKT_heads[j], -0.3);
                }
            }

            if constexpr (PRINT)
                std::cout << "SM iteration: " << j << std::endl;

            if constexpr (PRINT)
                printMatrix(decryptGPUMatrix(QKT_heads[j], keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize),
                            2, 2, "QKT_heads[j]: ", false);

            MatrixBootstrap(QKT_heads[j], conf.numSlots, conf.prescale);
            EvalSoftmax_Matrix(QKT_heads[j], ct_tokens[0][0], keys_.secretKey, cc.GetEvalKey(),
                               masks.mask_tokens[conf.token_length], masks.mask_broadcast2, masks.mask_layernorm[0],
                               masks.mask_max, conf.numSlots, conf.blockSize, conf.bStepAcc, conf.token_length, true,
                               test_case, layerNo);
            MatrixBootstrap(QKT_heads[j], conf.numSlots, conf.prescale);

            if constexpr (TIMING) {
                cudaDeviceSynchronize();
                end_gpu = std::chrono::high_resolution_clock::now();
                std::cout << "Softmax took: "
                          << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                          << " ms." << std::endl;
                start_gpu = std::chrono::high_resolution_clock::now();
            }
            if constexpr (PRINT)
                printMatrix(decryptGPUMatrix(QKT_heads[j], keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize),
                            2, 2, "QKT_heads[j]: ", false);

            if (j == 1) {
                MatrixRotate(V, conf.blockSize / 2);
                // MatrixRotate(QKT_heads[j], conf.blockSize / 2);
            }
            // CCMM_GPU_double_mask(QKT_heads[j], V, conf.blockSize, Sm_V_, precomp_gpu, masks, 0, conf.token_length, false);
            CCMM_GPU_masked(QKT_heads[j], V, conf.blockSize, Sm_V_, precomp_gpu, masks.head_masks[0]);

            if constexpr (PRINT)
                printMatrix(decryptGPUMatrix(Sm_V_[j], keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                            "Sm_V[j]: ", false);

            if constexpr (TIMING) {
                cudaDeviceSynchronize();
                end_gpu = std::chrono::high_resolution_clock::now();
                std::cout << "CCMM took: "
                          << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                          << " ms." << std::endl;
                start_gpu = std::chrono::high_resolution_clock::now();
            }
        }

        MatrixRotate(Sm_V_[1], -conf.blockSize / conf.num_heads);
        Sm_V = MatrixAdd(Sm_V_[0], Sm_V_[1]);
    }

    //////////////////////////////////////////////////////////////////////////////////////
    Sm_V = MatrixMask(Sm_V, masks.row_masks[conf.token_length]);
    if constexpr (PRINT)
        std::cout << "# limbs SmV: " << Sm_V[0][0].getLevel() << " " << Sm_V[0][0].NoiseLevel << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(Sm_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "SmV: ", false);

    MatrixBootstrap(Sm_V, conf.numSlots);

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "Boot took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }

    // Output CCMM
    dropMatrixLevel(Sm_V, conf.level_matmul);
    PCMM_2(Sm_V, weights_layer.Wo, conf.blockSize, O, precomp_gpu, weights_layer.bo,
           masks.row_masks[conf.token_length]);
    // PCMM_GPU(Sm_V, weights_layer.Wo, conf.blockSize, O, precomp_gpu, weights_layer.bo, masks.row_masks[conf.token_length]);

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "PCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    if constexpr (PRINT)
        std::cout << "# limbs O: " << O[0][0].getLevel() << " " << O[0][0].NoiseLevel << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(O, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_output: ", false);

    // Layer Norm
    O = MatrixAdd(O, tokens);

    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(O, conf.numSlots);

    if constexpr (PRINT)
        std::cout << "# limbs LNin: " << O[0][0].getLevel() << " " << O[0][0].NoiseLevel << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(O, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "LNin: ", false);
    tokens.clear();

    if (test_case == 2) {
        EvalLayerNorm_Matrix_DelayedInv(O, ct_tokens[0][0], keys_.secretKey, cc.GetEvalKey(), masks.mask_layernorm,
                                        masks.row_masks[conf.token_length], weights_layer.Wln1, weights_layer.bln1,
                                        conf.numSlots, conf.blockSize, conf.bStepAcc, true, factor);
        if constexpr (PRINT)
            std::cout << "# limbs LN: " << O[0][0].getLevel() << " " << O[0][0].NoiseLevel << std::endl;

        if constexpr (PRINT)
            std::cout << "# ------- bts ------- " << std::endl;
        MatrixBootstrap(O, conf.numSlots);
        if constexpr (PRINT)
            std::cout << "# limbs LN: " << O[0][0].getLevel() << " " << O[0][0].NoiseLevel << std::endl;

        if constexpr (PRINT)
            printMatrix(decryptGPUMatrix(O, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                        "LNout: ", false);
        if constexpr (PRINT)
            printMatrix(decryptGPUMatrix(factor, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                        "factor: ", false);

        if constexpr (TIMING) {
            cudaDeviceSynchronize();
            end_gpu = std::chrono::high_resolution_clock::now();
            std::cout << "LN-no-inv took: "
                      << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                      << std::endl;
            start_gpu = std::chrono::high_resolution_clock::now();
        }
        // Up PCMM
        PCMM_GPU_delayedInv(O, weights_layer.Wu, conf.blockSize, U, precomp_gpu, weights_layer.bu,
                            masks.row_masks[conf.token_length], factor);

        if constexpr (PRINT)
            std::cout << "# size U: " << U.size() << std::endl;
        if constexpr (PRINT)
            std::cout << "# limbs U: " << U[0][0].getLevel() << " " << U[0][0].NoiseLevel << std::endl;
        if constexpr (PRINT)
            printMatrix(decryptGPUMatrix(U, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                        "Up: ", false);

        if constexpr (TIMING) {
            cudaDeviceSynchronize();
            end_gpu = std::chrono::high_resolution_clock::now();
            std::cout << "PCMM took: "
                      << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                      << std::endl;
            start_gpu = std::chrono::high_resolution_clock::now();
        }
        // ReLU
        EvalGelu_Matrix(U, cc.GetEvalKey(), conf.numSlots);
        if constexpr (PRINT)
            std::cout << "# limbs r: " << U[0][0].getLevel() << " " << U[0][0].NoiseLevel << std::endl;
        if constexpr (PRINT)
            printMatrix(decryptGPUMatrix(U, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                        "GELU: ", false);

        if constexpr (TIMING) {
            cudaDeviceSynchronize();
            end_gpu = std::chrono::high_resolution_clock::now();
            std::cout << "Gelu took: "
                      << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                      << std::endl;
            start_gpu = std::chrono::high_resolution_clock::now();
        }
        // Down PCMM
        dropMatrixLevel(U, conf.level_matmul - 4);  // level adjustment
        PCMM_GPU_delayedInv(U, weights_layer.Wd, conf.blockSize, D, precomp_gpu, weights_layer.bd,
                            masks.row_masks[conf.token_length], factor);
        if constexpr (PRINT)
            printMatrix(decryptGPUMatrix(D, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                        "Down: ", false);
        if constexpr (PRINT)
            std::cout << "# limbs D: " << D[0][0].getLevel() << " " << D[0][0].NoiseLevel << std::endl;

        if constexpr (TIMING) {
            cudaDeviceSynchronize();
            end_gpu = std::chrono::high_resolution_clock::now();
            std::cout << "PCMM took: "
                      << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                      << std::endl;
            start_gpu = std::chrono::high_resolution_clock::now();
        }
    } else {
        EvalLayerNorm_Matrix(O, ct_tokens[0][0], keys_.secretKey, cc.GetEvalKey(), masks.mask_layernorm,
                             masks.row_masks[conf.token_length], weights_layer.Wln1, weights_layer.bln1, conf.numSlots,
                             conf.blockSize, conf.bStepAcc, true);
        if constexpr (PRINT)
            std::cout << "# limbs LN: " << O[0][0].getLevel() << " " << O[0][0].NoiseLevel << std::endl;

        if constexpr (PRINT)
            std::cout << "# ------- bts ------- " << std::endl;
        MatrixBootstrap(O, conf.numSlots);
        if constexpr (PRINT)
            std::cout << "# limbs LN: " << O[0][0].getLevel() << " " << O[0][0].NoiseLevel << std::endl;

        if constexpr (PRINT)
            printMatrix(decryptGPUMatrix(O, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                        "LNout: ", false);

        if constexpr (TIMING) {
            cudaDeviceSynchronize();
            end_gpu = std::chrono::high_resolution_clock::now();
            std::cout << "LN took: "
                      << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                      << std::endl;
            start_gpu = std::chrono::high_resolution_clock::now();
        }
        // Up PCMM
        PCMM_2(O, weights_layer.Wu, conf.blockSize, U, precomp_gpu, weights_layer.bu,
               masks.row_masks[conf.token_length]);
        // PCMM_GPU(O, weights_layer.Wu, conf.blockSize, U, precomp_gpu, weights_layer.bu, masks.row_masks[conf.token_length]);

        if constexpr (PRINT)
            std::cout << "# limbs U: " << U[0][0].getLevel() << " " << U[0][0].NoiseLevel << std::endl;
        if constexpr (PRINT)
            printMatrix(decryptGPUMatrix(U, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                        "Up: ", false);

        if constexpr (TIMING) {
            cudaDeviceSynchronize();
            end_gpu = std::chrono::high_resolution_clock::now();
            std::cout << "PCMM took: "
                      << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                      << std::endl;
            start_gpu = std::chrono::high_resolution_clock::now();
        }
        // ReLU
        EvalGelu_Matrix(U, cc.GetEvalKey(), conf.numSlots);
        if constexpr (PRINT)
            std::cout << "# limbs r: " << U[0][0].getLevel() << " " << U[0][0].NoiseLevel << std::endl;
        if constexpr (PRINT)
            printMatrix(decryptGPUMatrix(U, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                        "GELU: ", false);

        if constexpr (TIMING) {
            cudaDeviceSynchronize();
            end_gpu = std::chrono::high_resolution_clock::now();
            std::cout << "Gelu took: "
                      << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                      << std::endl;
            start_gpu = std::chrono::high_resolution_clock::now();
        }

        // Down PCMM
        dropMatrixLevel(U, conf.level_matmul - 4);
        PCMM_2(U, weights_layer.Wd, conf.blockSize, D, precomp_gpu, weights_layer.bd,
               masks.row_masks[conf.token_length]);
        // PCMM_GPU(U, weights_layer.Wd, conf.blockSize, D, precomp_gpu, weights_layer.bd, masks.row_masks[conf.token_length]);
        if constexpr (PRINT)
            printMatrix(decryptGPUMatrix(D, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                        "Down: ", false);
        if constexpr (PRINT)
            std::cout << "# limbs D: " << D[0][0].getLevel() << " " << D[0][0].NoiseLevel << std::endl;

        if constexpr (TIMING) {
            cudaDeviceSynchronize();
            end_gpu = std::chrono::high_resolution_clock::now();
            std::cout << "PCMM took: "
                      << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                      << std::endl;
            start_gpu = std::chrono::high_resolution_clock::now();
        }
    }

    // Layer Norm 2
    D = MatrixAdd(D, O);
    MatrixBootstrap(D, conf.numSlots);

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(D, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "LNin: ", false);

    EvalLayerNorm_Matrix(D, ct_tokens[0][0], keys_.secretKey, cc.GetEvalKey(), masks.mask_layernorm,
                         masks.row_masks[conf.token_length], weights_layer.Wln2, weights_layer.bln2, conf.numSlots,
                         conf.blockSize, conf.bStepAcc, true);

    if constexpr (PRINT)
        std::cout << "# limbs LN: " << D[0][0].getLevel() << " " << D[0][0].NoiseLevel << std::endl;
    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(D, conf.numSlots);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(D, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "LNout: ", false);
    if constexpr (PRINT)
        std::cout << "# limbs LN: " << D[0][0].getLevel() << " " << D[0][0].NoiseLevel << std::endl;

    if constexpr (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "LN took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                  << " ms." << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    return D;
}

void process_sentences_from_csv(std::string& file_path, std::string& output_file, std::string& model_name,
                                std::string& model_path, std::string& output_path, EncoderConfiguration& base_conf,
                                lbcrypto::PublicKey<lbcrypto::DCRTPoly>& pk, FIDESlib::CKKS::Context& GPUcc,
                                std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& ct_tokens,
                                PtWeights_GPU& weights_layer0, PtWeights_GPU& weights_layer1, PtMasks_GPU& masks,
                                MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
                                TransposePrecomputations_GPU& Tprecomp_gpu,
                                lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& sk, int test_case) {

    std::ifstream file(file_path);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file: " << file_path << std::endl;
        return;
    }

    size_t total_counter = 0;
    size_t correct_counter = 0;

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::string sentence;
        int label = -1;

        if (line.front() == '"') {
            size_t end_quote = line.find("\",");
            if (end_quote == std::string::npos)
                continue;
            sentence = line.substr(1, end_quote - 1);

            size_t label_start = end_quote + 2;
            size_t next_comma = line.find(',', label_start);
            if (next_comma != std::string::npos)
                label = std::stoi(line.substr(label_start, next_comma - label_start));
        } else {
            size_t comma1 = line.find(',');
            if (comma1 == std::string::npos)
                continue;
            sentence = line.substr(0, comma1);

            size_t comma2 = line.find(',', comma1 + 1);
            if (comma2 != std::string::npos)
                label = std::stoi(line.substr(comma1 + 1, comma2 - comma1 - 1));
        }

        if (sentence.empty() || label == -1)
            continue;

        EncoderConfiguration conf = base_conf;
        conf.token_length = tokenizer(sentence, model_name, model_path, output_file);

        std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> ct_tokens_clone;
        ct_tokens_clone.resize(ct_tokens.size());
        for (int i = 0; i < ct_tokens_clone.size(); i++) {
            ct_tokens_clone[i].resize(ct_tokens[i].size());
            for (int j = 0; j < ct_tokens_clone[i].size(); j++) {
                ct_tokens_clone[i][j] = ct_tokens[i][j]->Clone();
            }
        }

        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu;
        encryptMatrixtoGPU(model_path + "/" + output_file, tokens_gpu, pk, GPUcc, conf.numSlots, conf.blockSize,
                           conf.rows, conf.cols, conf.level_matmul);

        // file output
        std::ofstream outFile(output_path, std::ios::app);
        outFile << std::endl << "///////////////////////////////////////" << std::endl;
        outFile << "Input: '" << sentence << "'" << ", " << conf.token_length << std::endl
                << "Label: " << label << std::endl;
        outFile.close();

        // terminal output
        std::cout << std::endl << "///////////////////////////////////////" << std::endl;
        std::cout << "Input: '" << sentence << "'" << ", " << conf.token_length << std::endl
                  << "Label: " << label << std::endl;

        cudaDeviceSynchronize();
        auto start_gpu = std::chrono::high_resolution_clock::now();
        tokens_gpu = encoder_helmet(weights_layer0, precomp_gpu, Tprecomp_gpu, tokens_gpu, masks, conf, 0, test_case);
        tokens_gpu = encoder_helmet(weights_layer1, precomp_gpu, Tprecomp_gpu, tokens_gpu, masks, conf, 1, test_case);
        uint32_t class_pred = classifier(cc, tokens_gpu, sk, ct_tokens_clone, precomp_gpu, weights_layer1, masks,
                                         conf.numSlots, conf.blockSize, conf.token_length, true, output_path);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();

        tokens_gpu.clear();

        total_counter++;
        if (class_pred == static_cast<uint32_t>(label))
            correct_counter++;

        std::ofstream outFile2(output_path, std::ios::app);
        outFile2 << "took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()
                 << " ms." << std::endl;
        outFile2 << "Accuracy: " << correct_counter << "/" << total_counter << std::endl;
        outFile2.close();
        // terminal output
        std::cout << "Accuracy: " << correct_counter << "/" << total_counter << std::endl;
    }
}

static inline void ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !std::isspace(ch); }));
}
static inline void rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(), s.end());
}
static inline void trim(std::string& s) {
    ltrim(s);
    rtrim(s);
}

// data: <sentence ending at first '.'><comma><label><comma><optional id or rest>
void process_sentences_from_csv_cola(std::string& file_path, std::string& output_file, std::string& model_name,
                                     std::string& model_path, std::string& output_path, EncoderConfiguration& base_conf,
                                     lbcrypto::PublicKey<lbcrypto::DCRTPoly>& pk, FIDESlib::CKKS::Context& GPUcc,
                                     std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& ct_tokens,
                                     PtWeights_GPU& weights_layer0, PtWeights_GPU& weights_layer1, PtMasks_GPU& masks,
                                     MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
                                     TransposePrecomputations_GPU& Tprecomp_gpu,
                                     lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                     lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& sk, int test_case) {

    std::ifstream file(file_path);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file: " << file_path << std::endl;
        return;
    }

    size_t total_counter = 0;
    size_t correct_counter = 0;

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::string sentence;
        int label = -1;

        // --- Parse by first '.' as end-of-sentence ---
        // 1) find first '.', include it in the sentence
        size_t dot_pos = line.find('.');
        if (dot_pos == std::string::npos) {
            // No period -> malformed line; skip
            continue;
        }
        sentence = line.substr(0, dot_pos + 1);

        // 2) after the '.', there should be a comma separating the label
        size_t after_dot = dot_pos + 1;
        // skip spaces if any (e.g., ". ,1,260")
        while (after_dot < line.size() && std::isspace(static_cast<unsigned char>(line[after_dot])))
            ++after_dot;

        if (after_dot >= line.size() || line[after_dot] != ',') {
            // Expect a comma right after sentence end (allowing spaces). If not present, skip.
            continue;
        }

        size_t label_start = after_dot + 1;
        // skip spaces before label
        while (label_start < line.size() && std::isspace(static_cast<unsigned char>(line[label_start])))
            ++label_start;

        // label ends at next comma or end of line
        size_t label_end = line.find(',', label_start);
        std::string label_str = (label_end == std::string::npos) ? line.substr(label_start)
                                                                 : line.substr(label_start, label_end - label_start);
        trim(label_str);

        try {
            label = std::stoi(label_str);
        } catch (...) {
            // Non-integer label -> skip
            continue;
        }

        trim(sentence);
        if (sentence.empty() || label == -1)
            continue;

        EncoderConfiguration conf = base_conf;
        conf.token_length = tokenizer(sentence, model_name, model_path, output_file);

        std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> ct_tokens_clone;
        ct_tokens_clone.resize(ct_tokens.size());
        for (int i = 0; i < ct_tokens_clone.size(); i++) {
            ct_tokens_clone[i].resize(ct_tokens[i].size());
            for (int j = 0; j < ct_tokens_clone[i].size(); j++) {
                ct_tokens_clone[i][j] = ct_tokens[i][j]->Clone();
            }
        }

        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu;
        encryptMatrixtoGPU(model_path + "/" + output_file, tokens_gpu, pk, GPUcc, conf.numSlots, conf.blockSize,
                           conf.rows, conf.cols, conf.level_matmul);

        // file output
        std::ofstream outFile(output_path, std::ios::app);
        outFile << std::endl << "///////////////////////////////////////" << std::endl;
        outFile << "Input: '" << sentence << "'" << std::endl << "Label: " << label << std::endl;
        outFile.close();

        // terminal output
        std::cout << std::endl << "///////////////////////////////////////" << std::endl;
        std::cout << "Input: '" << sentence << "'" << std::endl << "Label: " << label << std::endl;

        cudaDeviceSynchronize();
        auto start_gpu = std::chrono::high_resolution_clock::now();
        tokens_gpu = encoder_helmet(weights_layer0, precomp_gpu, Tprecomp_gpu, tokens_gpu, masks, conf, 0, test_case);
        tokens_gpu = encoder_helmet(weights_layer1, precomp_gpu, Tprecomp_gpu, tokens_gpu, masks, conf, 1, test_case);
        uint32_t class_pred = classifier(cc, tokens_gpu, sk, ct_tokens_clone, precomp_gpu, weights_layer1, masks,
                                         conf.numSlots, conf.blockSize, conf.token_length, true, output_path);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();

        tokens_gpu.clear();

        total_counter++;
        if (class_pred == static_cast<uint32_t>(label))
            correct_counter++;

        std::ofstream outFile2(output_path, std::ios::app);
        outFile2 << "took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()
                 << " ms." << std::endl;
        outFile2 << "Accuracy: " << correct_counter << "/" << total_counter << std::endl;
        outFile2.close();
        // terminal output
        std::cout << "Accuracy: " << correct_counter << "/" << total_counter << std::endl;
    }
}

int32_t classifier(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
                   std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& input,
                   lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& privateKey,
                   std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& ct_tokens,
                   const MatrixMatrixProductPrecomputations_GPU& precomp, PtWeights_GPU& weights_layer,
                   PtMasks_GPU& masks, int numSlots, int blockSize, int token_length, bool bts,
                   std::string& output_path) {

    bool constexpr print = false;

    FIDESlib::CKKS::Context& GPUcc = input[0][0].cc;

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> result, result_f;
    PCMM_2(input, weights_layer.Wp, blockSize, result, precomp, weights_layer.bp, masks.row_masks[token_length]);

    if constexpr (print)
        std::cout << "# limbs: " << result[0][0].getLevel() << " " << result[0][0].NoiseLevel << std::endl;
    if (print)
        printMatrix(decryptGPUMatrix(result, keys_.secretKey, ct_tokens, numSlots, blockSize), 2, 2, "PCMM", false);

    evalTanh(result[0][0], GPUcc.GetEvalKey(), numSlots, -20, 20, true);  // -10, 10 -> -20, 20

    if constexpr (print)
        std::cout << "# limbs: " << result[0][0].getLevel() << " " << result[0][0].NoiseLevel << std::endl;
    if (print)
        printMatrix(decryptGPUMatrix(result, keys_.secretKey, ct_tokens, numSlots, blockSize), 2, 2, "Tanh", false);

    FIDESlib::CKKS::Ciphertext result_0(GPUcc), result_1(GPUcc);
    result_0.copy(result[0][0]);
    result_0.multPt(weights_layer.Wc[0][0], false);
    Accumulate(result_0, 4, 1, blockSize);
    result_0.addPt(weights_layer.bc[0][0]);

    FIDESlib::CKKS::RawCipherText raw_res;
    result_0.store(GPUcc, raw_res);
    auto result_gpu0(ct_tokens[0][0]->Clone());
    GetOpenFHECipherText(result_gpu0, raw_res);

    Plaintext weights_rotated(GPUcc), bias_rotated(GPUcc);
    weights_rotated.copy(weights_layer.Wc[0][0]);
    weights_rotated.automorph(blockSize);
    bias_rotated.copy(weights_layer.bc[0][0]);
    bias_rotated.automorph(blockSize);

    result_1.copy(result[0][0]);
    result_1.multPt(weights_rotated, false);
    Accumulate(result_1, 4, 1, blockSize);
    result_1.addPt(bias_rotated);

    FIDESlib::CKKS::RawCipherText raw_res1;
    result_1.store(GPUcc, raw_res1);
    auto result_gpu1(ct_tokens[0][0]->Clone());
    GetOpenFHECipherText(result_gpu1, raw_res1);

    try {
        lbcrypto::Plaintext pt_result_gpu0;
        context->Decrypt(privateKey, result_gpu0, &pt_result_gpu0);
        double result0 = pt_result_gpu0->GetRealPackedValue()[0];

        lbcrypto::Plaintext pt_result_gpu1;
        context->Decrypt(privateKey, result_gpu1, &pt_result_gpu1);
        double result1 = pt_result_gpu1->GetRealPackedValue()[0];

        int yhat = 1;
        if (result0 > result1) {
            yhat = 0;
        }

        std::ofstream outFile(output_path, std::ios::app);
        outFile << "logits: " << result0 << ", " << result1 << std::endl;
        outFile << "Class: " << yhat << std::endl;
        outFile.close();

        // terminal output
        std::cout << "logits: " << result0 << ", " << result1 << std::endl;
        std::cout << "Class: " << yhat << std::endl;
        return yhat;

    } catch (const std::exception& e) {
        // std::cerr << "none. Decryption failed: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // std::cerr << "none. Unknown error occurred during decryption." << std::endl;
        return -1;
    }
    std::cout << std::endl;
}

std::vector<int> GenerateRotationIndices_GPU(int blockSize, int bStep, int bStepAcc, int num_heads) {

    // JKLS MatMul rotation indices
    std::vector<int32_t> rotation_indices_MM = GenerateMatMulRotationIndices_GPU(blockSize, bStep);
    // Multi-head Attention rotation indices
    std::vector<int32_t> rotation_indices_MHA = GenerateMatMulRotationIndices_GPU(blockSize / num_heads, bStep);

    // Transpose rotation indices
    std::vector<int> rotation_indices_T = GenerateTransposeRotationIndices_GPU(blockSize, bStep);

    std::vector<int> rotsum_indices = {
        1,  2,  3,  4,   8,   16,  32,  64,  8192, 0,   -1,  -2,
        -3, -4, -8, -16, -32, -64, 127, -15, -31,  -47, -63, -127};  // 127 is for pooling, -blockSize for Concat

    std::vector<int> accum_indices = FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, 1, blockSize);
    std::vector<int> accum_indices2 = FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, blockSize, blockSize);
    std::vector<int> accum_indices3 = FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, blockSize, blockSize / 2);
    std::vector<int> accum_indices4 = FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, blockSize, blockSize / 4);
    std::vector<int> broad_indices = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize);
    std::vector<int> broad_indices2 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize / 2);
    std::vector<int> broad_indices3 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize / 4);
    std::vector<int> broad_indices4 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize / 8);
    std::vector<int> broad_indices5 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize / 16);  // 4
    std::vector<int> broad_indices6 =
        FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, blockSize, blockSize * blockSize);

    // if (blockSize == 128) rotsum_indices = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    // Merge the rotation indices and remove duplicates

    std::set<int32_t> merged_set(rotsum_indices.begin(), rotsum_indices.end());
    merged_set.insert(rotation_indices_MM.begin(), rotation_indices_MM.end());
    merged_set.insert(rotation_indices_MHA.begin(), rotation_indices_MHA.end());
    merged_set.insert(rotation_indices_T.begin(), rotation_indices_T.end());
    merged_set.insert(accum_indices.begin(), accum_indices.end());
    merged_set.insert(accum_indices2.begin(), accum_indices2.end());
    merged_set.insert(accum_indices3.begin(), accum_indices3.end());
    merged_set.insert(accum_indices4.begin(), accum_indices4.end());
    merged_set.insert(broad_indices.begin(), broad_indices.end());
    merged_set.insert(broad_indices2.begin(), broad_indices2.end());
    merged_set.insert(broad_indices3.begin(), broad_indices3.end());
    merged_set.insert(broad_indices4.begin(), broad_indices4.end());
    merged_set.insert(broad_indices5.begin(), broad_indices5.end());
    merged_set.insert(broad_indices6.begin(), broad_indices6.end());
    std::vector<int32_t> rotation_indices(merged_set.begin(), merged_set.end());

    return rotation_indices;
}

void MatrixBootstrap(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, int numSlots, bool input_prescaled) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            Bootstrap(matrix[i][j], numSlots, input_prescaled);
        }
    }
}

void MatrixSquare(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix,
                  const KeySwitchingKey& keySwitchingKey) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            matrix[i][j].mult(matrix[i][j], matrix[i][j], keySwitchingKey);
        }
    }
}

void MatrixMultScalar(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, double scale) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            matrix[i][j].multScalar(scale);
        }
    }
}

void MatrixAddScalar(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, double value) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            matrix[i][j].addScalar(value);
        }
    }
}

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixAdd(
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix,
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2) {

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> masked_matrix;
    masked_matrix.reserve(matrix.size());
    for (size_t i = 0; i < matrix.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix[0].size());
        for (size_t j = 0; j < matrix[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext masked_ct(matrix[i][j].cc);
            masked_ct.copy(matrix[i][j]);
            masked_ct.add(matrix2[i][j]);
            row.emplace_back(std::move(masked_ct));
        }
        masked_matrix.emplace_back(std::move(row));
    }
    return masked_matrix;
}

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixMaskSpecial(
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, PtMasks_GPU& masks, double scalar) {

    FIDESlib::CKKS::Context& cc = matrix[0][0].cc;
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> masked_matrix;
    masked_matrix.reserve(matrix.size());
    for (size_t i = 0; i < matrix.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix[0].size());
        for (size_t j = 0; j < matrix[0].size(); j++) {
            FIDESlib::CKKS::Plaintext mask_temp(cc);
            mask_temp.copy(masks.mask_max);      // 1s
            mask_temp.multScalar(scalar, true);  // Subtract except 1st zero

            FIDESlib::CKKS::Ciphertext masked_ct(cc);
            masked_ct.copy(matrix[i][j]);
            masked_ct.addPt(mask_temp);
            row.emplace_back(std::move(masked_ct));
        }
        masked_matrix.emplace_back(std::move(row));
    }
    return masked_matrix;
}

void MatrixRotate(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, int index) {
    FIDESlib::CKKS::Context& cc = matrix[0][0].cc;
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            matrix[i][j].rotate(index, cc.GetRotationKey(index));
        }
    }
}

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixMask(
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, FIDESlib::CKKS::Plaintext& mask) {

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> masked_matrix;
    masked_matrix.reserve(matrix.size());
    for (size_t i = 0; i < matrix.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix[0].size());
        for (size_t j = 0; j < matrix[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext masked_ct(matrix[i][j].cc);
            masked_ct.copy(matrix[i][j]);
            masked_ct.multPt(mask);
            // masked_ct.rescale();
            row.emplace_back(std::move(masked_ct));
        }
        masked_matrix.emplace_back(std::move(row));
    }
    return masked_matrix;
}

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixConcat(
    std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>>& matrices,
    std::vector<FIDESlib::CKKS::Plaintext>& masks, int blockSize) {

    if (matrices.empty() || masks.size() != matrices.size())
        throw std::invalid_argument("Matrix and mask size mismatch or empty input.");

    const size_t numRows = matrices[0].size();
    const size_t numCols = matrices[0][0].size();

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> result;
    result.reserve(numRows);

    for (size_t i = 0; i < numRows; ++i) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(numCols);

        for (size_t j = 0; j < numCols; ++j) {
            // Initialize with a copy of the first masked ciphertext
            FIDESlib::CKKS::Ciphertext sum(matrices[0][i][j].cc);
            sum.copy(matrices[0][i][j]);
            sum.multPt(masks[0]);

            // Accumulate masked ciphertexts from remaining matrices
            for (size_t k = 1; k < matrices.size(); ++k) {
                FIDESlib::CKKS::Ciphertext tmp(matrices[k][i][j].cc);
                tmp.copy(matrices[k][i][j]);
                tmp.multPt(masks[k]);
                sum.add(tmp);
            }

            row.emplace_back(std::move(sum));
        }
        result.emplace_back(std::move(row));
    }
    return result;
}

void MatrixMaskCT(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix,
                  std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& mask, const KeySwitchingKey& keySwitchingKey) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            matrix[i][j].mult(matrix[i][j], mask[i][j], keySwitchingKey);
        }
    }
}

void dropMatrixLevel(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& in, int level) {
    for (auto& row : in)
        for (auto& ct : row) {
            if (ct.NoiseLevel == 2)
                ct.rescale();
            if (ct.getLevel() > level) {
                ct.dropToLevel(level);
                assert(ct.getLevel() == level);
            }
        }
}

int tokenizer(const std::string& sentence, const std::string& model_name, const std::string& model_path,
              const std::string& output_filename) {
    std::string script_path = "../src/python/";
    std::string script;

    if (output_filename == "tokens1.txt")
        script = "ExtractEmbeddings1.py";
    else if (output_filename == "tokens2.txt")
        script = "ExtractEmbeddings2.py";
    else if (output_filename == "tokens3.txt")
        script = "ExtractEmbeddings3.py";
    else if (output_filename == "tokens4.txt")
        script = "ExtractEmbeddings4.py";
    else if (output_filename == "tokens5.txt")
        script = "ExtractEmbeddings5.py";

    else if (output_filename == "tokens_test1.txt")
        script = "ExtractEmbeddings_test1.py";
    else if (output_filename == "tokens_test2.txt")
        script = "ExtractEmbeddings_test2.py";
    else if (output_filename == "tokens_test3.txt")
        script = "ExtractEmbeddings_test3.py";
    else if (output_filename == "tokens_test4.txt")
        script = "ExtractEmbeddings_test4.py";
    else if (output_filename == "tokens_test5.txt")
        script = "ExtractEmbeddings_test5.py";

    else if (output_filename == "tokens_cola1.txt")
        script = "ExtractEmbeddings_cola1.py";
    else if (output_filename == "tokens_cola2.txt")
        script = "ExtractEmbeddings_cola2.py";
    else if (output_filename == "tokens_cola3.txt")
        script = "ExtractEmbeddings_cola3.py";
    else if (output_filename == "tokens_cola4.txt")
        script = "ExtractEmbeddings_cola4.py";

    else if (output_filename == "tokens_mrpc1.txt")
        script = "ExtractEmbeddings_mrpc1.py";
    else if (output_filename == "tokens_mrpc2.txt")
        script = "ExtractEmbeddings_mrpc2.py";
    else if (output_filename == "tokens_mrpc3.txt")
        script = "ExtractEmbeddings_mrpc3.py";
    else if (output_filename == "tokens_mrpc4.txt")
        script = "ExtractEmbeddings_mrpc4.py";

    else if (output_filename == "tokens_rte1.txt")
        script = "ExtractEmbeddings_rte1.py";
    else if (output_filename == "tokens_rte2.txt")
        script = "ExtractEmbeddings_rte2.py";
    else if (output_filename == "tokens_rte3.txt")
        script = "ExtractEmbeddings_rte3.py";
    else if (output_filename == "tokens_rte4.txt")
        script = "ExtractEmbeddings_rte4.py";

    else
        script = "ExtractEmbeddings.py";

    std::string cmd = "python3 " + script_path + script + " \"" + sentence + "\" \"" + model_name + "\" \"" +
                      model_path + "\" \"" + output_filename + "\"";

    std::array<char, 128> buffer;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe)
        throw std::runtime_error("popen() failed to run the tokenizer script");

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }

    int exitCode = pclose(pipe);
    if (exitCode != 0) {
        throw std::runtime_error("Tokenizer script failed with exit code: " + std::to_string(exitCode));
    }

    try {
        return std::stoi(result);
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] Failed to parse token count from script output: \"" << result << "\"" << std::endl;
        return 0;
    }
}

// New: escape double-quotes to keep popen() safe enough for our usage.
static std::string shell_escape_quotes(std::string s) {
    std::string t;
    t.reserve(s.size());
    for (char c : s)
        t += (c == '"' ? "\\\"" : std::string(1, c));
    return t;
}

int tokenizer_pair(const std::string& sentence1, const std::string& sentence2, const std::string& model_name,
                   const std::string& output_path, const std::string& output_filename) {
    const std::string script_path = "../src/python/";
    const std::string script = "ExtractEmbeddings_pair.py";  // new script below

    const std::string s1 = shell_escape_quotes(sentence1);
    const std::string s2 = shell_escape_quotes(sentence2);
    const std::string m = shell_escape_quotes(model_name);
    const std::string op = shell_escape_quotes(output_path);
    const std::string of = shell_escape_quotes(output_filename);

    const std::string cmd = "python3 " + script_path + script + " \"" + s1 + "\" \"" + s2 + "\" \"" + m + "\" \"" + op +
                            "\" \"" + of + "\"";

    std::array<char, 128> buffer{};
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe)
        throw std::runtime_error("popen() failed to run the tokenizer_pair script");

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    int exitCode = pclose(pipe);
    if (exitCode != 0) {
        throw std::runtime_error("tokenizer_pair script failed, exit code: " + std::to_string(exitCode));
    }

    try {
        return std::stoi(result);  // returns seq_len (including special tokens)
    } catch (...) {
        std::cerr << "[WARNING] Failed to parse token count from script output: \"" << result << "\"\n";
        return 0;
    }
}
size_t CountNumTokens(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        return 0;
    }

    size_t count = 0;
    std::string line;
    while (std::getline(file, line)) {
        // Trim leading/trailing whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        if (!line.empty()) {
            ++count;
        }
    }

    file.close();
    return count;
}

// TOKENIZER HELPERS

// --- helpers: trim / strip ---
static inline void lstrip(std::string& s) {
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
        ++i;
    if (i)
        s.erase(0, i);
}
static inline void rstrip(std::string& s) {
    size_t i = s.size();
    while (i > 0 && std::isspace(static_cast<unsigned char>(s[i - 1])))
        --i;
    if (i < s.size())
        s.erase(i);
}
static inline void strip(std::string& s) {
    rstrip(s);
    lstrip(s);
}

// --- minimal RFC4180 CSV row parser (returns columns) ---
static bool parse_csv_row(const std::string& line, std::vector<std::string>& outCols) {
    outCols.clear();
    std::string field;
    bool inQuotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (inQuotes) {
            if (c == '"') {
                // doubled quote -> literal quote
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    field.push_back('"');
                    ++i;
                } else {
                    inQuotes = false;
                }
            } else {
                field.push_back(c);
            }
        } else {
            if (c == ',') {
                outCols.push_back(field);
                field.clear();
            } else if (c == '"') {
                // opening quote only valid at field start
                if (field.empty())
                    inQuotes = true;
                else
                    field.push_back('"');  // treat as literal if mid-field
            } else {
                field.push_back(c);
            }
        }
    }
    outCols.push_back(field);
    return true;
}
// Collapses spaces, fixes tokenization artifacts, and cleans quotes at boundaries.
static void normalize_tokenized_punct(std::string& s) {
    auto is_space = [](char c) {
        return std::isspace(static_cast<unsigned char>(c));
    };

    // 0) Normalize newlines/tabs to single spaces
    for (char& c : s)
        if (c == '\t' || c == '\n' || c == '\r')
            c = ' ';

    // 1) Collapse consecutive spaces
    {
        std::string t;
        t.reserve(s.size());
        bool prevSpace = false;
        for (char c : s) {
            bool sp = is_space(c);
            if (!(sp && prevSpace))
                t.push_back(sp ? ' ' : c);
            prevSpace = sp;
        }
        s.swap(t);
    }

    // 2) Trim outer spaces first
    auto ltrim = [&](std::string& x) {
        size_t i = 0;
        while (i < x.size() && is_space(x[i]))
            ++i;
        if (i)
            x.erase(0, i);
    };
    auto rtrim = [&](std::string& x) {
        size_t i = x.size();
        while (i > 0 && is_space(x[i - 1]))
            --i;
        if (i < x.size())
            x.erase(i);
    };
    ltrim(s);
    rtrim(s);

    // 3) Drop *leading* noise: quotes, commas, spaces (handles: "" The ... ,  ,  " word)
    while (!s.empty() && (s.front() == '"' || s.front() == '\'' || s.front() == ',' || is_space(s.front())))
        s.erase(s.begin());
    // 4) Drop *trailing* noise: quotes, commas, spaces
    while (!s.empty() && (s.back() == '"' || s.back() == '\'' || s.back() == ',' || is_space(s.back())))
        s.pop_back();

    // 5) Remove spaces immediately BEFORE punctuation: "word ,", "word .", etc.
    {
        std::string t;
        t.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == ' ' && i + 1 < s.size()) {
                char p = s[i + 1];
                if (p == ',' || p == '.' || p == '!' || p == '?' || p == ';' || p == ':')
                    continue;  // skip this space
            }
            t.push_back(s[i]);
        }
        s.swap(t);
    }

    // 6) Remove spaces around apostrophes: "doesn ' t" -> "doesn't"
    {
        std::string t;
        t.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == ' ' && ((i + 1 < s.size() && s[i + 1] == '\'') || (i > 0 && s[i - 1] == '\''))) {
                continue;  // drop space adjacent to apostrophe
            }
            t.push_back(s[i]);
        }
        s.swap(t);
    }

    // 7) Tighten spaces right *inside* quotes:
    //    "\" word"  -> "\"word"
    //    "word \""  -> "word\""
    {
        // after opening quote
        std::string t;
        t.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            t.push_back(s[i]);
            if ((s[i] == '"' || s[i] == '\'') && i + 1 < s.size() && s[i + 1] == ' ') {
                // skip all spaces immediately after quote
                size_t j = i + 1;
                while (j < s.size() && s[j] == ' ')
                    ++j;
                if (j < s.size())
                    t.push_back(s[j]);
                i = j;
            }
        }
        s.swap(t);

        // before closing quote
        std::string u;
        u.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == ' ' && i + 1 < s.size() && (s[i + 1] == '"' || s[i + 1] == '\''))
                continue;  // drop space
            u.push_back(s[i]);
        }
        s.swap(u);
    }

    // 8) Final trim (in case step 3/4 created new edges)
    ltrim(s);
    rtrim(s);
}

void process_sentences_from_csv_mrpc(std::string& file_path, std::string& output_file, std::string& model_name,
                                     std::string& model_path, std::string& output_path, EncoderConfiguration& base_conf,
                                     lbcrypto::PublicKey<lbcrypto::DCRTPoly>& pk, FIDESlib::CKKS::Context& GPUcc,
                                     std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& ct_tokens,
                                     PtWeights_GPU& weights_layer0, PtWeights_GPU& weights_layer1, PtMasks_GPU& masks,
                                     MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
                                     TransposePrecomputations_GPU& Tprecomp_gpu,
                                     lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                     lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& sk, int test_case) {

    std::ifstream file(file_path);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file: " << file_path << std::endl;
        return;
    }

    size_t total_counter = 0;
    size_t correct_counter = 0;

    std::vector<std::string> cols;

    while (std::getline(file, line)) {
        rstrip(line);
        if (line.empty())
            continue;

        if (!parse_csv_row(line, cols))
            continue;
        if (cols.size() < 4)
            continue;

        std::string s1 = cols[0];
        std::string s2 = cols[1];
        std::string label_str = cols[2];
        std::string idx_str = cols[3];

        strip(s1);
        strip(s2);
        strip(label_str);
        strip(idx_str);
        normalize_tokenized_punct(s1);
        normalize_tokenized_punct(s2);

        int label;
        try {
            label = std::stoi(label_str);
        } catch (...) {
            continue;
        }

        // **pair** tokenization for MRPC!
        EncoderConfiguration conf = base_conf;
        conf.token_length = tokenizer_pair(s1, s2, model_name, model_path, output_file);

        std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> ct_tokens_clone;
        ct_tokens_clone.resize(ct_tokens.size());
        for (int i = 0; i < ct_tokens_clone.size(); i++) {
            ct_tokens_clone[i].resize(ct_tokens[i].size());
            for (int j = 0; j < ct_tokens_clone[i].size(); j++) {
                ct_tokens_clone[i][j] = ct_tokens[i][j]->Clone();
            }
        }

        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu;
        int tokens_size = 1 << static_cast<int>(std::ceil(std::log2(conf.token_length)));
        encryptMatrixtoGPU(model_path + "/" + output_file, tokens_gpu, pk, GPUcc, conf.numSlots, conf.blockSize,
                           tokens_size, conf.cols, conf.level_matmul);

        // file output
        {
            std::ofstream outFile(output_path, std::ios::app);
            if (!outFile) {
                std::cerr << "[ERROR] Could not open file for writing: " << output_path << "\n";
            } else {
                outFile << "\n///////////////////////////////////////\n";
                outFile << "Sentence1: \"" << s1 << "\"\n";
                outFile << "Sentence2: \"" << s2 << "\"\n";
                outFile << "Label: " << label << "\n";
                outFile << "Length: " << conf.token_length << "\n";
            }
        }

        // terminal output
        std::cout << "\n///////////////////////////////////////\n";
        std::cout << "Sentence1: \"" << s1 << "\"\n";
        std::cout << "Sentence2: \"" << s2 << "\"\n";
        std::cout << "Label: " << label << "\n";

        cudaDeviceSynchronize();
        auto start_gpu = std::chrono::high_resolution_clock::now();
        tokens_gpu = encoder_helmet(weights_layer0, precomp_gpu, Tprecomp_gpu, tokens_gpu, masks, conf, 0, test_case);
        tokens_gpu = encoder_helmet(weights_layer1, precomp_gpu, Tprecomp_gpu, tokens_gpu, masks, conf, 1, test_case);
        uint32_t class_pred = classifier(cc, tokens_gpu, sk, ct_tokens_clone, precomp_gpu, weights_layer1, masks,
                                         conf.numSlots, conf.blockSize, conf.token_length, true, output_path);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();

        tokens_gpu.clear();

        total_counter++;
        if (class_pred == static_cast<uint32_t>(label))
            correct_counter++;

        std::ofstream outFile2(output_path, std::ios::app);
        outFile2 << "took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()
                 << " ms." << std::endl;
        outFile2 << "Accuracy: " << correct_counter << "/" << total_counter << std::endl;
        outFile2.close();
        // terminal output
        std::cout << "Accuracy: " << correct_counter << "/" << total_counter << std::endl;
    }
}

// int tokenizer_mrpc(const std::string& sentence, const std::string& model_name,
//             const std::string& model_path, const std::string& output_filename) {
//     std::string script_path = "/projectnb/he/seyda/FIDESlib/src/python/";
//     std::string script = "ExtractEmbeddings_mrpc.py";

//     std::string cmd = "python3 " + script_path + script + " \"" +
//                     sentence + "\" \"" + model_name + "\" \"" + model_path + "\" \"" +
//                     output_filename + "\"";

//     std::array<char, 128> buffer;
//     std::string result;
//     FILE* pipe = popen(cmd.c_str(), "r");
//     if (!pipe)
//         throw std::runtime_error("popen() failed to run the tokenizer script");

//     while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
//         result += buffer.data();
//     }

//     int exitCode = pclose(pipe);
//     if (exitCode != 0) {
//         throw std::runtime_error("Tokenizer script failed with exit code: " + std::to_string(exitCode));
//     }

//     try {
//         return std::stoi(result);
//     } catch (const std::exception& e) {
//         std::cerr << "[WARNING] Failed to parse token count from script output: \"" << result << "\"" << std::endl;
//         return 0;
//     }
// }
}  // namespace FIDESlib::CKKS
