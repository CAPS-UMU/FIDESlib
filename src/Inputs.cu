#include "Inputs.cuh"

namespace FIDESlib::CKKS {

    struct PtWeights_GPU GetPtWeightsGPU(FIDESlib::CKKS::Context& GPUcc, lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
                                        const std::string& model_path, int layerNo, int numSlots, int blockSize,
                                        int rows, int cols, const int level, int num_heads) {

        PtWeights_GPU pt_weights_gpu;

        std::string path = std::string(model_path + "/layer") + std::to_string(layerNo);

        encodeMatrixtoGPU(path + "_Wk.txt", pt_weights_gpu.Wk, publicKey, GPUcc, numSlots, blockSize, rows, cols, level , false);
        encodeMatrixtoGPU(path + "_Wq.txt", pt_weights_gpu.Wq, publicKey, GPUcc, numSlots, blockSize, rows, cols, level , false);
        encodeMatrixtoGPU(path + "_Wv.txt", pt_weights_gpu.Wv, publicKey, GPUcc, numSlots, blockSize, rows, cols, level , false);
        encodeMatrixtoGPU(path + "_bk.txt", pt_weights_gpu.bk, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);
        encodeMatrixtoGPU(path + "_bq.txt", pt_weights_gpu.bq, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);
        encodeMatrixtoGPU(path + "_bv.txt", pt_weights_gpu.bv, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);

        encodeMatrixtoGPU(path + "_Wo.txt", pt_weights_gpu.Wo, publicKey, GPUcc, numSlots, blockSize, rows, cols, level , false);
        encodeMatrixtoGPU(path + "_Wu.txt", pt_weights_gpu.Wu, publicKey, GPUcc, numSlots, blockSize, rows, cols*4, level , false);
        encodeMatrixtoGPU(path + "_Wd.txt", pt_weights_gpu.Wd, publicKey, GPUcc, numSlots, blockSize, rows*4, cols, level , false);
        encodeMatrixtoGPU(path + "_bo.txt", pt_weights_gpu.bo, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);
        encodeMatrixtoGPU(path + "_bu.txt", pt_weights_gpu.bu, publicKey, GPUcc, numSlots, blockSize, rows, cols*4, level, true);
        encodeMatrixtoGPU(path + "_bd.txt", pt_weights_gpu.bd, publicKey, GPUcc, numSlots, blockSize, rows*4, cols, level, true);

        encodeMatrixtoGPU(path + "_Wln1.txt", pt_weights_gpu.Wln1, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);
        encodeMatrixtoGPU(path + "_bln1.txt", pt_weights_gpu.bln1, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);
        encodeMatrixtoGPU(path + "_Wln2.txt", pt_weights_gpu.Wln2, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);
        encodeMatrixtoGPU(path + "_bln2.txt", pt_weights_gpu.bln2, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);

        if (layerNo == 1){
            encodeMatrixtoGPU(model_path + "/Wp.txt", pt_weights_gpu.Wp, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, false);
            encodeMatrixtoGPU(model_path + "/bp.txt", pt_weights_gpu.bp, publicKey, GPUcc, numSlots, blockSize, rows, cols, level - 1, true);
            encodeMatrixtoGPU(model_path + "/Wc.txt", pt_weights_gpu.Wc, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);
            encodeMatrixtoGPU(model_path + "/bc.txt", pt_weights_gpu.bc, publicKey, GPUcc, numSlots, blockSize, rows, cols, level - 1, true);
        }
        return pt_weights_gpu;
    }

    PtMasks_GPU GetPtMasks_GPU(FIDESlib::CKKS::Context& cc,
                            lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
                            int numSlots, int blockSize, int token_length, int level) {

        // Token masks in Softmax
        std::vector<FIDESlib::CKKS::Plaintext> token_masks;
        token_masks.reserve(blockSize);

        for (int idx = 0; idx < blockSize; ++idx) {
            std::vector<double> token_mask = CreateBlockMask(numSlots, blockSize, idx, 1);

            auto raw_pt = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(token_mask));
            FIDESlib::CKKS::Plaintext pt(cc, raw_pt);

            token_masks.emplace_back(std::move(pt));
        }
        // Broadcast masks
        std::vector<double> mask(numSlots, 0.0);
        // double scale = 2.0 / (1e4 - 1);  // softmax 1/x poly approx boundaries
        double scale = 1;
        for (int i = 0; i < numSlots; i += blockSize/2)
            mask[i] = 1.0 * scale;
        auto raw_pt = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask, 1, cc.L - level));
        FIDESlib::CKKS::Plaintext GPU_broad_mask(cc, raw_pt);

        std::vector<double> mask2(numSlots, 0.0);
        for (int i = 0; i < numSlots; i += blockSize)
            mask2[i] = 1.0;
        raw_pt = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask2, 1, cc.L - level));
        FIDESlib::CKKS::Plaintext GPU_broad_mask2(cc, raw_pt);

        // Sm_max masks
        std::vector<double> mask_max(numSlots, 1.0);
        for (int i = 0; i < blockSize; i ++)
            mask_max[i] = 0;
        raw_pt = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask_max, 1, cc.L - level));
        FIDESlib::CKKS::Plaintext GPU_mask_max(cc, raw_pt);

        // LayerNorm masks
        std::vector<FIDESlib::CKKS::Plaintext> ln_masks;
        ln_masks.reserve(2);

        std::vector<double> mask_ln(numSlots, 0.0);
        for (int i = 0; i < numSlots; i += blockSize)
            mask_ln[i] = 1.0 / blockSize;
        auto raw_ln = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask_ln, 1, cc.L - level));
        FIDESlib::CKKS::Plaintext GPU_ln_mask(cc, raw_ln);
        ln_masks.emplace_back(std::move(GPU_ln_mask));

        std::vector<double> mask_ln2(numSlots, 0.0);
        scale = 2.0 / (1e2 - 0.01);  // ln 1/sqrt(x) poly approx boundaries
        // scale = 1;
        for (int i = 0; i < numSlots; i += blockSize)
            mask_ln2[i] = 1.0 / blockSize * scale;
        auto raw_ln2 = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask_ln2, 1, cc.L - level));
        FIDESlib::CKKS::Plaintext GPU_ln_mask2(cc, raw_ln2);
        ln_masks.emplace_back(std::move(GPU_ln_mask2));

        // Head masks
        std::vector<FIDESlib::CKKS::Plaintext> head_masks;
        head_masks.reserve(2);
        for (int h = 0; h < 2; ++h) {
            auto mask_head = CreateHeadMask(numSlots, blockSize, h);
            auto raw = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask_head, 1, cc.L - level - 1));
            head_masks.emplace_back(cc, raw);
        }

        // Head masks - transpose
        std::vector<FIDESlib::CKKS::Plaintext> head_masks_T;
        head_masks_T.reserve(2);
        std::vector<double> mask_head1_T(numSlots, 0.0);
        for (int i = 0; i < blockSize / 2; i += 1) {
            for (int j = 0; j < blockSize; j += 1) {
                mask_head1_T[i * blockSize + j] = 1.0;
            }
        }
        auto raw = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask_head1_T, 1, cc.L - level));
        head_masks_T.emplace_back(cc, raw);
        std::vector<double> mask_head2_T(numSlots, 1.0);
        for (int i = 0; i < blockSize / 2; i += 1) {
            for (int j = 0; j < blockSize; j += 1) {
                mask_head2_T[i * blockSize + j] = 0.0;
            }
        }
        raw = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask_head2_T, 1, cc.L - level));
        head_masks_T.emplace_back(cc, raw);

        // Row masks for PCMM biases
        std::vector<FIDESlib::CKKS::Plaintext> row_masks;
        row_masks.reserve(blockSize);

        for (int idx = 0; idx < blockSize; ++idx) {
            std::vector<double> mask_row(numSlots, 0.0);
            for (int i = 0; i < idx; i += 1) {
                for (int j = 0; j < blockSize; j += 1) {
                    mask_row[i * blockSize + j] = 1.0;
                }
            }

            auto raw_pt = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask_row, 1, cc.L - level - 1));
            FIDESlib::CKKS::Plaintext pt(cc, raw_pt);

            row_masks.emplace_back(std::move(pt));
        }


        // Row masks for PCMM_U
        std::vector<FIDESlib::CKKS::Plaintext> U_masks;
        U_masks.reserve(64);

        for (int idx = 0; idx < 64; ++idx) {
            std::vector<double> mask_U(numSlots, 0.0);

            for (int j = 0; j < blockSize; j += 1) {
                mask_U[idx * blockSize + j] = 1.0;
            }

            auto raw_pt = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask_U, 1, cc.L - level - 2));
            FIDESlib::CKKS::Plaintext pt(cc, raw_pt);

            U_masks.emplace_back(std::move(pt));
        }

        // PCMM masks
        std::vector<FIDESlib::CKKS::Plaintext> pcmm_masks;
        pcmm_masks.reserve(blockSize);
        for (int h = 0; h < blockSize; h++) {
            std::vector<double> mask_pcmm(numSlots, 0.0);
            int rowSize = numSlots / blockSize;
            for (int i = 0; i < rowSize; i++) {
                mask_pcmm[h * rowSize + i] = 1.0;
            }
            auto raw = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask_pcmm, 1, cc.L - level));
            pcmm_masks.emplace_back(cc, raw);
        }

        return PtMasks_GPU(
            std::move(token_masks),
            std::move(GPU_broad_mask),
            std::move(GPU_broad_mask2),
            std::move(GPU_mask_max),
            std::move(ln_masks),
            std::move(head_masks),
            std::move(head_masks_T),
            std::move(pcmm_masks),
            std::move(row_masks),
            std::move(U_masks)
        );
    }


    std::vector<std::string> read_sentences_from_csv(const std::string& file_path) {
        std::ifstream file(file_path);
        std::vector<std::string> sentences;
        std::string line;

        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open file: " << file_path << std::endl;
            return sentences;
        }

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::string sentence;
            int label = -1;

            if (line.front() == '"') {
                // Quoted sentence
                size_t end_quote = line.find("\",");
                if (end_quote == std::string::npos) {
                    std::cerr << "Malformed line (unclosed quote): " << line << std::endl;
                    continue;
                }

                sentence = line.substr(1, end_quote - 1);

                size_t label_start = end_quote + 2;
                size_t next_comma = line.find(',', label_start);
                if (next_comma != std::string::npos) {
                    try {
                        label = std::stoi(line.substr(label_start, next_comma - label_start));
                    } catch (...) {
                        std::cerr << "Invalid label in line: " << line << std::endl;
                        continue;
                    }
                }

            } else {
                // Unquoted sentence
                size_t comma1 = line.find(',');
                if (comma1 == std::string::npos) continue;

                sentence = line.substr(0, comma1);

                size_t comma2 = line.find(',', comma1 + 1);
                if (comma2 != std::string::npos) {
                    try {
                        label = std::stoi(line.substr(comma1 + 1, comma2 - comma1 - 1));
                    } catch (...) {
                        std::cerr << "Invalid label in line: " << line << std::endl;
                        continue;
                    }
                }
            }

            if (!sentence.empty() && label != -1) {
                std::cout << "Sentence: \"" << sentence <<  std::endl << "\"Label: " << label << std::endl;
                sentences.push_back(sentence);
            }
        }

        return sentences;
    }


    struct Weights_GPU GetWeightsGPU(FIDESlib::CKKS::Context& GPUcc, lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
                                    const std::string& model_path, int layerNo, int numSlots, int blockSize,
                                    int rows, int cols, const int level) {

        Weights_GPU weights_gpu;

        std::string path = std::string(model_path + "/layer") + std::to_string(layerNo);

        encryptMatrixtoGPU(path + "_Wk.txt", weights_gpu.Wk, publicKey, GPUcc, numSlots, blockSize, rows, cols, level - 2, false);
        encryptMatrixtoGPU(path + "_Wq.txt", weights_gpu.Wq, publicKey, GPUcc, numSlots, blockSize, rows, cols, level - 2, false);
        encryptMatrixtoGPU(path + "_Wv.txt", weights_gpu.Wv, publicKey, GPUcc, numSlots, blockSize, rows, cols, level - 2, false);
        encryptMatrixtoGPU(path + "_bk.txt", weights_gpu.bk, publicKey, GPUcc, numSlots, blockSize, rows, cols, level - 3, true);
        encryptMatrixtoGPU(path + "_bq.txt", weights_gpu.bq, publicKey, GPUcc, numSlots, blockSize, rows, cols, level - 3, true);
        encryptMatrixtoGPU(path + "_bv.txt", weights_gpu.bv, publicKey, GPUcc, numSlots, blockSize, rows, cols, level - 3, true);

        encryptMatrixtoGPU(path + "_Wo.txt", weights_gpu.Wo, publicKey, GPUcc, numSlots, blockSize, rows, cols, level - 2, false);
        encryptMatrixtoGPU(path + "_Wu.txt", weights_gpu.Wu, publicKey, GPUcc, numSlots, blockSize, rows, cols*4, level - 2, false);
        encryptMatrixtoGPU(path + "_Wd.txt", weights_gpu.Wd, publicKey, GPUcc, numSlots, blockSize, rows*4, cols, level - 2, false);
        encryptMatrixtoGPU(path + "_bo.txt", weights_gpu.bo, publicKey, GPUcc, numSlots, blockSize, rows, cols, level - 3, true);
        encryptMatrixtoGPU(path + "_bu.txt", weights_gpu.bu, publicKey, GPUcc, numSlots, blockSize, rows, cols*4, level - 3, true);
        encryptMatrixtoGPU(path + "_bd.txt", weights_gpu.bd, publicKey, GPUcc, numSlots, blockSize, rows*4, cols, level - 3, true);

        encryptMatrixtoGPU(path + "_Wln1.txt", weights_gpu.Wln1, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);
        encryptMatrixtoGPU(path + "_bln1.txt", weights_gpu.bln1, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);
        encryptMatrixtoGPU(path + "_Wln2.txt", weights_gpu.Wln2, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);
        encryptMatrixtoGPU(path + "_bln2.txt", weights_gpu.bln2, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, true);

        if (layerNo == 1){
            encryptMatrixtoGPU(model_path + "/Wp.txt", weights_gpu.Wp, publicKey, GPUcc, numSlots, blockSize, rows, cols, level-2, false);
            encryptMatrixtoGPU(model_path + "/bp.txt", weights_gpu.bp, publicKey, GPUcc, numSlots, blockSize, rows, cols, level-3, true);
            encryptMatrixtoGPU(model_path + "/Wc.txt", weights_gpu.Wc, publicKey, GPUcc, numSlots, blockSize, rows, cols, level, false);
            encryptMatrixtoGPU(model_path + "/bc.txt", weights_gpu.bc, publicKey, GPUcc, numSlots, blockSize, rows, cols, level-1, false);
        }
        return weights_gpu;
    }

    std::vector<std::vector<lbcrypto::Plaintext>> EncodeMatrix(const std::vector<std::vector<std::vector<double>>>& matrix,
                                                            lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey,
                                                            int level) {

        std::vector<std::vector<lbcrypto::Plaintext>> ptMatrix(matrix.size());
        auto cc = publicKey->GetCryptoContext();
        for (size_t i = 0; i < matrix.size(); i++) {
            for (size_t j = 0; j < matrix[0].size(); j++) {
                lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(matrix[i][j], 1, level);
                ptMatrix[i].emplace_back(ptxt1);
            }
        }
        return ptMatrix;
    }

    std::vector<double> getPCMM_bMatrix(std::vector<double> weights, int rowSize) {
        int slots = weights.size();
        std::vector<double> data(slots, 0.0);
        for (int j = 0; j < rowSize; ++j) {
            for (int i = 0; i < slots / rowSize; ++i) {
                data[rowSize * i + j] = weights[(rowSize * (slots / rowSize + i + j) + j) % slots];
            }
        }
        return data;
    }

    void encodeMatrixtoGPU(const std::string& filename, std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& pt_inputs_gpu,
                        lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, FIDESlib::CKKS::Context& GPUcc, int numSlots,
                        int blockSize, size_t rows, size_t cols, int level, bool if_repeat) {

        auto cc = publicKey->GetCryptoContext();

        std::vector<std::vector<double>> inputs;
        if (if_repeat) {
            load_bias(filename, inputs, rows, cols);
        } else {
            load_weights(filename, inputs, rows, cols);
        }

        std::vector<std::vector<std::vector<double>>> inputs_temp = extractAndLinearizeMatrix(inputs, numSlots, blockSize);

        if (!if_repeat) {
            for (auto& i : inputs_temp)
                for (auto& j : i)
                    j = getPCMM_bMatrix(j, blockSize);
        }
        auto pt_inputs = EncodeMatrix(inputs_temp, publicKey, GPUcc.L - level);

        pt_inputs_gpu.resize(pt_inputs.size());
        for (size_t i = 0; i < pt_inputs.size(); ++i) {
            pt_inputs_gpu[i].reserve(pt_inputs[0].size());
            for (size_t j = 0; j < pt_inputs[0].size(); ++j) {
                auto raw_pt = FIDESlib::CKKS::GetRawPlainText(cc, pt_inputs[i][j]);
                FIDESlib::CKKS::Plaintext GPUpt1(GPUcc, raw_pt);
                pt_inputs_gpu[i].emplace_back(std::move(GPUpt1));
            }
        }
    }

    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> encryptMatrixtoCPU(
        const std::string& filename, lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int numSlots, int blockSize,
        size_t rows, size_t cols, bool if_repeat) {

        auto cc = publicKey->GetCryptoContext();

        std::vector<std::vector<double>> inputs;
        if (if_repeat) {
            load_bias(filename, inputs, rows, cols);
        } else {
            load_weights(filename, inputs, rows, cols);
        }

        auto inputs_temp = extractAndLinearizeMatrix(inputs, numSlots, blockSize);
        auto inputs_cpu = EncryptMatrix(inputs_temp, publicKey, 0);

        return inputs_cpu;
    }

    void encryptMatrixtoGPU(const std::string& filename, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& inputs_gpu,
                            lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, FIDESlib::CKKS::Context& GPUcc,
                            int numSlots, int blockSize, size_t rows, size_t cols, int level, bool if_repeat) {

        auto cc = publicKey->GetCryptoContext();

        std::vector<std::vector<double>> inputs;
        if (if_repeat) {
            load_bias(filename, inputs, rows, cols);
        } else {
            load_weights(filename, inputs, rows, cols);
        }

        auto inputs_temp = extractAndLinearizeMatrix(inputs, numSlots, blockSize);
        auto ct_inputs = EncryptMatrix(inputs_temp, publicKey, GPUcc.L - level);
        // auto ct_inputs = EncryptMatrix(inputs_temp, publicKey, level);

        inputs_gpu.resize(ct_inputs.size());
        for (size_t i = 0; i < ct_inputs.size(); ++i) {
            inputs_gpu[i].reserve(ct_inputs[0].size());
            for (size_t j = 0; j < ct_inputs[0].size(); ++j) {
                auto raw = FIDESlib::CKKS::GetRawCipherText(cc, ct_inputs[i][j]);
                inputs_gpu[i].emplace_back(GPUcc, raw);
            }
        }
    }


    std::vector<std::vector<double>> decryptGPUMatrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& result_gpu, 
                        lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& privateKey,
                        std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& dummy,
                        int numSlots, int blockSize){
                        
        FIDESlib::CKKS::Context& GPUcc = result_gpu[0][0].cc;

        std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> result_cpu(result_gpu.size());
        for (size_t i = 0; i < result_gpu.size(); ++i) {
            result_cpu[i].reserve(result_gpu[0].size());
            for (size_t j = 0; j < result_gpu[0].size(); ++j) {
                auto ctxt = dummy[0][0]->Clone();

                FIDESlib::CKKS::RawCipherText raw_res;
                result_gpu[i][j].store(GPUcc, raw_res);
                auto result(ctxt);
                GetOpenFHECipherText(result, raw_res);
                result_cpu[i].emplace_back(result);
            }
        }
        auto result_decrypted = DecryptMatrix(result_cpu, privateKey, numSlots);
        auto final_result_cpu = convertToLargeMatrix(result_decrypted, blockSize);

        return final_result_cpu;
    }

    // Function to load .txt inputs into a 2-D matrix
    void load_weights(const std::string& filename, std::vector<std::vector<double>>& matrix_weights, int rows, int cols) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        matrix_weights.assign(rows, std::vector<double>(cols, 0.0));

        std::string line;
        size_t i = 0;

        while (std::getline(file, line) && i < static_cast<size_t>(rows)) {
            std::istringstream ss(line);
            double value;
            size_t j = 0;

            while (ss >> value && j < static_cast<size_t>(cols)) {
                matrix_weights[i][j] = value;
                j++;
            }
            i++;
        }
    }

    // Function to load .txt bias into a 2D matrix [rows][cols],
    // by reading multiple rows from file, and repeating them cyclically if needed
    void load_bias(const std::string& filename, std::vector<std::vector<double>>& bias_matrix, int rows, int cols) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<std::vector<double>> read_rows;
        std::string line;
        double value;

        // Read all lines and extract up to `cols` values per line
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            std::vector<double> row;
            row.reserve(cols);

            while (ss >> value && static_cast<int>(row.size()) < cols) {
                row.emplace_back(value);
            }

            // Zero pad if fewer than `cols` values
            while (static_cast<int>(row.size()) < cols) {
                row.emplace_back(0.0);
            }

            read_rows.emplace_back(std::move(row));
        }

        if (read_rows.empty()) {
            std::cerr << "No data found in file: " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        // Repeat the read rows cyclically to fill the full matrix
        bias_matrix.resize(rows);
        for (int i = 0; i < rows; ++i) {
            bias_matrix[i] = read_rows[i % read_rows.size()];
        }
    }



    std::vector<std::vector<double>> readGroundTruth(const std::string& filename) {
        std::ifstream file(filename);
        std::vector<std::vector<double>> matrix;
        std::string line;

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        while (std::getline(file, line)) {
            std::vector<double> row;
            std::stringstream ss(line);
            std::string val;
            while (std::getline(ss, val, ',')) {
                try {
                    row.push_back(std::stod(val));
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid number: " << val << std::endl;
                    continue;
                }
            }
            if (!row.empty())
                matrix.push_back(row);
        }

        return matrix;
    }

    std::vector<double> CreateBlockMask(size_t numSlots, size_t blockSize, size_t token_length, double value) {
        std::vector<double> mask(numSlots, 0.0);

        for (size_t blockStart = 0; blockStart < numSlots; blockStart += blockSize) {
            size_t limit = std::min(token_length, blockSize/2);
            for (size_t i = 0; i < limit && (blockStart + i) < numSlots; ++i) {
                mask[blockStart + i] = 1.0/value;               // head 1
                mask[blockStart + i + blockSize/2] = 1.0/value; // head 2
            }
        }
        return mask;
    }


    std::vector<double> CreateHeadMask(size_t numSlots, size_t blockSize, int head_no) {

        if (head_no == 0) {
            std::vector<double> mask(numSlots, 0.0);
            for (size_t blockStart = 0; blockStart < numSlots; blockStart += blockSize) {
                size_t end = std::min(blockStart + blockSize / 2, numSlots);
                for (size_t i = blockStart; i < end; ++i) {
                    mask[i] = 1.0;
                }
            }
            return mask;
        }
        else if (head_no == 1) {
            std::vector<double> mask(numSlots, 1.0);
            for (size_t blockStart = 0; blockStart < numSlots; blockStart += blockSize) {
                size_t end = std::min(blockStart + blockSize / 2, numSlots);
                for (size_t i = blockStart; i < end; ++i) {
                    mask[i] = 0.0;
                }
            }
            return mask;
        } 
    }    

}  // namespace FIDESlib::CKKS