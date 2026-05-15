#include "Transformer.cuh"

namespace FIDESlib::CKKS {

std::vector<std::vector<lbcrypto::Ciphertext<DCRTPoly>>> ct_tokens;
lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys_;
bool TIMING = true;

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> encoder(PtWeights_GPU& weights_layer,
  MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
  TransposePrecomputations_GPU& Tprecomp_gpu,
  std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& tokens,
  PtMasks_GPU& masks,
  EncoderConfiguration& conf,
  int layerNo) {
	constexpr bool PRINT  = false;
	constexpr bool TIMING = false;
	std::chrono::time_point<std::chrono::system_clock> start_gpu, end_gpu;

	if (TIMING) {
		cudaDeviceSynchronize();
		start_gpu = std::chrono::high_resolution_clock::now();
	}
	Context& cc = tokens[0][0].cc_;
	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> K, Q, V;
	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> GPUResult_QKT, GPUResult_Sm_V, GPUResult_Output, GPUResult_Up, GPUResult_Down;

	dropMatrixLevel(tokens, conf.level_matmul);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(tokens, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "tokens", false);

	PCMM_GPU(tokens, weights_layer.Wk, conf.blockSize, K, precomp_gpu, weights_layer.bk, masks.row_masks[conf.token_length]);
	PCMM_GPU(tokens, weights_layer.Wq, conf.blockSize, Q, precomp_gpu, weights_layer.bq, masks.row_masks[conf.token_length]);
	PCMM_GPU(tokens, weights_layer.Wv, conf.blockSize, V, precomp_gpu, weights_layer.bv, masks.row_masks[conf.token_length]);

	if (TIMING) {
		cudaDeviceSynchronize();
		end_gpu = std::chrono::high_resolution_clock::now();
		std::cout << "PCMM took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
		start_gpu = std::chrono::high_resolution_clock::now();
	}

	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(K, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "K: ", false);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(Q, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Q: ", false);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "V: ", false);

	////////////////////////////// Multi Head Attention /////////////////////////////////
	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> Sm_V, Sm_V2;

	MatrixBootstrap(Q, conf.numSlots, conf.prescale);
	MatrixBootstrap(K, conf.numSlots, conf.prescale);
	MatrixBootstrap(V, conf.numSlots, conf.prescale);

	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> QKT1, QKT2;

	dropMatrixLevel(Q, conf.level_matmul + 1);
	// dropMatrixLevel(Q2, conf.level_matmul+1);
	dropMatrixLevel(K, conf.level_matmul + 2);
	// dropMatrixLevel(K2, conf.level_matmul+1);

	auto Q1 = MatrixMask(Q, masks.head_masks[0]);
	auto Q2 = MatrixMask(Q, masks.head_masks[1]);

	auto K1 = MatrixMask(K, masks.head_masks[0]);
	auto K2 = MatrixMask(K, masks.head_masks[1]);

	auto K1_T = MatrixTranspose_GPU(std::move(K1), conf.blockSize, Tprecomp_gpu);
	auto K2_T = MatrixTranspose_GPU(std::move(K2), conf.blockSize, Tprecomp_gpu);

	CCMM_GPU(Q1, K1_T, conf.blockSize, QKT1, precomp_gpu);
	CCMM_GPU(Q2, K2_T, conf.blockSize, QKT2, precomp_gpu);

	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(QKT1, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT1: ", false);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(QKT2, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT2: ", false);

	if (TIMING) {
		cudaDeviceSynchronize();
		end_gpu = std::chrono::high_resolution_clock::now();
		std::cout << "CCMM took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
		start_gpu = std::chrono::high_resolution_clock::now();
	}

	// MatrixBootstrap(QKT1, conf.numSlots, conf.prescale);
	// MatrixBootstrap(QKT2, conf.numSlots, conf.prescale);

	FIDESlib::CKKS::Plaintext double_mask(QKT1[0][0].cc_);
	double_mask.copy(masks.row_masks[conf.token_length]);

	double_mask.multPt(double_mask, masks.head_masks[0], true);

	double_mask.multScalar(GetPreScaleFactor(cc, conf.numSlots));

	QKT1 = MatrixMask(QKT1, double_mask);
	QKT2 = MatrixMask(QKT2, double_mask);

	// offset for sst2
	if (layerNo == 1) {
		MatrixAddScalar(QKT1, -0.25);
		MatrixAddScalar(QKT2, -0.25);
	}

	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(QKT1, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT1: ", false);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(QKT2, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT2: ", false);

	MatrixBootstrap(QKT1, conf.numSlots, true);

	EvalSoftmax_Matrix(QKT1,
	  ct_tokens[0][0],
	  keys_.secretKey,
	  masks.mask_tokens[conf.token_length],
	  masks.mask_broadcast,
	  masks.mask_layernorm[0],
	  masks.mask_max,
	  conf.numSlots,
	  conf.blockSize,
	  conf.bStepAcc,
	  conf.token_length,
	  true);

	// MatrixBootstrap(QKT1, conf.numSlots, conf.prescale);

	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(QKT1, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT1: ", false);

	MatrixBootstrap(QKT2, conf.numSlots, true);
	EvalSoftmax_Matrix(QKT2,
	  ct_tokens[0][0],
	  keys_.secretKey,
	  masks.mask_tokens[conf.token_length],
	  masks.mask_broadcast,
	  masks.mask_layernorm[0],
	  masks.mask_max,
	  conf.numSlots,
	  conf.blockSize,
	  conf.bStepAcc,
	  conf.token_length,
	  true);
	// MatrixBootstrap(QKT2, conf.numSlots, conf.prescale);

	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(QKT2, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT2: ", false);

	double_mask.copy(masks.row_masks[conf.token_length]);
	double_mask.multPt(double_mask, masks.head_masks[0], true);

	dropMatrixLevel(QKT1, conf.level_matmul + 1);
	dropMatrixLevel(QKT2, conf.level_matmul + 1);
	QKT1 = MatrixMask(QKT1, double_mask);
	QKT2 = MatrixMask(QKT2, double_mask);

	if (TIMING) {
		cudaDeviceSynchronize();
		end_gpu = std::chrono::high_resolution_clock::now();
		std::cout << "Softmax took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
		start_gpu = std::chrono::high_resolution_clock::now();
	}

	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(QKT1, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT1: ", false);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(QKT2, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "QKT2: ", false);

	dropMatrixLevel(V, conf.level_matmul + 1);
	auto V1 = MatrixMask(V, masks.head_masks[0]);
	auto V2 = MatrixMask(V, masks.head_masks[1]);
	MatrixRotate(V2, conf.blockSize / conf.num_heads);

	CCMM_GPU(QKT1, V1, conf.blockSize, Sm_V, precomp_gpu);
	CCMM_GPU(QKT2, V2, conf.blockSize, Sm_V2, precomp_gpu);

	Plaintext mask(cc);
	mask.copy(masks.head_masks[0]);
	mask.multScalar(GetPreScaleFactor(cc, conf.numSlots), true);
	// Sm_V  = MatrixMask(Sm_V, masks.head_masks[0]);
	// Sm_V2 = MatrixMask(Sm_V2, masks.head_masks[0]);
	Sm_V  = MatrixMask(Sm_V, mask);
	Sm_V2 = MatrixMask(Sm_V2, mask);

	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(Sm_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Sm_V1: ", false);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(Sm_V2, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Sm_V2: ", false);
	MatrixRotate(Sm_V2, -conf.blockSize / conf.num_heads);

	Sm_V = MatrixAdd(Sm_V, Sm_V2);

	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(Sm_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Sm_V: ", false);

	if (TIMING) {
		cudaDeviceSynchronize();
		end_gpu = std::chrono::high_resolution_clock::now();
		std::cout << "CCMM took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
		start_gpu = std::chrono::high_resolution_clock::now();
	}

	//////////////////////////////////////////////////////////////////////////////////////

	K.clear();
	Q.clear();
	V.clear();
	QKT1.clear();
	QKT2.clear();
	MatrixBootstrap(Sm_V, conf.numSlots, true);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(Sm_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Sm_V: ", false);

	if (TIMING) {
		cudaDeviceSynchronize();
		end_gpu = std::chrono::high_resolution_clock::now();
		std::cout << "Boot took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
		start_gpu = std::chrono::high_resolution_clock::now();
	}
	// Output CCMM
	dropMatrixLevel(Sm_V, conf.level_matmul);
	PCMM_GPU(Sm_V, weights_layer.Wo, conf.blockSize, GPUResult_Output, precomp_gpu, weights_layer.bo, masks.row_masks[conf.token_length]);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Result_output: ", false);

	if (TIMING) {
		cudaDeviceSynchronize();
		end_gpu = std::chrono::high_resolution_clock::now();
		std::cout << "PCMM took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
		start_gpu = std::chrono::high_resolution_clock::now();
	}
	// Layer Norm
	GPUResult_Output = MatrixAdd(GPUResult_Output, tokens);
	tokens.clear();

	MatrixBootstrap(GPUResult_Output, conf.numSlots, false);

	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Result_output: ", false);

	EvalLayerNorm_Matrix(GPUResult_Output,
	  ct_tokens[0][0],
	  keys_.secretKey,
	  masks.mask_layernorm,
	  masks.row_masks[conf.token_length],
	  weights_layer.Wln1,
	  weights_layer.bln1,
	  conf.numSlots,
	  conf.blockSize,
	  conf.bStepAcc,
	  true);
	MatrixBootstrap(GPUResult_Output, conf.numSlots, false);
	if constexpr (PRINT)
		std::cout << "# ------- bts ------- " << std::endl;

	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "LN: ", false);

	if (TIMING) {
		cudaDeviceSynchronize();
		end_gpu = std::chrono::high_resolution_clock::now();
		std::cout << "LN took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
		start_gpu = std::chrono::high_resolution_clock::now();
	}
	// Up PCMM
	dropMatrixLevel(GPUResult_Output, conf.level_matmul);
	PCMM_GPU(GPUResult_Output, weights_layer.Wu, conf.blockSize, GPUResult_Up, precomp_gpu, weights_layer.bu, masks.row_masks[conf.token_length]);

	if (TIMING) {
		cudaDeviceSynchronize();
		end_gpu = std::chrono::high_resolution_clock::now();
		std::cout << "PCMM took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
		start_gpu = std::chrono::high_resolution_clock::now();
	}

	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(GPUResult_Up, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "RELU input: ", false);

	// dropMatrixLevel(GPUResult_Up, conf.level_matmul);

	// ReLU
	MatrixBootstrap(GPUResult_Up, conf.numSlots);
	EvalGelu_Matrix(GPUResult_Up, conf.numSlots);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(GPUResult_Up, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "RELU: ", false);

	if (TIMING) {
		cudaDeviceSynchronize();
		end_gpu = std::chrono::high_resolution_clock::now();
		std::cout << "Gelu took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
		start_gpu = std::chrono::high_resolution_clock::now();
	}
	// Down PCMM
	// MatrixBootstrap(GPUResult_Up, conf.numSlots);
	dropMatrixLevel(GPUResult_Up, conf.level_matmul);
	PCMM_GPU(GPUResult_Up, weights_layer.Wd, conf.blockSize, GPUResult_Down, precomp_gpu, weights_layer.bd, masks.row_masks[conf.token_length]);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Result_Down: ", false);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "Result_Down: ", false);

	if (TIMING) {
		cudaDeviceSynchronize();
		end_gpu = std::chrono::high_resolution_clock::now();
		std::cout << "PCMM took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
		start_gpu = std::chrono::high_resolution_clock::now();
	}
	// Layer Norm
	GPUResult_Down = MatrixAdd(GPUResult_Down, GPUResult_Output);

	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "LN input: ", false);

	MatrixBootstrap(GPUResult_Down, conf.numSlots, false);
	EvalLayerNorm_Matrix(GPUResult_Down,
	  ct_tokens[0][0],
	  keys_.secretKey,
	  masks.mask_layernorm,
	  masks.row_masks[conf.token_length],
	  weights_layer.Wln2,
	  weights_layer.bln2,
	  conf.numSlots,
	  conf.blockSize,
	  conf.bStepAcc,
	  true);

	if constexpr (PRINT)
		std::cout << "# ------- bts ------- " << std::endl;
	MatrixBootstrap(GPUResult_Down, conf.numSlots);
	if constexpr (PRINT)
		printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "LN: ", false);

	if (TIMING) {
		cudaDeviceSynchronize();
		end_gpu = std::chrono::high_resolution_clock::now();
		std::cout << "LN took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
		start_gpu = std::chrono::high_resolution_clock::now();
	}
	return GPUResult_Down;
}

// -------------------- Offline/Cluster Mode: Pre-tokenized processing --------------------

/**
 * Structure to hold pre-tokenized sample metadata from manifest.
 */
struct PretokenizedSample {
	std::string idx;
	int label;
	int token_length;
	std::string filename;
	std::string text;
};

/**
 * Parse a manifest line into a PretokenizedSample.
 * Format: idx,label,token_length,filename,text
 */
static bool parse_manifest_line(const std::string& line, PretokenizedSample& sample) {
	if (line.empty() || line[0] == '#')
		return false; // Skip comments and empty lines

	std::vector<std::string> parts;
	std::stringstream ss(line);
	std::string part;

	// Parse first 4 comma-separated fields
	for (int i = 0; i < 4 && std::getline(ss, part, ','); ++i) {
		parts.push_back(part);
	}

	// Rest is the text (may contain commas)
	if (std::getline(ss, part)) {
		parts.push_back(part);
	}

	if (parts.size() < 4)
		return false;

	try {
		sample.idx			= parts[0];
		sample.label		= std::stoi(parts[1]);
		sample.token_length = std::stoi(parts[2]);
		sample.filename		= parts[3];
		sample.text			= (parts.size() > 4) ? parts[4] : "";
		return true;
	} catch (...) {
		return false;
	}
}

void process_pretokenized_samples(const std::string& pretokenized_dir,
  const std::string& output_path,
  EncoderConfiguration& base_conf,
  lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
  FIDESlib::CKKS::Context& GPUcc,
  std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& ct_tokens,
  PtWeights_GPU& weights_layer0,
  PtWeights_GPU& weights_layer1,
  PtMasks_GPU& masks,
  MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
  TransposePrecomputations_GPU& Tprecomp_gpu,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& sk) {
	// Read manifest file
	std::string manifest_path = pretokenized_dir + "/manifest.txt";
	std::ifstream manifest_file(manifest_path);
	if (!manifest_file.is_open()) {
		std::cerr << "ERROR: Could not open manifest file: " << manifest_path << std::endl;
		return;
	}

	std::cout << "Processing pre-tokenized samples from: " << pretokenized_dir << std::endl;

	std::vector<PretokenizedSample> samples;
	std::string line;
	while (std::getline(manifest_file, line)) {
		PretokenizedSample sample;
		if (parse_manifest_line(line, sample)) {
			samples.push_back(sample);
		}
	}
	manifest_file.close();

	std::cout << "Found " << samples.size() << " samples in manifest." << std::endl;

	size_t total_counter = 0, correct_counter = 0;

	const size_t warmup_samples = (samples.size() < 3) ? samples.size() : 3;
	const size_t max_samples	  = (samples.size() < 6) ? samples.size() : 6;
	std::cout << "Warmup samples: " << warmup_samples << " (not measured)" << std::endl;
	for (size_t si = 0; si < max_samples; ++si) {
		const auto& sample = samples[si];
		const bool is_warmup = si < warmup_samples;
		try {
			// Validate token length
			if (sample.token_length <= 0 || sample.token_length > 128) {
				std::cerr << "[WARN] Skipping sample " << sample.idx << ": invalid token_length=" << sample.token_length << std::endl;
				continue;
			}

			// Clone ct_tokens for this sample
			std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> ct_tokens_clone;
			ct_tokens_clone.resize(ct_tokens.size());
			for (size_t i = 0; i < ct_tokens.size(); ++i) {
				ct_tokens_clone[i].resize(ct_tokens[i].size());
				for (size_t j = 0; j < ct_tokens[i].size(); ++j) {
					ct_tokens_clone[i][j] = ct_tokens[i][j]->Clone();
				}
			}

			// Set configuration for this sample
			EncoderConfiguration conf = base_conf;
			conf.token_length		  = sample.token_length;

			// Terminal output
			std::cout << "\n///////////////////////////////////////\n";
			std::cout << "Sample idx: " << sample.idx << std::endl;
			std::cout << "Text: " << sample.text << std::endl;
			std::cout << "Token length: " << sample.token_length << std::endl;
			std::cout << "Label: " << sample.label << std::endl;

			// Log to file
			{
				std::ofstream outFile(output_path, std::ios::app);
				outFile << "\n///////////////////////////////////////\n";
				outFile << "Sample idx: " << sample.idx << "\n";
				outFile << "Text: " << sample.text << "\n";
				outFile << "Token length: " << sample.token_length << "\n";
				outFile << "Label: " << sample.label << "\n";
			}

			// Load pre-tokenized embeddings from file
			std::string embedding_path = pretokenized_dir + "/" + sample.filename;
			std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu;
			encryptMatrixtoGPU(embedding_path, tokens_gpu, publicKey, GPUcc, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul);

			// Run inference
			cudaDeviceSynchronize();
			auto start_gpu = std::chrono::high_resolution_clock::now();

			tokens_gpu = encoder(weights_layer0, precomp_gpu, Tprecomp_gpu, tokens_gpu, masks, conf, 0);
			tokens_gpu = encoder(weights_layer1, precomp_gpu, Tprecomp_gpu, tokens_gpu, masks, conf, 1);

			int32_t class_pred = classifier(
			  cc, tokens_gpu, sk, ct_tokens_clone, precomp_gpu, weights_layer1, masks, conf.numSlots, conf.blockSize, conf.token_length, true, output_path, conf);

			cudaDeviceSynchronize();
			auto end_gpu = std::chrono::high_resolution_clock::now();

			tokens_gpu.clear();

			std::cout << class_pred << " vs " << sample.label << std::endl;

			if (is_warmup) {
				std::cout << "Warmup sample (not measured)." << std::endl;
				std::ofstream outFile(output_path, std::ios::app);
				outFile << "Warmup sample (not measured).\n";
			} else {
				++total_counter;
				std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count() << " ms." << std::endl;
				if (class_pred >= 0 && class_pred == sample.label)
					++correct_counter;

				{
					std::ofstream outFile(output_path, std::ios::app);
					outFile << "took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count() << " ms.\n";
					outFile << "Accuracy: " << correct_counter << "/" << total_counter << "\n";
				}
				std::cout << "Accuracy: " << correct_counter << "/" << total_counter << std::endl;
			}
		} catch (const std::exception& e) {
			std::cerr << "[EXCEPTION] " << e.what() << " — skipping sample " << sample.idx << ".\n";
		} catch (...) {
			std::cerr << "[EXCEPTION] unknown — skipping sample " << sample.idx << ".\n";
		}
	}

	std::cout << "\n========================================\n";
	std::cout << "Final Accuracy: " << correct_counter << "/" << total_counter << std::endl;
	{
		std::ofstream outFile(output_path, std::ios::app);
		outFile << "\n========================================\n";
		outFile << "Final Accuracy: " << correct_counter << "/" << total_counter << "\n";
	}
}

int32_t classifier(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
  std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& input,
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& privateKey,
  std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& ct_tokens,
  const MatrixMatrixProductPrecomputations_GPU& precomp,
  PtWeights_GPU& weights_layer,
  PtMasks_GPU& masks,
  int numSlots,
  int blockSize,
  int token_length,
  bool bts,
  const std::string& output_path,
  const EncoderConfiguration& conf) {

	bool constexpr PRINT = false;

	FIDESlib::CKKS::Context& GPUcc = input[0][0].cc_;

	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> result, result_f;
	// PCMM_2(input, weights_layer.Wp, blockSize, result, precomp, weights_layer.bp, masks.row_masks[token_length]);
	if (input[0][0].NoiseLevel == 2) {
		for (auto& i : input)
			for (auto& j : i)
				j.rescale();
	}
	dropMatrixLevel(input, conf.level_matmul);
	PCMM_GPU(input, weights_layer.Wp, blockSize, result, precomp, weights_layer.bp, masks.row_masks[token_length]);

	if (PRINT)
		printMatrix(decryptGPUMatrix(result, keys_.secretKey, ct_tokens, numSlots, blockSize), 2, 2, "PCMM", false);

	evalTanh(result[0][0], numSlots, -40, 40, true); // -10, 10 -> -20, 20

	if (PRINT)
		printMatrix(decryptGPUMatrix(result, keys_.secretKey, ct_tokens, numSlots, blockSize), 2, 2, "Tanh", false);

	FIDESlib::CKKS::Ciphertext result_0(GPUcc), result_1(GPUcc);
	result_0.copy(result[0][0]);
	result_0.multPt(weights_layer.Wc[0][0], false);
	Accumulate(result_0, 4, 1, blockSize);
	result_0.addPt(weights_layer.bc[0][0]);

	FIDESlib::CKKS::RawCipherText raw_res;
	result_0.store(raw_res);
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
	result_1.store(raw_res1);
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
		std::cerr << "Decryption failed: " << e.what() << std::endl;
		return -1;
	} catch (...) {
		std::cerr << "Unknown error occurred during decryption." << std::endl;
		return -1;
	}
	std::cout << std::endl;
}

std::vector<int> GenerateRotationIndices_GPU(int blockSize, int bStep, int bStepAcc, int colSize, int N) {

	if (colSize == 0) {
		colSize = blockSize;
	}
	// JKLS MatMul rotation indices
	std::vector<int32_t> rotation_indices_MM  = GenerateMatMulRotationIndices_GPU(blockSize, bStep, blockSize);
	std::vector<int32_t> rotation_indices_MM2 = GenerateMatMulRotationIndices_GPU(blockSize, bStep, colSize);
	// Multi-head Attention rotation indices
	std::vector<int32_t> rotation_indices_MHA = GenerateMatMulRotationIndices_GPU(64, bStep, colSize); // d_k = 64

	// Transpose rotation indices
	std::vector<int> rotation_indices_T = GenerateTransposeRotationIndices_GPU(blockSize, bStep);

	std::vector<int> rotsum_indices = { 1, 2, 3, 4, 8, 16, 32, 64, 8192, 0, -1, -2, -3, -4, -8, -16, -32, -64, 127, -15, -31, -47, -63, -127 }; // 127 is for pooling, -blockSize for Concat

	std::vector<int> accum_indices	= FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, 1, blockSize);
	std::vector<int> accum_indices2 = FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, blockSize, blockSize);
	std::vector<int> accum_indices3 = FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, blockSize, blockSize / 2);
	std::vector<int> accum_indices4 = FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, blockSize, blockSize / 4);
	std::vector<int> broad_indices	= FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize);
	std::vector<int> broad_indices2 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize / 2);
	std::vector<int> broad_indices3 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize / 4);
	std::vector<int> broad_indices4 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize / 8);
	std::vector<int> broad_indices5 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize / 16); // 4
	std::vector<int> broad_indices6 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, blockSize, blockSize * blockSize);

	// if (blockSize == 128) rotsum_indices = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};

	// Merge the rotation indices and remove duplicates

	std::set<int32_t> merged_set(rotsum_indices.begin(), rotsum_indices.end());
	merged_set.insert(rotation_indices_MM.begin(), rotation_indices_MM.end());
	merged_set.insert(rotation_indices_MM2.begin(), rotation_indices_MM2.end());
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

	std::set<int32_t> normalized_set;
	for (auto i : merged_set) {
		normalized_set.insert(FIDESlib::CKKS::normalyzeIndex(i, blockSize * colSize, N));
	}
	std::vector<int32_t> rotation_indices(normalized_set.begin(), normalized_set.end());
	return rotation_indices;
}

void MatrixBootstrap(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, int numSlots, bool input_prescaled) {
	for (size_t i = 0; i < matrix.size(); i++) {
		for (size_t j = 0; j < matrix[0].size(); j++) {
			Bootstrap(matrix[i][j], numSlots, input_prescaled);
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

void MatrixMultScalar(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, double value) {
	for (size_t i = 0; i < matrix.size(); i++) {
		for (size_t j = 0; j < matrix[0].size(); j++) {
			matrix[i][j].multScalar(value);
		}
	}
}

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>
MatrixAdd(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2) {

	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> masked_matrix;
	masked_matrix.reserve(matrix.size());
	for (size_t i = 0; i < matrix.size(); i++) {
		std::vector<FIDESlib::CKKS::Ciphertext> row;
		row.reserve(matrix[0].size());
		for (size_t j = 0; j < matrix[0].size(); j++) {
			FIDESlib::CKKS::Ciphertext masked_ct(matrix[i][j].cc_);
			masked_ct.copy(matrix[i][j]);
			masked_ct.add(matrix2[i][j]);
			row.emplace_back(std::move(masked_ct));
		}
		masked_matrix.emplace_back(std::move(row));
	}
	return masked_matrix;
}

void MatrixRotate(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, int index) {
	FIDESlib::CKKS::ContextData& cc = matrix[0][0].cc;
	for (size_t i = 0; i < matrix.size(); i++) {
		for (size_t j = 0; j < matrix[0].size(); j++) {
			matrix[i][j].rotate(index);
		}
	}
}

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixMask(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, FIDESlib::CKKS::Plaintext& mask) {

	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> masked_matrix;
	masked_matrix.reserve(matrix.size());
	for (size_t i = 0; i < matrix.size(); i++) {
		std::vector<FIDESlib::CKKS::Ciphertext> row;
		row.reserve(matrix[0].size());
		for (size_t j = 0; j < matrix[0].size(); j++) {
			FIDESlib::CKKS::Ciphertext masked_ct(matrix[i][j].cc_);
			masked_ct.copy(matrix[i][j]);
			masked_ct.multPt(mask);
			// masked_ct.rescale();
			row.emplace_back(std::move(masked_ct));
		}
		masked_matrix.emplace_back(std::move(row));
	}
	return masked_matrix;
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

} // namespace FIDESlib::CKKS
