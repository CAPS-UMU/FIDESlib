//
// Created by seyda on 6/6/25.
//

#include <cstdlib>
#include <filesystem>
#include <string>

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"
#include <CKKS/KeySwitchingKey.cuh>

#include "Transformer.cuh"

using namespace std;
using namespace FIDESlib::CKKS;
using namespace lbcrypto;
using namespace std::chrono;

namespace FIDESlib::Testing {

class MMTests : public GeneralParametrizedTest {};

TEST_P(MMTests, MatrixMultiplication) {

	bool verbose = true;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	// Store in member variable GPUcc
	FIDESlib::CKKS::Context cc_			   = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUccData = *cc_;

	// Parameters
	GPUccData.batch	 = 128;
	int matmul_level = 8;
	int bStepAcc	 = 4;
	int numSlots	 = cc->GetEncodingParams()->GetBatchSize();
	int blockSize	 = 128;
	int bStep		 = 16;
	int num_heads	 = 2;
	size_t rows		 = 128;
	size_t cols		 = 128;
	int token_length = 22;

	// Inputs - use pre-existing tokens file (path relative to build dir)
	std::string model_path	= "../weights/weights-bert-tiny-sst2";
	std::string output_file = "pretokenized/sample_0000.txt";

	// Keys
	cc->EvalMultKeyGen(keys.secretKey);
	auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(cc_);
	eval_key_gpu.Initialize(eval_key);
	GPUccData.AddEvalKey(std::move(eval_key_gpu));

	std::vector<int32_t> rotation_indices = GenerateRotationIndices_GPU(blockSize, bStep, bStepAcc, 0, cc->GetRingDimension());
	GenAndAddRotationKeys(cc, keys, cc_, rotation_indices);

	struct PtMasks_GPU masks = GetPtMasks_GPU(cc_, cc, numSlots, blockSize, matmul_level + 1);

	// // Bootstrapping Precomputation
	// cc->EvalBootstrapSetup({3, 3}, {4, 4}, numSlots);
	// cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

	// FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys_, numSlots, GPUcc);

	auto ct_tokens = encryptMatrixtoCPU(model_path + "/" + output_file, keys.publicKey, numSlots, blockSize, rows, cols, false);
	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu;
	encryptMatrixtoGPU(model_path + "/" + output_file, tokens_gpu, keys.publicKey, cc_, numSlots, blockSize, rows, cols, matmul_level);

	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> K, Q, QK_T;
	std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>> QKT_heads;

	struct PtWeights_GPU weights_layer0 = GetPtWeightsGPU(cc_, keys.publicKey, model_path, 0, numSlots, blockSize, rows, cols, matmul_level, num_heads);

	// Bootstrap(tokens_gpu[0][0z], numSlots, false);
	printMatrix(decryptGPUMatrix(tokens_gpu, keys.secretKey, ct_tokens, numSlots, blockSize), 2, 2, "tokens_gpu: ", false, 10);

	// ------- PCMM on GPU ------
	if (verbose)
		std::cout << "Precomp: ";
	auto start_gpu = std::chrono::high_resolution_clock::now();
	// struct MatrixMatrixProductPrecomputations_GPU precomp_gpu = getMatrixMatrixProductPrecomputations_GPU(cc_, cc, masks, blockSize, bStep, matmul_level+1, matmul_level+1, false, numSlots);
	struct MatrixMatrixProductPrecomputations_GPU precomp_gpu =
	  getMatrixMatrixProductPrecomputations_GPU(cc_, cc, blockSize, bStep, matmul_level, matmul_level - 4, false, numSlots);
	TransposePrecomputations_GPU Tprecomp_gpu = getMatrixTransposePrecomputations_GPU(cc_, cc, blockSize, bStep, matmul_level - 3);
	cudaDeviceSynchronize();
	auto end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;

	dropMatrixLevel(tokens_gpu, matmul_level);
	int N = 1;

	std::cout << "# limbs: " << tokens_gpu[0][0].getLevel() + 1 << " " << tokens_gpu[0][0].NoiseLevel << std::endl;
	std::cout << "PCMM 1: " << std::endl;

	start_gpu = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++) {
		CudaCheckErrorMod;
		PCMM_GPU(tokens_gpu, weights_layer0.Wq, blockSize, K, precomp_gpu, weights_layer0.bk);
		//	PCMM_GPU(tokens_gpu, weights_layer0.Wq, blockSize, K, precomp_gpu, weights_layer0.bk);
		CudaCheckErrorMod;
		PCMM_GPU(tokens_gpu, weights_layer0.Wq, blockSize, Q, precomp_gpu, weights_layer0.bq);
		CudaCheckErrorMod;
	}
	cudaDeviceSynchronize();
	end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) / N << " ms." << std::endl;

	std::cout << "# limbs: " << K[0][0].getLevel() + 1 << " " << K[0][0].NoiseLevel << std::endl;
	printMatrix(decryptGPUMatrix(K, keys.secretKey, ct_tokens, numSlots, blockSize), 2, 2, "K: ", false, 4, false);
	printMatrix(decryptGPUMatrix(Q, keys.secretKey, ct_tokens, numSlots, blockSize), 2, 2, "Q: ", false, 4, false);

	std::cout << "Transpose: " << std::endl;

	CudaCheckErrorMod;
	auto K_T = /*std::move(K);*/ MatrixTranspose_GPU(std::move(K), blockSize, Tprecomp_gpu);
	CudaCheckErrorMod;

	printMatrix(decryptGPUMatrix(K_T, keys.secretKey, ct_tokens, numSlots, blockSize), 2, 2, "K_T: ", false, 4, true);

	std::cout << "CCMM 1: " << std::endl;

	CudaCheckErrorMod;
	CCMM_GPU(Q, K_T, blockSize, QK_T, precomp_gpu);
	CudaCheckErrorMod;

	printMatrix(decryptGPUMatrix(QK_T, keys.secretKey, ct_tokens, numSlots, blockSize), 32, 32, "QK_T: ", false, 4, true);

	cc_->clearAuxilarPoly();
	cc_->precom.monomialCache.clear();
	cc_->clearAutomorphismKeys();
	cc_->clearBootPrecomputation();
	cc_->clearEvalMultKeys();
	cc_->clearParamSwitchKeys();
}

INSTANTIATE_TEST_SUITE_P(LLMTests, MMTests, testing::Values(tparams64_15_LLM_flex, tparams64_16_LLM_flex));
} // namespace FIDESlib::Testing
