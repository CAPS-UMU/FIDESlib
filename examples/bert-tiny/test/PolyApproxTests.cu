//
// Created by seyda on 5/8/25.
//

#include <cstdlib>
#include <filesystem>
#include <string>

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"
#include <CKKS/KeySwitchingKey.cuh>

#include "CKKS/AccumulateBroadcast.cuh"
#include "MatMul.h"
#include "PolyApprox.cuh"
#include "Transformer.cuh"

using namespace std;
using namespace FIDESlib::CKKS;

namespace FIDESlib::Testing {

class PolyApproxTests : public GeneralParametrizedTest {};

TEST_P(PolyApproxTests, PolynomialApproximation) {
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, ENCAPS);
	// Store in member variable GPUcc
	FIDESlib::CKKS::Context cc_			   = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUccData = *cc_;

	int numSlots  = cc->GetEncodingParams()->GetBatchSize();
	int blockSize = int(sqrt(numSlots));

	GPUccData.batch	 = 128;
	int matmul_level = GPUccData.L - 3;
	int bStep		 = 4;
	int bStepAcc	 = 4;
	int rows		 = blockSize;
	int cols		 = blockSize;
	cc->EvalMultKeyGen(keys.secretKey);
	auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(cc_);
	eval_key_gpu.Initialize(eval_key);
	GPUccData.AddEvalKey(std::move(eval_key_gpu));
	std::vector<int32_t> rotation_indices = GenerateRotationIndices_GPU(blockSize, bStep, bStepAcc);
	GenAndAddRotationKeys(cc, keys, cc_, rotation_indices);

	// // Bootstrapping Precomputation
	bool bts = true;

	if (bts) {
		matmul_level = 12;

		cc->EvalBootstrapSetup(
		  { 3, 3 }, { 16, 16 }, numSlots, 0, true, GetMultiplicativeDepthByCoeffVector(GPUccData.GetCoeffsChebyshev(), false) + GPUccData.GetDoubleAngleIts());
		cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);
		FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, cc_);
	}

	// --------- Encrypt Inputs (paths relative to build dir) -------
	const std::string model_path = "../weights/weights-bert-tiny-sst2";
	int token_length			 = 40;

	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> QKT_SM, QKT_LN;

	auto ct_sm = encryptMatrixtoCPU(std::string(model_path + "/intermediate-results/QKT_SM.txt"), keys.publicKey, numSlots, blockSize, rows, cols, matmul_level);
	auto ct_ln = encryptMatrixtoCPU(std::string(model_path + "/intermediate-results/QKT_LN.txt"), keys.publicKey, numSlots, blockSize, rows, cols, matmul_level);

	encryptMatrixtoGPU(std::string(model_path + "/intermediate-results/QKT_SM.txt"), QKT_SM, keys.publicKey, cc_, numSlots, blockSize, rows, cols, matmul_level);
	encryptMatrixtoGPU(std::string(model_path + "/intermediate-results/QKT_LN.txt"), QKT_LN, keys.publicKey, cc_, numSlots, blockSize, rows, cols, matmul_level);

	struct PtMasks_GPU masks = GetPtMasks_GPU(cc_, cc, numSlots, blockSize, matmul_level + 1);
	// struct MatrixMatrixProductPrecomputations_GPU precomp_gpu = getMatrixMatrixProductPrecomputations_GPU(cc_, cc, blockSize, bStep, matmul_level, matmul_level, false, 0);

	std::vector<std::vector<FIDESlib::CKKS::Plaintext>> Wln, bln;
	encodeMatrixtoGPU(std::string(model_path + "/layer0_Wln1.txt"), Wln, keys.publicKey, cc_, numSlots, blockSize, rows, cols, matmul_level, true);
	encodeMatrixtoGPU(std::string(model_path + "/layer0_bln1.txt"), bln, keys.publicKey, cc_, numSlots, blockSize, rows, cols, matmul_level, true);

	// ------- Softmax Evaluation on GPU ------
	std::cout << "Softmax: " << std::endl;
	printMatrix(decryptGPUMatrix(QKT_SM, keys.secretKey, ct_sm, numSlots, blockSize), 2, 2, "Input ", false);

	EvalSoftmax_Matrix(
	  QKT_SM, ct_sm[0][0], keys.secretKey, masks.mask_tokens[token_length], masks.mask_broadcast, masks.mask_layernorm[0], masks.mask_max, numSlots, blockSize, bStepAcc, token_length, bts);
	cudaDeviceSynchronize();
	std::cout << "# limbs: " << QKT_SM[0][0].getLevel() << " " << QKT_SM[0][0].NoiseLevel << std::endl;
	printMatrix(decryptGPUMatrix(QKT_SM, keys.secretKey, ct_sm, numSlots, blockSize), 2, 2, "Output: ", false);

	cc_->clearAuxilarPoly();
	cc_->precom.monomialCache.clear();
	cc_->clearAutomorphismKeys();
	cc_->clearBootPrecomputation();
	cc_->clearEvalMultKeys();
	cc_->clearParamSwitchKeys();

	// printMatrix(decryptGPUMatrix(QKT_LN, keys.secretKey, ct_ln, numSlots, blockSize), 2, 2, "LN input ", false);

	// EvalLayerNorm_Matrix(QKT_LN, ct_ln[0][0], keys.secretKey, masks.mask_layernorm, masks.row_masks[token_length], Wln, bln, numSlots, blockSize, bStepAcc, bts);

	// cudaDeviceSynchronize();
	// std::cout << "# limbs: " << QKT_LN[0][0].getLevel() << " " << QKT_LN[0][0].NoiseLevel << std::endl;
	// printMatrix(decryptGPUMatrix(QKT_LN, keys.secretKey, ct_ln, numSlots, blockSize), 2, 32, "LN output: ", false, 3, true);

	// MatrixAddScalar(QKT_SM_, -0.2);
	// QKT_SM_ = MatrixMaskSpecial(QKT_SM_, masks, -0.15);

	// make sure to adjust the token length

	// EvalSoftmax_Matrix(QKT_SM_, ct_sm[0][0], keys.secretKey, masks.mask_tokens[token_length], masks.mask_broadcast, masks.mask_layernorm[0], masks.mask_max,
	// numSlots, blockSize, bStepAcc, token_length, bts, 0, 1); cudaDeviceSynchronize(); printMatrix(decryptGPUMatrix(QKT_SM_, keys.secretKey, ct_sm, numSlots,
	// blockSize), 2, 2, "Layer 0, 0: ", false);

	// EvalSoftmax_Matrix(QKT_SM2, ct_sm[0][0], keys.secretKey, masks.mask_tokens[token_length], masks.mask_broadcast, masks.mask_layernorm[0], masks.mask_max,
	// numSlots, blockSize, bStepAcc, token_length, bts, 1, 1); cudaDeviceSynchronize(); printMatrix(decryptGPUMatrix(QKT_SM2, keys.secretKey, ct_sm, numSlots,
	// blockSize), 2, 2, "Layer 1, 1: ", false);

	// MatrixAddScalar(QKT_SM2_, -0.2);

	// EvalSoftmax_Matrix(QKT_SM2_, ct_sm2[0][0], keys.secretKey, masks.mask_tokens[token_length], masks.mask_broadcast, masks.mask_layernorm[0],
	// masks.mask_max, numSlots, blockSize, bStepAcc, token_length, bts, 0, 1); cudaDeviceSynchronize(); printMatrix(decryptGPUMatrix(QKT_SM2_, keys.secretKey,
	// ct_sm, numSlots, blockSize), 2, 2, "Layer 1, 0: ", false);

	// MatrixBootstrap(QKT_SM, numSlots);
	// MatrixAddScalar(QKT_SM, -0.2);

	// // ------- LayerNorm Evaluation on GPU ------
	// std::cout << "LayerNorm: " << std::endl;

	// std::cout << "# limbs: " << QKT_LN[0][0].getLevel() << " " << QKT_LN[0][0].NoiseLevel << std::endl;
	// printMatrix(decryptGPUMatrix(QKT_LN, keys.secretKey, ct_ln, numSlots, blockSize), 2, 2, "Input: ", false);

	// EvalLayerNorm_Matrix(QKT_LN, ct_ln[0][0], keys.secretKey, GPUcc.GetEvalKey(), masks.mask_layernorm, masks.row_masks[token_length], Wln, bln, numSlots,
	// blockSize, bStepAcc, bts); cudaDeviceSynchronize();

	// std::cout << "# limbs: " << QKT_LN[0][0].getLevel() << " " << QKT_LN[0][0].NoiseLevel << std::endl;
	// printMatrix(decryptGPUMatrix(QKT_LN, keys.secretKey, ct_ln, numSlots, blockSize), 2, 2, "Output: ", false);
}

INSTANTIATE_TEST_SUITE_P(LLMTests, PolyApproxTests, testing::Values(tparams64_15_LLM_flex));
} // namespace FIDESlib::Testing
