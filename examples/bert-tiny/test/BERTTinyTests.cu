//
// Created by seyda on 6/10/25.
//

#include <cstdlib>
#include <filesystem>
#include <string>

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"
#include <CKKS/KeySwitchingKey.cuh>

#include "MatMul.cuh"
#include "PolyApprox.cuh"
#include "Transformer.cuh"
#include "Transpose.cuh"

#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/CoeffsToSlots.cuh"

using namespace FIDESlib::CKKS;

namespace FIDESlib::Testing {

class TransformerTests1 : public GeneralParametrizedTest {};

TEST_P(TransformerTests1, EmbeddingGeneration) {

	const bool sparse_encaps = true;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, ENCAPS);
	// Store in member variable GPUcc (defined in GeneralParametrizedTest)
	FIDESlib::CKKS::Context cc_ = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	ASSERT_NE(cc_, nullptr) << "Failed to create GPU context";
	FIDESlib::CKKS::ContextData& GPUccData = *cc_;

	GPUccData.batch = 1024;

	// ------- Model + paths (relative to build dir) -------
	std::string model_path		  = "../weights/weights-bert-tiny-sst2";
	std::string tokens_path		  = model_path + "/tokens_sst2.txt";
	std::string pretokenized_path = model_path + "/pretokenized";

	std::string output_path = "a1.txt";

	// ------- Generate Keys and Move to GPU --------
	cc->EvalMultKeyGen(keys.secretKey);
	auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(cc_);
	eval_key_gpu.Initialize(eval_key);
	GPUccData.AddEvalKey(std::move(eval_key_gpu));

	EncoderConfiguration conf{ .numSlots = cc->GetEncodingParams()->GetBatchSize(), .blockSize = int(sqrt(cc->GetEncodingParams()->GetBatchSize())), .token_length = 63 };

	std::vector<int32_t> rotation_indices = GenerateRotationIndices_GPU(conf.blockSize, conf.bStep, conf.bStepAcc);
	GenAndAddRotationKeys(cc, keys, cc_, rotation_indices);

	ct_tokens = encryptMatrixtoCPU(tokens_path, keys.publicKey, conf.numSlots, conf.blockSize, conf.rows, conf.cols);

	// Bootstrapping Precomputation
	cc->EvalBootstrapSetup(
	  { 3, 3 }, { 16, 16 }, conf.numSlots, 0, true, GetMultiplicativeDepthByCoeffVector(GPUccData.GetCoeffsChebyshev(), false) + GPUccData.GetDoubleAngleIts());
	cc->EvalBootstrapKeyGen(keys.secretKey, conf.numSlots);
	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, conf.numSlots, cc_);

	// Loading weights and biases
	struct PtMasks_GPU masks = GetPtMasks_GPU(cc_, cc, conf.numSlots, conf.blockSize, conf.level_matmul - 1);

	struct PtWeights_GPU weights_layer0 =
	  GetPtWeightsGPU(cc_, keys.publicKey, model_path, 0, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul - 1, conf.num_heads);
	struct PtWeights_GPU weights_layer1 =
	  GetPtWeightsGPU(cc_, keys.publicKey, model_path, 1, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul - 1, conf.num_heads);

	struct MatrixMatrixProductPrecomputations_GPU precomp_gpu =
	  getMatrixMatrixProductPrecomputations_GPU(cc_, cc, conf.blockSize, conf.bStep, conf.level_matmul, conf.level_matmul, conf.prescale, conf.numSlots);

	TransposePrecomputations_GPU Tprecomp_gpu = getMatrixTransposePrecomputations_GPU(cc_, cc, conf.blockSize, conf.bStep, conf.level_matmul - 2);

	process_pretokenized_samples(
	  pretokenized_path, output_path, conf, keys.publicKey, cc_, ct_tokens, weights_layer0, weights_layer1, masks, precomp_gpu, Tprecomp_gpu, cc, keys.secretKey);

	cc_->clearAuxilarPoly();
	cc_->precom.monomialCache.clear();
	cc_->clearAutomorphismKeys();
	cc_->clearBootPrecomputation();
	cc_->clearEvalMultKeys();
	cc_->clearParamSwitchKeys();
}

// INSTANTIATE_TEST_SUITE_P(LLMTests, TransformerTests1, testing::Values(tparams64_15_LLM_flex));
INSTANTIATE_TEST_SUITE_P(LLMTests, TransformerTests1, testing::Values(tparams64_16_LLM_sq_flex));
} // namespace FIDESlib::Testing
