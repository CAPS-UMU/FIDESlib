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

class TransformerTests2 : public GeneralParametrizedTest {};

TEST_P(TransformerTests2, EmbeddingGeneration) {

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
	// Store in member variable GPUcc
	FIDESlib::CKKS::Context cc_ = GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), generalTestParams.GPUs);
	cc_->batch					= 100;

	// ------- Generate Keys and Move to GPU--------
	cc->EvalMultKeyGen(keys.secretKey);
	auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(cc_);
	eval_key_gpu.Initialize(eval_key);
	cc_->AddEvalKey(std::move(eval_key_gpu));

	// Paths relative to build dir
	std::string model_path		  = "../weights/weights-bert-tiny-sst2";
	std::string pretokenized_path = model_path + "/pretokenized";

	// Use a fixed token length for testing (will be overridden by pretokenized data)
	int token_length = 20;

	EncoderConfiguration conf{ .numSlots = (int)cc->GetEncodingParams()->GetBatchSize(), .blockSize = int(sqrt(cc->GetEncodingParams()->GetBatchSize())), .token_length = token_length };

	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu;
	encryptMatrixtoGPU(std::string(model_path + "/tokens_sst2.txt"), tokens_gpu, keys.publicKey, cc_, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul);

	if (conf.verbose)
		std::cout << "Block size: " << conf.blockSize << std::endl;
	std::vector<int32_t> rotation_indices = GenerateRotationIndices_GPU(conf.blockSize, conf.bStep, conf.bStepAcc);
	GenAndAddRotationKeys(cc, keys, cc_, rotation_indices);

	// Bootstrapping Precomputation
	cc->EvalBootstrapSetup({ conf.levelsCtS, conf.levelsStC }, { conf.bStepBoot, conf.bStepBoot }, conf.numSlots);
	cc->EvalBootstrapKeyGen(keys.secretKey, conf.numSlots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, conf.numSlots, cc_);

	// Loading weights and biases
	struct PtMasks_GPU masks = GetPtMasks_GPU(cc_, cc, conf.numSlots, conf.blockSize, conf.level_matmul + 1);

	struct PtWeights_GPU weights_layer0 =
	  GetPtWeightsGPU(cc_, keys.publicKey, model_path, 0, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul + 1, conf.num_heads);
	struct PtWeights_GPU weights_layer1 =
	  GetPtWeightsGPU(cc_, keys.publicKey, model_path, 1, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul + 1, conf.num_heads);

	struct MatrixMatrixProductPrecomputations_GPU precomp_gpu =
	  getMatrixMatrixProductPrecomputations_GPU(cc_, cc, conf.blockSize, conf.bStep, conf.level_matmul + 1, conf.level_matmul + 1, conf.prescale, conf.numSlots);

	TransposePrecomputations_GPU Tprecomp_gpu = getMatrixTransposePrecomputations_GPU(cc_, cc, conf.blockSize, conf.bStep, conf.level_matmul);

	ct_tokens = encryptMatrixtoCPU(std::string(model_path + "/tokens_sst2.txt"), keys.publicKey, conf.numSlots, conf.blockSize, conf.rows, conf.cols);

	std::string output_path = "a.txt";
	process_pretokenized_samples(
	  pretokenized_path, output_path, conf, keys.publicKey, cc_, ct_tokens, weights_layer0, weights_layer1, masks, precomp_gpu, Tprecomp_gpu, cc, keys.secretKey);

	cc_->clearAuxilarPoly();
	cc_->precom.monomialCache.clear();
	cc_->clearAutomorphismKeys();
	cc_->clearBootPrecomputation();
	cc_->clearEvalMultKeys();
	cc_->clearParamSwitchKeys();
}

INSTANTIATE_TEST_SUITE_P(LLMTests, TransformerTests2, testing::Values(tparams64_15_LLM_flexext));
} // namespace FIDESlib::Testing
