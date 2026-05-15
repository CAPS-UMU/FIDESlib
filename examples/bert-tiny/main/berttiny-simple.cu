#include "utils.cuh"

namespace fs = std::filesystem;
using namespace FIDESlib::CKKS;

int main(const int argc, char** argv) {
	init_devices_from_env(); // Read GPU count from FIDESLIB_NUM_GPUS env var

	// Context
	create_cpu_context();
	lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys = cc->KeyGen();
	const int numSlots						   = static_cast<int>(cc->GetEncodingParams()->GetBatchSize());
	const int blockSize						   = static_cast<int>(sqrt(numSlots));
	EncoderConfiguration conf{ .numSlots = numSlots, .blockSize = blockSize, .token_length = 40, .bStep = 16, .bStepBoot = 16, .bStepAcc = 4 };
	keys_								= keys;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, FIDESlib::ENCAPS);
	FIDESlib::CKKS::Context cc_			= FIDESlib::CKKS::GenCryptoContextGPU(params.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc	= *cc_;
	prepare_cpu_context(cc_, keys, conf.numSlots, conf.blockSize, conf);
	// Context

	prepare_gpu_context_bert(cc_, keys, conf);
	GPUcc.batch = 100;

	// Paths
	const std::string dataset				= "sst2";
	std::string tokens_file					= "pretokenized/sample_0000.txt"; // "it 's a charming and often affecting journey ." (token_length=12)
	std::string model_name					= "bert-tiny-" + dataset;
	const fs::path model_path_fs			= fs::path(root_dir) / ("weights/weights-" + model_name);
	const std::filesystem::path tokens_path = model_path_fs / tokens_file;
	std::string model_path					= model_path_fs.string();
	std::string output_path					= "simple_out.txt";

	// Loading weights and biases
	struct PtMasks_GPU masks = GetPtMasks_GPU(cc_, cc, conf.numSlots, conf.blockSize, conf.level_matmul + 1);

	struct PtWeights_GPU weights_layer0 =
	  GetPtWeightsGPU(cc_, keys.publicKey, model_path, 0, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul + 1, conf.num_heads);
	struct PtWeights_GPU weights_layer1 =
	  GetPtWeightsGPU(cc_, keys.publicKey, model_path, 1, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul + 1, conf.num_heads);

	struct MatrixMatrixProductPrecomputations_GPU precomp_gpu =
	  getMatrixMatrixProductPrecomputations_GPU(cc_, cc, conf.blockSize, conf.bStep, conf.level_matmul + 2, conf.level_matmul + 2, conf.prescale, conf.numSlots);

	TransposePrecomputations_GPU Tprecomp_gpu = getMatrixTransposePrecomputations_GPU(cc_, cc, conf.blockSize, conf.bStep, conf.level_matmul);

	// Loading tokens
	ct_tokens = encryptMatrixtoCPU(tokens_path.string(), keys.publicKey, conf.numSlots, conf.blockSize, conf.rows, conf.cols);
	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu;
	encryptMatrixtoGPU(tokens_path.string(), tokens_gpu, keys.publicKey, cc_, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul);

	// --------- Timing BERT-Tiny ---------
	cudaDeviceSynchronize();
	auto start_gpu = std::chrono::high_resolution_clock::now();
	tokens_gpu	   = encoder(weights_layer0, precomp_gpu, Tprecomp_gpu, tokens_gpu, masks, conf, 0);
	cudaDeviceSynchronize();
	auto end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "Encoder 1 took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
	start_gpu  = std::chrono::high_resolution_clock::now();
	tokens_gpu = encoder(weights_layer1, precomp_gpu, Tprecomp_gpu, tokens_gpu, masks, conf, 1);

	cudaDeviceSynchronize();
	end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "Encoder 2 took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
	start_gpu			= std::chrono::high_resolution_clock::now();
	uint32_t class_pred = classifier(
	  cc, tokens_gpu, keys.secretKey, ct_tokens, precomp_gpu, weights_layer1, masks, conf.numSlots, conf.blockSize, conf.token_length, true, output_path, conf);
	cudaDeviceSynchronize();
	end_gpu = std::chrono::high_resolution_clock::now();
	std::cout << "Classifier took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms." << std::endl;
	// ------------------------------------

	DeregisterAllContexts();
	return 0;
}