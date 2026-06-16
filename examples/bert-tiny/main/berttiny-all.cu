#include "utils.cuh"

namespace fs = std::filesystem;
using namespace FIDESlib::CKKS;

int main(const int argc, char** argv) {
	init_devices_from_env(); // Read GPU count from FIDESLIB_NUM_GPUS env var

	// ----- Dataset + model paths -----
	const fs::path model_path_fs = fs::path(root_dir) / "weights/weights-bert-tiny-sst2";
	std::string model_path		 = model_path_fs.string();

	// Pre-tokenized samples directory
	const fs::path pretokenized_path = model_path_fs / "pretokenized";

	// Dummy tokens file (for ct_tokens template)
	const fs::path tokens_path = model_path_fs / "tokens_sst2.txt";

	// ----- Output path -----
	std::string output_path = "all_out.txt";
	std::ofstream outFile(output_path);
	if (!outFile.is_open()) {
		std::cerr << "Error: could not create file " << output_path << std::endl;
		return 1;
	}
	outFile.close();

	create_cpu_context();
	lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys = cc->KeyGen();
	keys_									   = keys;
	const int numSlots						   = static_cast<int>(cc->GetEncodingParams()->GetBatchSize());
	const int blockSize						   = static_cast<int>(sqrt(numSlots));
	EncoderConfiguration conf{ .numSlots = numSlots, .blockSize = blockSize, .token_length = 63 };

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, FIDESlib::ENCAPS);
	FIDESlib::CKKS::Context cc_			= FIDESlib::CKKS::GenCryptoContextGPU(params.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc	= *cc_;
	prepare_cpu_context(cc_, keys, conf.numSlots, conf.blockSize, conf);
	prepare_gpu_context_bert(cc_, keys, conf);

	// Loading weights and biases
	struct PtMasks_GPU masks = GetPtMasks_GPU(cc_, cc, conf.numSlots, conf.blockSize, conf.level_matmul + 1);

	struct PtWeights_GPU weights_layer0 =
	  GetPtWeightsGPU(cc_, keys.publicKey, model_path, 0, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul, conf.num_heads);
	struct PtWeights_GPU weights_layer1 =
	  GetPtWeightsGPU(cc_, keys.publicKey, model_path, 1, conf.numSlots, conf.blockSize, conf.rows, conf.cols, conf.level_matmul, conf.num_heads);

	struct MatrixMatrixProductPrecomputations_GPU precomp_gpu =
	  getMatrixMatrixProductPrecomputations_GPU(cc_, cc, conf.blockSize, conf.bStep, conf.level_matmul, conf.level_matmul, conf.prescale, conf.numSlots);
	TransposePrecomputations_GPU Tprecomp_gpu = getMatrixTransposePrecomputations_GPU(cc_, cc, conf.blockSize, conf.bStep, conf.level_matmul + 1);

	ct_tokens = encryptMatrixtoCPU(tokens_path.string(), keys.publicKey, conf.numSlots, conf.blockSize, conf.rows, conf.cols);

	// Process pre-tokenized samples (no Python runtime required)
	process_pretokenized_samples(
	  pretokenized_path.string(), output_path, conf, keys.publicKey, cc_, ct_tokens, weights_layer0, weights_layer1, masks, precomp_gpu, Tprecomp_gpu, cc, keys.secretKey);

	DeregisterAllContexts();
	return 0;
}