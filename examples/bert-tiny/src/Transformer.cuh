#ifndef FIDESLIB_CKKS_TRANSFORMER_CUH
#define FIDESLIB_CKKS_TRANSFORMER_CUH

#include <CKKS/AccumulateBroadcast.cuh>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/Context.cuh>
#include <CKKS/LinearTransform.cuh>
#include <CKKS/Plaintext.cuh>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <optional>
#include <type_traits>

#include "Inputs.cuh"
#include "MatMul.cuh"
#include "MatMul.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using namespace lbcrypto;

namespace FIDESlib::CKKS {

extern std::vector<std::vector<lbcrypto::Ciphertext<DCRTPoly>>> ct_tokens;
extern lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys_;
extern bool TIMING;

struct EncoderConfiguration {
	bool verbose = true;
	int numSlots;
	int blockSize;
	int token_length;
	int bStep			= 16;
	int num_heads		= 2;
	uint32_t bStepBoot	= 16;
	int bStepAcc		= 4;
	uint32_t levelsStC	= 3;
	uint32_t levelsCtS	= 3;
	int level_matmul	= 4;
	bool prescale		= false;
	int level_transpose = 4;
	size_t rows			= 128;
	size_t cols			= 128;
};

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> encoder(PtWeights_GPU& weights_layer,
  MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
  TransposePrecomputations_GPU& Tprecomp_gpu,
  std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& tokens,
  PtMasks_GPU& masks,
  EncoderConfiguration& conf,
  int layerNo);

void MatrixBootstrap(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, int numSlots, bool input_prescaled = false);

void MatrixAddScalar(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, double value);

void MatrixMultScalar(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, double value);

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>
MatrixAdd(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2);

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>
MatrixConcat(std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>>& matrices, std::vector<FIDESlib::CKKS::Plaintext>& masks, int blockSize);

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixMask(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, FIDESlib::CKKS::Plaintext& mask);

void MatrixRotate(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, int index);

void dropMatrixLevel(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& in, int level);

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
  const EncoderConfiguration& conf);

std::vector<int> GenerateRotationIndices_GPU(int blockSize, int bstep, int bStepAcc, int colSize = 0);

/**
 * Process pre-tokenized samples from manifest file (for cluster use without Python).
 *
 * @param pretokenized_dir Directory containing manifest.txt and sample_XXXX.txt files
 * @param output_path Path to write results
 * @param base_conf Encoder configuration
 * @param publicKey Public key for encryption
 * @param GPUcc GPU crypto context
 * @param ct_tokens Token ciphertexts (template for cloning)
 * @param weights_layer0 Layer 0 weights
 * @param weights_layer1 Layer 1 weights
 * @param masks Masks for operations
 * @param precomp_gpu Matrix-matrix product precomputations
 * @param Tprecomp_gpu Transpose precomputations
 * @param cc CPU crypto context
 * @param sk Secret key for decryption
 */
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
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& sk);

} // namespace FIDESlib::CKKS

#endif // FIDESLIB_CKKS_TRANSFORMER_CUH