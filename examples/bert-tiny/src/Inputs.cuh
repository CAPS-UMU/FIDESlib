//
// Created by seyda on 7/13/25.
//
#ifndef FIDESLIB_BERT_TINY_INPUTS_CUH
#define FIDESLIB_BERT_TINY_INPUTS_CUH

#include <cmath>
#include <iostream>
#include <vector>

#include "MatMul.h"
#include <CKKS/Ciphertext.cuh>
#include <CKKS/Context.cuh>
#include <CKKS/Plaintext.cuh>
#include <pke/openfhe.h>

namespace FIDESlib::CKKS {

struct PtWeights_GPU {
	std::vector<std::vector<FIDESlib::CKKS::Plaintext>> Wk, Wq, Wv, Wo, Wu, Wd, Wln1, Wln2, Wc, Wp;
	std::vector<std::vector<FIDESlib::CKKS::Plaintext>> bk, bq, bv, bo, bu, bd, bln1, bln2, bc, bp;
};

struct Weights_GPU {
	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> Wk, Wq, Wv, Wo, Wu, Wd, Wln1, Wln2, Wc, Wp;
	std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> bk, bq, bv, bo, bu, bd, bln1, bln2, bc, bp;
};

struct PtMasks_GPU {
	std::vector<FIDESlib::CKKS::Plaintext> mask_tokens, mask_layernorm, mask_broadcast;
	FIDESlib::CKKS::Plaintext mask_max;
	std::vector<FIDESlib::CKKS::Plaintext> head_masks, head_masks_T;
	std::vector<FIDESlib::CKKS::Plaintext> row_masks;

	// Delete copy constructor
	PtMasks_GPU(const PtMasks_GPU&)			   = delete;
	PtMasks_GPU& operator=(const PtMasks_GPU&) = delete;

	// Allow only move
	PtMasks_GPU(PtMasks_GPU&&)			  = default;
	PtMasks_GPU& operator=(PtMasks_GPU&&) = default;

	// Move constructor
	PtMasks_GPU(std::vector<FIDESlib::CKKS::Plaintext>&& token,
	  FIDESlib::CKKS::Plaintext&& max,
	  std::vector<FIDESlib::CKKS::Plaintext>&& broadcast,
	  std::vector<FIDESlib::CKKS::Plaintext>&& layernorm,
	  std::vector<FIDESlib::CKKS::Plaintext>&& heads,
	  std::vector<FIDESlib::CKKS::Plaintext>&& heads_T,
	  std::vector<FIDESlib::CKKS::Plaintext>&& row)
	: mask_tokens(std::move(token)), mask_max(std::move(max)), mask_broadcast(std::move(broadcast)), mask_layernorm(std::move(layernorm)),
	  head_masks(std::move(heads)), head_masks_T(std::move(heads_T)), row_masks(std::move(row)) {
	}
};

PtMasks_GPU GetPtMasks_GPU(FIDESlib::CKKS::Context& GPUcc, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, int numSlots, int blockSize, int level = 1);

PtWeights_GPU GetPtWeightsGPU(FIDESlib::CKKS::Context& GPUcc,
  lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
  const std::string& model_path,
  int layername,
  int numSlots,
  int blockSize,
  int rows,
  int cols,
  int level,
  int num_heads,
  int colSize = 0);

Weights_GPU GetWeightsGPU(FIDESlib::CKKS::Context& GPUcc,
  lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
  const std::string& model_path,
  int layername,
  int numSlots,
  int blockSize,
  int rows,
  int cols,
  int level,
  int colSize = 0);

std::vector<double> getPCMM_bMatrix(std::vector<double> weights, int rowSize);

std::vector<std::vector<lbcrypto::Plaintext>>
EncodeMatrix(const std::vector<std::vector<std::vector<double>>>& matrix, lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey, int level);

void encodeMatrixtoGPU(const std::string& filename,
  std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& pt_inputs_gpu,
  lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
  FIDESlib::CKKS::Context& GPUcc,
  int numSlots,
  int blockSize,
  size_t rows,
  size_t cols,
  int level		 = 0,
  bool if_repeat = false,
  int colSize	 = 0);

std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>
encryptMatrixtoCPU(const std::string& filename, lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int numSlots, int blockSize, size_t rows, size_t cols, bool if_repeat = false, int colSize = 0);

void encryptMatrixtoGPU(const std::string& filename,
  std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& inputs_gpu,
  lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
  FIDESlib::CKKS::Context& GPUcc,
  int numSlots,
  int blockSize,
  size_t rows,
  size_t cols,
  int level		 = 0,
  bool if_repeat = false,
  int colSize	 = 0);

std::vector<std::vector<double>> decryptGPUMatrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& result_gpu,
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& privateKey,
  std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& dummy,
  int numSlots,
  int blockSize,
  int colSize = 0);

std::vector<double> CreateHeadMask(size_t numSlots, size_t blockSize, int head_no);

void load_weights(const std::string& filename, std::vector<std::vector<double>>& matrix_weights, int rows, int cols);

// void load_bias(const std::string& filename, std::vector<double>& bias, int cols);
void load_bias(const std::string& filename, std::vector<std::vector<double>>& bias_matrix, int rows, int cols);

std::vector<std::vector<double>> readGroundTruth(const std::string& filename);

std::vector<std::string> read_sentences_from_csv(const std::string& filename);

std::vector<double> CreateBlockMask(size_t numSlots, size_t blockSize, size_t token_length, double value);

// void MatrixRealMask(std::vector<double>& matrix, int mask_index, int numSlots, int blockSize);

} // namespace FIDESlib::CKKS

#endif // FIDESLIB_BERT_TINY_INPUTS_CUH