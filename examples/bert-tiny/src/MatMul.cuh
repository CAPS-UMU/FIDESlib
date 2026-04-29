#ifndef FIDESLIB_CKKS_MATMUL_CUH
#define FIDESLIB_CKKS_MATMUL_CUH

#include <CKKS/Ciphertext.cuh>
#include <CKKS/Context.cuh>
#include <CKKS/Plaintext.cuh>
#include <cassert>
#include <iostream>
#include <optional>

#include "Inputs.cuh"
#include "MatMul.h"
#include "PolyApprox.cuh"
#include "Transpose.cuh"

#define LOW_MEM true
#define BSGS true

using namespace lbcrypto;

namespace FIDESlib::CKKS {

struct MatrixMatrixProductPrecomputations_GPU {
	int rowSize;
	std::vector<std::vector<Plaintext>> sigmaPlaintexts;
	std::vector<std::vector<Plaintext>> sigmaPlaintexts_CC;
	std::vector<Plaintext> tauPlaintexts;
	std::vector<std::vector<Plaintext>> phiPlaintexts, phiPlaintexts_new;

#if BSGS
	int bStep;
	std::vector<Plaintext*> pts_1, pts_1_cc, pts_1_r, pts_1_head0, pts_1_head1;
	std::vector<Plaintext*> pts_2, pts_2_head0, pts_2_head1;
	std::vector<Plaintext> pts_1_head0_storage, pts_1_head1_storage;
	std::vector<Plaintext> pts_2_head0_storage, pts_2_head1_storage;

	std::vector<Plaintext*> pts_3_1, pts_3_1_new;
	std::vector<Plaintext*> pts_3_2, pts_3_2_new;
#endif

	MatrixMatrixProductPrecomputations_GPU(const MatrixMatrixProductPrecomputations_GPU&)			 = delete;
	MatrixMatrixProductPrecomputations_GPU& operator=(const MatrixMatrixProductPrecomputations_GPU&) = delete;

	MatrixMatrixProductPrecomputations_GPU(MatrixMatrixProductPrecomputations_GPU&&) noexcept			 = default;
	MatrixMatrixProductPrecomputations_GPU& operator=(MatrixMatrixProductPrecomputations_GPU&&) noexcept = default;

	MatrixMatrixProductPrecomputations_GPU()  = default;
	~MatrixMatrixProductPrecomputations_GPU() = default;
};

void CCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
  std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2,
  uint32_t rowSize,
  std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
  const MatrixMatrixProductPrecomputations_GPU& precomp);

void PCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
  std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2,
  uint32_t rowSize,
  std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
  const MatrixMatrixProductPrecomputations_GPU& precomp,
  std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias);

// PCMM with (masked) bias
void PCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
  std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2,
  uint32_t rowSize,
  std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
  const MatrixMatrixProductPrecomputations_GPU& precomp,
  std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias,
  Plaintext& mask_row);

MatrixMatrixProductPrecomputations_GPU getMatrixMatrixProductPrecomputations_GPU(FIDESlib::CKKS::Context& GPUcc,
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
  int rowSize,
  int bStep,
  int levelCP,
  int levelCC,
  bool fuse_boot_prescale_CCMM,
  int slots);

std::vector<int> GenerateMatMulRotationIndices_GPU(int rowSize, int bStep, int colSize = 0);

FIDESlib::CKKS::Ciphertext rotsum_GPU(FIDESlib::CKKS::Ciphertext& in, int blockSize, int padding);

} // namespace FIDESlib::CKKS

#endif // FIDESLIB_CKKS_MATMUL_CUH