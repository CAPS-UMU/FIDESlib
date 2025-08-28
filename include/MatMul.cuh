#pragma once

#include <CKKS/Context.cuh>
#include <cassert>
#include <iostream>
#include <optional>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Plaintext.cuh"

#include "MatMul.h"
#include "PolyApprox.cuh"
#include "Transpose.cuh"
#include "Inputs.cuh"

#define LOW_MEM true
#define BSGS true

using namespace lbcrypto;

namespace FIDESlib::CKKS {

struct MatrixMatrixProductPrecomputations_GPU {
    int rowSize;
    std::vector<std::vector<Plaintext>> sigmaPlaintexts;
    //std::vector<std::vector<double>> tauVectors;
    std::vector<Plaintext> tauPlaintexts;
    std::vector<std::vector<Plaintext>> phiPlaintexts, phiPlaintexts_new;

    // std::vector<std::vector<Plaintext>> weightLinearTransform; //

#if BSGS
    int bStep;
    std::vector<Plaintext*> pts_1, pts_1_head0, pts_1_head1;
    std::vector<Plaintext*> pts_2, pts_2_head0, pts_2_head1;
    std::vector<Plaintext> pts_1_head0_storage, pts_1_head1_storage;
    std::vector<Plaintext> pts_2_head0_storage, pts_2_head1_storage;

    std::vector<Plaintext*> pts_3_1, pts_3_1_new;
    std::vector<Plaintext*> pts_3_2, pts_3_2_new;
#endif

    MatrixMatrixProductPrecomputations_GPU(const MatrixMatrixProductPrecomputations_GPU&) = delete;
    MatrixMatrixProductPrecomputations_GPU& operator=(const MatrixMatrixProductPrecomputations_GPU&) = delete;

    MatrixMatrixProductPrecomputations_GPU(MatrixMatrixProductPrecomputations_GPU&&) noexcept = default;
    MatrixMatrixProductPrecomputations_GPU& operator=(MatrixMatrixProductPrecomputations_GPU&&) noexcept = default;

    MatrixMatrixProductPrecomputations_GPU() = default;
    ~MatrixMatrixProductPrecomputations_GPU() = default;
};

void CCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp);

void CCMM_GPU_double_mask(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp, PtMasks_GPU& masks, 
              bool head_no, int token_length, bool if_transpose);
              
// PCMM with delayed normalization (factored and masked) bias
void PCMM_GPU_delayedInv(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias,
              Plaintext& mask_row, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& factor);


void PCMM_2(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias,
              Plaintext& mask_row);


void CCMM_GPU_masked(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp, Plaintext& mask_pt);


void PCMMSquare_GPU(FIDESlib::CKKS::Ciphertext& cMat1, const FIDESlib::CKKS::Plaintext& pMat2, uint32_t rowSize,
                    FIDESlib::CKKS::Ciphertext& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp);

void PCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias);

// PCMM with (masked) bias
void PCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias,
              Plaintext& mask_row);

void PCMM_GPU_QKV_merged(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
            std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>>& matrix2, uint32_t rowSize,
            std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>>& product,
            const MatrixMatrixProductPrecomputations_GPU& precomp,
            std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>>& bias);

// masked bias
void PCMM_GPU_QKV_merged(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
                std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>>& matrix2, uint32_t rowSize,
                std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>>& product,
                const MatrixMatrixProductPrecomputations_GPU& precomp,
                std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>>& bias, Plaintext& mask_row); 


void PCMM_GPU_NoBias(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp);

              
MatrixMatrixProductPrecomputations_GPU getMatrixMatrixProductPrecomputations_GPU(
    FIDESlib::CKKS::Context& GPUcc, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, 
    PtMasks_GPU& masks, int rowSize, int bStep,
    int levelCP, int levelCC, bool fuse_boot_prescale_CCMM, int slots);

std::vector<int> GenerateMatMulRotationIndices_GPU(int rowSize, int bStep);

FIDESlib::CKKS::Ciphertext rotsum_GPU(FIDESlib::CKKS::Ciphertext& in, int blockSize, int padding);

void CombineQKV_2D_to_3(PtWeights_GPU& weights_layer, FIDESlib::CKKS::Context& cc, std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>>& WeightsQKV,
            std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>>& BiasQKV);
}  // namespace FIDESlib::CKKS