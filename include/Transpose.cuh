#ifndef TRANSPOSE_CUH
#define TRANSPOSE_CUH
#include <cinttypes>
#include <vector>
//#include "CKKS/Plaintext.cuh"
#include "CKKS/forwardDefs.cuh"
#include "openfhe.h"
#include <CKKS/Context.cuh>

namespace FIDESlib::CKKS {

struct TransposePrecomputations_GPU {
    int rowSize;
    std::vector<Plaintext> diagPlaintexts;
    int bStep;
    std::vector<Plaintext*> pts_1;

    // Disable copy constructor and assignment
    TransposePrecomputations_GPU(const TransposePrecomputations_GPU&) = delete;
    TransposePrecomputations_GPU& operator=(const TransposePrecomputations_GPU&) = delete;

    // Enable move constructor and assignment
    TransposePrecomputations_GPU(TransposePrecomputations_GPU&&) noexcept = default;
    TransposePrecomputations_GPU& operator=(TransposePrecomputations_GPU&&) noexcept = default;

    TransposePrecomputations_GPU() = default;
    ~TransposePrecomputations_GPU() = default;
};

void MatrixTransposeSquare_GPU(Ciphertext& cMat1, uint32_t rowSize, const TransposePrecomputations_GPU& precomp);

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixTranspose_GPU(
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>&& matrix1, uint32_t rowSize,
    const TransposePrecomputations_GPU& precomp);

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixTranspose_GPU_masked(
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1, uint32_t rowSize,
    const TransposePrecomputations_GPU& precomp, Plaintext& mask_pt) ;

TransposePrecomputations_GPU getMatrixTransposePrecomputations_GPU(Context& GPUcc,
                            lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
                            int rowSize, int bStep, int level);

std::vector<int> GenerateTransposeRotationIndices_GPU(int rowSize, int bStep);
}  // namespace FIDESlib::CKKS
#endif  //TRANSPOSE_CUH