#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include "Transpose.cuh"

// #include <cuda/std/__concepts/swappable.h>
#include "CKKS/Ciphertext.cuh"

namespace FIDESlib::CKKS {
void MatrixTransposeSquare_GPU(Ciphertext& cMat1, uint32_t rowSize, const TransposePrecomputations_GPU& precomp) {
    LinearTransform(cMat1, 2 * rowSize - 1, precomp.bStep, precomp.pts_1, (int)(rowSize - 1),
                    -(int)((rowSize - 1) * (rowSize - 1)));
}
std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixTranspose_GPU(
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>&& matrix1, uint32_t rowSize,
    const TransposePrecomputations_GPU& precomp) {

    for (auto& i : matrix1) {
        for (auto& j : i) {
            MatrixTransposeSquare_GPU(j, rowSize, precomp);
        }
    }

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> res(matrix1.size());
    for (int i = 0; i < matrix1.size(); i++) {
        for (int j = 0; j < matrix1[i].size(); j++) {
            res[i].emplace_back(std::move(matrix1[j][i]));
        }
    }

    return res;
}


void MatrixTransposeSquare_GPU_masked(Ciphertext& cMat1, uint32_t rowSize, const TransposePrecomputations_GPU& precomp, Plaintext& mask_pt) {
    LinearTransform(cMat1, 2 * rowSize - 1, precomp.bStep, precomp.pts_1, (int)(rowSize - 1),
                    -(int)((rowSize - 1) * (rowSize - 1)), mask_pt);
}
std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixTranspose_GPU_masked(
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1, 
    uint32_t rowSize,
    const TransposePrecomputations_GPU& precomp,
    Plaintext& mask_pt) 
    {
        for (auto& row : matrix1) {
            for (auto& ct : row) {
                MatrixTransposeSquare_GPU_masked(ct, rowSize, precomp, mask_pt);
            }
        }

        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> res(matrix1.size());
        for (size_t i = 0; i < matrix1.size(); ++i) {
            res[i].reserve(matrix1[i].size());
            for (size_t j = 0; j < matrix1[i].size(); ++j) {
                res[i].emplace_back(std::move(matrix1[j][i]));
            }
        }

        return res;
    }

struct MatrixTransposePrecomputations {
    int rowSize;
    std::vector<lbcrypto::Plaintext> diagPlaintexts;
};

TransposePrecomputations_GPU convertToGPUPrecomputations(Context& GPUcc,
                                                         lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
                                                         MatrixTransposePrecomputations cpuPrecomp, int bStep,
                                                         const int level) {
    struct TransposePrecomputations_GPU gpuPrecomp;
    gpuPrecomp.rowSize = cpuPrecomp.rowSize;

    gpuPrecomp.bStep = bStep;

    {
        auto pt_rots =
            GetLinearTransformPlaintextRotationIndices(2 * gpuPrecomp.rowSize - 1, bStep, (gpuPrecomp.rowSize - 1),
                                                       -(gpuPrecomp.rowSize - 1) * (gpuPrecomp.rowSize - 1));

        for (int i = 0; i < 2 * gpuPrecomp.rowSize - 1; ++i) {
            const auto& sigmaPt_ = cpuPrecomp.diagPlaintexts[i];
            auto sigmaPt = context->MakeCKKSPackedPlaintext(
                lbcrypto::Rotate(sigmaPt_->GetCKKSPackedValue(), pt_rots[i]), 1, GPUcc.L - level);
            auto raw_sigma = FIDESlib::CKKS::GetRawPlainText(context, sigmaPt);
            FIDESlib::CKKS::Plaintext sigma_gpu(GPUcc, raw_sigma);
            gpuPrecomp.diagPlaintexts.emplace_back(std::move(sigma_gpu));
        }

        gpuPrecomp.pts_1.resize(gpuPrecomp.rowSize * 2 - 1);
        for (int i = 0; i < 2 * gpuPrecomp.rowSize - 1; ++i) {
            gpuPrecomp.pts_1[i] = &gpuPrecomp.diagPlaintexts[i];
        }
    }
    return gpuPrecomp;
}

MatrixTransposePrecomputations getMatrixTransposePrecomputations(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, int rowSize) {
    std::vector<std::vector<double>> diagDiagonals(2 * rowSize - 1);

    for (int i = 0; i < 2 * rowSize - 1; ++i) {
        diagDiagonals[i].assign(rowSize * rowSize, 0.0);
        int obj_diff = rowSize - 1 - i;
        int i_ = std::max(rowSize - 1 - i, 0);
        for (int j = i_ - obj_diff; j < rowSize && i_ < rowSize; ++j, ++i_) {
            diagDiagonals[i][i_ * rowSize + j] = 1.0;
        }
    }

    std::vector<lbcrypto::Plaintext> diagPlaintexts(diagDiagonals.size());

    for (int i = 0; i < 2 * rowSize - 1; i++) {
        lbcrypto::Plaintext ptxtSigma = context->MakeCKKSPackedPlaintext(diagDiagonals[i]);
        diagPlaintexts[i] = ptxtSigma;
    }

    struct MatrixTransposePrecomputations precomp;
    precomp.rowSize = rowSize;
    precomp.diagPlaintexts = diagPlaintexts;

    return precomp;
}
TransposePrecomputations_GPU getMatrixTransposePrecomputations_GPU(Context& GPUcc,
                                                                   lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
                                                                   int rowSize, int bStep, const int level) {
    // First get CPU precomputations
    MatrixTransposePrecomputations cpuPrecomp = getMatrixTransposePrecomputations(context, rowSize);

    // Then convert to GPU precomputations
    return convertToGPUPrecomputations(GPUcc, context, cpuPrecomp, bStep, level);
}
std::vector<int> GenerateTransposeRotationIndices_GPU(int rowSize, int bStep) {
    return GetLinearTransformRotationIndices(bStep, rowSize - 1, -(rowSize - 1) * (rowSize - 1));
}

}  // namespace FIDESlib::CKKS