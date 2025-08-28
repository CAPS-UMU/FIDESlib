#include "CKKS/LinearTransform.cuh"
#include "MatMul.cuh"

namespace FIDESlib::CKKS {

#if LOW_MEM

void CCMMSquare_GPU(FIDESlib::CKKS::Ciphertext& cMat1, FIDESlib::CKKS::Ciphertext& cMat2, uint32_t rowSize,
                    FIDESlib::CKKS::Ciphertext& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp) {

    // Carlos A.D. TODO: Equalize levels (+ noise degree) of inputs to reduce unnecessary complexity in computations

    FIDESlib::CKKS::Context& cc = cMat1.cc;
    // Step 1-1: Linear transform for first matrix
    FIDESlib::CKKS::Ciphertext linearTransform1(cc);

#if BSGS
    linearTransform1.copy(cMat1);

    // std::cout << "#1: " << linearTransform1.getLevel() << ", " << linearTransform1.NoiseLevel << std::endl;
    LinearTransform(linearTransform1, 2 * rowSize - 1, precomp.bStep, precomp.pts_1, 1, -(int)rowSize + 1);
    // std::cout << "#1: " << linearTransform1.getLevel() << ", " << linearTransform1.NoiseLevel << std::endl;

#else
    linearTransform1.multPt(cMat1, precomp.sigmaPlaintexts[0][0]);

    FIDESlib::CKKS::Ciphertext prev_rotation(cc);
    FIDESlib::CKKS::Ciphertext productCt(cc);

    prev_rotation.copy(cMat1);
    for (size_t i = 1; i < rowSize; i++) {
        prev_rotation.rotate(1, cc.GetRotationKey(1));

        productCt.multPt(prev_rotation, precomp.sigmaPlaintexts[0][i]);
        linearTransform1.add(productCt);
    }

    prev_rotation.copy(cMat1);
    for (size_t i = 1; i < rowSize; i++) {
        prev_rotation.rotate(-1, cc.GetRotationKey(-1));

        productCt.multPt(prev_rotation, precomp.sigmaPlaintexts[1][i - 1]);
        linearTransform1.add(productCt);
    }
#endif

    // Step 1-2: Linear transform for second matrix
    FIDESlib::CKKS::Ciphertext linearTransform2(cc);

#if BSGS
    linearTransform2.copy(cMat2);

    // std::cout << "#2: " << linearTransform2.getLevel() << ", " << linearTransform2.NoiseLevel << std::endl;
    LinearTransform(linearTransform2, rowSize, precomp.bStep, precomp.pts_2, rowSize, 0);
    // std::cout << "#2: " << linearTransform2.getLevel() << ", " << linearTransform2.NoiseLevel << std::endl;
#else
    linearTransform2.multPt(cMat2, precomp.tauPlaintexts[0]);

    prev_rotation.copy(cMat2);

    for (size_t i = 1; i < rowSize; i++) {
        prev_rotation.rotate(rowSize, cc.GetRotationKey(rowSize));

        productCt.multPt(prev_rotation, precomp.tauPlaintexts[i /* * rowSize */]);
        linearTransform2.add(productCt);
    }
#endif

    // cProduct.copy(linearTransform1);
    //
    // Steps 2 and 3: Initial computation

#if BSGS
    Ciphertext aux(cc);
    aux.rotate(linearTransform1, -(int)rowSize /*+ 1*/, cc.GetRotationKey(-(int)rowSize /*+ 1*/));
    LinearTransformSpecial(linearTransform1, aux, linearTransform2, rowSize, precomp.bStep, precomp.pts_3_1,
                           precomp.pts_3_2, 1, rowSize);

    // std::cout << "#3: " << linearTransform1.getLevel() << ", " << linearTransform1.NoiseLevel << std::endl;

    cProduct.copy(linearTransform1);
#else
    cProduct.mult(linearTransform1, linearTransform2, cc.GetEvalKey());
    prev_rotation.copy(linearTransform1);
    FIDESlib::CKKS::Ciphertext prev_rotation_minus_row(cc);
    prev_rotation_minus_row.rotate(prev_rotation, -(int)rowSize, cc.GetRotationKey(-(int)rowSize));
    FIDESlib::CKKS::Ciphertext productCt2(cc);

    FIDESlib::CKKS::Ciphertext prev_rotation2(cc);
    prev_rotation2.copy(linearTransform2);

    for (size_t i = 1; i < rowSize; i++) {
        // Step 2
        prev_rotation.rotate(1, linearTransform1.cc.GetRotationKey(1));

        productCt.multPt(prev_rotation, precomp.phiPlaintexts[i][0]);

        prev_rotation_minus_row.rotate(1, linearTransform1.cc.GetRotationKey(1));

        productCt2.multPt(prev_rotation_minus_row, precomp.phiPlaintexts[i][1]);

        productCt.add(productCt2);

        prev_rotation2.rotate(rowSize, cc.GetRotationKey(rowSize));

        // Step 3
        productCt.mult(prev_rotation2, cc.GetEvalKey());
        cProduct.add(productCt);
    }

#endif
}

#else

void CCMMSquare_GPU(FIDESlib::CKKS::Ciphertext& cMat1, FIDESlib::CKKS::Ciphertext& cMat2, uint32_t rowSize,
                    FIDESlib::CKKS::Ciphertext& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp) {

    // Step 1-1: Linear transform for first matrix
    FIDESlib::CKKS::Ciphertext linearTransform1(cMat1.cc);

    for (size_t i = 1; i < rowSize; i++) {
        FIDESlib::CKKS::Ciphertext rotatedCt(cMat1.cc);
        rotatedCt.copy(cMat1);
        rotatedCt.rotate(i, cMat1.cc.GetRotationKey(i));

        FIDESlib::CKKS::Ciphertext productCt(rotatedCt.cc);
        productCt.multPt(rotatedCt, precomp.sigmaPlaintexts[0][i]);
        linearTransform1.add(productCt);

        rotatedCt.copy(cMat1);
        rotatedCt.rotate(-i, cMat1.cc.GetRotationKey(-i));

        productCt.multPt(rotatedCt, precomp.sigmaPlaintexts[0][rowSize * rowSize - i]);
        linearTransform1.add(productCt);
    }

    // Step 1-2: Linear transform for second matrix
    FIDESlib::CKKS::Ciphertext linearTransform2(cMat2.cc);
    linearTransform2.multPt(cMat2, precomp.tauPlaintexts[0]);

    for (size_t i = 1; i < rowSize; i++) {
        FIDESlib::CKKS::Ciphertext rotatedCt(cMat2.cc);
        rotatedCt.copy(cMat2);
        rotatedCt.rotate(i * rowSize, cMat2.cc.GetRotationKey(i * rowSize));

        FIDESlib::CKKS::Ciphertext productCt(rotatedCt.cc);
        productCt.multPt(rotatedCt, precomp.tauPlaintexts[i * rowSize]);
        linearTransform2.add(productCt);
    }

    // Steps 2 and 3: Initial computation
    cProduct.mult(linearTransform1, linearTransform2, linearTransform2.cc.GetEvalKey());

    for (size_t i = 1; i < rowSize; i++) {
        // Step 2
        FIDESlib::CKKS::Ciphertext rotatedCt(linearTransform1.cc);
        rotatedCt.copy(linearTransform1);
        rotatedCt.rotate(i, linearTransform1.cc.GetRotationKey(i));

        FIDESlib::CKKS::Ciphertext productCt1(rotatedCt.cc);
        productCt1.multPt(rotatedCt, precomp.phiPlaintexts[i][0]);

        rotatedCt.copy(linearTransform1);
        rotatedCt.rotate(i - rowSize, linearTransform1.cc.GetRotationKey(i - rowSize));

        FIDESlib::CKKS::Ciphertext productCt2(rotatedCt.cc);
        productCt2.multPt(rotatedCt, precomp.phiPlaintexts[i][1]);

        FIDESlib::CKKS::Ciphertext linearTransformPhi(productCt1.cc);
        linearTransformPhi.add(productCt1, productCt2);

        FIDESlib::CKKS::Ciphertext linearTransformPsi(linearTransform2.cc);
        linearTransformPsi.copy(linearTransform2);
        linearTransformPsi.rotate(i * rowSize, linearTransform2.cc.GetRotationKey(i * rowSize));

        // Step 3
        FIDESlib::CKKS::Ciphertext tempProduct(linearTransformPhi.cc);
        tempProduct.mult(linearTransformPhi, linearTransformPsi, linearTransformPsi.cc.GetEvalKey());
        cProduct.add(tempProduct);
    }
}

#endif

// matrix multiplication
void CCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp) {

    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix2[0].size());

        for (size_t j = 0; j < matrix2[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext dotProd(matrix1[i][0].cc);
            CCMMSquare_GPU(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);
            for (size_t k = 1; k < matrix2.size(); k++) {
                FIDESlib::CKKS::Ciphertext dotProdNew(matrix1[i][k].cc);
                CCMMSquare_GPU(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);
                dotProd.add(dotProdNew);
            }
            row.emplace_back(std::move(dotProd));
        }
        product.emplace_back(std::move(row));
    }
}

void CCMMSquare_GPU_masked(FIDESlib::CKKS::Ciphertext& cMat1, FIDESlib::CKKS::Ciphertext& cMat2, uint32_t rowSize,
                    FIDESlib::CKKS::Ciphertext& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp, Plaintext& mask_pt) {

    FIDESlib::CKKS::Context& cc = cMat1.cc;
    // Step 1-1: Linear transform for first matrix
    FIDESlib::CKKS::Ciphertext linearTransform1(cc);

    linearTransform1.copy(cMat1);

    // std::cout << "#1: " << linearTransform1.getLevel() << ", " << linearTransform1.NoiseLevel << std::endl;
    LinearTransform(linearTransform1, 2 * rowSize - 1, precomp.bStep, precomp.pts_1, 1, -(int)rowSize + 1);
    // std::cout << "#1: " << linearTransform1.getLevel() << ", " << linearTransform1.NoiseLevel << std::endl;

    // Step 1-2: Linear transform for second matrix
    FIDESlib::CKKS::Ciphertext linearTransform2(cc);
    linearTransform2.copy(cMat2);

    // std::cout << "#2: " << linearTransform2.getLevel() << ", " << linearTransform2.NoiseLevel << std::endl;
    LinearTransform(linearTransform2, rowSize, precomp.bStep, precomp.pts_2, rowSize, 0);
    // std::cout << "#2: " << linearTransform2.getLevel() << ", " << linearTransform2.NoiseLevel << std::endl;

    // Steps 2 and 3: Initial computation
    Ciphertext aux(cc);
    aux.rotate(linearTransform1, -(int)rowSize /*+ 1*/, cc.GetRotationKey(-(int)rowSize /*+ 1*/));
    LinearTransformSpecial(linearTransform1, aux, linearTransform2, rowSize, precomp.bStep, precomp.pts_3_1,
                           precomp.pts_3_2, 1, rowSize, mask_pt);

    // std::cout << "#3: " << linearTransform1.getLevel() << ", " << linearTransform1.NoiseLevel << std::endl;

    cProduct.copy(linearTransform1);

}

// masked matrix multiplication
void CCMM_GPU_masked(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp, Plaintext& mask_pt) {
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> cProduct;
    cProduct.reserve(matrix1.size());
    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix2[0].size());

        for (size_t j = 0; j < matrix2[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext dotProd(matrix1[i][0].cc);
            CCMMSquare_GPU_masked(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp, mask_pt);
            for (size_t k = 1; k < matrix2.size(); k++) {
                FIDESlib::CKKS::Ciphertext dotProdNew(matrix1[i][k].cc);
                CCMMSquare_GPU_masked(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp, mask_pt);
                dotProd.add(dotProdNew);
            }
            row.emplace_back(std::move(dotProd));
        }
        cProduct.emplace_back(std::move(row));
    }
    product.emplace_back(std::move(cProduct));
}



// double masking, in and out
void CCMMSquare_GPU_double_mask(FIDESlib::CKKS::Ciphertext& cMat1, FIDESlib::CKKS::Ciphertext& cMat2, uint32_t rowSize,
                    FIDESlib::CKKS::Ciphertext& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp, 
                    PtMasks_GPU& masks, bool head_no, int token_length, bool if_transpose) {

    FIDESlib::CKKS::Context& cc = cMat1.cc;
    // Step 1-1: Linear transform for first matrix
    FIDESlib::CKKS::Ciphertext linearTransform1(cc);
    linearTransform1.copy(cMat1);

    // std::cout << "#1: " << linearTransform1.getLevel() << ", " << linearTransform1.NoiseLevel << std::endl;
    LinearTransform(linearTransform1, 2 * rowSize - 1, precomp.bStep, precomp.pts_1, 1, -(int)rowSize + 1, masks.head_masks[head_no]);
    // std::cout << "#1: " << linearTransform1.getLevel() << ", " << linearTransform1.NoiseLevel << std::endl;

    // Step 1-2: Linear transform for second matrix
    FIDESlib::CKKS::Ciphertext linearTransform2(cc);
    linearTransform2.copy(cMat2);

    // std::cout << "#2: " << linearTransform2.getLevel() << ", " << linearTransform2.NoiseLevel << std::endl;
    // LinearTransform(linearTransform2, rowSize, precomp.bStep, precomp.pts_2, rowSize, 0, mask_pt);

    if (if_transpose) {
        LinearTransform(linearTransform2, rowSize, precomp.bStep, precomp.pts_2, rowSize, 0);
    }
    else {
        LinearTransform(linearTransform2, rowSize, precomp.bStep, precomp.pts_2, rowSize, 0, masks.head_masks[head_no]);
    }

    // std::cout << "#2: " << linearTransform2.getLevel() << ", " << linearTransform2.NoiseLevel << std::endl;

    // Steps 2 and 3: Initial computation
    Ciphertext aux(cc);
    aux.rotate(linearTransform1, -(int)rowSize /*+ 1*/, cc.GetRotationKey(-(int)rowSize /*+ 1*/));
    // LinearTransformSpecial(linearTransform1, aux, linearTransform2, rowSize, precomp.bStep, precomp.pts_3_1,
    //                        precomp.pts_3_2, 1, rowSize);

    Plaintext masks_double(cc);
    masks_double.copy(masks.head_masks[0]);
    masks_double.multPt(masks_double, masks.row_masks[token_length], true);

    LinearTransformSpecial(linearTransform1, aux, linearTransform2, rowSize, precomp.bStep, precomp.pts_3_1,
                           precomp.pts_3_2, 1, rowSize, masks_double);

    // std::cout << "#3: " << linearTransform1.getLevel() << ", " << linearTransform1.NoiseLevel << std::endl;
    
    cProduct.copy(linearTransform1);

}

// masked matrix multiplication
void CCMM_GPU_double_mask(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp, PtMasks_GPU& masks, bool head_no, int token_length, bool if_transpose) {
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> cProduct;
    cProduct.reserve(matrix1.size());
    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix2[0].size());

        for (size_t j = 0; j < matrix2[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext dotProd(matrix1[i][0].cc);
            CCMMSquare_GPU_double_mask(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp, masks, head_no, token_length, if_transpose);
            for (size_t k = 1; k < matrix2.size(); k++) {
                FIDESlib::CKKS::Ciphertext dotProdNew(matrix1[i][k].cc);
                CCMMSquare_GPU_double_mask(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp, masks, head_no, token_length, if_transpose);
                dotProd.add(dotProdNew);
            }
            row.emplace_back(std::move(dotProd));
        }
        cProduct.emplace_back(std::move(row));
    }
    product.emplace_back(std::move(cProduct));
}


void PCMMSquare_GPU(FIDESlib::CKKS::Ciphertext& cMat1, const FIDESlib::CKKS::Plaintext& pMat2, uint32_t rowSize,
                    FIDESlib::CKKS::Ciphertext& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp) {

    FIDESlib::CKKS::Context& cc = cMat1.cc;
    // Step 1-1: Linear transform for first matrix
    FIDESlib::CKKS::Ciphertext linearTransform1(cc);
    linearTransform1.copy(cMat1);
    LinearTransform(linearTransform1, 2 * rowSize - 1, precomp.bStep, precomp.pts_1, 1, -(int)rowSize + 1);

    // Step 1-2: Linear transform for second matrix
    FIDESlib::CKKS::Plaintext linearTransform2(cc);
    linearTransform2.copy(pMat2);

    //LinearTransformPt(linearTransform2, cc, rowSize, precomp.bStep, precomp.pts_2, rowSize, 0);

    // Steps 2 and 3: Initial computation
    Ciphertext aux(cc);
    aux.rotate(linearTransform1, -(int)rowSize /*+ 1*/, cc.GetRotationKey(-(int)rowSize /*+ 1*/));
    LinearTransformSpecialPt(linearTransform1, aux, linearTransform2, rowSize, precomp.bStep, precomp.pts_3_1,
                             precomp.pts_3_2, 1,
                             rowSize);  // LT2 is now plaintext
    if (linearTransform1.NoiseLevel == 2)
        linearTransform1.rescale();
    // linearTransform1.dropToLevel(cMat1.getLevel() - 3);
    cProduct.copy(linearTransform1);
}

// PCMM with bias
void PCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias) {
    product.clear();
    product.reserve(matrix1.size());
    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix2[0].size());
        for (size_t j = 0; j < matrix2[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext dotProd(matrix1[i][0].cc);
            PCMMSquare_GPU(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);
            dotProd.addPt(bias[0][j]);
            for (size_t k = 1; k < matrix2.size(); k++) {
                FIDESlib::CKKS::Ciphertext dotProdNew(matrix1[i][k].cc);
                PCMMSquare_GPU(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);
                dotProd.add(dotProdNew);
            }
            row.emplace_back(std::move(dotProd));
        }
        product.emplace_back(std::move(row));
    }

}


// PCMM with (masked) bias
void PCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias,
              Plaintext& mask_row) {
    product.clear();
    product.reserve(matrix1.size());
    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix2[0].size());
        for (size_t j = 0; j < matrix2[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext dotProd(matrix1[i][0].cc);
            PCMMSquare_GPU(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);

            Plaintext bias_masked(matrix1[i][0].cc);
            bias_masked.copy(bias[0][j]);
            if (mask_row.NoiseLevel > 1) { mask_row.rescale(); }
            bias_masked.multPt(bias_masked, mask_row, true);

            dotProd.addPt(bias_masked);
            for (size_t k = 1; k < matrix2.size(); k++) {
                FIDESlib::CKKS::Ciphertext dotProdNew(matrix1[i][k].cc);
                PCMMSquare_GPU(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);
                dotProd.add(dotProdNew);
            }
            row.emplace_back(std::move(dotProd));
        }
        product.emplace_back(std::move(row));
    }

}



// void PCMMSquare_GPU_masked(FIDESlib::CKKS::Ciphertext& cMat1, const FIDESlib::CKKS::Plaintext& pMat2, uint32_t rowSize,
//                     std::vector<FIDESlib::CKKS::Ciphertext>& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp, std::vector<Plaintext>& mask_pt) {

//     FIDESlib::CKKS::Context& cc = cMat1.cc;
//     // Step 1-1: Linear transform for first matrix
//     FIDESlib::CKKS::Ciphertext linearTransform1(cc);
//     linearTransform1.copy(cMat1);
//     LinearTransform(linearTransform1, 2 * rowSize - 1, precomp.bStep, precomp.pts_1, 1, -(int)rowSize + 1);

//     // Step 1-2: Linear transform for second matrix
//     FIDESlib::CKKS::Plaintext linearTransform2(cc);
//     linearTransform2.copy(pMat2);

//     //LinearTransformPt(linearTransform2, cc, rowSize, precomp.bStep, precomp.pts_2, rowSize, 0);

//     // Steps 2 and 3: Initial computation
//     Ciphertext aux(cc);
//     aux.rotate(linearTransform1, -(int)rowSize /*+ 1*/, cc.GetRotationKey(-(int)rowSize /*+ 1*/));
    
    
//     std::vector<FIDESlib::CKKS::Ciphertext> results;
//     results.reserve(mask_pt.size());
//     LinearTransformSpecialPt(linearTransform1, aux, linearTransform2, rowSize, precomp.bStep, precomp.pts_3_1,
//                              precomp.pts_3_2, 1, rowSize, mask_pt, results);  

//     for (int i = 0; i < results.size(); i++){
//         if (results[i].NoiseLevel == 2)
//             results[i].rescale();
//     }

//     // linearTransform1.dropToLevel(cMat1.getLevel() - 3);
//     cProduct.reserve(results.size());
//     for (int i = 0; i < results.size(); i++){
//         cProduct.emplace_back(std::move(results[i]));
//     }
// }

// // PCMM with bias, masked
// void PCMM_GPU_masked(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
//               std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
//               std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>>& product,
//               const MatrixMatrixProductPrecomputations_GPU& precomp,
//               std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias, std::vector<Plaintext>& mask_pt) {

//     product.reserve(matrix1.size());

//     for (size_t i = 0; i < matrix1.size(); i++) {
//         std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> row;
//         row.reserve(matrix2[0].size());
//         for (size_t j = 0; j < matrix2[0].size(); j++) {
//             std::vector<FIDESlib::CKKS::Ciphertext> dotProd;
//             PCMMSquare_GPU_masked(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp, mask_pt);            
//             for (size_t k = 1; k < matrix2.size(); k++) {
//                 std::vector<FIDESlib::CKKS::Ciphertext> dotProdNew;
//                 PCMMSquare_GPU_masked(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp, mask_pt);
//                 for (int m = 0; m < dotProd.size(); m++){
//                     dotProd[m].add(dotProdNew[m]);
//                 }
//             }
//             for (int m = 0; m < dotProd.size(); m++){
//                 dotProd[m].addPt(bias[0][j]);
//             }
//             row.emplace_back(std::move(dotProd));
//         }
//         product.emplace_back(std::move(row));
//     }
// }

///// Q, K, V merged 
void PCMMSquare_GPU_merged(FIDESlib::CKKS::Ciphertext& cMat1, const std::vector<FIDESlib::CKKS::Plaintext>& pMat2, uint32_t rowSize,
                    std::vector<FIDESlib::CKKS::Ciphertext>& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp) {

    FIDESlib::CKKS::Context& cc = cMat1.cc;

    // std::cout << "Step 1-1: ";
    // auto start_gpu = std::chrono::high_resolution_clock::now();

    // Step 1-1: Linear transform for first matrix
    FIDESlib::CKKS::Ciphertext linearTransform1(cc);
    linearTransform1.copy(cMat1);
    LinearTransform(linearTransform1, 2 * rowSize - 1, precomp.bStep, precomp.pts_1, 1, -(int)rowSize + 1);

    // Step 1-2: Linear transform for second matrix
    std::vector<FIDESlib::CKKS::Plaintext> linearTransform2;
    for (int i = 0; i < pMat2.size(); i++){
        Plaintext lt(cc);
        lt.copy(pMat2[i]);
        linearTransform2.emplace_back(std::move(lt));
    }

    // auto end_gpu = std::chrono::high_resolution_clock::now();
    // double time1 = (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count());
        
    // std::cout << "Step 2 and 3: ";
    // start_gpu = std::chrono::high_resolution_clock::now();

    // Steps 2 and 3: Initial computation
    Ciphertext aux(cc);
    aux.rotate(linearTransform1, -(int)rowSize /*+ 1*/, cc.GetRotationKey(-(int)rowSize /*+ 1*/));
    
    FIDESlib::CKKS::Ciphertext cProduct1(cc), cProduct2(cc), cProduct3(cc);

    cProduct.reserve(pMat2.size());
    // K
    LinearTransformSpecialPt(linearTransform1, aux, linearTransform2[0], rowSize, precomp.bStep, precomp.pts_3_1,
                             precomp.pts_3_2, 1, rowSize, cProduct1);  
    cProduct.emplace_back(std::move(cProduct1));

    // end_gpu = std::chrono::high_resolution_clock::now();
    // double time2 = (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count());

    // std::cout << "ratio: " << time1 / time2 << std::endl;
        
    // Q
    LinearTransformSpecialPt(linearTransform1, aux, linearTransform2[1], rowSize, precomp.bStep, precomp.pts_3_1,
                             precomp.pts_3_2, 1, rowSize, cProduct2);  
    cProduct.emplace_back(std::move(cProduct2));
    // V
    LinearTransformSpecialPt(linearTransform1, aux, linearTransform2[2], rowSize, precomp.bStep, precomp.pts_3_1,
                             precomp.pts_3_2, 1, rowSize, cProduct3);  
    cProduct.emplace_back(std::move(cProduct3));

    for (int i = 0; i < cProduct.size(); i++){
        if (cProduct[i].NoiseLevel == 2)
            cProduct[i].rescale();
    }
}

    void PCMM_GPU_QKV_merged(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
                std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>>& matrix2, uint32_t rowSize,
                std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>>& product,
                const MatrixMatrixProductPrecomputations_GPU& precomp,
                std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>>& bias) {

        product.clear();
        product.reserve(3);

        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> product1, product2, product3;
        product1.reserve(matrix1.size());
        product2.reserve(matrix1.size());
        product3.reserve(matrix1.size());

        for (size_t i = 0; i < matrix1.size(); i++) {
            std::vector<FIDESlib::CKKS::Ciphertext> row, row2, row3;
            row.reserve(matrix2[0].size());
            row2.reserve(matrix2[0].size());
            row3.reserve(matrix2[0].size());
            for (size_t j = 0; j < matrix2[0].size(); j++) {
                std::vector<FIDESlib::CKKS::Ciphertext> dotProd;
                dotProd.reserve(3);
                PCMMSquare_GPU_merged(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);

                for (size_t m = 0; m < dotProd.size(); ++m)
                    dotProd[m].addPt(bias[0][j][m]);

                for (size_t k = 1; k < matrix2.size(); k++) {
                    std::vector<FIDESlib::CKKS::Ciphertext> dotProdNew;
                    dotProdNew.reserve(3);
                    PCMMSquare_GPU_merged(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);

                    for (size_t m = 0; m < dotProd.size(); ++m)
                        dotProd[m].add(dotProdNew[m]);
                }
                row.emplace_back(std::move(dotProd[0]));
                row2.emplace_back(std::move(dotProd[1]));
                row3.emplace_back(std::move(dotProd[2]));
            }
            product1.emplace_back(std::move(row));
            product2.emplace_back(std::move(row2));
            product3.emplace_back(std::move(row3));
        }
        product.emplace_back(std::move(product1));
        product.emplace_back(std::move(product2));
        product.emplace_back(std::move(product3));
    }


    // masked bias
    void PCMM_GPU_QKV_merged(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
                std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>>& matrix2, uint32_t rowSize,
                std::vector<std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>>& product,
                const MatrixMatrixProductPrecomputations_GPU& precomp,
                std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>>& bias, Plaintext& mask_row) {

        product.clear();
        product.reserve(3);

        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> product1, product2, product3;
        product1.reserve(matrix1.size());
        product2.reserve(matrix1.size());
        product3.reserve(matrix1.size());

        for (size_t i = 0; i < matrix1.size(); i++) {
            std::vector<FIDESlib::CKKS::Ciphertext> row, row2, row3;
            row.reserve(matrix2[0].size());
            row2.reserve(matrix2[0].size());
            row3.reserve(matrix2[0].size());
            for (size_t j = 0; j < matrix2[0].size(); j++) {
                std::vector<FIDESlib::CKKS::Ciphertext> dotProd;
                dotProd.reserve(3);
                PCMMSquare_GPU_merged(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);

                for (size_t m = 0; m < dotProd.size(); ++m){
                    Plaintext bias_masked(matrix1[i][0].cc);
                    bias_masked.copy(bias[0][j][m]);
                    bias_masked.multPt(bias_masked, mask_row, true);
                    
                    dotProd[m].addPt(bias_masked);

                    // dotProd[m].addPt(bias[0][j][m]);

                }
                for (size_t k = 1; k < matrix2.size(); k++) {
                    std::vector<FIDESlib::CKKS::Ciphertext> dotProdNew;
                    dotProdNew.reserve(3);
                    PCMMSquare_GPU_merged(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);

                    for (size_t m = 0; m < dotProd.size(); ++m)
                        dotProd[m].add(dotProdNew[m]);
                }
                row.emplace_back(std::move(dotProd[0]));
                row2.emplace_back(std::move(dotProd[1]));
                row3.emplace_back(std::move(dotProd[2]));
            }
            product1.emplace_back(std::move(row));
            product2.emplace_back(std::move(row2));
            product3.emplace_back(std::move(row3));
        }
        product.emplace_back(std::move(product1));
        product.emplace_back(std::move(product2));
        product.emplace_back(std::move(product3));
    }


// PCMM without bias
void PCMM_GPU_NoBias(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp) {

    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix2[0].size());
        for (size_t j = 0; j < matrix2[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext dotProd(matrix1[i][0].cc);
            PCMMSquare_GPU(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);
            for (size_t k = 1; k < matrix2.size(); k++) {
                FIDESlib::CKKS::Ciphertext dotProdNew(matrix1[i][k].cc);
                PCMMSquare_GPU(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);
                dotProd.add(dotProdNew);
            }
            row.emplace_back(std::move(dotProd));
        }
        product.emplace_back(std::move(row));
    }
}

// Function to convert lbcrypto precomputations to GPU precomputations
struct MatrixMatrixProductPrecomputations_GPU convertToGPUPrecomputations(
    FIDESlib::CKKS::Context& GPUcc, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
    const MatrixMatrixProductPrecomputations& cpuPrecomp, PtMasks_GPU& masks, int bStep, int levelCP, int levelCC,
    bool fuse_boot_prescale_CCMM, const int slots) {

    // Adjustment for the new PCMM
    levelCP = levelCP + 1;
    levelCC = levelCC + 1;

    struct MatrixMatrixProductPrecomputations_GPU gpuPrecomp;
    gpuPrecomp.rowSize = cpuPrecomp.rowSize;

#if BSGS
    gpuPrecomp.bStep = bStep;
#endif

    gpuPrecomp.sigmaPlaintexts.resize(2);
    // Convert sigma plaintexts
    {
        auto pt_rots =
            GetLinearTransformPlaintextRotationIndices(2 * gpuPrecomp.rowSize - 1, bStep, 1, -gpuPrecomp.rowSize + 1);

        for (int i = 0; i < gpuPrecomp.rowSize; ++i) {

            const auto& sigmaPt_ = cpuPrecomp.sigmaPlaintexts[i];
#if BSGS
#if !EXT
            auto sigmaPt = context->MakeCKKSPackedPlaintext(
                Rotate(sigmaPt_->GetCKKSPackedValue(), pt_rots[i + gpuPrecomp.rowSize - 1]), 1, GPUcc.L - levelCP);
#else
            auto sigmaPt =
                encodeExt(context, Rotate(sigmaPt_->GetCKKSPackedValue(), pt_rots[i + gpuPrecomp.rowSize - 1]), 1,
                          GPUcc.L - level);
#endif
#endif
            auto raw_sigma = FIDESlib::CKKS::GetRawPlainText(context, sigmaPt);
            FIDESlib::CKKS::Plaintext sigma_gpu(GPUcc, raw_sigma);
            gpuPrecomp.sigmaPlaintexts[0].emplace_back(std::move(sigma_gpu));
        }

        for (int i = 0; i < gpuPrecomp.rowSize - 1; ++i) {
            const auto& sigmaPt_ = cpuPrecomp.sigmaPlaintexts[gpuPrecomp.rowSize * gpuPrecomp.rowSize - 1 - i];
#if BSGS
#if !EXT
            auto sigmaPt = context->MakeCKKSPackedPlaintext(
                Rotate(sigmaPt_->GetCKKSPackedValue(), pt_rots[gpuPrecomp.rowSize - 2 - i]), 1, GPUcc.L - levelCP);
#else

            auto sigmaPt =
                encodeExt(context, Rotate(sigmaPt_->GetCKKSPackedValue(), pt_rots[gpuPrecomp.rowSize - 2 - i]), 1,
                          GPUcc.L - level + 1);
#endif
#endif
            auto raw_sigma = FIDESlib::CKKS::GetRawPlainText(context, sigmaPt);
            FIDESlib::CKKS::Plaintext sigma_gpu(GPUcc, raw_sigma);
            gpuPrecomp.sigmaPlaintexts[1].emplace_back(std::move(sigma_gpu));
        }
#if BSGS
        gpuPrecomp.pts_1.resize(gpuPrecomp.rowSize * 2 - 1);
        for (int i = 0; i < gpuPrecomp.rowSize - 1; ++i) {
            gpuPrecomp.pts_1[gpuPrecomp.rowSize - 2 - i] = &gpuPrecomp.sigmaPlaintexts[1][i];
        }
        for (int i = 0; i < gpuPrecomp.rowSize; ++i) {
            gpuPrecomp.pts_1[i + gpuPrecomp.rowSize - 1] = &gpuPrecomp.sigmaPlaintexts[0][i];
        }
#endif
    }
    {
        auto pt_rots = GetLinearTransformPlaintextRotationIndices(gpuPrecomp.rowSize, bStep, gpuPrecomp.rowSize);

        for (int i = 0; i < gpuPrecomp.rowSize; ++i) {
            const auto& tauPt_ = cpuPrecomp.tauPlaintexts[i * gpuPrecomp.rowSize];
            auto values = tauPt_->GetCKKSPackedValue();
            if (fuse_boot_prescale_CCMM) {
                double scale = FIDESlib::CKKS::GetPreScaleFactor(GPUcc, slots);
                //std::cout << scale << std::endl;
                for (auto& k : values)
                    k *= scale;
            } else {
                // for (auto& k : values)
                //     k *= 0.001;
            }

#if BSGS
#if !EXT
            auto tauPt = context->MakeCKKSPackedPlaintext(Rotate(values, pt_rots[i]), 1, GPUcc.L - levelCC);
#else

            auto tauPt = encodeExt(context, Rotate(values, pt_rots[i]), 1, GPUcc.L - level);
#endif
#endif
            {
                auto raw_tau = FIDESlib::CKKS::GetRawPlainText(context, tauPt);
                FIDESlib::CKKS::Plaintext tau_gpu(GPUcc, raw_tau);
                gpuPrecomp.tauPlaintexts.emplace_back(std::move(tau_gpu));
            }
        }

#if BSGS
        gpuPrecomp.pts_2.resize(gpuPrecomp.rowSize);
        for (int i = 0; i < gpuPrecomp.rowSize; ++i) {
            gpuPrecomp.pts_2[i] = &gpuPrecomp.tauPlaintexts[i];
        }
#endif
    }

    {
        auto pt_rots = GetLinearTransformPlaintextRotationIndices(gpuPrecomp.rowSize, bStep, 1, 0);
        // Convert phi plaintexts

        int i = 0;
        for (const auto& phiVec : cpuPrecomp.phiPlaintexts) {
            std::vector<FIDESlib::CKKS::Plaintext> phiGpuVec;
            int j = 0;
            for (const auto& phiPt_ : phiVec) {
                const auto& tauPt_ = cpuPrecomp.tauPlaintexts[i * gpuPrecomp.rowSize];
                auto values = phiPt_->GetCKKSPackedValue();
#if BSGS
#if !EXT
                auto phiPt = context->MakeCKKSPackedPlaintext(Rotate(values, pt_rots[i]), 1, GPUcc.L - levelCP + 1);
#else
                auto phiPt = context->MakeCKKSPackedPlaintext(Rotate(phiPt_->GetCKKSPackedValue(), pt_rots[i]), 1,
                                                              GPUcc.L - level);
                //        auto phiPt = encodeExt(context, Rotate(phiPt_->GetCKKSPackedValue(), pt_rots[i]), 1, GPUcc.L - level);
#endif
#endif
                auto raw_phi = FIDESlib::CKKS::GetRawPlainText(context, phiPt);
                FIDESlib::CKKS::Plaintext phi_gpu(GPUcc, raw_phi);

                // NEW PCMM!!!!
                // phi_gpu.multPt(phi_gpu, masks.U_masks[0], true);
                phiGpuVec.emplace_back(std::move(phi_gpu));
                ++j;
            }
            gpuPrecomp.phiPlaintexts.emplace_back(std::move(phiGpuVec));
            ++i;
        }

#if BSGS
        gpuPrecomp.pts_3_2.resize(gpuPrecomp.rowSize);
        for (int i = 1; i < gpuPrecomp.rowSize; ++i) {
            gpuPrecomp.pts_3_2[i] = &gpuPrecomp.phiPlaintexts[i][1];
        }
        gpuPrecomp.pts_3_1.resize(gpuPrecomp.rowSize);
        for (int i = 0; i < gpuPrecomp.rowSize; ++i) {
            gpuPrecomp.pts_3_1[i] = &gpuPrecomp.phiPlaintexts[i][0];
        }
#endif
    }


        {
        auto pt_rots = GetLinearTransformPlaintextRotationIndices(gpuPrecomp.rowSize, bStep, 1, 0);
        // Convert phi plaintexts

        int i = 0;
        for (const auto& phiVec : cpuPrecomp.phiPlaintexts_new) {
            std::vector<FIDESlib::CKKS::Plaintext> phiGpuVec;
            int j = 0;
            for (const auto& phiPt_ : phiVec) {
                const auto& tauPt_ = cpuPrecomp.tauPlaintexts[i * gpuPrecomp.rowSize];
                auto values = phiPt_->GetCKKSPackedValue();
#if BSGS
#if !EXT
                auto phiPt = context->MakeCKKSPackedPlaintext(Rotate(values, pt_rots[i]), 1, GPUcc.L - levelCP + 1);
#else
                auto phiPt = context->MakeCKKSPackedPlaintext(Rotate(phiPt_->GetCKKSPackedValue(), pt_rots[i]), 1,
                                                              GPUcc.L - level);
                //        auto phiPt = encodeExt(context, Rotate(phiPt_->GetCKKSPackedValue(), pt_rots[i]), 1, GPUcc.L - level);
#endif
#endif
                auto raw_phi = FIDESlib::CKKS::GetRawPlainText(context, phiPt);
                FIDESlib::CKKS::Plaintext phi_gpu(GPUcc, raw_phi);
                phiGpuVec.emplace_back(std::move(phi_gpu));
                ++j;
            }
            gpuPrecomp.phiPlaintexts_new.emplace_back(std::move(phiGpuVec));
            ++i;

        }

#if BSGS
        gpuPrecomp.pts_3_2_new.resize(gpuPrecomp.rowSize);
        for (int i = 1; i < gpuPrecomp.rowSize; ++i) {
            gpuPrecomp.pts_3_2_new[i] = &gpuPrecomp.phiPlaintexts_new[i][1];
        }
        gpuPrecomp.pts_3_1_new.resize(gpuPrecomp.rowSize);
        for (int i = 0; i < gpuPrecomp.rowSize; ++i) {
            gpuPrecomp.pts_3_1_new[i] = &gpuPrecomp.phiPlaintexts_new[i][0];
        }
#endif
    }


    //gpuPrecomp.tauVectors.resize(gpuPrecomp.rowSize);
    //for (int i = 0; i < gpuPrecomp.rowSize; ++i) {
    //        gpuPrecomp.tauVectors[i] = cpuPrecomp.tauVectors[i];
    //    }

    return gpuPrecomp;
}

// Direct GPU version of getMatrixMatrixProductPrecomputations
struct MatrixMatrixProductPrecomputations_GPU getMatrixMatrixProductPrecomputations_GPU(
    FIDESlib::CKKS::Context& GPUcc, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, PtMasks_GPU& masks, int rowSize, int bStep,
    const int levelCP, const int levelCC, const bool fuse_boot_prescale_CCMM, const int slots) {

    // First get CPU precomputations
    MatrixMatrixProductPrecomputations cpuPrecomp = getMatrixMatrixProductPrecomputations(context, rowSize);

    // Then convert to GPU precomputations
    return convertToGPUPrecomputations(GPUcc, context, cpuPrecomp, masks, bStep, levelCP, levelCC, fuse_boot_prescale_CCMM,
                                       slots);
}

std::vector<int> GenerateMatMulRotationIndices_GPU(int rowSize, int bStep) {
    std::set<int> indices;
#if LOW_MEM

#if BSGS
    std::vector<int> aux = GetLinearTransformRotationIndices(bStep, 1, -rowSize + 1);
    std::vector<int> aux2 = GetLinearTransformRotationIndices(bStep, rowSize, 0);
    for (auto& i : aux)
        indices.insert(i);
    for (auto& i : aux2)
        indices.insert(i);
    indices.insert(-rowSize);

    // For the special transform for steps 2 and 3
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);
    if (gStep - 1 != 0 && rowSize - 1 != 0)
        indices.insert((gStep - 1) * bStep * (rowSize - 1));
    if (rowSize - 1 != 0)
        indices.insert(-bStep * (rowSize - 1));
#else
    indices.insert(1);
    indices.insert(-1);
    indices.insert(rowSize);
    indices.insert(-rowSize);
#endif

#else
    for (size_t i = 1; i < rowSize; i++) {
        indices.insert(i);
        indices.insert(-i);
        indices.insert(i * rowSize);
        indices.insert(i - rowSize);
    }
#endif

    std::vector<int> indicesList(indices.begin(), indices.end());
    return indicesList;
}

FIDESlib::CKKS::Ciphertext rotsum_GPU(FIDESlib::CKKS::Ciphertext& in, int blockSize, int padding) {
    { std::cout << "Replace RotSum with CKKS::Accumulate " << std::endl; }
    Context& cc = in.cc;

    FIDESlib::CKKS::Ciphertext prev_rotation(cc);
    prev_rotation.copy(in);

    for (int i = 0; i < std::log2(blockSize); ++i) {
        int rot_index = padding * (1 << i);
        FIDESlib::CKKS::Ciphertext rotated(cc);
        rotated.copy(prev_rotation);
        rotated.rotate(rot_index, cc.GetRotationKey(rot_index));
        prev_rotation.add(rotated);
    }

    return prev_rotation;
}

// Combine 2-D grids into [I][J][3]
void CombineQKV_2D_to_3( PtWeights_GPU& weights_layer, FIDESlib::CKKS::Context& cc, 
            std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>>& WeightsQKV, // [I][J][3]
            std::vector<std::vector<std::vector<FIDESlib::CKKS::Plaintext>>>& BiasQKV     // [I][J][3]
) {
    using PT = FIDESlib::CKKS::Plaintext;
    const size_t I = weights_layer.Wk.size();
    const size_t J = weights_layer.Wk[0].size();

    WeightsQKV.clear(); BiasQKV.clear();
    WeightsQKV.resize(I);
    BiasQKV.resize(I);

    for (size_t i = 0; i < I; ++i) {
        WeightsQKV[i].resize(J);
        BiasQKV[i].resize(J);
        for (size_t j = 0; j < J; ++j) {
            WeightsQKV[i][j].reserve(3);
            BiasQKV[i][j].reserve(3);

            // K, Q, V
            WeightsQKV[i][j].emplace_back(cc);
            WeightsQKV[i][j].back().copy(weights_layer.Wk[i][j]);

            WeightsQKV[i][j].emplace_back(cc);
            WeightsQKV[i][j].back().copy(weights_layer.Wq[i][j]);

            WeightsQKV[i][j].emplace_back(cc);
            WeightsQKV[i][j].back().copy(weights_layer.Wv[i][j]);

            // bk, bq, bv
            BiasQKV[i][j].emplace_back(cc);
            BiasQKV[i][j].back().copy(weights_layer.bk[i][j]);

            BiasQKV[i][j].emplace_back(cc);
            BiasQKV[i][j].back().copy(weights_layer.bq[i][j]);

            BiasQKV[i][j].emplace_back(cc);
            BiasQKV[i][j].back().copy(weights_layer.bv[i][j]);
        }
    }
}


// Asymmetrical PCMM, square
void PCMMSquare_2(FIDESlib::CKKS::Ciphertext& cMat1, const FIDESlib::CKKS::Plaintext& pMat2, uint32_t rowSize,
                    FIDESlib::CKKS::Ciphertext& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp) {

    FIDESlib::CKKS::Context& cc = cMat1.cc;

    FIDESlib::CKKS::Plaintext linearTransform2(cc);
    linearTransform2.copy(pMat2);

    // Step 2:
    Ciphertext linearTransform1(cc);
    linearTransform1.copy(cMat1);

    Ciphertext aux(cc);
    aux.rotate(linearTransform1, -(int)rowSize, cc.GetRotationKey(-(int)rowSize));
    
    LinearTransformSpecialPt_2(linearTransform1, aux, linearTransform2, rowSize, precomp.bStep, precomp.pts_3_1_new,
                             precomp.pts_3_2_new, 1, rowSize); 

    if (linearTransform1.NoiseLevel == 2)
        linearTransform1.rescale();

    cProduct.copy(linearTransform1);
}

// Asymmetrical PCMM 
void PCMM_2(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias,
              Plaintext& mask_row) {
    product.clear();
    product.reserve(matrix1.size());
    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix2[0].size());
        for (size_t j = 0; j < matrix2[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext dotProd(matrix1[i][0].cc);
            PCMMSquare_2(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);

            Plaintext bias_masked(matrix1[i][0].cc);
            bias_masked.copy(bias[0][j]);
            if (mask_row.NoiseLevel > 1) { mask_row.rescale(); }
            bias_masked.multPt(bias_masked, mask_row, true);

            dotProd.addPt(bias_masked);
            for (size_t k = 1; k < matrix2.size(); k++) {
                FIDESlib::CKKS::Ciphertext dotProdNew(matrix1[i][k].cc);
                PCMMSquare_2(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);
                dotProd.add(dotProdNew);
            }
            row.emplace_back(std::move(dotProd));
        }
        product.emplace_back(std::move(row));
    }

}

// // Asymmetrical PCMM with delayed normalization and row-masked bias
void PCMM_2_delayedInv(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias,
              Plaintext& mask_row, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& factor) {
    product.clear();
    product.reserve(matrix1.size());

    FIDESlib::CKKS::Context& cc = matrix1[0][0].cc;

    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix2[0].size());
        for (size_t j = 0; j < matrix2[0].size(); j++) {
            auto factor_ind = std::min(i,j);
            FIDESlib::CKKS::Ciphertext dotProd(cc);
            PCMMSquare_2(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);

            Plaintext bias_masked(cc);
            bias_masked.copy(bias[0][j]);
            if (mask_row.NoiseLevel > 1) { mask_row.rescale(); }
            bias_masked.multPt(bias_masked, mask_row, true);

            Ciphertext tmp_bias(cc);
            tmp_bias.copy(factor[factor_ind][factor_ind]);
            tmp_bias.dropToLevel(bias_masked.c0.getLevel());
            tmp_bias.multPt(bias_masked);
            dotProd.dropToLevel(tmp_bias.getLevel());
            dotProd.add(tmp_bias);

            for (size_t k = 1; k < matrix2.size(); k++) {
                FIDESlib::CKKS::Ciphertext dotProdNew(matrix1[i][k].cc);
                PCMMSquare_2(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);
                dotProd.add(dotProdNew);
            }
            row.emplace_back(std::move(dotProd));
        }
        product.emplace_back(std::move(row));
    }

}

// PCMM with delayed Inv (factored and masked) bias
void PCMM_GPU_delayedInv(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias,
              Plaintext& mask_row, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& factor) {
    product.clear();
    product.reserve(matrix1.size());

    FIDESlib::CKKS::Context& cc = matrix1[0][0].cc;

    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix2[0].size());
        for (size_t j = 0; j < matrix2[0].size(); j++) {
            auto factor_ind = std::min(i,j);
            FIDESlib::CKKS::Ciphertext dotProd(cc);
            PCMMSquare_GPU(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);

            Plaintext bias_masked(cc);
            bias_masked.copy(bias[0][j]);
            if (mask_row.NoiseLevel > 1) { mask_row.rescale(); }
            // std::cout << "bias[i][j]: "  << bias[i][j].c0.getLevel() << ", " << bias[i][j].NoiseLevel << std::endl;
            // std::cout << "mask_row: "  << mask_row.c0.getLevel() << ", " << mask_row.NoiseLevel << std::endl;
            bias_masked.multPt(bias_masked, mask_row, true);

            Ciphertext tmp_bias(cc);
            tmp_bias.copy(factor[factor_ind][factor_ind]);
            // std::cout << "bias_masked: "  << bias_masked.c0.getLevel() << ", " << bias_masked.NoiseLevel << std::endl;
            // std::cout << "factor[i][j]: "  << factor[factor_ind][factor_ind].getLevel() << ", " << factor[factor_ind][factor_ind].NoiseLevel << std::endl;
            tmp_bias.dropToLevel(bias_masked.c0.getLevel());
            tmp_bias.multPt(bias_masked);
            dotProd.dropToLevel(tmp_bias.getLevel());
            dotProd.add(tmp_bias);

            for (size_t k = 1; k < matrix2.size(); k++) {
                FIDESlib::CKKS::Ciphertext dotProdNew(matrix1[i][k].cc);
                PCMMSquare_GPU(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);
                dotProd.add(dotProdNew);
            }
            row.emplace_back(std::move(dotProd));
        }
        product.emplace_back(std::move(row));
    }

}


}  // namespace FIDESlib::CKKS