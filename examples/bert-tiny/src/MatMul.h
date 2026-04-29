// @author TPOC: contact@palisade-crypto.org
//
// @copyright Copyright (c) 2021, Duality Technologies Inc.
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. THIS SOFTWARE IS
// PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THISvector<
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// PS: The following code is not included in the main branch of OpenFHE v1.3,
//      expected to be released in v1.4

#ifndef OPENFHE_MATRIX_MULT_EXAMPLES_H
#define OPENFHE_MATRIX_MULT_EXAMPLES_H

#define PROFILE

#include <CKKS/Bootstrap.cuh>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <iterator>
#include <math.h>
#include <openfhe.h>
#include <string>
#include <vector>

namespace FIDESlib::CKKS {

struct MatrixMatrixProductPrecomputations {
	int rowSize;
	int colSize;
	std::vector<lbcrypto::Plaintext> sigmaPlaintexts;
	std::vector<lbcrypto::Plaintext> tauPlaintexts;
	// std::vector<std::vector<double>> tauVectors;
	std::vector<std::vector<lbcrypto::Plaintext>> phiPlaintexts, phiPlaintexts_new;
}; // MatrixMatrixProductPrecomputations;

std::vector<std::vector<double>> generateRandomMatrix(size_t numRows, size_t numCols, unsigned int seed);

std::vector<int32_t> GenerateMatMulRotationIndices(uint32_t rowSize);

// Helper methods to get permutation matrices for matrix multiplication
template <typename Element> std::vector<std::vector<Element>> getDiagonals(std::vector<std::vector<Element>> matrix);

std::vector<std::vector<double>> getSigmaPermutationMatrix(size_t rowSize, size_t colSize);

std::vector<std::vector<double>> getTauPermutationMatrix(size_t rowSize, size_t colSize);

std::vector<std::vector<double>> getPhiDiagonals(size_t rowSize, size_t colSize, size_t numRotations);

struct MatrixMatrixProductPrecomputations getMatrixMatrixProductPrecomputations(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, int rowSize, int colSize = 0);

void MatrixMatrixProductSquare(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& cMat1,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& cMat2,
  uint32_t rowSize,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& cProduct,
  struct MatrixMatrixProductPrecomputations precomp);

void MatrixMatrixProduct(std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& matrix1,
  std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& matrix2,
  uint32_t rowSize,
  std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& product,
  struct MatrixMatrixProductPrecomputations precomp);

void MatrixMatrixProductwithBias(std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& matrix1,
  std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& matrix2,
  uint32_t rowSize,
  std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& product,
  struct MatrixMatrixProductPrecomputations precomp,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> bias);

std::vector<double> extractAndLinearizeMatrixBlock(const std::vector<std::vector<double>>& matrix,
  size_t numSlots,
  size_t rowSize, // #cols per block
  size_t colSize, // #rows per block
  size_t offsetRows,
  size_t offsetCols);

std::vector<std::vector<std::vector<double>>> extractAndLinearizeMatrix(const std::vector<std::vector<double>>& matrix, size_t numSlots, size_t rowSize, size_t colSize = 0);

std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>
EncryptMatrix(const std::vector<std::vector<std::vector<double>>>& matrix, lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey, int level = 0);
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EncryptVector(const std::vector<double>& bias, lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey);

std::vector<std::vector<std::vector<double>>>
DecryptMatrix(const std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& matrix, lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey, int numSlots);

// Function to convert blocked matrix (given in 3D std vector structure) into a 2D large matrix (in human-readable format)
std::vector<std::vector<double>> convertToLargeMatrix(const std::vector<std::vector<std::vector<double>>>& blockedMatrix, size_t rowSize, size_t colSize = 0);

bool compareMatrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, double epsilon);
void printMatrix(const std::vector<std::vector<double>>& matrix,
  uint32_t horizontalPrintSize,
  uint32_t verticalPrintSize,
  const std::string& label,
  bool fullPrint,
  int precision = 5,
  bool if_first = false);

std::vector<std::vector<double>> clear_MM(std::vector<std::vector<double>> matrix1,
  std::vector<std::vector<double>> matrix2,
  const std::vector<std::vector<double>>& bias = std::vector<std::vector<double>>());

// std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> naive_CCMM(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context,
// std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> rows, const lbcrypto::Plaintext &weight, const lbcrypto::Plaintext &bias);

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> rotsum(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& in, int slots, int padding);

//     vector<double> clear_naive_MM(const vector<double>& input, const vector<double>& weights, const vector<double>& bias, const vector<double>& bias, int d_in = 128, int d_out = 128);
} // namespace FIDESlib::CKKS

#endif // OPENFHE_MATRIX_MULT_EXAMPLES_H
