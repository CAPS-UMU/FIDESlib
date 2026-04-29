//
// Created by seyda on 5/19/25.
//
#ifndef FIDESLIB_BERT_TINY_POLYAPPROX_CUH
#define FIDESLIB_BERT_TINY_POLYAPPROX_CUH

#include <cmath>
#include <iostream>
#include <vector>

#include "MatMul.cuh"
#include <CKKS/AccumulateBroadcast.cuh>
#include <CKKS/ApproxModEval.cuh>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/Context.cuh>
#include <pke/openfhe.h>

namespace FIDESlib::CKKS {

void EvalSoftmax_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu,
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey,
  FIDESlib::CKKS::Plaintext& mask_token,
  std::vector<FIDESlib::CKKS::Plaintext>& mask_broadcast,
  FIDESlib::CKKS::Plaintext& mask_mean,
  FIDESlib::CKKS::Plaintext& mask_max,
  int numSlots,
  int blockSize,
  int bStepAcc,
  int token_length,
  bool bts		   = false,
  int test_case	   = 0,
  int layerNo	   = 0,
  int long_input   = 1,
  double num_sigma = 0);

void EvalLayerNorm_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu,
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey,
  std::vector<FIDESlib::CKKS::Plaintext>& mask_ln,
  FIDESlib::CKKS::Plaintext& mask_row,
  std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& weight,
  std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias,
  int numSlots,
  int blockSize,
  int bStepAcc,
  bool bts = false);

void EvalGelu_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, int numSlots, int test_case = 0);

void EvalTanh_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, int numSlots, double lower_bound, double upper_bound, bool bts = false);

void evalTanh(FIDESlib::CKKS::Ciphertext& ctxt, int numSlots, double lower_bound, double upper_bound, bool bts = false);

void NewtonRaphsonInv(FIDESlib::CKKS::Ciphertext& ctxt,
  FIDESlib::CKKS::Ciphertext& initial,
  int num_iter,
  FIDESlib::CKKS::Ciphertext& final,
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu,
  lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey);

void NewtonRaphsonInvSqrt(FIDESlib::CKKS::Ciphertext& ctxt, FIDESlib::CKKS::Ciphertext& initial, int num_iter);

} // namespace FIDESlib::CKKS

#endif // FIDESLIB_BERT_TINY_POLYAPPROX_CUH