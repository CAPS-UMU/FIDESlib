#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include "Inputs.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "MatMul.cuh"
#include "pke/openfhe.h"
#include "CKKS/AccumulateBroadcast.cuh"

namespace FIDESlib::CKKS {

void EvalSoftmax_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu,lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey, 
                        const KeySwitchingKey& keySwitchingKey, FIDESlib::CKKS::Plaintext& mask_token, FIDESlib::CKKS::Plaintext& mask_broadcast, FIDESlib::CKKS::Plaintext& mask_mean, FIDESlib::CKKS::Plaintext& mask_max,
                        int numSlots, int blockSize, int bStepAcc, int token_length, bool bts = false, int test_case = 0, int long_input = 0);
                        
void EvalLayerNorm_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu, lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey,
                          const KeySwitchingKey& keySwitchingKey, std::vector<FIDESlib::CKKS::Plaintext>& mask_ln, FIDESlib::CKKS::Plaintext& mask_row,
                          std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& weight,
                          std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias, int numSlots, int blockSize,
                          int bStepAcc, bool bts = false);
void EvalLayerNorm_Matrix_DelayedInv(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu, lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey,
                        const KeySwitchingKey& keySwitchingKey, std::vector<FIDESlib::CKKS::Plaintext>& mask_ln, FIDESlib::CKKS::Plaintext& mask_row,
                        std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& weight,
                        std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias, int numSlots, int blockSize,
                        const int bStepAcc, bool bts, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& factor);

void EvalGelu_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, const KeySwitchingKey& keySwitchingKey,
                     int numSlots);

void EvalTanh_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, const KeySwitchingKey& keySwitchingKey, 
                int numSlots, double lower_bound, double upper_bound, bool bts = false);


void evalTanh(FIDESlib::CKKS::Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, int numSlots,
              double lower_bound, double upper_bound, bool bts = false);

void evalSqrtTaylor(FIDESlib::CKKS::Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey);

void NewtonRaphsonInv(FIDESlib::CKKS::Ciphertext& ctxt, FIDESlib::CKKS::Ciphertext& initial, const KeySwitchingKey& keySwitchingKey, int num_iter, 
                    FIDESlib::CKKS::Ciphertext& final, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_cpu, lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey);

void NewtonRaphsonInvSqrt(FIDESlib::CKKS::Ciphertext& ctxt, FIDESlib::CKKS::Ciphertext& initial, const KeySwitchingKey& keySwitchingKey, int num_iter);

}  // namespace FIDESlib::CKKS