#ifndef FIDESLIB_LINEARTRANSFORM_CUH
#define FIDESLIB_LINEARTRANSFORM_CUH

#include <vector>
#include "forwardDefs.cuh"
#include "CKKS/AccumulateBroadcast.cuh"

namespace FIDESlib::CKKS {

void LinearTransform(Ciphertext& ctxt, int rowSize, int bStep, std::vector<Plaintext*> pts, 
                        int stride = 1, int offset = 0);

void LinearTransform(Ciphertext& ctxt, int rowSize, int bStep, std::vector<Plaintext*> pts, int stride,
                                     int offset, Plaintext& mask_pt);

std::vector<int> GetLinearTransformRotationIndices(int bStep, int stride = 1, int offset = 0);
std::vector<int> GetLinearTransformPlaintextRotationIndices(int rowSize, int bStep, int stride = 1, int offset = 0);

void LinearTransformSpecial(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt2,
                            FIDESlib::CKKS::Ciphertext& ctxt3, int rowSize, int bStep, std::vector<Plaintext*> pts1,
                            std::vector<Plaintext*> pts2, int stride, int stride3);

void LinearTransformSpecial(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt2,
                                            FIDESlib::CKKS::Ciphertext& ctxt3, int rowSize, int bStep,
                                            std::vector<Plaintext*> pts1, std::vector<Plaintext*> pts2, int stride,
                                            int stride3, Plaintext& mask_pt);

void LinearTransformSpecialPt(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt3,
                              FIDESlib::CKKS::Plaintext& ptxt, int rowSize, int bStep, std::vector<Plaintext*> pts1,
                              std::vector<Plaintext*> pts2, int stride, int stride3);

// Q,K,V merged
void LinearTransformSpecialPt(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt2,
                                              FIDESlib::CKKS::Plaintext& ptxt, int rowSize, int bStep,
                                              std::vector<Plaintext*> pts1, std::vector<Plaintext*> pts2, int stride,
                                              int stride3, FIDESlib::CKKS::Ciphertext& cProduct);

void LinearTransformSpecialPt_2(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt2,
                                              FIDESlib::CKKS::Plaintext& ptxt, int rowSize, int bStep,
                                              std::vector<Plaintext*> pts1, std::vector<Plaintext*> pts2, int stride,
                                              int stride3);

// void LinearTransform_Batched(Ciphertext& ctxt,
//                                              int rowSize, int bStep,
//                                              std::vector<Plaintext*> pts,
//                                              int stride, int offset);
}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_LINEARTRANSFORM_CUH