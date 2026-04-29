//
// Created by carlosad on 7/05/25.
//

#ifndef FIDESLIB_LINEARTRANSFORM_CUH
#define FIDESLIB_LINEARTRANSFORM_CUH

#include "forwardDefs.cuh"
#include <vector>

namespace FIDESlib::CKKS {

void LinearTransform(Ciphertext& ctxt, int rowSize, int bStep, const std::vector<Plaintext*>& pts, int stride = 1, int offset = 0);

template <CiphertextPtr ptrT, PlaintextPtr ptrU>
void LinearTransform(CiphertextBatch<ptrT>& ctxt, int rowSize, int bStep, const PlaintextBatch<ptrU>& pts, int stride = 1, int offset = 0);

std::vector<int> GetLinearTransformRotationIndices(int bStep, int stride = 1, int offset = 0);
std::vector<int> GetLinearTransformPlaintextRotationIndices(int rowSize, int bStep, int stride = 1, int offset = 0);

// ConvolutionTransform: Like LinearTransform but with INVERTED order (rotate then sum, forward loop)
void ConvolutionTransform(Ciphertext& ctxt, int rowSize, int bStep, const std::vector<Plaintext*>& pts, int stride, const std::vector<int>& indexes, uint32_t gStep);

std::vector<int> GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep);

// SpecialConvolutionTransform: Like ConvolutionTransform but with special masking logic
// After each gStep's bStep sum: 3 rotations with additions + mask multiplication before accumulation
void SpecialConvolutionTransform(Ciphertext& ctxt,
  int rowSize,
  int bStep,
  const std::vector<Plaintext*>& pts,
  Plaintext& mask,
  int stride,
  int maskRotationStride,
  const std::vector<int>& indexes,
  uint32_t gStep);

void LinearTransformSpecial(FIDESlib::CKKS::Ciphertext& ctxt1,
  FIDESlib::CKKS::Ciphertext& ctxt2,
  FIDESlib::CKKS::Ciphertext& ctxt3,
  int rowSize,
  int bStep,
  std::vector<Plaintext*> pts1,
  std::vector<Plaintext*> pts2,
  int stride,
  int stride3);
/*
	void LinearTransformPt(FIDESlib::CKKS::Plaintext& ptxt, FIDESlib::CKKS::Context& cc, int rowSize, int bStep,
										std::vector<Plaintext*> pts, int stride, int offset);
*/
void LinearTransformSpecialPt(FIDESlib::CKKS::Ciphertext& ctxt1,
  FIDESlib::CKKS::Ciphertext& ctxt3,
  FIDESlib::CKKS::Plaintext& ptxt,
  int rowSize,
  int bStep,
  std::vector<Plaintext*> pts1,
  std::vector<Plaintext*> pts2,
  int stride,
  int stride3);

} // namespace FIDESlib::CKKS
#endif // FIDESLIB_LINEARTRANSFORM_CUH