//
// Created by carlosad on 7/05/25.
//

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include "CudaUtils.cuh"

#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

namespace FIDESlib::CKKS {
template <CiphertextPtr ptrT, PlaintextPtr ptrU>
void DotProductPtInternal(std::vector<std::shared_ptr<Ciphertext>>& result,
  const std::vector<ptrT>& a,
  const std::vector<ptrU>& b,
  const int red_n,
  const int pt_reuse_stride,
  const int pt_different_stride,
  const bool ext) {
	std::vector<RNSPoly*> cts, pts, res;
	cts.reserve(a.size() * 2);
	for (auto& i : a) {
		cts.push_back(&i->c0);
		cts.push_back(&i->c1);
	}
	res.reserve(result.size() * 2);
	for (auto& i : result) {
		res.push_back(&i->c0);
		res.push_back(&i->c1);
	}
	pts.reserve(b.size());
	for (auto& i : b) {
		if (i)
			pts.push_back(&i->c0);
		else
			pts.push_back(nullptr);
	}

	RNSPoly::LTdotProductPtBatch(res, cts, pts, red_n, pt_different_stride, pt_reuse_stride, 1.0, ext);

	for (size_t i = 0; i < result.size(); ++i) {
		result[i]->multMetadata(
		  *a[(i / pt_different_stride) * red_n], *b[(i % pt_different_stride + (i / (pt_reuse_stride * pt_different_stride)) * pt_different_stride) * red_n]);
		for (int j = 1; j < red_n; ++j) {
			if (b[(i % pt_different_stride + (i / (pt_reuse_stride * pt_different_stride)) * pt_different_stride) * red_n + j]) {
				result[i]->slots = std::max(result[i]->slots, a[(i / pt_different_stride) * red_n + j]->slots);
				result[i]->slots =
				  std::max(result[i]->slots, b[(i % pt_different_stride + (i / (pt_reuse_stride * pt_different_stride)) * pt_different_stride) * red_n + j]->slots);
			}
		}
	}
}
} // namespace FIDESlib::CKKS

void FIDESlib::CKKS::LinearTransform(Ciphertext& ctxt, int rowSize, int bStep, const std::vector<Plaintext*>& pts, int stride, int offset) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	assert(pts.size() >= rowSize);
	for (auto i : pts) {
		assert(i != nullptr);
	}
	Context& cc_	= ctxt.cc_;
	ContextData& cc = ctxt.cc;
	uint32_t gStep	= ceil(static_cast<double>(rowSize) / bStep);

	if (ctxt.NoiseLevel == 2)
		ctxt.rescale();

	{
		std::vector<Ciphertext> fastRotation;
		fastRotation.reserve(bStep);
		for (size_t i = fastRotation.size(); i < static_cast<size_t>(bStep); ++i) {
			fastRotation.emplace_back(cc_);
			fastRotation.back().growToLevel(ctxt.getLevel());
			fastRotation.back().dropToLevel(ctxt.getLevel(), true);
			fastRotation.back().extend(false);
		}

		{
			std::vector<Ciphertext*> fastRotationPtr;

			bool ext = true;
			{
				std::vector<int> indexes;
				for (int i = 0; i < bStep; ++i) {
					fastRotationPtr.push_back(&fastRotation[i]);
					indexes.push_back(i * stride);
				}

				if (bStep == 1)
					ext = false;
				for (auto& i : pts) {
					if (!i->c0.isModUp()) {
						ext = false;
					}
				}

				ctxt.rotate_hoisted(indexes, fastRotationPtr, ext);
			}

			constexpr bool MODDOWN_HOIST = true;
			constexpr bool ONLY_C1		 = true;
			constexpr bool FUSED		 = true;
			if constexpr (FUSED) {
				assert(rowSize == pts.size());
				std::vector<Plaintext*> Aptr(bStep * gStep, nullptr);
				for (uint32_t j = 0; j < gStep; ++j) {
					for (int i = 0; i < bStep; ++i) {
						if (bStep * j + i < static_cast<uint32_t>(rowSize))
							Aptr[bStep * j + i] = pts[bStep * j + i];
					}
				}

				std::vector<std::shared_ptr<Ciphertext>> results;
				results.reserve(gStep);

				for (uint32_t i = 0; i < gStep; ++i) {
					results.emplace_back(std::make_shared<Ciphertext>(cc_));
					results.back()->growToLevel(ctxt.getLevel());
					results.back()->dropToLevel(ctxt.getLevel(), true);
					results.back()->extend(false);
				}

				for (auto& i : results) {
					for (uint32_t j = 0; j < i->c0.GPU.size(); ++j) {
						results[0]->c0.GPU[j].s.wait(i->c0.GPU[j].s);
						results[0]->c0.GPU[j].s.wait(i->c1.GPU[j].s);
					}
				}
				{
					for (auto& i : fastRotationPtr) {
						for (uint32_t j = 0; j < i->c0.GPU.size(); ++j) {
							results[0]->c0.GPU[j].s.wait(i->c0.GPU[j].s);
							results[0]->c0.GPU[j].s.wait(i->c1.GPU[j].s);
						}
					}
				}

				// CudaCheckErrorMod;

				DotProductPtInternal<Ciphertext*, Plaintext*>(results, fastRotationPtr, Aptr, bStep, 1, gStep, MODDOWN_HOIST && ext);

				// CudaCheckErrorMod;

				// todo: for correctness, fastRotationPtr ciphertexts should not be modified until this point

				for (auto& i : results) {
					for (uint32_t j = 0; j < i->c0.GPU.size(); ++j) {
						i->c0.GPU[j].s.wait(results[0]->c0.GPU[j].s);
						i->c1.GPU[j].s.wait(results[0]->c0.GPU[j].s);
					}
				}
				{
					for (auto& i : fastRotationPtr) {
						for (uint32_t j = 0; j < i->c0.GPU.size(); ++j) {
							i->c0.GPU[j].s.wait(results[0]->c0.GPU[j].s);
							i->c1.GPU[j].s.wait(results[0]->c0.GPU[j].s);
						}
					}
				}

				for (uint32_t j = gStep - 1; j < gStep; --j) {

					if (j != gStep - 1) {
						if (results[j + 1]->c1.isModUp())
							results[j]->extend();
						results[j]->add(*results[j + 1]);
						results.pop_back();
					}

					if (j > 0) {
						if ((bStep * stride) % (cc.N / 2) != 0) {
							if (results[j]->c1.isModUp())
								if (ONLY_C1)
									results[j]->c1.moddown(true, false);
								else
									results[j]->modDown(false);
							results[j]->rotate((int)bStep * stride, false);
						}
					} else if (offset != 0) {
						if (results[j]->c1.isModUp())
							if (ONLY_C1)
								results[j]->c1.moddown(true, false);
							else
								results[j]->modDown(false);
						results[j]->rotate(offset, true);
					} else {
						if (results[j]->c1.isModUp())
							results[j]->modDown(false);
					}
				}

				// results[0]->modDown(false);
				ctxt.copy(*results[0]);
			} else {

				assert(rowSize == pts.size());
				Ciphertext inner(cc_);
				for (uint32_t j = gStep - 1; j < gStep; --j) {
					int n = 1;
					// inner.multPt(ctxt, A[bStep * j], false);
					for (uint32_t i = 1; i < bStep; i++) {
						if (bStep * j + i < rowSize) {
							n++;
							// inner.addMultPt(fastRotation[i - 1], A[bStep * j + i], false);
						}
					}

					if (fastRotation.at(0).getLevel() > inner.getLevel()) {
						inner.c0.grow(fastRotation.at(0).getLevel(), false);
						inner.c1.grow(fastRotation.at(0).getLevel(), false);
					} else {
						inner.dropToLevel(fastRotation.at(0).getLevel(), true);
					}

					inner.dotProductPt(fastRotation.data(), (Plaintext**)pts.data() + j * bStep, n, ext);

					if (j == gStep - 1) {
						ctxt.copy(inner);
					} else {
						if (!ext)
							inner.extend();
						ctxt.add(inner);
					}

					if (j > 0) {
						if (ctxt.c1.isModUp()) {
							ctxt.c1.moddown(true, false);
						}
						if (stride * bStep % (cc.N / 2) != 0)
							ctxt.rotate((int)stride * bStep, false);
					}
				}

				if (offset != 0) {
					if (ctxt.c1.isModUp()) {
						ctxt.c1.moddown(true, false);
					}
					ctxt.rotate(offset, true);
				} else {
					if (ctxt.c1.isModUp()) {
						ctxt.modDown(false);
					}
				}
			}
		}
	}
}

// ConvolutionTransform: Like LinearTransform but with INVERTED order
// LinearTransform: sum then rotate (backward loop j=gStep-1 to 0)
// ConvolutionTransform: rotate then sum (forward loop j=0 to gStep-1)
// If gStep > 8, divides into blocks of 8, processes each block, rotates and accumulates.
void FIDESlib::CKKS::ConvolutionTransform(Ciphertext& ctxt, int rowSize, int bStep, const std::vector<Plaintext*>& pts, int stride, const std::vector<int>& indexes, uint32_t gStep) {

	assert(pts.size() >= rowSize);
	for (auto i : pts) {
		assert(i != nullptr);
	}
	Context& cc_ = ctxt.cc_;
	// ContextData& cc = ctxt.cc;

	if (ctxt.NoiseLevel == 2)
		ctxt.rescale();

	// Internal block size for DotProductPtInternal
	constexpr uint32_t INTERNAL_GSTEP = 8;

	// Calculate number of blocks needed
	uint32_t blockCount = (gStep + INTERNAL_GSTEP - 1) / INTERNAL_GSTEP;

	{
		std::vector<Ciphertext> fastRotation;
		fastRotation.reserve(bStep);
		for (size_t i = fastRotation.size(); i < static_cast<size_t>(bStep); ++i) {
			fastRotation.emplace_back(cc_);
			fastRotation.back().growToLevel(ctxt.getLevel());
			fastRotation.back().extend(false);
		}

		{
			std::vector<Ciphertext*> fastRotationPtr;

			bool ext = true;
			{
				for (int i = 0; i < bStep; ++i) {
					fastRotationPtr.push_back(&fastRotation[i]);
				}

				if (bStep == 1)
					ext = false;
				for (auto& i : pts) {
					if (!i->c0.isModUp()) {
						ext = false;
					}
				}

				ctxt.rotate_hoisted(indexes, fastRotationPtr, ext);
			}

			constexpr bool MODDOWN_HOIST = true;
			// constexpr bool ONLY_C1		 = true;

			assert(rowSize == pts.size());

			// Vector to store results from each block
			std::vector<std::shared_ptr<Ciphertext>> blockResults;
			blockResults.reserve(blockCount);

			// Process each block of INTERNAL_GSTEP
			for (uint32_t blockIdx = 0; blockIdx < blockCount; ++blockIdx) {
				uint32_t blockStart		   = blockIdx * INTERNAL_GSTEP;
				uint32_t blockEnd		   = std::min(blockStart + INTERNAL_GSTEP, gStep);
				uint32_t currentBlockGStep = blockEnd - blockStart;

				// Build Aptr for this block
				std::vector<Plaintext*> Aptr(bStep * currentBlockGStep, nullptr);
				for (uint32_t j = 0; j < currentBlockGStep; ++j) {
					for (int i = 0; i < bStep; ++i) {
						uint32_t globalIdx = bStep * (blockStart + j) + i;
						if (globalIdx < static_cast<uint32_t>(rowSize))
							Aptr[bStep * j + i] = pts[globalIdx];
					}
				}

				// Create result ciphertexts for this block
				std::vector<std::shared_ptr<Ciphertext>> results;
				results.reserve(currentBlockGStep);
				for (uint32_t i = 0; i < currentBlockGStep; ++i) {
					results.emplace_back(std::make_shared<Ciphertext>(cc_));
					results.back()->growToLevel(ctxt.getLevel());
					results.back()->extend(false);
				}
				for (auto& i : results) {
					for (size_t j = 0; j < i->c0.GPU.size(); ++j) {
						results[0]->c0.GPU[j].s.wait(i->c0.GPU[j].s);
						results[0]->c0.GPU[j].s.wait(i->c1.GPU[j].s);
					}
				}

				for (auto& i : fastRotationPtr) {
					for (size_t j = 0; j < i->c0.GPU.size(); ++j) {
						results[0]->c0.GPU[j].s.wait(i->c0.GPU[j].s);
						results[0]->c0.GPU[j].s.wait(i->c1.GPU[j].s);
					}
				}

				// Call DotProductPtInternal with the current block's gStep
				DotProductPtInternal<Ciphertext*, Plaintext*>(results, fastRotationPtr, Aptr, bStep, 1, static_cast<int>(currentBlockGStep), MODDOWN_HOIST && ext);

				for (auto& i : results) {
					for (size_t j = 0; j < i->c0.GPU.size(); ++j) {
						i->c0.GPU[j].s.wait(results[0]->c0.GPU[j].s);
						i->c1.GPU[j].s.wait(results[0]->c0.GPU[j].s);
					}
				}

				// Intra-block rotation: rotate by (currentBlockGStep - j) * stride
				for (uint32_t j = 0; j < currentBlockGStep; ++j) {
					int rotation = (int)stride * (currentBlockGStep - j);
					if (ctxt.normalyzeIndex(rotation) != 0)
						results[j]->rotate(rotation, true);
				}

				// Intra-block accumulation
				std::shared_ptr<Ciphertext> blockAccumulated = std::make_shared<Ciphertext>(cc_);
				blockAccumulated->growToLevel(ctxt.getLevel());

				if (currentBlockGStep % 2 == 0) {
					uint32_t activeStep = currentBlockGStep;
					while (activeStep > 1) {
						for (uint32_t j = 0; j < activeStep / 2; ++j) {
							if (results[j]->c0.isModUp())
								results[j + activeStep / 2]->extend();
							results[j]->add(*results[j + activeStep / 2]);
						}
						activeStep /= 2;
					}
					blockAccumulated->copy(*results[0]);
				} else {
					for (uint32_t j = 0; j < currentBlockGStep; ++j) {
						if (j == 0) {
							blockAccumulated->copy(*results[0]);
						} else {
							if (blockAccumulated->c0.isModUp())
								results[j]->extend();
							blockAccumulated->add(*results[j]);
						}
					}
				}

				blockResults.push_back(blockAccumulated);
			}

			// Inter-block rotation and accumulation
			// Each block result needs to be rotated by (blockCount - 1 - blockIdx) * INTERNAL_GSTEP * stride
			if (blockCount > 1) {
				int baseRotation = INTERNAL_GSTEP * stride;

				for (uint32_t blockIdx = 0; blockIdx < blockCount - 1; ++blockIdx) {
					int totalBlocks = blockCount - 1 - blockIdx;
					int rotation	= totalBlocks * baseRotation;

					if (ctxt.normalyzeIndex(rotation) != 0) {
						blockResults[blockIdx]->rotate(rotation, true);
					}
				}

				// Accumulate all block results using binary tree if possible
				if (blockCount % 2 == 0) {
					uint32_t activeCount = blockCount;
					while (activeCount > 1) {
						for (uint32_t j = 0; j < activeCount / 2; ++j) {
							if (blockResults[j]->c0.isModUp())
								blockResults[j + activeCount / 2]->extend();
							blockResults[j]->add(*blockResults[j + activeCount / 2]);
						}
						activeCount /= 2;
					}
				} else {
					for (uint32_t j = 1; j < blockCount; ++j) {
						if (blockResults[0]->c0.isModUp())
							blockResults[j]->extend();
						blockResults[0]->add(*blockResults[j]);
					}
				}
			}

			// Final moddown AFTER all rotations (if needed)
			if (blockResults[0]->c1.isModUp())
				blockResults[0]->modDown(false);

			ctxt.copy(*blockResults[0]);
		}
	}
}

// SpecialConvolutionTransform: Like ConvolutionTransform but with special masking logic
// After each gStep's bStep sum: 3 rotations with additions + mask multiplication before accumulation
// This is used for layer 0 in ResNet which has a different accumulation pattern
void FIDESlib::CKKS::SpecialConvolutionTransform(Ciphertext& ctxt,
  int rowSize,
  int bStep,
  const std::vector<Plaintext*>& pts,
  Plaintext& mask,
  int stride,
  int maskRotationStride,
  const std::vector<int>& indexes,
  uint32_t gStep) {

	assert(pts.size() >= rowSize);
	for (auto i : pts) {
		assert(i != nullptr);
	}
	Context& cc_ = ctxt.cc_;
	// ContextData& cc = ctxt.cc;

	if (ctxt.NoiseLevel == 2)
		ctxt.rescale();

	// Internal block size for DotProductPtInternal (same as ConvolutionTransform)
	constexpr uint32_t INTERNAL_GSTEP = 8;

	// Calculate number of blocks needed
	uint32_t blockCount = (gStep + INTERNAL_GSTEP - 1) / INTERNAL_GSTEP;

	{
		std::vector<Ciphertext> fastRotation;
		fastRotation.reserve(bStep);
		for (size_t i = fastRotation.size(); i < static_cast<uint32_t>(bStep); ++i) {
			fastRotation.emplace_back(cc_);
			fastRotation.back().growToLevel(ctxt.getLevel());
			fastRotation.back().extend(false);
		}

		{
			std::vector<Ciphertext*> fastRotationPtr;

			bool ext = true;
			{
				for (int i = 0; i < bStep; ++i) {
					fastRotationPtr.push_back(&fastRotation[i]);
				}

				if (bStep == 1)
					ext = false;
				for (auto& i : pts) {
					if (!i->c0.isModUp()) {
						ext = false;
					}
				}

				ctxt.rotate_hoisted(indexes, fastRotationPtr, ext);
			}

			constexpr bool MODDOWN_HOIST = true;
			// constexpr bool ONLY_C1		 = true;

			assert(rowSize == pts.size());

			// Vector to store results from each block
			std::vector<std::shared_ptr<Ciphertext>> blockResults;
			blockResults.reserve(blockCount);

			// Process each block of INTERNAL_GSTEP
			for (uint32_t blockIdx = 0; blockIdx < blockCount; ++blockIdx) {
				uint32_t blockStart		   = blockIdx * INTERNAL_GSTEP;
				uint32_t blockEnd		   = std::min(blockStart + INTERNAL_GSTEP, gStep);
				uint32_t currentBlockGStep = blockEnd - blockStart;

				// Build Aptr for this block
				std::vector<Plaintext*> Aptr(bStep * currentBlockGStep, nullptr);
				for (uint32_t j = 0; j < currentBlockGStep; ++j) {
					for (int i = 0; i < bStep; ++i) {
						uint32_t globalIdx = bStep * (blockStart + j) + i;
						if (globalIdx < static_cast<uint32_t>(rowSize))
							Aptr[bStep * j + i] = pts[globalIdx];
					}
				}

				// Create result ciphertexts for this block
				std::vector<std::shared_ptr<Ciphertext>> results;
				results.reserve(currentBlockGStep);
				for (uint32_t i = 0; i < currentBlockGStep; ++i) {
					results.emplace_back(std::make_shared<Ciphertext>(cc_));
					results.back()->growToLevel(ctxt.getLevel());
					results.back()->extend(false);
				}
				for (auto& i : results) {
					for (size_t j = 0; j < i->c0.GPU.size(); ++j) {
						results[0]->c0.GPU[j].s.wait(i->c0.GPU[j].s);
						results[0]->c0.GPU[j].s.wait(i->c1.GPU[j].s);
					}
				}

				for (auto& i : fastRotationPtr) {
					for (size_t j = 0; j < i->c0.GPU.size(); ++j) {
						results[0]->c0.GPU[j].s.wait(i->c0.GPU[j].s);
						results[0]->c0.GPU[j].s.wait(i->c1.GPU[j].s);
					}
				}

				// Call DotProductPtInternal with the current block's gStep
				DotProductPtInternal<Ciphertext*, Plaintext*>(results, fastRotationPtr, Aptr, bStep, 1, static_cast<int>(currentBlockGStep), MODDOWN_HOIST && ext);

				for (auto& i : results) {
					for (size_t j = 0; j < i->c0.GPU.size(); ++j) {
						i->c0.GPU[j].s.wait(results[0]->c0.GPU[j].s);
						i->c1.GPU[j].s.wait(results[0]->c0.GPU[j].s);
					}
				}

				// Intra-block rotation: rotate by (currentBlockGStep - j) * stride
				for (uint32_t j = 0; j < currentBlockGStep; ++j) {
					std::shared_ptr<Ciphertext> temp = std::make_shared<Ciphertext>(cc_);
					temp->copy(*results[j]);
					// First rotation and add
					results[j]->rotate(maskRotationStride, true);
					temp->add(*results[j]);

					// Second rotation and add
					results[j]->rotate(maskRotationStride, true);
					temp->add(*results[j]);

					// Apply mask multiplication
					temp->multPt(mask, false);

					// Replace the result with the processed version
					results[j] = temp;

					int rotation = (int)stride * (currentBlockGStep - j);
					if (ctxt.normalyzeIndex(rotation) != 0)
						results[j]->rotate(rotation, true);
				}

				// Intra-block accumulation
				std::shared_ptr<Ciphertext> blockAccumulated = std::make_shared<Ciphertext>(cc_);
				blockAccumulated->growToLevel(ctxt.getLevel());

				if (currentBlockGStep % 2 == 0) {
					uint32_t activeStep = currentBlockGStep;
					while (activeStep > 1) {
						for (uint32_t j = 0; j < activeStep / 2; ++j) {
							if (results[j]->c0.isModUp())
								results[j + activeStep / 2]->extend();
							results[j]->add(*results[j + activeStep / 2]);
						}
						activeStep /= 2;
					}
					blockAccumulated->copy(*results[0]);
				} else {
					for (uint32_t j = 0; j < currentBlockGStep; ++j) {
						if (j == 0) {
							blockAccumulated->copy(*results[0]);
						} else {
							if (blockAccumulated->c0.isModUp())
								results[j]->extend();
							blockAccumulated->add(*results[j]);
						}
					}
				}

				blockResults.push_back(blockAccumulated);
			}

			// Inter-block rotation and accumulation
			// Each block result needs to be rotated by (blockCount - 1 - blockIdx) * INTERNAL_GSTEP * stride
			if (blockCount > 1) {
				int baseRotation = INTERNAL_GSTEP * stride;

				for (uint32_t blockIdx = 0; blockIdx < blockCount - 1; ++blockIdx) {
					int totalBlocks = blockCount - 1 - blockIdx;
					int rotation	= totalBlocks * baseRotation;

					if (ctxt.normalyzeIndex(rotation) != 0) {
						blockResults[blockIdx]->rotate(rotation, true);
					}
				}

				// Accumulate all block results using binary tree if possible
				if (blockCount % 2 == 0) {
					uint32_t activeCount = blockCount;
					while (activeCount > 1) {
						for (uint32_t j = 0; j < activeCount / 2; ++j) {
							if (blockResults[j]->c0.isModUp())
								blockResults[j + activeCount / 2]->extend();
							blockResults[j]->add(*blockResults[j + activeCount / 2]);
						}
						activeCount /= 2;
					}
				} else {
					for (uint32_t j = 1; j < blockCount; ++j) {
						if (blockResults[0]->c0.isModUp())
							blockResults[j]->extend();
						blockResults[0]->add(*blockResults[j]);
					}
				}
			}

			// Final moddown AFTER all rotations (if needed)
			if (blockResults[0]->c1.isModUp())
				blockResults[0]->modDown(false);

			ctxt.copy(*blockResults[0]);
			ctxt.rescale();
		}
	}
}

std::vector<int> FIDESlib::CKKS::GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep) {
	std::vector<int> res;
	// Internal block size for DotProductPtInternal
	constexpr uint32_t INTERNAL_GSTEP = 8;

	// Intra-block rotations: stride * k for k in [1, INTERNAL_GSTEP]
	// We need rotations up to the max block size used, which is min(gStep, INTERNAL_GSTEP)
	// But to be safe and simple, we can generate up to INTERNAL_GSTEP if gStep >= INTERNAL_GSTEP
	uint32_t maxIntra = std::min(gStep, INTERNAL_GSTEP);
	for (uint32_t k = 1; k < maxIntra; ++k) {
		res.push_back(stride * k);
	}

	// Inter-block rotations: stride * INTERNAL_GSTEP * k for k in [1, blockCount - 1]
	uint32_t blockCount = (gStep + INTERNAL_GSTEP - 1) / INTERNAL_GSTEP;
	if (blockCount > 1) {
		int baseRotation = INTERNAL_GSTEP * stride;
		for (uint32_t k = 1; k < blockCount; ++k) {
			res.push_back(baseRotation * k);
		}
	}

	return res;
}

namespace FIDESlib::CKKS {

/*
template <CiphertextPtr ptrT, PlaintextPtr ptrU>
void LinearTransform(CiphertextBatch<ptrT>& ctxt, int rowSize, int bStep, const PlaintextBatch<ptrU>& pts, int stride,
					 int offset) {
	CudaNvtxRange r(std::string{sc::current().function_name()});
	assert(pts.cts.size() >= rowSize);
	Context& cc_ = ctxt.conf.cc_;
	ContextData& cc = *cc_;
	uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

	if (ctxt.conf.scale_degree == 2)
		ctxt.Rescale();

	std::vector<int> indexes;
	for (int i = 0; i < bStep; ++i) {
		indexes.push_back(i * stride);
	}

	bool ext = true;
	if (bStep == 1)
		ext = false;
	for (auto& i : pts.cts) {
		if (i && !i->c0.isModUp()) {
			ext = false;
		}
	}

	assert(pts.conf.dims.size() >= 2);
	assert(pts.conf.dims[pts.conf.dims.size() - 1].size == bStep);
	assert(pts.conf.dims[pts.conf.dims.size() - 2].size == gStep);
	assert(bStep * gStep == rowSize);

	auto outSize = ctxt.conf.GetDimSizes();
	outSize.push_back(gStep);

	std::vector<CiphertextBatch<std::shared_ptr<Ciphertext>>> inner_by_step(gStep);

	{
		auto fastRotation = ctxt.HoistedRotate({indexes}, ext);

		{
			auto inner = fastRotation.DotProductPt(pts, outSize, {(int)gStep});
			auto aux_conf = inner.conf;
			aux_conf.dims.pop_back();
			for (int i = 0; i < gStep; ++i) {
				inner_by_step[i].conf = aux_conf;
				inner_by_step[i].cts.resize(ctxt.cts.size());
				for (int j = 0; j < ctxt.cts.size(); ++j) {
					inner_by_step[i].cts[j] = inner.cts[j * gStep + i];
				}
				for (int j = 0; j < inner_by_step[i].cts[0]->c0.GPU.size(); ++j) {
					inner_by_step[i].cts[0]->c0.GPU[j].s.wait(inner.cts[0]->c0.GPU[j].s);
				}
			}
		}
	}

	//fastRotation = {};

	for (uint32_t j = gStep - 1; j < gStep; --j) {

		if (j != gStep - 1) {
			inner_by_step[j].Add(inner_by_step[j + 1]);
			inner_by_step.pop_back();
		}

		if (j > 0) {
			if ((stride * bStep) % (cc.N / 2) != 0) {
				if (inner_by_step[j].conf.isExt)
					inner_by_step[j].ModDownC1(false);
				inner_by_step[j].Rotate({(int)stride * bStep}, false);
			}
		} else if (offset != 0) {
			if (inner_by_step[j].conf.isExt)
				inner_by_step[j].ModDownC1(false);
			inner_by_step[j].Rotate({offset}, true);
		} else {
			if (inner_by_step[j].conf.isExt)
				inner_by_step[j].ModDown(false);
		}
	}

	ctxt.Copy(inner_by_step[0]);
}
*/
/*
template void LinearTransform<Ciphertext*, Plaintext*>(CiphertextBatch<Ciphertext*>& ctxt, int rowSize, int bStep,
													   const PlaintextBatch<Plaintext*>& pts, int stride, int offset);
template void LinearTransform<Ciphertext*, std::shared_ptr<Plaintext>>(
	CiphertextBatch<Ciphertext*>& ctxt, int rowSize, int bStep, const PlaintextBatch<std::shared_ptr<Plaintext>>& pts,
	int stride, int offset);
template void LinearTransform<std::shared_ptr<Ciphertext>, Plaintext*>(
	CiphertextBatch<std::shared_ptr<Ciphertext>>& ctxt, int rowSize, int bStep, const PlaintextBatch<Plaintext*>& pts,
	int stride, int offset);
template void LinearTransform<std::shared_ptr<Ciphertext>, std::shared_ptr<Plaintext>>(
	CiphertextBatch<std::shared_ptr<Ciphertext>>& ctxt, int rowSize, int bStep,
	const PlaintextBatch<std::shared_ptr<Plaintext>>& pts, int stride, int offset);
*/

} // namespace FIDESlib::CKKS

#if 0
void FIDESlib::CKKS::LinearTransformSpecial(FIDESlib::CKKS::Ciphertext& ctxt1,
  FIDESlib::CKKS::Ciphertext& ctxt2,
  FIDESlib::CKKS::Ciphertext& ctxt3,
  int rowSize,
  int bStep,
  std::vector<Plaintext*> pts1,
  std::vector<Plaintext*> pts2,
  int stride,
  int stride3) {
	constexpr bool PRINT = false;
	if constexpr (PRINT)
		std::cout << std::endl << "LinearTransformSpecial ";

	if constexpr (PRINT) {
		std::cout << "ctxt1 ";
		for (auto& j : ctxt1.c0.GPU)
			for (auto& k : j.limb) {
				SWITCH(k, printThisLimb(1));
			}
		std::cout << std::endl;
		for (auto& j : ctxt1.c0.GPU)
			for (auto& k : j.SPECIALlimb) {
				SWITCH(k, printThisLimb(1));
			}
		std::cout << std::endl;
	}

	if constexpr (PRINT) {
		std::cout << "ctxt2 ";
		for (auto& j : ctxt2.c0.GPU)
			for (auto& k : j.limb) {
				SWITCH(k, printThisLimb(1));
			}
		std::cout << std::endl;
		for (auto& j : ctxt2.c0.GPU)
			for (auto& k : j.SPECIALlimb) {
				SWITCH(k, printThisLimb(1));
			}
		std::cout << std::endl;
	}

	if constexpr (PRINT) {
		std::cout << "ctxt3 ";
		for (auto& j : ctxt3.c0.GPU)
			for (auto& k : j.limb) {
				SWITCH(k, printThisLimb(1));
			}
		std::cout << std::endl;
		for (auto& j : ctxt3.c0.GPU)
			for (auto& k : j.SPECIALlimb) {
				SWITCH(k, printThisLimb(1));
			}
		std::cout << std::endl;
	}

	CudaNvtxRange r(std::string{ sc::current().function_name() });
	assert(pts1.size() >= rowSize);
	for (auto i : pts1) {
		assert(i != nullptr);
	}
	assert(pts2.size() >= rowSize);

	Context& cc_	= ctxt1.cc_;
	ContextData& cc = ctxt1.cc;
	uint32_t gStep	= ceil(static_cast<double>(rowSize) / bStep);

	if (ctxt1.NoiseLevel == 2)
		ctxt1.rescale();
	if (ctxt2.NoiseLevel == 2)
		ctxt2.rescale();
	if (ctxt3.NoiseLevel == 2)
		ctxt3.rescale();

	std::vector<Ciphertext> fastRotation;

	for (int i = fastRotation.size(); i < (bStep - 1) * 3; ++i)
		fastRotation.emplace_back(cc_);

	std::vector<Ciphertext*> fastRotationPtr2;
	std::vector<Ciphertext*> fastRotationPtr1;
	std::vector<int> indexes;
	for (int i = 1; i < bStep; ++i) {
		fastRotationPtr1.push_back(&fastRotation[i - 1]);
		fastRotationPtr2.push_back(&fastRotation[(bStep - 1) + i - 1]);
		indexes.push_back(i * stride);
	}

	ctxt1.rotate_hoisted(indexes, fastRotationPtr1, false);
	ctxt2.rotate_hoisted(indexes, fastRotationPtr2, false);

	/*
	std::vector<Ciphertext> fastRotation3;
	for (int i = 0; i < bStep - 1; ++i)
		fastRotation3.emplace_back(cc);
*/
	std::vector<Ciphertext*> fastRotationPtr3;
	std::vector<int> indexes3;
	std::vector<KeySwitchingKey*> keys3;
	for (int i = 1; i < bStep; ++i) {
		fastRotationPtr3.push_back(&fastRotation[2 * (bStep - 1) + i - 1]);
		indexes3.push_back(i * stride3);
	}

	Ciphertext result(cc_);
	Ciphertext inner(cc_);
	Ciphertext aux(cc_);

	if ((gStep - 1) * bStep * (rowSize - 1) != 0)
		ctxt3.rotate((gStep - 1) * bStep * (rowSize - 1), true);

	for (uint32_t j = gStep - 1; j < gStep; --j) {

		inner.multPt(ctxt1, *pts1[bStep * j], false);
		if (bStep * j != 0)
			inner.addMultPt(ctxt2, *pts2[bStep * j], true);
		inner.mult(ctxt3, false, false);

		if constexpr (PRINT) {
			std::cout << "inner " << bStep * j << " ";
			for (auto& j : inner.c0.GPU)
				for (auto& k : j.limb) {
					SWITCH(k, printThisLimb(1));
				}
			std::cout << std::endl;
			for (auto& j : inner.c0.GPU)
				for (auto& k : j.SPECIALlimb) {
					SWITCH(k, printThisLimb(1));
				}
			std::cout << std::endl;
		}

		for (uint32_t i = 1; i < bStep; i++) {
			if (bStep * j + i < rowSize) {
				if (i == 1) {
					int size = std::min((int)bStep - 1, (int)(rowSize - (bStep * j + i)));
					if (size < bStep - 1) {
						auto keys3_			   = keys3;
						auto indexes3_		   = indexes3;
						auto fastRotationPtr3_ = fastRotationPtr3;

						keys3_.resize(size);
						indexes3_.resize(size);
						fastRotationPtr3_.resize(size);

						ctxt3.rotate_hoisted(indexes3_, fastRotationPtr3_, false);
					} else {
						ctxt3.rotate_hoisted(indexes3, fastRotationPtr3, false);
					}
				}
				aux.multPt(fastRotation[i - 1], *pts1[bStep * j + i], false);
				aux.addMultPt(fastRotation[(bStep - 1) + i - 1], *pts2[bStep * j + i], true);
				aux.mult(fastRotation[2 * (bStep - 1) + i - 1], false, false);
				if constexpr (PRINT) {
					std::cout << "inner " << bStep * j + i << " ";
					for (auto& j : aux.c0.GPU)
						for (auto& k : j.limb) {
							SWITCH(k, printThisLimb(1));
						}
					std::cout << std::endl;
					for (auto& j : aux.c0.GPU)
						for (auto& k : j.SPECIALlimb) {
							SWITCH(k, printThisLimb(1));
						}
					std::cout << std::endl;
				}
				inner.add(aux);
			}
		}

		if (j == gStep - 1) {
			result.copy(inner);
		} else {
			result.add(inner);
		}
		result.modDown(false);
		if (j > 0) {
			result.rotate(stride * bStep, false);
			ctxt3.rotate(-bStep * (rowSize - 1), true);
		}
	}

	ctxt1.copy(result);
	CudaCheckErrorModNoSync;
}
#else
void FIDESlib::CKKS::LinearTransformSpecial(FIDESlib::CKKS::Ciphertext& ctxt1,
  FIDESlib::CKKS::Ciphertext& ctxt2,
  FIDESlib::CKKS::Ciphertext& ctxt3,
  int rowSize,
  int bStep,
  std::vector<Plaintext*> pts1,
  std::vector<Plaintext*> pts2,
  int stride,
  int stride3) {
	constexpr bool PRINT = false;
	if constexpr (PRINT)
		std::cout << std::endl << "LinearTransformSpecial ";

	if constexpr (PRINT) {
		std::cout << "ctxt1 ";
		for (auto& j : ctxt1.c0.GPU)
			for (auto& k : j.limb) {
				SWITCH(k, printThisLimb(1));
			}
		std::cout << std::endl;
		for (auto& j : ctxt1.c0.GPU)
			for (auto& k : j.SPECIALlimb) {
				SWITCH(k, printThisLimb(1));
			}
		std::cout << std::endl;
	}

	if constexpr (PRINT) {
		std::cout << "ctxt2 ";
		for (auto& j : ctxt2.c0.GPU)
			for (auto& k : j.limb) {
				SWITCH(k, printThisLimb(1));
			}
		std::cout << std::endl;
		for (auto& j : ctxt2.c0.GPU)
			for (auto& k : j.SPECIALlimb) {
				SWITCH(k, printThisLimb(1));
			}
		std::cout << std::endl;
	}

	if constexpr (PRINT) {
		std::cout << "ctxt3 ";
		for (auto& j : ctxt3.c0.GPU)
			for (auto& k : j.limb) {
				SWITCH(k, printThisLimb(1));
			}
		std::cout << std::endl;
		for (auto& j : ctxt3.c0.GPU)
			for (auto& k : j.SPECIALlimb) {
				SWITCH(k, printThisLimb(1));
			}
		std::cout << std::endl;
	}

	CudaNvtxRange r(std::string{ sc::current().function_name() });
	assert(pts1.size() >= rowSize);
	for (auto i : pts1) {
		assert(i != nullptr);
	}
	assert(pts2.size() >= rowSize);

	Context& cc_	= ctxt1.cc_;
	ContextData& cc = ctxt1.cc;
	uint32_t gStep	= ceil(static_cast<double>(rowSize) / bStep);

	if (ctxt1.NoiseLevel == 2)
		ctxt1.rescale();
	if (ctxt2.NoiseLevel == 2)
		ctxt2.rescale();
	if (ctxt3.NoiseLevel == 2)
		ctxt3.rescale();

	std::vector<Ciphertext> fastRotation;

	for (int i = fastRotation.size(); i < (bStep - 1) * 4 + 1; ++i)
		fastRotation.emplace_back(cc_);

	std::vector<Ciphertext*> fastRotationPtr2;
	std::vector<Ciphertext*> fastRotationPtr1;
	std::vector<int> indexes;
	for (int i = 1; i < bStep; ++i) {
		fastRotationPtr1.push_back(&fastRotation[i - 1]);
		fastRotationPtr2.push_back(&fastRotation[(bStep - 1) + i - 1]);
		indexes.push_back(i * stride);
	}

	ctxt1.rotate_hoisted(indexes, fastRotationPtr1, false);
	ctxt2.rotate_hoisted(indexes, fastRotationPtr2, false);

	/*
	std::vector<Ciphertext> fastRotation3;
	for (int i = 0; i < bStep - 1; ++i)
		fastRotation3.emplace_back(cc);
*/
	std::vector<Ciphertext*> fastRotationPtr3;
	std::vector<int> indexes3;
	std::vector<KeySwitchingKey*> keys3;
	for (int i = 1; i < bStep; ++i) {
		fastRotationPtr3.push_back(&fastRotation[2 * (bStep - 1) + i - 1]);
		indexes3.push_back(i * stride3);
	}

	std::vector<Ciphertext*> zeroRot_andFastRotationPtr3;
	zeroRot_andFastRotationPtr3.push_back(&ctxt3);
	for (int i = 1; i < bStep; ++i) {
		zeroRot_andFastRotationPtr3.push_back(fastRotationPtr3[i - 1]);
	}

	std::vector<Ciphertext*> auxVector;
	for (int i = 0; i < bStep; ++i) {
		auxVector.push_back(&fastRotation[3 * (bStep - 1) + i]);
	}

	Ciphertext result(cc_);
	// Ciphertext inner(cc_);
	Ciphertext aux(cc_);

	if ((gStep - 1) * bStep * (rowSize - 1) != 0)
		ctxt3.rotate((gStep - 1) * bStep * (rowSize - 1), true);

	for (uint32_t j = gStep - 1; j < gStep; --j) {

		auxVector[0]->multPt(ctxt1, *pts1[bStep * j], false);
		if (bStep * j != 0)
			auxVector[0]->addMultPt(ctxt2, *pts2[bStep * j], false);

		// auxVector[0]->rescale();
		//  auxVector[0]->mult(ctxt3, false, false);

		for (uint32_t i = 1; i < static_cast<uint32_t>(bStep); i++) {
			if (bStep * j + i < static_cast<uint32_t>(rowSize)) {
				if (i == 1) {
					int size = std::min((int)bStep - 1, (int)(rowSize - (bStep * j + i)));
					if (size < bStep - 1) {
						auto keys3_			   = keys3;
						auto indexes3_		   = indexes3;
						auto fastRotationPtr3_ = fastRotationPtr3;

						keys3_.resize(size);
						indexes3_.resize(size);
						fastRotationPtr3_.resize(size);

						ctxt3.rotate_hoisted(indexes3_, fastRotationPtr3_, false);
					} else {
						ctxt3.rotate_hoisted(indexes3, fastRotationPtr3, false);
					}
				}
				auxVector[i]->multPt(*fastRotationPtr1[i - 1], *pts1[bStep * j + i], false);
				auxVector[i]->addMultPt(*fastRotationPtr2[i - 1], *pts2[bStep * j + i], false);
				// auxVector[i]->mult(*fastRotationPtr3[i - 1], false, false);
				// auxVector[i]->rescale();
				// auxVector[0]->add(*auxVector[i]);
			}
		}

		// CudaCheckErrorMod;

		aux.dotProduct(auxVector, zeroRot_andFastRotationPtr3, true);

		// CudaCheckErrorMod;
		/*
		 for (uint32_t i = 0; i < bStep; i++) {
			if (bStep * j + i < rowSize) {
				auxVector[i]->mult(*zeroRot_andFastRotationPtr3[i], false, false);
				if (i == 0) {
					aux.copy(*auxVector[i]);
				} else {
					aux.add(*auxVector[i]);
				}
			}
		}
		*/

		if (j == gStep - 1) {
			result.copy(aux);
		} else {
			result.add(aux);
		}
		if (j > 0) {
			if (result.c1.isModUp())
				result.c1.moddown();
			result.rotate(stride * bStep, false);
			ctxt3.rotate(-bStep * (rowSize - 1), true);
		} else {
			result.modDown();
		}
	}

	result.rescale();
	result.NoiseFactor = cc.param.ScalingFactorRealBig[result.getLevel()];
	ctxt1.copy(result);
	CudaCheckErrorModNoSync;
}
#endif

std::vector<int> FIDESlib::CKKS::GetLinearTransformRotationIndices(int bStep, int stride, int offset) {
	std::vector<int> res(bStep + (offset != 0));
	for (int i = 1; i <= bStep; ++i)
		res[i - 1] = i * stride;
	if (offset != 0)
		res[bStep] = offset;
	return res;
}

std::vector<int> FIDESlib::CKKS::GetLinearTransformPlaintextRotationIndices(int rowSize, int bStep, int stride, int offset) {
	std::vector<int> res(rowSize);
	uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

	for (uint32_t j = 0; j < gStep; j++) {
		for (int i = 0; i < bStep; ++i) {
			if (i + j * bStep < static_cast<uint32_t>(rowSize))
				res[i + j * bStep] = -bStep * j * stride - offset;
		}
	}
	return res;
}

/*
void FIDESlib::CKKS::LinearTransformPt(FIDESlib::CKKS::Plaintext& ptxt, FIDESlib::CKKS::Context& cc, int rowSize,
									   int bStep, std::vector<Plaintext*> pts, int stride, int offset) {

	CudaNvtxRange r(std::string{sc::current().function_name()});

	assert(pts.size() >= rowSize);
	for (auto i : pts) {
		assert(i != nullptr);
	}

	uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

	std::vector<Plaintext> fastRotation;
	for (int i = 0; i < bStep - 1; ++i)
		fastRotation.emplace_back(cc);

	std::vector<Plaintext*> fastRotationPtr;
	std::vector<int> indexes;
	std::vector<KeySwitchingKey*> keys;

	for (int i = 1; i < bStep; ++i) {
		fastRotationPtr.push_back(&fastRotation[i - 1]);
		keys.push_back(&cc.GetRotationKey(i * stride));
		indexes.push_back(i * stride);
	}
	ptxt.rotate_hoisted(indexes, fastRotationPtr);

	Plaintext result(cc);
	Plaintext inner(cc);

	for (uint32_t j = gStep - 1; j < gStep; --j) {
		Plaintext temp(cc);
		temp.copy(ptxt);
		temp.multPt(*pts[bStep * j], false);
		inner.copy(temp);
		for (uint32_t i = 1; i < bStep; i++) {
			if (bStep * j + i < rowSize) {
				Plaintext tmp(cc);
				tmp.copy(fastRotation[i - 1]);
				tmp.multPt(*pts[(bStep * j + i)], false);
				inner.addPt(tmp);
			}
		}
		if (j > 0) {
			if (j == gStep - 1) {
				result.copy(inner);
			} else {
				Plaintext tmp(cc);
				tmp.copy(result);
				inner.addPt(tmp);
				result.copy(inner);
				// result.addPt(inner); // the d-tour here is due to the level adjustment logic in the RNSPoly structure
			}
			result.automorph(stride * bStep);
		} else {
			if (gStep == 1) {
				result.copy(inner);
			} else {
				Plaintext tmp(cc);
				tmp.copy(result);
				inner.addPt(tmp);
				result.copy(inner);
				// result.addPt(inner); // the d-tour here is due to the level adjustment logic in the RNSPoly structure
			}
		}
	}
	if (offset != 0) {
		result.automorph(offset);
	}
	result.rescale();
	ptxt.copy(result);
}
*/

void FIDESlib::CKKS::LinearTransformSpecialPt(FIDESlib::CKKS::Ciphertext& ctxt1,
  FIDESlib::CKKS::Ciphertext& ctxt2,
  FIDESlib::CKKS::Plaintext& ptxt,
  int rowSize,
  int bStep,
  std::vector<Plaintext*> pts1,
  std::vector<Plaintext*> pts2,
  int stride,
  int stride3) {

	CudaNvtxRange r(std::string{ sc::current().function_name() });
	assert(pts1.size() >= rowSize);
	for (auto i : pts1) {
		assert(i != nullptr);
	}
	assert(pts2.size() >= rowSize);

	Context& cc_	= ctxt1.cc_;
	ContextData& cc = ctxt1.cc;
	uint32_t gStep	= ceil(static_cast<double>(rowSize) / bStep);

	if (ctxt1.NoiseLevel == 2)
		ctxt1.rescale();
	if (ctxt2.NoiseLevel == 2)
		ctxt2.rescale();

	std::vector<Ciphertext> fastRotation;

	for (int i = fastRotation.size(); i < (bStep - 1) * 2; ++i)
		fastRotation.emplace_back(cc_);

	/*
	std::vector<Ciphertext> fastRotation1;
	for (int i = 0; i < bStep - 1; ++i)
		fastRotation1.emplace_back(cc);
	std::vector<Ciphertext> fastRotation2;
	for (int i = 0; i < bStep - 1; ++i)
		fastRotation2.emplace_back(cc);
*/
	std::vector<Ciphertext*> fastRotationPtr1;
	std::vector<Ciphertext*> fastRotationPtr2;
	std::vector<int> indexes;
	for (int i = 1; i < bStep; ++i) {
		fastRotationPtr1.push_back(&fastRotation[i - 1]);
		fastRotationPtr2.push_back(&fastRotation[(bStep - 1) + i - 1]);
		indexes.push_back(i * stride);
	}

	ctxt1.rotate_hoisted(indexes, fastRotationPtr1, false);
	ctxt2.rotate_hoisted(indexes, fastRotationPtr2, false);

	std::vector<Plaintext> fastRotation3;
	for (int i = 0; i < bStep - 1; ++i)
		fastRotation3.emplace_back(cc_);

	std::vector<Plaintext*> fastRotationPtr3;
	std::vector<int> indexes3;
	for (int i = 1; i < bStep; ++i) {
		fastRotationPtr3.push_back(&fastRotation3[i - 1]);
		indexes3.push_back(i * stride3);
	}

	Ciphertext result(cc_);
	Ciphertext inner(cc_);
	Ciphertext aux(cc_);

	if ((gStep - 1) * bStep * (rowSize - 1) != 0)
		ptxt.automorph((gStep - 1) * bStep * (rowSize - 1));

	for (uint32_t j = gStep - 1; j < gStep; --j) {

		inner.multPt(ctxt1, *pts1[bStep * j], false);
		if (bStep * j != 0)
			inner.addMultPt(ctxt2, *pts2[bStep * j], false);

		// inner.rescale();
		inner.multPt(ptxt, false, true);
		for (uint32_t i = 1; i < static_cast<uint32_t>(bStep); i++) {

			if (bStep * j + i < static_cast<uint32_t>(rowSize)) {

				if (i == 1) {
					int size = std::min((int)bStep - 1, (int)(rowSize - (bStep * j + i)));
					if (size < bStep - 1) {
						// auto keys3_ = keys3;
						auto indexes3_ = indexes3;

						// keys3_.resize(size);
						indexes3_.resize(size);
						// ctxt2.rotate_hoisted(keys3_, indexes3_, fastRotationPtr3);
						ptxt.rotate_hoisted(indexes3_, fastRotationPtr3);
					} else {
						// ctxt2.rotate_hoisted(keys3, indexes3, fastRotationPtr3);
						ptxt.rotate_hoisted(indexes3, fastRotationPtr3);
					}
				}
				aux.multPt(fastRotation[i - 1], *pts1[bStep * j + i], false);
				aux.addMultPt(fastRotation[(bStep - 1) + i - 1], *pts2[bStep * j + i], false);
				// aux.rescale();
				aux.multPt(fastRotation3[i - 1], false, true);

				inner.add(aux);
			}
		}

		if (j == gStep - 1) {
			result.copy(inner);
		} else {
			result.add(inner);
		}
		if (j > 0) {
			result.rotate(stride * bStep, true);
			ptxt.automorph(-bStep * (rowSize - 1));
		}
	}

	result.rescale();
	// result.rescale();
	result.NoiseFactor = cc.param.ScalingFactorRealBig[result.getLevel()];

	ctxt1.copy(result);
}
