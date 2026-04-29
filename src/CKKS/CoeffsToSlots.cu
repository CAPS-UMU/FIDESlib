//
// Created by carlosad on 27/11/24.
//

#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include <ranges>
#include <vector>

#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

using namespace FIDESlib::CKKS;

constexpr bool BATCHED = false;

void FIDESlib::CKKS::EvalLinearTransform(Ciphertext& ctxt, int slots, bool decode) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	constexpr bool PRINT		 = false;
	FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	ContextData& cc				 = ctxt.cc;

	if constexpr (BATCHED) {
		/*
		CiphertextBatch<Ciphertext*> bctxt = {.cts = {&ctxt},
											  .conf = {.cc_ = ctxt.cc_,
													   .dims = {{.size = 1}},
													   .level = ctxt.getLevel(),
													   .scale_degree = ctxt.NoiseLevel,
													   .isExt = ctxt.c0.isModUp()}};

		auto& LTconf = cc.GetBootPrecomputation(slots).LT;
		PlaintextBatch<Plaintext*> bptxt = {
			.conf = {.cc_ = ctxt.cc_,
					 .dims = {{.size = 1},
							  {.size = (LTconf.slots + LTconf.bStep - 1) / LTconf.bStep},
							  {.size = LTconf.bStep}},
					 .level = ctxt.getLevel(),
					 .scale_degree = ctxt.NoiseLevel,
					 .isExt = (decode ? LTconf.invA : LTconf.A)[0].c0.isModUp()}};
		for (auto& i : (decode ? LTconf.invA : LTconf.A)) {
			bptxt.cts.push_back(&i);
		}
		int rowSize_padded = bptxt.conf.dims[1].size * bptxt.conf.dims[2].size;
		while (bptxt.cts.size() < rowSize_padded)
			bptxt.cts.push_back(nullptr);

		LinearTransform(bctxt, rowSize_padded, LTconf.bStep, bptxt, 1, 0);
		*/
	} else {
		// Computing the baby-step bStep and the giant-step gStep.
		uint32_t bStep = cc.GetBootPrecomputation(slots).LT.bStep;
		uint32_t gStep = ceil(static_cast<double>(slots) / bStep);

		// uint32_t M = cc.N * 2;
		// uint32_t N = cc.N;

		std::vector<Ciphertext> fastRotation;

		for (auto i = fastRotation.size(); i < bStep; ++i)
			fastRotation.emplace_back(cc_);

		std::vector<Ciphertext*> fastRotationPtr;
		std::vector<int> indexes;
		for (uint32_t i = 0; i < bStep; ++i) {
			fastRotationPtr.push_back(&fastRotation[i]);
			indexes.push_back(i);
		}

		if constexpr (PRINT) {
			cudaDeviceSynchronize();
			std::cout << "Input LT ";
			for (auto& j : ctxt.c0.GPU) {
				cudaSetDevice(j.device);
				for (auto& i : j.limb) {
					SWITCH(i, printThisLimb(1));
				}
			}
			std::cout << std::endl;
			cudaDeviceSynchronize();
		}

		bool ext = true;
		if (bStep == 1)
			ext = false;
		for (auto& i : decode ? cc.GetBootPrecomputation(slots).LT.invA : cc.GetBootPrecomputation(slots).LT.A) {
			if (!i.c0.isModUp()) {
				ext = false;
			}
		}

		if constexpr (PRINT) {
			cudaDeviceSynchronize();
			for (int i = 0; i < bStep; ++i) {
				std::cout << "In hoistRotation ";
				for (auto& j : fastRotation[i].c0.GPU) {
					cudaSetDevice(j.device);
					for (auto& k : j.limb) {
						SWITCH(k, printThisLimb(1));
					}
				}
				std::cout << std::endl;
				for (auto& j : fastRotation[i].c0.GPU) {
					cudaSetDevice(j.device);
					for (auto& k : j.SPECIALlimb) {
						SWITCH(k, printThisLimb(1));
					}
				}
				std::cout << std::endl;
			}
			cudaDeviceSynchronize();
		}

		ctxt.rotate_hoisted(indexes, fastRotationPtr, ext);

		if constexpr (PRINT) {
			cudaDeviceSynchronize();
			for (int i = 0; i < bStep; ++i) {
				std::cout << "Out hoistRotation ";
				for (auto& j : fastRotation[i].c0.GPU) {
					cudaSetDevice(j.device);
					for (auto& k : j.limb) {
						SWITCH(k, printThisLimb(1));
					}
				}
				std::cout << std::endl;
				for (auto& j : fastRotation[i].c0.GPU) {
					cudaSetDevice(j.device);
					for (auto& k : j.SPECIALlimb) {
						SWITCH(k, printThisLimb(1));
					}
				}
				std::cout << std::endl;
			}
			cudaDeviceSynchronize();
		}
		Ciphertext inner(cc_);
		std::vector<Plaintext>& A = decode ? cc.GetBootPrecomputation(slots).LT.invA : cc.GetBootPrecomputation(slots).LT.A;

		std::vector<Plaintext*> Aptr(slots, nullptr);
		for (uint32_t j = 0; j < gStep; ++j) {
			for (uint32_t i = 0; i < bStep; ++i) {
				if (bStep * j + i < static_cast<uint32_t>(slots))
					Aptr[bStep * j + i] = &(A[bStep * j + i]);
			}
		}
		LinearTransform(ctxt, slots, bStep, Aptr, 1, 0);
	}
}

void FIDESlib::CKKS::EvalCoeffsToSlots(Ciphertext& ctxt, int slots, bool decode) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	constexpr bool PRINT = false;
	// FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	ContextData& cc = ctxt.cc;

	if constexpr (PRINT) {
		cudaDeviceSynchronize();
		std::cout << "Input stc ";
		for (auto& j : ctxt.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(1));
			}
		}
		std::cout << std::endl;
		cudaDeviceSynchronize();
	}
	//  No need for Encrypted Bit Reverse
	// Ciphertext& result = ctxt;
	// hoisted automorphisms
	if (ctxt.NoiseLevel == 2)
		ctxt.rescale();

	for (BootstrapPrecomputation::LTstep& step : (decode ? cc.GetBootPrecomputation(slots).StC : cc.GetBootPrecomputation(slots).CtS)) {
		// computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)

		if constexpr (AFFINE_LT && BATCHED) {
			/*
			CiphertextBatch<Ciphertext*> bctxt = {.cts = {&ctxt},
												  .conf = {.cc_ = ctxt.cc_,
														   .dims = {{.size = 1}},
														   .level = ctxt.getLevel(),
														   .scale_degree = ctxt.NoiseLevel,
														   .isExt = ctxt.c0.isModUp()}};

			if (bctxt.conf.scale_degree == 2)
				bctxt.Rescale();

			PlaintextBatch<Plaintext*> bptxt = {
				.conf = {
					.cc_ = ctxt.cc_,
					.dims = {{.size = 1}, {.size = (step.slots + step.bStep - 1) / step.bStep}, {.size = step.bStep}},
					.level = ctxt.getLevel(),
					.scale_degree = ctxt.NoiseLevel,
					.isExt = step.A[0].c0.isModUp()}};
			for (auto& i : step.A) {
				bptxt.cts.push_back(&i);
			}
			int rowSize_padded = bptxt.conf.dims[1].size * bptxt.conf.dims[2].size;
			while (bptxt.cts.size() < rowSize_padded)
				bptxt.cts.push_back(nullptr);

			LinearTransform(bctxt, rowSize_padded, step.bStep, bptxt, step.rotIn[1] - step.rotIn[0], step.rotOut[0]);
			*/
		} else {

			{

				assert(step.slots == step.A.size());
				std::vector<Plaintext*> Aptr(step.slots, nullptr);
				for (int j = 0; j < step.gStep; ++j) {
					for (int i = 0; i < step.bStep; ++i) {
						if (step.bStep * j + i < step.slots)
							Aptr[step.bStep * j + i] = &(step.A[step.bStep * j + i]);
					}
				}

				int stride = step.bStep > 1 ? step.rotIn[1] - step.rotIn[0] : step.rotOut[1] - step.rotOut[0];
				int offset = step.rotOut[0];
				{ LinearTransform(ctxt, step.slots, step.bStep, Aptr, stride, offset); }
			}
		}
	}
}
