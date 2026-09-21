//
// Created by carlosad on 12/06/25.
//

#include "CKKS/AccumulateBroadcast.cuh"

#include "CKKS/Context.cuh"

#include <bit>
#include <stdexcept>
#include <string>

namespace {
// Accumulate/Broadcast advance the rotation stride by `logbStep = bit_width(bStep) - 1` and
// generate `bStep - 1` hoisted rotations per round, so they are only correct when bStep is a
// power of two >= 2: bStep == 1 gives logbStep == 0 and the round loop never terminates, and a
// non-power-of-two bStep sums some shifts twice. Reject those instead of hanging or returning
// wrong results.
void checkBStep(const int bStep, const char* where) {
	if (bStep < 2 || (bStep & (bStep - 1)) != 0) {
		throw std::invalid_argument(std::string(where) + ": bStep must be a power of two >= 2, got " + std::to_string(bStep));
	}
}
} // namespace

void AccumulateCascadeImpl(FIDESlib::CKKS::Ciphertext& ctxt, const int bStep, const int stride, const int size, const int startFactor) {
	checkBStep(bStep, "Accumulate");
	if (startFactor <= 0 || size <= 0) {
		return;
	}

	FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	std::vector<FIDESlib::CKKS::Ciphertext> aux;
	for (int i = 0; i < bStep - 1; ++i) {
		aux.emplace_back(cc_);
	}

	int logbStep = std::bit_width((uint32_t)bStep) - 1;
	for (int s = startFactor; s < size; s <<= logbStep) {
		std::vector<int> indexes;
		std::vector<FIDESlib::CKKS::Ciphertext*> auxptr;
		for (int idx = stride * s; idx < stride * size && idx < bStep * stride * s; idx += stride * s) {
			indexes.push_back(idx);
			auxptr.emplace_back(&aux[idx / stride / s - 1]);
		}

		if (indexes.empty()) {
			continue;
		}

		ctxt.rotate_hoisted(indexes, auxptr, true);
		ctxt.extend();
		for (size_t i = 0; i < indexes.size(); ++i) {
			ctxt.add(*auxptr[i]);
		}
		ctxt.modDown(false);
	}

	if (size * stride == ctxt.slots)
		ctxt.slots = stride * startFactor;
} // namespace

std::vector<int> FIDESlib::CKKS::GetAccumulateRotationIndices(const int bStep, const int stride, const int size) {
	checkBStep(bStep, "GetAccumulateRotationIndices");
	std::vector<int> indices;
	int logbStep = std::bit_width((uint32_t)bStep) - 1;
	for (int s = stride; s < stride * size; s <<= logbStep) {
		for (int idx = s; idx < s * bStep && idx < stride * size; idx += s) {
			indices.push_back(idx);
		}
	}
	return indices;
}

std::vector<int> FIDESlib::CKKS::GetbroadcastRotationIndices(const int bStep, const int initsize, const int outsize) {
	checkBStep(bStep, "GetbroadcastRotationIndices");
	const int size	 = outsize / initsize;
	const int stride = initsize;
	std::vector<int> indices;

	int logbStep = std::bit_width((uint32_t)bStep) - 1;
	for (int s = stride; s < stride * size; s <<= logbStep) {
		if (stride * s * bStep >= stride * size) {
			for (int i = 1; i <= bStep; ++i) {
				int idx = -(i * s) + 1;
				if (-idx < outsize) {
					indices.push_back(idx);
				}
			}
		} else {
			for (int idx = s; idx < s * bStep && idx < stride * size; idx += s) {
				indices.push_back(idx);
			}
		}
	}
	return indices;
}

void FIDESlib::CKKS::Accumulate(Ciphertext& ctxt, const int bStep, const int stride, const int size) {
	checkBStep(bStep, "Accumulate");
	Context& cc_ = ctxt.cc_;
	// ContextData& cc = ctxt.cc;
	std::vector<Ciphertext> aux;

	for (int i = 0; i < bStep - 1; ++i) {
		aux.emplace_back(cc_);
	}

	int logbStep = std::bit_width((uint32_t)bStep) - 1;
	for (int s = 1; s < size; s <<= logbStep) {
		std::vector<int> indexes;
		std::vector<Ciphertext*> auxptr;
		for (int idx = stride * s; idx < stride * size && idx < bStep * stride * s; idx += stride * s) {
			// std::cout << idx << std::endl;
			indexes.push_back(idx);
			auxptr.emplace_back(&aux[idx / stride / s - 1]);
		}
		ctxt.rotate_hoisted(indexes, auxptr, true);
		// ctxt.extend();
		for (size_t i = 0; i < indexes.size(); ++i) {
			ctxt.add(*auxptr[i]);
		}
		ctxt.c1.moddown();
	}
	if (ctxt.c0.isModUp())
		ctxt.c0.moddown();

	if (size * stride == ctxt.slots)
		ctxt.slots = stride;
}

void FIDESlib::CKKS::Accumulate(Ciphertext& ctxt, const int bStep, const int stride, const int size, const int startFactor) {
	AccumulateCascadeImpl(ctxt, bStep, stride, size, startFactor);
}

void FIDESlib::CKKS::Broadcast(Ciphertext& ctxt, const int bStep, const int initsize, const int outsize) {
	checkBStep(bStep, "Broadcast");
	const int size	 = outsize / initsize;
	const int stride = initsize;
	Context& cc_	 = ctxt.cc_;
	// ContextData& cc	 = ctxt.cc;
	std::vector<Ciphertext> aux;

	for (int i = 0; i < bStep; ++i) {
		aux.emplace_back(cc_);
	}

	int logbStep = std::bit_width((uint32_t)bStep) - 1;
	for (int s = 1; s < size; s <<= logbStep) {
		std::vector<int> indexes;
		std::vector<Ciphertext*> auxptr;
		if (s * bStep >= size) {
			for (int i = 1; i <= bStep; ++i) {
				int idx = -(i * stride * s) + 1;
				if (-idx < outsize) {
					//  std::cout << idx << std::endl;
					indexes.push_back(idx);
					auxptr.emplace_back(&aux[i - 1]);
				}
			}
			ctxt.rotate_hoisted(indexes, auxptr, true);
			ctxt.add(*auxptr[0], *auxptr[1]);
			for (size_t i = 2; i < indexes.size(); ++i) {
				ctxt.add(*auxptr[i]);
			}

		} else {
			for (int idx = stride * s; idx < stride * size && idx < bStep * stride * s; idx += stride * s) {
				//  std::cout << idx << std::endl;
				indexes.push_back(idx);
				auxptr.emplace_back(&aux[idx / stride / s - 1]);
			}
			ctxt.rotate_hoisted(indexes, auxptr, true);
			ctxt.extend();
			for (size_t i = 0; i < indexes.size(); ++i) {
				ctxt.add(*auxptr[i]);
			}
		}
		ctxt.c1.moddown(false);
	}
	if (ctxt.c0.isModUp())
		ctxt.c0.moddown();
	
	if (outsize == ctxt.slots)
		ctxt.slots = initsize;
}