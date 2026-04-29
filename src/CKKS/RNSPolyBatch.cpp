//
// Created by carlosad on 1/10/25.
//
#include <errno.h>

#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"

#include <omp.h>
#include <stdexcept>

namespace FIDESlib::CKKS {

void RNSPoly::addBatchManyToOne(std::vector<RNSPoly*>& polya, const std::vector<RNSPoly*>& polyb, int stride, double usage, bool sub, bool exta, bool extb) {
#pragma omp parallel for num_threads(polya[0]->cc.GPUid.size())
	for (size_t i = 0; i < polya[0]->cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)polya[0]->cc.GPUid.size());
		std::vector<LimbPartition*> parta;
		parta.reserve(polya.size());
		std::vector<LimbPartition*> partb;
		partb.reserve(polyb.size());
		for (auto j : polya) {
			parta.push_back(&j->GPU[i]);
		}
		for (auto j : polyb) {
			partb.push_back(&j->GPU[i]);
		}
		LimbPartition::addBatchManyToOne(parta, partb, stride, usage, sub, exta, extb);
	}
}

void RNSPoly::multPtBatchManyToOne(std::vector<RNSPoly*>& polya, const std::vector<RNSPoly*>& polyb, int stride, double usage) {
#pragma omp parallel for num_threads(polya[0]->cc.GPUid.size())
	for (size_t i = 0; i < polya[0]->cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)polya[0]->cc.GPUid.size());
		std::vector<LimbPartition*> parta;
		parta.reserve(polya.size());
		std::vector<LimbPartition*> partb;
		partb.reserve(polyb.size());
		for (auto j : polya) {
			parta.push_back(&j->GPU[i]);
		}
		for (auto j : polyb) {
			partb.push_back(&j->GPU[i]);
		}
		LimbPartition::multPtBatchManyToOne(parta, partb, stride, usage);
	}
}

void RNSPoly::addScalarBatchManyToOne(std::vector<RNSPoly*>& polya, const std::vector<std::vector<unsigned long int>>& vector, int stride, double usage) {
#pragma omp parallel for num_threads(polya[0]->cc.GPUid.size())
	for (size_t i = 0; i < polya[0]->cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)polya[0]->cc.GPUid.size());
		std::vector<LimbPartition*> parta;
		parta.reserve(polya.size());
		for (auto j : polya) {
			parta.push_back(&j->GPU[i]);
		}
		LimbPartition::addScalarBatchManyToOne(parta, vector, stride, usage);
	}
}

void RNSPoly::multScalarBatchManyToOne(std::vector<RNSPoly*>& polya,
  const std::vector<std::vector<unsigned long int>>& vector,
  const std::vector<std::vector<unsigned long int>>& vectors,
  int stride,
  double usage) {
#pragma omp parallel for num_threads(polya[0]->cc.GPUid.size())
	for (size_t i = 0; i < polya[0]->cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)polya[0]->cc.GPUid.size());
		std::vector<LimbPartition*> parta;
		parta.reserve(polya.size());
		for (auto j : polya) {
			parta.push_back(&j->GPU[i]);
		}
		LimbPartition::multScalarBatchManyToOne(parta, vector, vectors, stride, usage);
	}
}

void RNSPoly::LTdotProductPtBatch(std::vector<RNSPoly*>& out, const std::vector<RNSPoly*>& in, const std::vector<RNSPoly*>& pt, int bStep, int gStep, int stride, double usage, bool ext) {
	ContextData& cc = out[0]->cc;
	if (gStep <= 8) {
#pragma omp parallel for num_threads(out[0]->cc.GPUid.size())
		for (size_t i = 0; i < out[0]->cc.GPUid.size(); ++i) {
			assert(omp_get_num_threads() == (int)out[0]->cc.GPUid.size());
			std::vector<LimbPartition*> outs;
			std::vector<LimbPartition*> ins;
			std::vector<LimbPartition*> pts;
			outs.reserve(out.size());
			for (auto j : out) {
				outs.push_back(&j->GPU[i]);
			}
			ins.reserve(in.size());
			for (auto j : in) {
				ins.push_back(&j->GPU[i]);
			}
			pts.reserve(pt.size());
			for (auto j : pt) {
				if (j)
					pts.push_back(&j->GPU[i]);
				else
					pts.push_back(nullptr);
			}

			LimbPartition::LTdotProductPtBatch(outs, ins, pts, bStep, gStep, stride, usage, ext);
		}

		for (auto i : out) {
			i->SetModUp(ext);
		}
	} else {
		if (1) {
			int num_LT = out.size() / (2 * gStep);
			for (int g_in = 0; g_in < gStep; g_in += 8) {
				int g_internal = std::min(8, gStep - g_in);
				// int num_out	   = num_LT * stride * g_in;
#pragma omp parallel for num_threads(out[0]->cc.GPUid.size())
				for (size_t i = 0; i < out[0]->cc.GPUid.size(); ++i) {
					assert(omp_get_num_threads() == (int)out[0]->cc.GPUid.size());
					std::vector<LimbPartition*> outs;
					std::vector<LimbPartition*> ins;
					std::vector<LimbPartition*> pts;
					outs.reserve(out.size());

					for (int i_ = 0; i_ < num_LT; ++i_) {
						for (int j = 0; j < stride; ++j) {
							for (int k = g_in; k < g_in + g_internal; ++k) {
								outs.push_back(&out[2 * (i_ * stride * gStep + j * gStep + k)]->GPU[i]);
								outs.push_back(&out[2 * (i_ * stride * gStep + j * gStep + k) + 1]->GPU[i]);
								// data_ptrs[offset_out_c0 + i * stride * gStep + j * gStep + k] = out[2 * (i * stride * gStep + j * gStep + k)]->limbptr.data;
							}
						}
					}

					ins.reserve(in.size());
					for (auto j : in) {
						ins.push_back(&j->GPU[i]);
					}
					pts.reserve(pt.size());

					for (int i_ = 0; i_ < num_LT; i_++) {
						for (int j = g_in; j < g_in + g_internal; ++j) {
							for (int k = 0; k < bStep; ++k) {
								pts.push_back(pt[i_ * gStep * bStep + j * bStep + k] ? &pt[i_ * gStep * bStep + j * bStep + k]->GPU[i] : nullptr);
							}
						}
					}

					cudaSetDevice(cc.GPUid[i]);

					outs[0]->s.wait(out[0]->GPU[i].s);
					pts[0]->s.wait(pt[0]->GPU[i].s);
					LimbPartition::LTdotProductPtBatch(outs, ins, pts, bStep, g_internal, stride, usage, ext);
					out[0]->GPU[i].s.wait(outs[0]->s);
					pt[0]->GPU[i].s.wait(pts[0]->s);
				}
			}

			for (auto i : out) {
				i->SetModUp(ext);
			}

		} else { // TODO correct for stride != 1
			assert(stride == 1);
			// ---- Added safety checks ----
			// The fallback implementation assumes an even number of output polynomials.
			assert(out.size() % 2 == 0 && "LTdotProductPtBatch fallback requires an even number of output polynomials");

			// const std::size_t nBatches = out.size() / 2; // Number of output pairs processed
			//  Plain‑text vector must be large enough to provide `bStep` factors per batch.
			// const std::size_t minPtSize = static_cast<std::size_t>(bStep) * nBatches;
			// assert(pt.size() >= minPtSize && "Plain‑text vector `pt` is too small for the requested batch configuration");

			// Input vector must contain enough operands for all batches.
			// For each group of `gStep` batches we need `2 * bStep` inputs per batch.
			// const std::size_t groups	= (nBatches + static_cast<std::size_t>(gStep) - 1) / static_cast<std::size_t>(gStep);
			// const std::size_t minInSize = static_cast<std::size_t>(bStep) * 2 * groups;
			// assert(in.size() >= minInSize && "Input vector `in` is too small for the requested batch configuration");
			// ---- End of safety checks ----

			// #pragma omp parallel for
			for (uint32_t i = 0; i < out.size() / 2; ++i) {
				std::vector<const RNSPoly*> c0s;
				std::vector<const RNSPoly*> c1s;
				std::vector<const RNSPoly*> pts_;
				for (int j = 0; j < bStep; ++j) {
					if (!pt[bStep * i + j])
						break;
					c0s.emplace_back(in[bStep * 2 * (i / gStep) + 2 * j]);
					c1s.emplace_back(in[bStep * 2 * (i / gStep) + 2 * j + 1]);
					pts_.emplace_back(pt[bStep * i + j]);
				}

				out[2 * i]->dotProductPt(*out[2 * i + 1], c0s, c1s, pts_, ext);
			}
		}
	}
}

void RNSPoly::fusedHoistedRotateBatch(std::vector<RNSPoly*>& out,
  const std::vector<RNSPoly*>& in,
  const std::vector<RNSPoly*>& ksk_a,
  const std::vector<RNSPoly*>& ksk_b,
  const std::vector<int>& indexes,
  int stride,
  double usage,
  bool c0_modup) {

	uint32_t n = indexes.size();
	std::vector<int> index(n);
	for (uint32_t j = 0; j < n; ++j) {
		index[j] = in[0]->automorph_index_precomp(indexes[j]);
	}

#pragma omp parallel for num_threads(out[0]->cc.GPUid.size())
	for (size_t i = 0; i < out[0]->cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)out[0]->cc.GPUid.size());
		std::vector<LimbPartition*> outs;
		std::vector<LimbPartition*> ins;
		std::vector<LimbPartition*> ksk_as;
		std::vector<LimbPartition*> ksk_bs;

		outs.reserve(out.size());
		for (auto j : out) {
			outs.push_back(&j->GPU[i]);
		}
		ins.reserve(in.size());
		for (auto j : in) {
			ins.push_back(&j->GPU[i]);
		}
		ksk_as.reserve(ksk_a.size());
		for (auto j : ksk_a) {
			ksk_as.push_back(j ? &j->GPU[i] : nullptr);
		}
		ksk_bs.reserve(ksk_b.size());
		for (auto j : ksk_b) {
			ksk_bs.push_back(j ? &j->GPU[i] : nullptr);
		}

		LimbPartition::fusedHoistedRotateBatch(outs, ins, ksk_as, ksk_bs, index, n, stride, usage, c0_modup);

		for (auto j : out) {
			j->SetModUp(true);
		}
	}
}

} // namespace FIDESlib::CKKS