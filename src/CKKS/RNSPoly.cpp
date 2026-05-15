//
// Created by carlosad on 25/04/24.
//
#include <errno.h>

#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"

#include <omp.h>
#include <stdexcept>

#include "../parallel_for.hpp"

// #define OMP omp_disabled
#define OMP omp
/**
#define OMP_ASSERT(x) \
	do {              \
		x;            \
	} while (0)
*/
#define OMP_ASSERT(x) assert(x);

namespace FIDESlib::CKKS {
void RNSPoly::grow(int new_level, bool single_malloc, bool constant) {
	if (level >= new_level)
		return;
	level = new_level;

	// single_malloc = false;
	// if (level == -1) {
	// std::cout << "from 0" << std::endl;
	if (!constant && (!single_malloc || (GPU.at(0).limb.size() > 0)) && GPU.at(0).bufferLIMB == nullptr) {
		// TODO fix bug (check that limb size matches level)
		int init = 0;
		for (auto& g : GPU)
			init += g.limb.size();

		for (auto& g : GPU) {
			g.generateLimbToLevel(new_level);
		}
		// for (int i = init; i <= new_level; ++i) {
		//     GPU.at(cc.limbGPUid.at(i).x).generateLimb();
		// }
	} else {
		// #pragma omp parallel for num_threads(GPU.size())
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			//          OMP_ASSERT(omp_get_num_threads() == (int)GPU.size());
			if (!constant) {
				GPU.at(i).generateLimbSingleMalloc();
			} else {
				GPU.at(i).generateLimbConstant();
			}
		}
	}
}

RNSPoly::RNSPoly(ContextData& context, int level, bool single_malloc, bool def_stream) : uid(next_uid++), cc(context), level(-1) {

	// #pragma omp parallel for num_threads(context.GPUid.size())
	for (size_t i = 0u; i < context.GPUid.size(); ++i) {
		cudaSetDevice(context.GPUid.at(i));
		GPU.emplace_back(context, uid, &this->level, i, def_stream);
	}
	assert(level >= -1 && level <= cc.L);
	grow(level, single_malloc);
}

RNSPoly::RNSPoly(ContextData& context, const std::vector<std::vector<uint64_t>>& data) : RNSPoly(context, data.size() - 1) {

	assert(data.size() <= cc.prime.size());
	std::vector<uint64_t> moduli(data.size());
	for (size_t i = 0; i < data.size(); ++i)
		moduli[i] = cc.prime[i].p;
	load(data, moduli);
}

RNSPoly::RNSPoly(RNSPoly&& src) noexcept : uid(src.uid), cc(src.cc), level(src.level), modUp(src.modUp), GPU(std::move(src.GPU)) {
	for (auto& g : GPU) {
		g.level = &(this->level);
	}
	// src.setLevel(-2);
}

int32_t RNSPoly::getLevel() const {
	return level;
}

void RNSPoly::freeSpecialLimbs() {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).freeSpecialLimbs();
	}
	this->SetModUp(false);
}

void RNSPoly::generateSpecialLimbs(const bool zero_out, const bool for_communication) {
	if (!GPU[0].SPECIALlimb.empty()) {
		if (zero_out) {
#pragma omp parallel for num_threads(cc.GPUid.size())
			for (size_t i = 0; i < cc.GPUid.size(); ++i) {
				assert(omp_get_num_threads() == (int)cc.GPUid.size());
				GPU[i].generateSpecialLimb(zero_out, for_communication);
			}
		}
		return;
	}

#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[i].generateSpecialLimb(zero_out, for_communication);
	}

	if (PEER_ACCESS && for_communication) {
		std::vector<void*> cpu_ptr(GPU[0].SPECIALlimb.size(), nullptr);
		for (uint32_t j = 0; j < cc.splitSpecialMeta.size(); ++j) {
			for (uint32_t g = 0; g < cc.splitSpecialMeta[j].size(); ++g) {
				for (uint32_t i = 0; i < cc.specialMeta[j].size(); ++i) {
					if (cc.specialMeta[0][i].id == cc.splitSpecialMeta[j][g].id) {
						cpu_ptr[i] = GPU[j].SPECIALlimb[i].index() == U32 ? (void*)std::get<U32>(GPU[j].SPECIALlimb[i]).v.data :
																			(void*)std::get<U64>(GPU[j].SPECIALlimb[i]).v.data;
					}
				}
			}
		}

		if (GPU[0].SPECIALlimb.size() * sizeof(void*) > 0) {
			CudaCheckErrorMod;
			for (size_t g = 0; g < GPU.size(); ++g) {
				// std::cout << GPU[g].DECOMPlimbptr[i].data << " " << cpu_ptr.data() << " " << GPU[g].DECOMPmeta.at(i).size() * sizeof(void*) << " "
				//		  << cudaMemcpyHostToDevice << " " << GPU[g].s.ptr() << std::endl;
				cudaSetDevice(cc.GPUid[g]);
				cudaMemcpyAsync(GPU[g].SPECIALlimbptr.data, cpu_ptr.data(), GPU[g].SPECIALmeta.size() * sizeof(void*), cudaMemcpyHostToDevice, GPU[g].s.ptr());
				CudaCheckErrorMod;
			}
		}
	}
}

void RNSPoly::generateDecompAndDigit(bool iskey) {

	if (!GPU[0].DECOMPlimb[0].empty())
		return;

#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[i].generateAllDecompAndDigit(iskey);
	}

	/** To support peer access kernels we only need to change the pointers to the other libs*/
	if (PEER_ACCESS) {
		for (size_t i = 0; i < GPU[0].DECOMPmeta.size(); ++i) {
			std::vector<void*> cpu_ptr(GPU[0].DECOMPmeta.at(i).size(), nullptr);
			for (uint32_t j = 0; j < GPU[0].DECOMPlimb[i].size(); ++j) {
				for (size_t g = 0; g < cc.meta.size(); ++g) {
					for (auto& m : cc.meta[g]) {
						if (m.id == cc.decompMeta[0][i][j].id) {
							cpu_ptr[j] = GPU[g].DECOMPlimb[i][j].index() == U32 ? (void*)std::get<U32>(GPU[g].DECOMPlimb[i][j]).v.data :
																				  (void*)std::get<U64>(GPU[g].DECOMPlimb[i][j]).v.data;
						}
					}
				}
			}

			if (GPU[0].DECOMPmeta.at(i).size() * sizeof(void*) > 0) {
				CudaCheckErrorMod;
				for (size_t g = 0; g < GPU.size(); ++g) {
					// std::cout << GPU[g].DECOMPlimbptr[i].data << " " << cpu_ptr.data() << " " << GPU[g].DECOMPmeta.at(i).size() * sizeof(void*) << " "
					//		  << cudaMemcpyHostToDevice << " " << GPU[g].s.ptr() << std::endl;
					cudaSetDevice(cc.GPUid[g]);
					cudaMemcpyAsync(
					  GPU[g].DECOMPlimbptr[i].data, cpu_ptr.data(), GPU[g].DECOMPmeta.at(i).size() * sizeof(void*), cudaMemcpyHostToDevice, GPU[g].s.ptr());
					CudaCheckErrorMod;
				}
			}
		}
	}
}

void RNSPoly::loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data, const std::vector<std::vector<uint64_t>>& moduli) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).loadDecompDigit(data, moduli);
	}
}

void RNSPoly::store(std::vector<std::vector<uint64_t>>& data) {
	data.resize(level + 1);
	for (size_t i = 0; i < data.size(); ++i) {
		// auto& rec = cc.meta[cc.limbGPUid[i].x][cc.limbGPUid[i].y];
		cudaSetDevice(GPU[cc.limbGPUid[i].x].device);
		SWITCH(GPU[cc.limbGPUid[i].x].limb[cc.limbGPUid[i].y], store_convert(data[i]));
	}
}

bool RNSPoly::isModUp() const {
	return modUp;
}

void RNSPoly::SetModUp(bool newValue) {
	modUp = newValue;
}

void RNSPoly::scaleByP() {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).scaleByP();
	}
	this->SetModUp(true);
}

void RNSPoly::multNoModdownEnd(RNSPoly& c0, const RNSPoly& bc0, const RNSPoly& bc1, const RNSPoly& in, const RNSPoly& aux) {
	assert(in.isModUp());
	assert(aux.isModUp());
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).multNoModdownEnd(c0.GPU.at(i), bc0.GPU.at(i), bc1.GPU.at(i), in.GPU.at(i), aux.GPU.at(i));
	}
	this->SetModUp(true);
	c0.SetModUp(true);
}

void RNSPoly::binomialMult(RNSPoly& c1, RNSPoly& in, const RNSPoly& d0, const RNSPoly& d1, bool moddown, bool square) {
	assert(!this->isModUp() && !c1.isModUp() && !d0.isModUp() && !d1.isModUp());

	if (!moddown) {
		this->generateSpecialLimbs(true, false);
		CudaCheckErrorMod;
		c1.generateSpecialLimbs(true, false);
	}

#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).binomialMult(c1.GPU.at(i), in.GPU.at(i), d0.GPU.at(i), d1.GPU.at(i), !moddown, square);
	}
	this->SetModUp(!moddown);
	c1.SetModUp(!moddown);
	in.SetModUp(!moddown);
}

void RNSPoly::add(const RNSPoly& p) {

	if (p.isModUp() && !this->isModUp()) {
		// std::cout << "Adapt non modup destination add" << std::endl;
		generateSpecialLimbs(false, false);
		// scaleByP();
	}
	assert(level <= p.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).add(p.GPU.at(i), this->isModUp(), p.isModUp());
	}
	this->SetModUp(this->isModUp() || p.isModUp());
}

void RNSPoly::sub(const RNSPoly& p) {
	assert(level <= p.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).sub(p.GPU.at(i));
	}
}

void RNSPoly::modup() {
	//  assert(GPU.size() == 1 || 0 == "ModUp Multi-GPU not implemented.");
	RNSPoly& aux = cc.getKeySwitchAux2();

	generateDecompAndDigit(false);
	aux.generateDecompAndDigit(false);

	std::vector<std::atomic_uint64_t> thread_stop_buffer(cc.GPUid.size() * 8);
	std::vector<std::atomic_uint64_t*> thread_stop(cc.GPUid.size(), nullptr);
	for (uint32_t k = 0; k < cc.GPUid.size(); ++k) {
		thread_stop[k]			  = &thread_stop_buffer[8 * k];
		thread_stop_buffer[8 * k] = 0;
	}

	std::vector<Stream*> external_s;
	for (uint32_t i = 0; i < GPU.size(); ++i) {
		external_s.push_back(&GPU[i].s);
	}

	std::vector<uint64_t*> buffergather;
	for (uint32_t i = 0; i < cc.GPUid.size(); ++i) {
		buffergather.push_back(GPU.at(i).bufferGATHER);
	}
	if (cc.GPUid.size() == 1) {
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			GPU.at(i).modupMGPU(aux.GPU.at(i), buffergather, thread_stop, external_s);
		}
	} else {
#pragma omp parallel num_threads(GPU.size())
		{
			int i = omp_get_thread_num();

			if (omp_get_num_threads() != (int)GPU.size())
				throw std::invalid_argument("OMP didn't create enough threads");
			assert(omp_get_num_threads() == (int)GPU.size());
			assert(static_cast<size_t>(i) < GPU.size());
			GPU.at(i).modupMGPU(aux.GPU.at(i), buffergather, thread_stop, external_s);
		}
	}
}

void RNSPoly::sync() {
	for (auto& i : GPU) {
		for (auto& j : i.limb) {
			cudaStreamSynchronize(STREAM(j).ptr());
		}
		cudaStreamSynchronize(i.s.ptr());
	}
}

void RNSPoly::rescale() {
	//    assert(GPU.size() == 1 && "Rescale Multi-GPU not implemented.");
	if (GPU.size() == 1) {
		for (auto& i : GPU) {
			i.rescale();
		}
		level -= 1;
	} else {
		int more_than_0 = 0;
		for (size_t i = 0; i < GPU.size(); ++i)
			if (GPU[i].getLimbSize(level) != 0)
				more_than_0++;

		if (more_than_0 == 1) {
			for (auto& i : GPU) {
				if (i.getLimbSize(level) > 0)
					i.rescale();
			}
		} else {
#pragma omp parallel num_threads(GPU.size())
			{
#pragma omp for
				for (size_t i = 0; i < GPU.size(); ++i) {
					if (omp_get_num_threads() != (int)GPU.size())
						throw std::invalid_argument("OMP didn't create enough threads");
					assert(omp_get_num_threads() == (int)GPU.size());
					GPU[i].rescaleMGPU();
				}
			}
		}
		level -= 1;
	}
}

void RNSPoly::rescaleDouble(RNSPoly& poly) {
	//    assert(GPU.size() == 1 && "Rescale Multi-GPU not implemented.");
	if (0 && GPU.size() == 1) {
		for (auto& i : GPU) {
			i.rescale();
		}
		level -= 1;
		for (auto& i : poly.GPU) {
			i.rescale();
		}
		poly.level -= 1;
	} else {

		int more_than_0 = 0;
		for (size_t i = 0; i < GPU.size(); ++i)
			if (GPU[i].getLimbSize(level) > 0)
				more_than_0++;

		if (0 && more_than_0 == 1) {
			for (size_t i = 0; i < GPU.size(); ++i) {
				if (GPU[i].getLimbSize(level) > 0) {
					GPU[i].rescale();
					poly.GPU[i].rescale();
				}
			}
		} else {

			if (0 && MEMCPY_PEER) {

				int id = cc.limbGPUid[level].x;
				GPU[id].doubleRescaleMGPU(poly.GPU[id]);
				for (size_t i = 0; i < GPU.size(); ++i) {

					// if (omp_get_num_threads() != (int)GPU.size())
					//     throw std::invalid_argument("OMP didn't create enough threads");
					// assert(omp_get_num_threads() == (int)GPU.size());
					// GPU[i].rescaleMGPU();
					if ((int32_t)i != id)
						GPU[i].doubleRescaleMGPU(poly.GPU[i]);
				}
			} else {
#pragma omp parallel num_threads(GPU.size())
				{
					int i = omp_get_thread_num();

					if (omp_get_num_threads() != (int)GPU.size())
						throw std::invalid_argument("OMP didn't create enough threads");
					assert(omp_get_num_threads() == (int)GPU.size());
					assert(static_cast<size_t>(i) < GPU.size());
					// GPU[i].rescaleMGPU();
					GPU[i].doubleRescaleMGPU(poly.GPU[i]);
				}
			}
			level -= 1;
			poly.level -= 1;
		}
	}
}

void RNSPoly::multPt(const RNSPoly& p, bool rescale) {
	if (rescale) {
		if (GPU.size() == 1) {
			for (size_t i = 0; i < GPU.size(); ++i) {
				GPU.at(i).multPt(p.GPU.at(i));
			}
			--level;
		} else {
#pragma omp parallel for num_threads(GPU.size())
			for (size_t i = 0; i < GPU.size(); ++i) {
				assert(omp_get_num_threads() == (int)GPU.size());
				GPU.at(i).multElement(p.GPU.at(i));
				GPU.at(i).rescaleMGPU();
			}
			--level;
		}
	} else {
#pragma omp parallel for num_threads(cc.GPUid.size())
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			assert(omp_get_num_threads() == (int)cc.GPUid.size());
			GPU.at(i).multElement(p.GPU.at(i));
		}
	}
}

template <ALGO algo> void RNSPoly::NTT(int batch, bool sync) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).NTT<algo>(batch, sync);
	}
}

#define YY(algo) template void RNSPoly::NTT<algo>(int batch, bool sync);

#include "ntt_types.inc"

#undef YY

template <ALGO algo> void RNSPoly::INTT(int batch, bool sync) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).INTT<algo>(batch, sync);
	}
}

#define YY(algo) template void RNSPoly::INTT<algo>(int batch, bool sync);

#include "ntt_types.inc"

#undef YY

/*
std::array<RNSPoly, 2> RNSPoly::dotKSK(const KeySwitchingKey& ksk) {
	constexpr bool PRINT = false;
	Out(KEYSWITCH, "dotKSK in");

	std::array<RNSPoly, 2> result{RNSPoly(cc, level, true), RNSPoly(cc, level, true)};
	result[0].generateSpecialLimbs(false);
	result[1].generateSpecialLimbs(false);

	if constexpr (PRINT)
		for (auto& i : ksk.b.GPU) {
			for (auto& j : i.DECOMPlimb) {
				for (auto& k : j) {
					SWITCH(k, printThisLimb(1));
				}
			}

			for (auto& j : i.DIGITlimb) {
				for (auto& k : j) {
					SWITCH(k, printThisLimb(1));
				}
			}
		}
	for (size_t i = 0; i < GPU.size(); ++i) {
		dotKSKinto(result[0], ksk.b, level);
		dotKSKinto(result[1], ksk.a, level);
	}

	Out(KEYSWITCH, "dotKSK out");
	return result;
}
*/

void RNSPoly::multElement(const RNSPoly& poly) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).multElement(poly.GPU.at(i));
	}
}

void RNSPoly::multElement(const RNSPoly& poly1, const RNSPoly& poly2) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).multElement(poly1.GPU.at(i), poly2.GPU.at(i));
	}
}

void RNSPoly::mult1AddMult23Add4(const RNSPoly& poly1, const RNSPoly& poly2, const RNSPoly& poly3, const RNSPoly& poly4) {

#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).mult1AddMult23Add4(poly1.GPU.at(i), poly2.GPU.at(i), poly3.GPU.at(i), poly4.GPU.at(i));
	}
}

void RNSPoly::mult1Add2(const RNSPoly& poly1, const RNSPoly& poly2) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).mult1Add2(poly1.GPU.at(i), poly2.GPU.at(i));
	}
}

void RNSPoly::dotKSKinto(RNSPoly& acc, const RNSPoly& ksk, const RNSPoly* limbsrc) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		acc.GPU.at(i).dotKSK(GPU.at(i), ksk.GPU.at(i), false, limbsrc ? &limbsrc->GPU.at(i) : nullptr);
	}
}

void RNSPoly::multModupDotKSK(RNSPoly& c1, const RNSPoly& c1tilde, RNSPoly& c0, const RNSPoly& c0tilde, const KeySwitchingKey& key) {
	assert(GPU.size() == 1 && "multModupDotKSK Multi-GPU not implemented.");
	assert(c1.level <= c1tilde.level);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		assert(level == c1.level);
		assert(level == c1tilde.level);
		assert(level == c0.level);
		assert(level == c0tilde.level);
	}
	generateDecompAndDigit(false);
	c0.generateSpecialLimbs(false, false);
	c1.generateSpecialLimbs(false, false);
	for (size_t i = 0; i < GPU.size(); ++i) {
		GPU.at(i).multModupDotKSK(c1.GPU.at(i), c1tilde.GPU.at(i), c0.GPU.at(i), c0tilde.GPU.at(i), key.a.GPU.at(i), key.b.GPU.at(i));
	}
	c0.SetModUp(true);
	c1.SetModUp(true);
}

void RNSPoly::rotateModupDotKSK(RNSPoly& c0, RNSPoly& c1, const KeySwitchingKey& key) {
	assert(GPU.size() == 1 && "rotateModupDotKSK Multi-GPU not implemented.");
	generateDecompAndDigit(false);
	c0.generateSpecialLimbs(false, false);
	c1.generateSpecialLimbs(false, false);
	for (size_t i = 0; i < GPU.size(); ++i) {
		GPU.at(i).rotateModupDotKSK(c1.GPU.at(i), c0.GPU.at(i), key.a.GPU.at(i), key.b.GPU.at(i));
	}
	c1.SetModUp(true);
	c0.SetModUp(true);
}

template <ALGO algo> void RNSPoly::moddown(bool ntt, bool free, int aux_num) {
	if (!this->isModUp()) {
		std::cout << "RNSPoly calling MOdDown on non-modup polynomial." << std::endl;
	}
	assert(this->isModUp());

	if (cc.GPUid.size() == 1) {
		for (int i = 0; i < (int)GPU.size(); ++i) {
			GPU.at(i).moddown<algo>(cc.getModdownAux(aux_num).GPU.at(i), ntt, free);
		}
	} else {
		RNSPoly& aux = cc.getModdownAux(aux_num);
		bool regular = true;
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			regular = regular && (cc.specialMeta.at(i).size() == cc.splitSpecialMeta.at(i).size());
		}

		std::vector<uint64_t*> bufferSpecial;
		for (uint32_t i = 0; i < GPU.size(); ++i) {
			bufferSpecial.push_back(aux.GPU[i].bufferSPECIAL);
		}

#pragma omp parallel num_threads(GPU.size())
		{
			int i = omp_get_thread_num();

			if (omp_get_num_threads() != (int)GPU.size())
				throw std::invalid_argument("OMP didn't create enough threads");
			assert(omp_get_num_threads() == (int)GPU.size());
			assert(static_cast<size_t>(i) < GPU.size());

			if (!regular) {
				GPU.at(i).moddownMGPU(aux.GPU.at(i), ntt, free, bufferSpecial);
			} else {
				GPU.at(i).moddown(aux.GPU.at(i), ntt, free);
			}
		}
		// assert(nullptr == "ModDown Multi-GPU not implemented.");
	}
	this->SetModUp(false);
}

#define YY(algo) template void RNSPoly::moddown<algo>(bool ntt, bool free, int aux_num);

#include "ntt_types.inc"

#undef YY

int RNSPoly::automorph_index_precomp(const int idx) const {
	return modpow(5, 2 * cc.N - idx, cc.N * 2);
}

void RNSPoly::automorph(const int idx, const int br, RNSPoly* src) {
	int k = automorph_index_precomp(idx);
	// int k2 = modpow(5, idx, cc.N * 2);

	// std::cout << k << " " << k2 << std::endl;
	if (src && src->isModUp() && !this->isModUp()) {
		generateSpecialLimbs(false, false);
	}
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).automorph(k, br, src ? &src->GPU.at(i) : nullptr, src ? src->isModUp() : this->isModUp());
	}
	if (src)
		this->SetModUp(src->isModUp());
}

RNSPoly& RNSPoly::dotKSKInPlace(const KeySwitchingKey& ksk, RNSPoly* limb_src) {
	constexpr bool PRINT = false;
	Out(KEYSWITCH, "dotKSK in");

	if (cc.GPUid.size() == 1) {
		if (limb_src) {
			std::cerr << "RNSPoly::dotKSKInPlace: limb_src: parameter ignored, fix this" << std::endl;
		}
		// RNSPoly result{RNSPoly(cc, level, true)};
		cc.getKeySwitchAux2().setLevel(level);
		cc.getKeySwitchAux2().generateSpecialLimbs(false, false);
		generateSpecialLimbs(false, false);
		if constexpr (PRINT)
			for (auto& i : ksk.b.GPU) {
				for (auto& j : i.DECOMPlimb) {
					for (auto& k : j) {
						SWITCH(k, printThisLimb(1));
					}
				}

				for (auto& j : i.DIGITlimb) {
					for (auto& k : j) {
						SWITCH(k, printThisLimb(1));
					}
				}
			}
		// dotKSKinto(cc.getKeySwitchAux2(), ksk.b, level);
		// dotKSKInPlace(ksk.a, level);

		this->dotKSKfused(cc.getKeySwitchAux2(), *this, ksk.a, ksk.b, limb_src ? limb_src : this);
	} else {

		RNSPoly& aux = cc.getKeySwitchAux2();
		aux.setLevel(level);
		aux.generateSpecialLimbs(false, false);
		generateSpecialLimbs(false, false);
		this->dotKSKfused(aux, *this, ksk.a, ksk.b, limb_src ? limb_src : this);
	}

	this->SetModUp(true);
	cc.getKeySwitchAux2().SetModUp(true);
	Out(KEYSWITCH, "dotKSK out");
	return cc.getKeySwitchAux2();
}

/*
void RNSPoly::dotKSKInPlace(const RNSPoly& ksk_b, int level) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).dotKSK(GPU.at(i), ksk_b.GPU.at(i), level, true);
	}
}
*/

void RNSPoly::setLevel(const int level) {
	assert(level >= -1 && level <= cc.L);
	this->level = level;
}

void RNSPoly::modupInto(RNSPoly& poly) {
	assert(level == poly.level);
	auto& aux = cc.getKeySwitchAux2();
	aux.setLevel(level);

	if (GPU.size() > 1 || true) {
		poly.copy(*this);
		poly.modup();
	} else {
#pragma omp parallel for num_threads(cc.GPUid.size())
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			assert(omp_get_num_threads() == (int)cc.GPUid.size());
			GPU.at(i).modupInto(poly.GPU.at(i), aux.GPU.at(i));
		}
	}
}

RNSPoly& RNSPoly::dotKSKInPlaceFrom(RNSPoly& poly, const KeySwitchingKey& ksk, const RNSPoly* limbsrc) {
	constexpr bool PRINT = false;
	Out(KEYSWITCH, "dotKSK in");

	assert(level == poly.level);
	cc.getKeySwitchAux2().setLevel(level);
	cc.getKeySwitchAux2().generateSpecialLimbs(false, false);
	generateSpecialLimbs(false, false);
	if constexpr (PRINT)
		for (auto& i : ksk.b.GPU) {
			for (auto& j : i.DECOMPlimb) {
				for (auto& k : j) {
					SWITCH(k, printThisLimb(1));
				}
			}
			for (auto& j : i.DIGITlimb) {
				for (auto& k : j) {
					SWITCH(k, printThisLimb(1));
				}
			}
		}
	// poly.dotKSKinto(cc.getKeySwitchAux2(), ksk.b, level, limbsrc ? limbsrc : this);
	// poly.dotKSKinto(*this, ksk.a, level, limbsrc ? limbsrc : this);

	this->dotKSKfused(cc.getKeySwitchAux2(), poly, ksk.a, ksk.b, limbsrc ? limbsrc : this);

	cc.getKeySwitchAux2().SetModUp(true);
	this->SetModUp(true);
	Out(KEYSWITCH, "dotKSK out");
	return cc.getKeySwitchAux2();
}

void RNSPoly::multScalar(std::vector<uint64_t>& vector1) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[i].multScalar(vector1);
	}
}

void RNSPoly::add(const RNSPoly& a, const RNSPoly& b) {
	assert(level <= a.level);
	assert(level <= b.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).add(a.GPU.at(i), b.GPU.at(i), a.isModUp(), b.isModUp());
	}

	this->SetModUp(a.isModUp() || b.isModUp());
}

void RNSPoly::squareElement(const RNSPoly& poly) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).squareElement(poly.GPU.at(i));
	}
}

void RNSPoly::binomialSquareFold(RNSPoly& c0_res, const RNSPoly& c2_key_switched_0, const RNSPoly& c2_key_switched_1) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).binomialSquareFold(c0_res.GPU.at(i), c2_key_switched_0.GPU.at(i), c2_key_switched_1.GPU.at(i));
	}
}

void RNSPoly::addScalar(std::vector<uint64_t>& vector1) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[i].addScalar(vector1);
	}
}

void RNSPoly::subScalar(std::vector<uint64_t>& vector1) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[i].subScalar(vector1);
	}
}

void RNSPoly::copy(const RNSPoly& poly) {
	// std::cout << "Copy level: " << poly.level << std::endl;
	this->dropToLevel(poly.level);
	this->grow(poly.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).copyLimb(poly.GPU.at(i));
		if (poly.isModUp())
			GPU.at(i).copySpecialLimb(poly.GPU.at(i));
	}
	this->SetModUp(poly.isModUp());
}

/** Copy contents without extra checks or resizing */
void RNSPoly::copyShallow(const RNSPoly& poly) {
	this->level = poly.level;
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).copyLimb(poly.GPU.at(i));
	}
}

void RNSPoly::dropToLevel(int level) {
	if (0 && GPU.at(0).bufferLIMB == nullptr) {
		for (auto& g : GPU) {
			cudaSetDevice(g.device);
			int limbSize = g.getLimbSize(level);
			while ((int)g.limb.size() > limbSize) {
				g.dropLimb();
			}
		}
	}
	if (this->level > level)
		this->level = level;
}

void RNSPoly::addMult(const RNSPoly& poly, const RNSPoly& poly1) {
	assert(level <= poly1.level && level <= poly.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).addMult(poly.GPU.at(i), poly1.GPU.at(i));
	}
}

void RNSPoly::load(const std::vector<std::vector<uint64_t>>& data, const std::vector<uint64_t>& moduli) {
	int limbsize  = 0;
	int Slimbsize = 0;
	for (int i = 0; i < (int)data.size(); ++i) {
		if (i <= cc.L && moduli[i] == cc.prime.at(i).p) {
			limbsize++;
		} else {
			Slimbsize++;
		}
	}
	// std::cout << "Load " << limbsize << " limbs" << std::endl;

	assert(limbsize - 1 <= cc.L);
	if (level < limbsize - 1)
		grow(limbsize - 1, false);
	if (level > limbsize - 1)
		dropToLevel(limbsize - 1);
	assert(level == limbsize - 1);
	for (int i = 0; i < limbsize; ++i) {
		// std::cout << "Load limb" << i << " into gpu " << cc.limbGPUid[i].x << std::endl;
		assert(moduli[i] == cc.prime.at(i).p);
		cudaSetDevice(GPU[cc.limbGPUid[i].x].device);
		SWITCH(GPU[cc.limbGPUid[i].x].limb[cc.limbGPUid[i].y], load_convert(data[i]));
	}

	if ((int)data.size() > limbsize)
		generateSpecialLimbs(false, false);
	for (int i = limbsize; i < (int)data.size(); ++i) {
		for (auto& j : GPU) {
			cudaSetDevice(j.device);
			assert(moduli[i] == cc.specialPrime.at(i - limbsize).p);
			SWITCH(j.SPECIALlimb[i - limbsize], load_convert(data[i]));
		}
	}
}

void RNSPoly::loadConstant(const std::vector<std::vector<uint64_t>>& data, const std::vector<uint64_t>& moduli) {
	int limbsize  = 0;
	int Slimbsize = 0;
	for (int i = 0; i < (int)data.size(); ++i) {
		if (i <= cc.L && moduli[i] == cc.prime.at(i).p) {
			limbsize++;
		} else {
			Slimbsize++;
		}
	}

	assert(limbsize <= cc.L + 1);
	if (level < limbsize - 1) {
		grow(limbsize - 1, false, true);
	} else {
		dropToLevel(limbsize - 1);
	}
	assert(level == limbsize - 1);
	for (int i = 0; i < limbsize; ++i) {
		assert(moduli[i] == cc.prime.at(i).p);
		cudaSetDevice(GPU[cc.limbGPUid[i].x].device);
		SWITCH(GPU[cc.limbGPUid[i].x].limb[cc.limbGPUid[i].y], load_convert(data[i]));
	}

	if ((int)data.size() > limbsize) {
		generatePartialSpecialLimbs();
		this->SetModUp(true);
	}
	for (size_t i = limbsize; i < data.size(); ++i) {
		for (size_t j = 0; j < GPU.size(); ++j) {
			for (size_t k = 0; k < cc.splitSpecialMeta.at(j).size(); ++k) {
				if (cc.specialPrime.at(cc.splitSpecialMeta.at(j).at(k).id - cc.L - 1).p == moduli[i]) {
					cudaSetDevice(GPU[j].device);
					SWITCH(GPU[j].SPECIALlimb[k], load_convert(data[i]));
				}
			}
		}
	}
}

void RNSPoly::broadcastLimb0() {
	if (cc.GPUid.size() == 1) {
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			GPU.at(i).broadcastLimb0();
		}
	} else {
#pragma omp parallel for num_threads(cc.GPUid.size())
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			assert(omp_get_num_threads() == (int)cc.GPUid.size());
			GPU.at(i).broadcastLimb0_mgpu();
		}
	}
}

void RNSPoly::evalLinearWSum(uint32_t n, std::vector<const RNSPoly*>& vec, std::vector<uint64_t>& elem) {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		std::vector<const LimbPartition*> ps(n);
		for (int j = 0; j < (int)n; ++j) {
			ps[j] = &vec[j]->GPU.at(i);
		}
		GPU.at(i).evalLinearWSum(n, ps, elem);
	}
}

void RNSPoly::squareModupDotKSK(RNSPoly& c0, RNSPoly& c1, const KeySwitchingKey& key) {
	assert(GPU.size() == 1 && "squareModupDotKSK Multi-GPU not implemented.");
	generateDecompAndDigit(false);
	c0.generateSpecialLimbs(false, false);
	c1.generateSpecialLimbs(false, false);
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU.at(i).squareModupDotKSK(c1.GPU.at(i), c0.GPU.at(i), key.a.GPU.at(i), key.b.GPU.at(i));
	}
	c0.SetModUp(true);
	c1.SetModUp(true);
}

void RNSPoly::generatePartialSpecialLimbs() {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t i = 0; i < cc.GPUid.size(); ++i) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[i].generatePartialSpecialLimb();
	}
}

void RNSPoly::dotKSKfused(RNSPoly& out2, const RNSPoly& digitSrc, const RNSPoly& ksk_a, const RNSPoly& ksk_b, const RNSPoly* source) {
	RNSPoly& out1	   = *this;
	const RNSPoly& src = source ? *source : *this;
	if (cc.GPUid.size() == 1) {
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			out1.GPU[i].dotKSKfusedMGPU(out2.GPU[i], digitSrc.GPU[i], ksk_a.GPU[i], ksk_b.GPU[i], src.GPU[i]);
		}
	} else {
#pragma omp parallel for num_threads(cc.GPUid.size())
		for (size_t i = 0; i < cc.GPUid.size(); ++i) {
			assert(omp_get_num_threads() == (int)cc.GPUid.size());
			out1.GPU[i].dotKSKfusedMGPU(out2.GPU[i], digitSrc.GPU[i], ksk_a.GPU[i], ksk_b.GPU[i], src.GPU[i]);
		}
	}
}

void RNSPoly::dotProductPt(RNSPoly& c1_, const std::vector<const RNSPoly*>& c0s_, const std::vector<const RNSPoly*>& c1s_, const std::vector<const RNSPoly*>& pts_, const bool ext) {

	if (ext) {
		generateSpecialLimbs(false, false);
		c1_.generateSpecialLimbs(false, false);
	}
	int n = pts_.size();
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t j = 0; j < cc.GPUid.size(); ++j) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		std::vector<const LimbPartition*> c0s(n, nullptr), c1s(n, nullptr), pts(n, nullptr);
		for (int i = 0; i < n; ++i) {
			c0s[i] = &(c0s_[i]->GPU[j]);
			c1s[i] = &(c1s_[i]->GPU[j]);
			pts[i] = &(pts_[i]->GPU[j]);
		}
		GPU[j].dotProductPt(c1_.GPU[j], c0s, c1s, pts, ext);
	}
	c1_.SetModUp(ext);
	this->SetModUp(ext);
}

RNSPoly& RNSPoly::dotProduct(RNSPoly& c1,
  const RNSPoly& kskb,
  const RNSPoly& kska,
  const std::vector<const RNSPoly*>& c0in,
  const std::vector<const RNSPoly*>& c1in,
  const std::vector<const RNSPoly*>& d0in,
  const std::vector<const RNSPoly*>& d1in,
  bool ext_in,
  bool ext_out) {

	auto& c2 = cc.getKeySwitchAux();

	if (ext_in) {
		generateSpecialLimbs(false, false);
		c1.generateSpecialLimbs(false, false);
	}
	c2.setLevel(c1.getLevel());

	int n = c0in.size();
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t j = 0; j < cc.GPUid.size(); ++j) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		std::vector<const LimbPartition*> c0s(n, nullptr), c1s(n, nullptr), d0s(n, nullptr), d1s(n, nullptr);
		for (int i = 0; i < n; ++i) {
			c0s[i] = &(c0in[i]->GPU[j]);
			c1s[i] = &(c1in[i]->GPU[j]);
			d0s[i] = &(d0in[i]->GPU[j]);
			d1s[i] = &(d1in[i]->GPU[j]);
		}
		GPU[j].binomialDotProduct(c1.GPU[j], c2.GPU[j], c0s, c1s, d0s, d1s, ext_in);
	}

	SetModUp(ext_in);
	c1.SetModUp(ext_in);
	c2.SetModUp(ext_in);

	return c2;
}

void RNSPoly::hoistedRotationFused(std::vector<int> indexes,
  std::vector<RNSPoly*>& c0,
  std::vector<RNSPoly*>& c1,
  const std::vector<RNSPoly*>& ksk_a,
  const std::vector<RNSPoly*>& ksk_b,
  const RNSPoly& src_c0,
  const RNSPoly& src_c1) {
	uint32_t n = indexes.size();
	for (uint32_t j = 0; j < n; ++j) {
		c0[j]->generateSpecialLimbs(false, false);
		c1[j]->generateSpecialLimbs(false, false);
		indexes[j] = indexes[j] == 2 * cc.N - 1 ? 2 * cc.N - 1 : automorph_index_precomp(indexes[j]);
	}
	//    assert(src_c0.isModUp() == false);

#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t j = 0; j < cc.GPUid.size(); ++j) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		std::vector<LimbPartition*> c0s(n, nullptr), c1s(n, nullptr), ksk_as(n, nullptr), ksk_bs(n, nullptr);
		for (uint32_t i = 0; i < n; ++i) {
			c0s[i]	  = &(c0[i]->GPU[j]);
			c1s[i]	  = &(c1[i]->GPU[j]);
			ksk_as[i] = &(ksk_a[i]->GPU[j]);
			ksk_bs[i] = &(ksk_b[i]->GPU[j]);
		}
		GPU[j].fusedHoistRotate(n, indexes, c0s, c1s, ksk_as, ksk_bs, src_c0.GPU[j], src_c1.GPU[j], src_c0.isModUp());
	}

	for (uint32_t j = 0; j < n; ++j) {
		c0[j]->SetModUp(true);
		c1[j]->SetModUp(true);
	}
}

/*
void RNSPoly::generateGatherLimbs() {
#pragma omp parallel for num_threads(cc.GPUid.size())
	for (size_t j = 0; j < cc.GPUid.size(); ++j) {
		assert(omp_get_num_threads() == (int)cc.GPUid.size());
		GPU[j].generateGatherLimb(false);
	}
}
*/

RNSPoly& RNSPoly::modup_ksk_moddown_mgpu(const KeySwitchingKey& key, const bool moddown) {
	RNSPoly& aux = cc.getKeySwitchAux2();
	aux.setLevel(level);
	RNSPoly& aux_limbs1 = cc.getModdownAux(0);
	RNSPoly& aux_limbs2 = cc.getModdownAux(1);

	static std::vector<std::vector<std::vector<std::pair<uint64_t, TimelineSemaphore*>>>> signals;
	/*
		if (signals.size() < 2 * (cc.dnum + 2) || (signals.size() > 0 && signals[0].size() < cc.GPUid.size())) {
			signals.resize(2 * (cc.dnum + 2));
			for (int k = 0; k < cc.GPUid.size(); ++k) {
				cudaSetDevice(cc.GPUid[k]);
				cudaDeviceSynchronize();
			}
			for (auto& i : signals) {
				i.resize(std::max(i.size(), cc.GPUid.size()));
				parallel_for(0, cc.GPUid.size(), 1, [&](int j) {
					// for (int j = 0; j < cc.GPUid.size(); ++j) {
					i[j].resize(std::max(i.size(), cc.GPUid.size()));
					// cudaSetDevice(cc.GPUid[j]);
					// cudaDeviceSynchronize();
					for (int k = 0; k < cc.GPUid.size(); ++k) {
						if (i[j][k].second != nullptr) {

							cudaFreeHost(i[j][k].second);
							i[j][k].second = nullptr;
							CudaCheckErrorModNoSync;
							// cudaFree(i[j][k].second);
						}
						CudaCheckErrorModNoSync;
						cudaHostAlloc(&i[j][k].second, sizeof(TimelineSemaphore), cudaHostAllocPortable);
						CudaCheckErrorModNoSync;
						// cudaMalloc(&i[j][k].second, sizeof(TimelineSemaphore));
						i[j][k].second->value = 0;
						// cudaMemset(i[j][k].second, 0, 128);
						CudaCheckErrorModNoSync;
						i[j][k].first = 1;
					}
				});
			}
		}
	*/
	std::vector<std::atomic_uint64_t> thread_stop_buffer(cc.GPUid.size() * 8);
	std::vector<std::atomic_uint64_t*> thread_stop(cc.GPUid.size(), nullptr);
	for (uint32_t k = 0; k < cc.GPUid.size(); ++k) {
		thread_stop[k]			  = &thread_stop_buffer[8 * k];
		thread_stop_buffer[8 * k] = 0;
	}

	if (1 || moddown) {

		std::vector<uint64_t*> bufferGather;
		std::vector<uint64_t*> bufferSpecial_c0;
		std::vector<uint64_t*> bufferSpecial_c1;
		std::vector<Stream*> external_s;
		std::vector<Stream*> external_s0;
		for (uint32_t i = 0; i < GPU.size(); ++i) {
			bufferGather.push_back(GPU[i].bufferGATHER);
			bufferSpecial_c0.push_back(aux_limbs2.GPU[i].bufferSPECIAL);
			bufferSpecial_c1.push_back(aux_limbs1.GPU[i].bufferSPECIAL);
			external_s.push_back(&GPU[i].s);
			external_s0.push_back(&aux.GPU[i].s);
		}

		if (!MEMCPY_PEER || !GRAPH_CAPTURE) {
#pragma omp parallel num_threads(GPU.size())
			{
				int j = omp_get_thread_num();

				if (omp_get_num_threads() != (int)GPU.size())
					throw std::invalid_argument("OMP didn't create enough threads");
				assert(omp_get_num_threads() == (int)GPU.size());
				assert(static_cast<size_t>(j) < GPU.size());
				GPU[j].modup_ksk_moddown_mgpu(
				  aux.GPU[j], key.a.GPU[j], key.b.GPU[j], aux_limbs1.GPU[j], aux_limbs2.GPU[j], moddown, bufferGather, bufferSpecial_c0, bufferSpecial_c1, external_s, signals, thread_stop, external_s0);
			}
		} else {

			// #pragma omp parallel num_threads(GPU.size())
			//{
			parallel_for(0, cc.GPUid.size(), 1, [&](int j) {
				// for (int j = 0; j < cc.GPUid.size(); ++j) {
				// int j = omp_get_thread_num();
				// if (omp_get_num_threads() != (int)GPU.size())
				//     throw std::invalid_argument("OMP didn't create enough threads");
				// assert(omp_get_num_threads() == (int)GPU.size());
				// assert(j < GPU.size());
				GPU[j].modup_ksk_moddown_mgpu(
				  aux.GPU[j], key.a.GPU[j], key.b.GPU[j], aux_limbs1.GPU[j], aux_limbs2.GPU[j], moddown, bufferGather, bufferSpecial_c0, bufferSpecial_c1, external_s, signals, thread_stop, external_s0);
			});
		}

		if (MEMCPY_PEER) {
			for (auto& i : signals) {
				for (uint32_t j = 0; j < cc.GPUid.size(); ++j) {
					cudaSetDevice(cc.GPUid[j]);

					for (uint32_t k = 0; k < cc.GPUid.size(); ++k) {
						i[j][k].first = i[j][k].first + 4;
					}
				}
			}
		}
		this->SetModUp(!moddown);
		aux.SetModUp(!moddown);
		return aux;
	} else {
		this->modup();
		RNSPoly& result = this->dotKSKInPlace(key, nullptr);
		if (moddown) {
			result.moddown(true, false);
			this->moddown(true, false);
		}
		return result;
	}
}

} // namespace FIDESlib::CKKS