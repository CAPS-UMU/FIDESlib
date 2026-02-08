//
// Created by carlosad on 24/04/24.
//

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Plaintext.cuh"
#include <omp.h>
#if defined(__clang__)
#include <experimental/source_location>
using sc				  = std::experimental::source_location;
constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc				  = std::source_location;
constexpr int PREFIX_SIZE = 23;
#endif

namespace FIDESlib::CKKS {

bool hoistRotateFused		  = true;
constexpr bool RESCALE_DOUBLE = true;

enum OPS {
	NOP,
	ADD,
	ADDPT,
	MULT,
	MULTPT,
	RESCALE,
	ROTATE,
	COPY,
	SQUARE,
	ADDSCALAR,
	MULTSCALAR,
	ADDMULTPT,
	ADDMULTPTINS,
	WSUM,
	WSUMINPUTS,
	CONJUGATE,
	HOISTEDROTATE,
	HOISTEDROTATEOUTS,
};

constexpr std::array<const char*, 18> opstr{ "                   Noop: ",
	"                   HAdd: ",
	"                  AddPt: ",
	"                   Mult: ",
	"                 MultPt: ", // 5
	"                Rescale: ",
	"                 Rotate: ",
	"                   Copy: ",
	"                 Square: ",
	"              ScalarAdd: ", // 10
	"             ScalarMult: ",
	"              AddMultPt: ",
	"     AddMultPt (inputs): ",
	"                   WSum: ",
	"          WSum (inputs): ", // 15
	"              Conjugate: ",
	"          HoistedRotate: ",
	"HoistedRotate (outputs): " };

std::map<OPS, int> op_count;

Ciphertext::Ciphertext(Ciphertext&& ct_moved) noexcept
: my_range(std::move(ct_moved.my_range)), keyID(std::move(ct_moved.keyID)), cc_(ct_moved.cc_), cc(*cc_), c0(std::move(ct_moved.c0)), c1(std::move(ct_moved.c1)),
  NoiseFactor(ct_moved.NoiseFactor), NoiseLevel(ct_moved.NoiseLevel), slots(ct_moved.slots) {
}

Ciphertext::Ciphertext(Context& cc)
: my_range(loc, LIFETIME), cc_((assert(cc != nullptr), CudaNvtxStart(std::string{ sc::current().function_name() }.substr()), cc)), cc(*cc_),
  c0(cc->getAuxilarPoly()), c1(cc->getAuxilarPoly()) {
	c0.dropToLevel(-1);
	c1.dropToLevel(-1);
	c0.SetModUp(false);
	c1.SetModUp(false);
	CudaNvtxStop();
}

Ciphertext::Ciphertext(Context& cc, const RawCipherText& rawct) : Ciphertext(cc) {
	this->load(rawct);
}

Ciphertext::~Ciphertext() {
	if (!c1.GPU.empty())
		cc.returnAuxilarPoly(std::move(c1));
	if (!c0.GPU.empty())
		cc.returnAuxilarPoly(std::move(c0));
}

void Ciphertext::copyMetadata(const Ciphertext& a) {
	assert(this->getLevel() == a.getLevel());
	this->slots		  = a.slots;
	this->keyID		  = a.keyID;
	this->NoiseLevel  = a.NoiseLevel;
	this->NoiseFactor = a.NoiseFactor;
}

void Ciphertext::addMetadata(const Ciphertext& a, const Ciphertext& b) {
	assert(this->getLevel() == a.getLevel());
	if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
		assert(a.getLevel() == b.getLevel());
		// assert(abs(a.NoiseFactor - b.NoiseFactor) < a.NoiseFactor / 1e9);
	}
	this->slots = std::max(a.slots, b.slots);
	assert(a.keyID == b.keyID);
	this->keyID = a.keyID;
	assert(a.NoiseLevel == b.NoiseLevel);
	this->NoiseLevel = a.NoiseLevel;

	this->NoiseFactor = a.NoiseFactor;
}

void Ciphertext::addMetadata(const Ciphertext& a, const Plaintext& b) {
	assert(this->getLevel() == a.getLevel());
	if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
		assert(a.getLevel() == b.c0.getLevel());
		// assert(a.NoiseFactor == b.NoiseFactor);
	}
	this->slots = std::max(a.slots, b.slots);
	this->keyID = a.keyID;
	assert(a.NoiseLevel == b.NoiseLevel);
	this->NoiseLevel  = a.NoiseLevel;
	this->NoiseFactor = a.NoiseFactor;
}

void Ciphertext::multMetadata(const Ciphertext& a, const Ciphertext& b) {
	assert(this->getLevel() == a.getLevel());
	if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
		assert(a.getLevel() == b.getLevel());
		// assert(a.NoiseFactor == b.NoiseFactor);
	}
	this->slots = std::max(a.slots, b.slots);
	assert(a.keyID == b.keyID);
	this->keyID = a.keyID;
	assert(a.NoiseLevel == b.NoiseLevel);
	this->NoiseLevel  = a.NoiseLevel + b.NoiseLevel;
	this->NoiseFactor = a.NoiseFactor * b.NoiseFactor;
}

void Ciphertext::multMetadata(const Ciphertext& a, const Plaintext& b) {
	assert(this->getLevel() == a.getLevel());
	if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
		assert(a.getLevel() == b.c0.getLevel());
		// assert(a.NoiseFactor == b.NoiseFactor);
	}
	this->slots = std::max(a.slots, b.slots);
	this->keyID = a.keyID;
	assert(a.NoiseLevel == b.NoiseLevel);
	this->NoiseLevel = a.NoiseLevel + b.NoiseLevel;

	this->NoiseFactor = a.NoiseFactor * b.NoiseFactor;
}

int Ciphertext::normalyzeIndex(int index) const {
	index = index % slots;
	if (index < 0)
		index += slots;
	if (index > slots / 2)
		index += cc.N / 2 - slots;
	return index;
}

void Ciphertext::add(const Ciphertext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	assert(keyID == b.keyID);
	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		if (!adjustForAddOrSub(b)) {
			Ciphertext b_(cc_);
			b_.copy(b);
			if (b_.adjustForAddOrSub(*this))
				add(b_);
			else
				assert(false);
			return;
		}
	}

	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		assert(this->getLevel() == b.getLevel());
	} else if (getLevel() > b.getLevel()) {
		c0.dropToLevel(b.getLevel());
		c1.dropToLevel(b.getLevel());
	}
	op_count[OPS::ADD]++;

	c0.add(b.c0);
	c1.add(b.c1);

	this->addMetadata(*this, b);
}

void Ciphertext::sub(const Ciphertext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	assert(keyID == b.keyID);
	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		if (!adjustForAddOrSub(b)) {
			Ciphertext b_(cc_);
			b_.copy(b);
			if (b_.adjustForAddOrSub(*this))
				sub(b_);
			else
				assert(false);
			return;
		}
	}

	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		assert(this->getLevel() == b.getLevel());
	} else if (getLevel() > b.getLevel()) {
		c0.dropToLevel(b.getLevel());
		c1.dropToLevel(b.getLevel());
	}
	op_count[OPS::ADD]++;

	c0.sub(b.c0);
	c1.sub(b.c1);

	this->addMetadata(*this, b);
}

void Ciphertext::addPt(const Plaintext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {
		if (b.NoiseLevel == 1 && NoiseLevel == 2 && b.c0.getLevel() == getLevel() - 1) {
			this->rescale();
		}

		if (b.c0.getLevel() != this->getLevel()) {
			Plaintext b_(cc_);
			if (!b_.adjustPlaintextToCiphertext(b, *this)) {
				assert(false);
			} else {
				addPt(b_);
			}
			return;
		}
	}
	assert(NoiseLevel == b.NoiseLevel);
	op_count[OPS::ADDPT]++;

	c0.add(b.c0);

	this->addMetadata(*this, b);
}

void Ciphertext::subPt(const Plaintext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {
		if (b.NoiseLevel == 1 && NoiseLevel == 2 && b.c0.getLevel() == getLevel() - 1) {
			this->rescale();
		}

		if (b.c0.getLevel() != this->getLevel()) {
			Plaintext b_(cc_);
			if (!b_.adjustPlaintextToCiphertext(b, *this)) {
				assert(false);
			} else {
				addPt(b_);
			}
			return;
		}
	}
	assert(NoiseLevel == b.NoiseLevel);
	op_count[OPS::ADDPT]++;

	c0.sub(b.c0);

	this->addMetadata(*this, b);
}

void Ciphertext::load(const RawCipherText& rawct) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	keyID = rawct.keyid;
	c0.load(rawct.sub_0, rawct.moduli);
	c1.load(rawct.sub_1, rawct.moduli);

	NoiseLevel	= rawct.NoiseLevel;
	NoiseFactor = rawct.Noise;
	slots		= rawct.slots;
}

void Ciphertext::store(RawCipherText& rawct) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());

	CKKS::SetCurrentContext(cc_);
	cudaDeviceSynchronize();
	rawct.numRes = c0.getLevel() + 1;
	rawct.sub_0.resize(rawct.numRes);
	rawct.sub_1.resize(rawct.numRes);
	c0.store(rawct.sub_0);
	c1.store(rawct.sub_1);
	rawct.N = cc.N;
	c0.sync();
	c1.sync();

	rawct.NoiseLevel = NoiseLevel;
	rawct.Noise		 = NoiseFactor;
	rawct.keyid		 = keyID;
	rawct.slots		 = slots; // TODO store other interesting metadata
	cudaDeviceSynchronize();
}

void Ciphertext::modDown(bool free) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	c0.moddown(true, false, 0);
	c1.moddown(true, false, 1);
	if (free) {
		c0.freeSpecialLimbs();
		c1.freeSpecialLimbs();
	}
}

void Ciphertext::modUp() {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	// c0.modup();
	c1.modup();
}

void Ciphertext::multPt(const Plaintext& b, bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	constexpr bool PRINT = false;
	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		if constexpr (PRINT)
			std::cout << "multPt: Rescale input ciphertext" << std::endl;
		if (NoiseLevel == 2)
			this->rescale();
	}

	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		if (b.c0.getLevel() != this->getLevel() || b.NoiseLevel == 2 /*!hasSameScalingFactor(b)*/) {
			Plaintext b_(cc_);
			if constexpr (PRINT)
				std::cout << "multPt: adjust input plaintext" << std::endl;

			// if (!this->adju)
			if (!b_.adjustPlaintextToCiphertext(b, *this)) {
				if constexpr (PRINT)
					std::cout << "multPt: FAILED!" << std::endl;
				assert(false);
			} else {
				if (NoiseLevel == 2)
					this->rescale();
				if (b_.NoiseLevel == 2) {
					if constexpr (PRINT)
						std::cout << "multPt: Rescale input plaintext" << std::endl;
					b_.rescale();
				}
				multPt(b_, rescale);
			}
			return;
		}
	}

	assert(NoiseLevel < 2);
	assert(b.NoiseLevel < 2);
	op_count[OPS::MULTPT]++;

	c0.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL);
	c1.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL);

	this->multMetadata(*this, b);
	if (rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
		NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
		NoiseLevel -= 1;
	}
}

void Ciphertext::rescale() {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	// assert(this->NoiseLevel == 2);
	if (cc.rescaleTechnique != FIXEDMANUAL) {
		// this wouldn't do anything in OpenFHE
	}
	op_count[OPS::RESCALE]++;

	if constexpr (RESCALE_DOUBLE) {
		c0.rescaleDouble(c1);
	} else {
		c0.rescale();
		c1.rescale();
	}

	// Manage metadata
	NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
	NoiseLevel -= 1;
}

/** "in" needs to have Digit and Gather limbs pre-generated */
RNSPoly& MGPUkeySwitchCore(RNSPoly& in, const KeySwitchingKey& kskEval, const bool moddown) {
	{
		RNSPoly& aux = in.modup_ksk_moddown_mgpu(kskEval, moddown);

		return aux;
	}
}

void Ciphertext::mult(const Ciphertext& b, bool rescale, const bool moddown) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	assert(keyID == b.keyID);
	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		if (!adjustForMult(b)) {
			Ciphertext b_(cc_);
			b_.copy(b);
			if (b_.adjustForMult(*this))
				mult(b_, rescale, moddown);
			else
				assert(false);
			return;
		}
	}
	assert(NoiseLevel == 1);
	assert(NoiseLevel == b.NoiseLevel);
	op_count[OPS::MULT]++;
	/*
	if (getLevel() > b.getLevel()) {
		this->c0.dropToLevel(b.getLevel());
		this->c1.dropToLevel(b.getLevel());
	}
	*/
	// assert(c0.getLevel() <= b.c0.getLevel());
	// assert(c1.getLevel() <= b.c1.getLevel());
	Out(KEYSWITCH, " start ");
	assert(this->NoiseLevel == 1);
	assert(b.NoiseLevel == 1);

	KeySwitchingKey& kskEval = cc.GetEvalKey(keyID);

	if (0 && cc.GPUid.size() == 1) {
		if constexpr (0) {
			constexpr bool PRINT = true;
			bool SELECT			 = true;

			cc.getKeySwitchAux().setLevel(c1.getLevel());
			cc.getKeySwitchAux().multElement(c1, b.c1);

			if constexpr (PRINT) {
				if (SELECT) {
					cudaDeviceSynchronize();
					std::cout << "GPU: " << 0 << "Input data ";
					for (size_t j = 0; j < cc.getKeySwitchAux().GPU[0].limb.size(); ++j) {
						std::cout << cc.getKeySwitchAux().GPU[0].meta[j].id;
						SWITCH(cc.getKeySwitchAux().GPU[0].limb[j], printThisLimb(2));
					}
					std::cout << std::endl;
					cudaDeviceSynchronize();
				}
			}

			cc.getKeySwitchAux().modup();

			if constexpr (PRINT) {
				if (SELECT) {
					cudaDeviceSynchronize();
					std::cout << "GPU: " << 0 << "Out ModUp after NTT ";
					for (size_t j = 0; j < cc.getKeySwitchAux().GPU[0].DIGITlimb.size(); ++j) {
						for (size_t i = 0; i < cc.getKeySwitchAux().GPU[0].DIGITlimb[j].size(); ++i) {
							std::cout << cc.getKeySwitchAux().GPU[0].DIGITmeta[j][i].id;
							SWITCH(cc.getKeySwitchAux().GPU[0].DIGITlimb[j][i], printThisLimb(2));
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;
					cudaDeviceSynchronize();
				}
			}

			auto& aux0 = cc.getKeySwitchAux().dotKSKInPlace(kskEval, nullptr);

			if constexpr (PRINT) {
				if (SELECT) {
					cudaDeviceSynchronize();
					std::cout << "GPU out KSK specials: ";
					for (const auto& j : { &aux0, &cc.getKeySwitchAux() }) {
						for (const auto& k : j->GPU) {
							for (auto& i : k.SPECIALlimb) {
								SWITCH(i, printThisLimb(2));
							}
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;
					cudaDeviceSynchronize();
				}
			}
			if constexpr (PRINT) {
				if (SELECT) {
					cudaDeviceSynchronize();
					std::cout << "GPU out KSK limbs: ";
					for (const auto& j : { &aux0, &cc.getKeySwitchAux() }) {
						for (const auto& k : j->GPU) {
							for (auto& i : k.limb) {
								SWITCH(i, printThisLimb(2));
							}
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;
					cudaDeviceSynchronize();
				}
			}

			cc.getKeySwitchAux().moddown(true, false, 0);
			aux0.moddown(true, false, 1);
			c1.mult1AddMult23Add4(b.c0, c0, b.c1, cc.getKeySwitchAux());
			c0.mult1Add2(b.c0, aux0);
			// c1.binomialSquareFold(c0, aux0, cc.getKeySwitchAux());
			if (rescale) {
				this->rescale();
			}
			/*
			cudaDeviceSynchronize();
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			cudaDeviceSynchronize();
			cc.getKeySwitchAux().multElement(c1, b.c1);
			cudaDeviceSynchronize();
			cc.getKeySwitchAux().modup();
			cudaDeviceSynchronize();
			auto& aux0 = cc.getKeySwitchAux().dotKSKInPlace(kskEval, c0.getLevel(), nullptr);
			cudaDeviceSynchronize();
			cc.getKeySwitchAux().moddown(true, false);
			cudaDeviceSynchronize();
			c1.mult1AddMult23Add4(b.c0, c0, b.c1, cc.getKeySwitchAux());  // Read 4 first for better cache locality.
			cudaDeviceSynchronize();
			aux0.moddown(true, false);
			cudaDeviceSynchronize();
			c0.mult1Add2(b.c0, aux0);
			cudaDeviceSynchronize();

			if (rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
				this->rescale();
			}
			 */
		} else if (false) {
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			cc.getKeySwitchAux().multModupDotKSK(c1, b.c1, c0, b.c0, kskEval);
			{ // TODO MAD Figure 4: add before fused ModDown+Rescale
			}
			if (moddown)
				c1.moddown(true, false);
			if (moddown)
				c0.moddown(true, false);

			// Manage metadata
			this->multMetadata(*this, b);

			if (moddown && rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL)
				this->rescale();
		} else if (false) {
			RNSPoly& in = cc.getKeySwitchAux();
			in.setLevel(c1.getLevel());
			in.multElement(c1, b.c1);

			RNSPoly& aux = MGPUkeySwitchCore(in, kskEval, moddown);

			if (moddown) {
				c1.mult1AddMult23Add4(b.c0, c0, b.c1, in); // Read 4 first for better cache locality.
				c0.mult1Add2(b.c0, aux);
			} else {
				c1.multNoModdownEnd(c0, b.c0, b.c1, in, aux);
			}

			// Manage metadata
			this->multMetadata(*this, b);
			if (moddown && rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
				this->rescale();
			}
		} else {
			RNSPoly& in = cc.getKeySwitchAux();
			in.setLevel(c1.getLevel());

			c0.binomialMult(c1, in, b.c0, b.c1, moddown, &b == this);

			RNSPoly& aux = MGPUkeySwitchCore(in, kskEval, moddown);

			c0.add(aux);
			c1.add(in);

			// Manage metadata
			this->multMetadata(*this, b);
			if (moddown && rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
				this->rescale();
			}
		}
	} else {
		constexpr bool PRINT = false;

		if constexpr (PRINT)
			std::cout << "Init mult" << std::endl;
		RNSPoly& in = cc.getKeySwitchAux();
		in.setLevel(c1.getLevel());
		c0.binomialMult(c1, in, b.c0, b.c1, moddown, &b == this);

		RNSPoly& aux = MGPUkeySwitchCore(in, kskEval, moddown);
		c0.add(aux);
		c1.add(in);

		// Manage metadata
		this->multMetadata(*this, b);
		if (moddown && rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
			this->rescale();
		}
		if constexpr (PRINT)
			std::cout << "End mult" << std::endl;
		if constexpr (PRINT)
			CudaCheckErrorMod;
	}

	Out(KEYSWITCH, " finish ");
}

void Ciphertext::square(bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	Out(KEYSWITCH, " start ");

	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {
		if (NoiseLevel == 2)
			this->rescale();
	}
	assert(this->NoiseLevel == 1);
	op_count[OPS::SQUARE]++;

	KeySwitchingKey& kskEval = cc.GetEvalKey(keyID);

	if (cc.GPUid.size() == 1) {
		if constexpr (0) {
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			cc.getKeySwitchAux().squareElement(c1);
			cc.getKeySwitchAux().modup();
			auto& aux0 = cc.getKeySwitchAux().dotKSKInPlace(kskEval, nullptr);
			cc.getKeySwitchAux().moddown(true, false);
			aux0.moddown(true, false);
			// c1.mult1AddMult23Add4(c0, c0, c1, cc.getKeySwitchAux());
			c1.binomialSquareFold(c0, aux0, cc.getKeySwitchAux());
			this->multMetadata(*this, *this);
			if (rescale) {
				this->rescale();
			}
		} else if constexpr (0) {
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			cc.getKeySwitchAux().squareModupDotKSK(c0, c1, kskEval);

			c1.moddown(true, false);

			c0.moddown(true, false);
			this->multMetadata(*this, *this);
			if (rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL)
				this->rescale();

		} else {
			this->mult(*this, rescale);
		}
	} else {
		if (0) {
			RNSPoly& in = cc.getKeySwitchAux();
			in.setLevel(c1.getLevel());
			in.squareElement(c1);

			RNSPoly& aux = MGPUkeySwitchCore(in, kskEval, true);

			c1.binomialSquareFold(c0, aux, in);
			if (rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
				this->rescale();
			}
			this->multMetadata(*this, *this);
		} else {
			this->mult(*this, rescale);
		}
	}
	// Manage metadata
	Out(KEYSWITCH, " finish ");
}

void Ciphertext::multScalarNoPrecheck(const double c, bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::MULTSCALAR]++;

	auto elem = cc.ElemForEvalMult(c0.getLevel(), c);
	c0.multScalar(elem);
	c1.multScalar(elem);

	// Manage metadata
	NoiseLevel += 1;
	NoiseFactor *= cc.param.ScalingFactorReal.at(c0.getLevel());
	if (rescale && cc.rescaleTechnique == FIXEDAUTO) {
		this->rescale();
	}
}

void Ciphertext::multScalar(const double c, bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {
		if (NoiseLevel == 2)
			this->rescale();
	}
	assert(this->NoiseLevel == 1);
	multScalarNoPrecheck(c, rescale && cc.rescaleTechnique == FIXEDMANUAL);
}

void Ciphertext::addScalar(const double c) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::ADDSCALAR]++;

	auto elem = cc.ElemForEvalAddOrSub(c0.getLevel(), std::abs(c), this->NoiseLevel);

	if (c < 0.0) {
		for (auto i = 0u; i < elem.size(); ++i) {
			elem[i] = cc.prime[i].p - elem[i];
		}
	}
	// if (c >= 0.0) {
	if (c != 0.0)
		c0.addScalar(elem);
	//} else {
	//    c0.subScalar(elem);
	//}
}

void Ciphertext::addScalar(const std::complex<double> c) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    CKKS::SetCurrentContext(cc_);
    op_count[OPS::ADDSCALAR]++;

    auto elem1 = cc.ElemForEvalAddOrSub(c0.getLevel(), std::fabs(c.real()), this->NoiseLevel);
    auto elem2 = cc.ElemForEvalAddOrSub(c0.getLevel(), std::fabs(c.imag()), this->NoiseLevel);

    uint32_t sizeQl = c0.getLevel() + 1;

    std::vector<uint64_t> moduli(sizeQl);
    for (int i = 0; i < sizeQl; ++i)
        moduli[i] = cc.prime[i].p;

    std::vector<std::vector<uint64_t>> data;
    for (uint32_t i = 0; i < sizeQl; i++) {
        std::vector<uint64_t> vec(cc.N, 0);
		// Code works for positive values only, doesn't work for negative values.
        vec[0]            = (c.real() > 0) ? elem1[i] % (moduli[i]) : (-elem1[i] % (moduli[i]) + moduli[i]) % (moduli[i]);
        vec[cc.N / 2]        = (c.imag() > 0) ? elem2[i] % (moduli[i]) : (-elem2[i] % (moduli[i]) + moduli[i]) % (moduli[i]);
        // End code that only works for positive values.
		data.push_back(vec);
    }

    RNSPoly elemComplex(cc, data);
    elemComplex.NTT(cc.batch, true);
    c0.add(elemComplex);
}

void Ciphertext::multScalar(const std::complex<double> c, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    CKKS::SetCurrentContext(cc_);
    op_count[OPS::MULTSCALAR]++;

    if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT ||
        cc.rescaleTechnique == FIXEDAUTO) {
        if (NoiseLevel == 2)
            this->rescale();
    }
    assert(this->NoiseLevel == 1);

    auto elem1 = cc.ElemForEvalMult(c0.getLevel(), c.real());
    auto elem2 = cc.ElemForEvalMult(c0.getLevel(), c.imag());

    auto level0 = c0.getLevel();
    auto level1 = c1.getLevel();
    RNSPoly real0(cc, level0), imag0(cc, level0), real1(cc, level1), imag1(cc, level1);
    real0.copy(c0);
    imag0.copy(c0);
    real1.copy(c1);
    imag1.copy(c1);

    real0.multScalar(elem1);
    imag0.multScalar(elem2);
    real1.multScalar(elem1);
    imag1.multScalar(elem2);

    uint32_t N               = cc.N;
    uint32_t M               = 2 * N;

    std::vector<std::vector<uint64_t>> poly;
    uint32_t power        = M / 4;
    uint32_t powerReduced = power % M;
    uint32_t index        = power % N;
    for (int i = 0; i <= c0.getLevel(); ++i) {
        std::vector<uint64_t> vec(N, 0);
        vec[index]       = powerReduced < N ? 1 : cc.prime[0].p - 1;
        poly.push_back(vec);
    }
    RNSPoly monomial(cc, poly);
    monomial.NTT(cc.batch, true);

    imag0.multElement(monomial);
    imag1.multElement(monomial);
    c0.add(real0, imag0);
    c1.add(real1, imag1);

    // Manage metadata
    NoiseLevel += 1;
    NoiseFactor *= cc.param.ScalingFactorReal.at(c0.getLevel());
    if (rescale) {
        NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel());
        NoiseLevel -= 1;
    }

    if (rescale) {
        c0.rescale();
        c1.rescale();
    }
}

void Ciphertext::automorph(const int index, const int br) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	c0.automorph(index, br, nullptr);
	c1.automorph(index, br, nullptr);
}

void Ciphertext::extend(bool init) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	c0.generateSpecialLimbs(init && !c0.isModUp(), false);
	c1.generateSpecialLimbs(init && !c1.isModUp(), false);

	if (init) {
		if (!c0.isModUp())
			c0.scaleByP();
		if (!c1.isModUp())
			c1.scaleByP();
	}
}

void Ciphertext::rotate(const int index_, const bool moddown) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::ROTATE]++;
	int index = normalyzeIndex(index_);

	assert(index != 0);
	constexpr bool PRINT	= false;
	KeySwitchingKey& kskRot = cc.GetRotationKey(index, keyID, slots);

	if (0 && cc.GPUid.size() == 1) {
		if constexpr (0) {
			if constexpr (PRINT) {
				std::cout << "Output Automorph 1.";
				for (auto& j : c1.GPU)
					for (auto& i : j.limb) {
						SWITCH(i, printThisLimb(2));
					}
			}
			c1.modupInto(cc.getKeySwitchAux());
			RNSPoly& aux0 = c1.dotKSKInPlaceFrom(cc.getKeySwitchAux(), kskRot);
			c1.moddown(true, false, 0);
			if constexpr (PRINT) {
				std::cout << "Output Automorph 1.";
				for (auto& j : c1.GPU)
					for (auto& i : j.limb) {
						SWITCH(i, printThisLimb(2));
					}
			}
			c1.automorph(index, 1, nullptr);

			aux0.moddown(true, false, 1);
			if constexpr (PRINT) {
				std::cout << "c0\n";
				for (auto& j : c0.GPU)
					for (auto& i : j.limb) {
						SWITCH(i, printThisLimb(2));
					}
			}
			c0.add(aux0);
			if constexpr (PRINT) {
				std::cout << "Output KeySwitch 0.";
				for (auto& j : aux0.GPU)
					for (auto& i : j.limb) {
						SWITCH(i, printThisLimb(2));
					}
			}

			if constexpr (PRINT) {
				std::cout << "Output Add 0.";
				for (auto& j : c0.GPU)
					for (auto& i : j.limb) {
						SWITCH(i, printThisLimb(2));
					}
			}
			c0.automorph(index, 1, nullptr);
			if constexpr (PRINT) {
				std::cout << "Output Rot 0.";
				for (auto& j : c0.GPU)
					for (auto& i : j.limb) {
						SWITCH(i, printThisLimb(2));
					}
			}
		} else if constexpr (0) {
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			cc.getKeySwitchAux().rotateModupDotKSK(c0, c1, kskRot);

			if (moddown)
				c1.moddown(true, false, 0);
			c1.automorph(index, 1, nullptr);
			if (moddown)
				c0.moddown(true, false, 1);
			c0.automorph(index, 1, nullptr);
		} else if (0) {
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			c1.modupInto(cc.getKeySwitchAux());
			RNSPoly& aux0 = c1.dotKSKInPlaceFrom(cc.getKeySwitchAux(), kskRot);
			c1.moddown(true, false, 0);
			c1.automorph(index, 1, nullptr);

			aux0.moddown(true, false, 1);
			c0.add(aux0);
			c0.automorph(index, 1, nullptr);
		} else {
			auto& in0 = cc.getKeySwitchAux2();
			auto& in1 = cc.getKeySwitchAux();
			in1.copy(c1);
			in1.modup();
			// c1.modupInto(cc.getKeySwitchAux());
			in0.copy(c0);

			std::vector<int> index_;
			std::vector<RNSPoly*> c0_out;
			std::vector<RNSPoly*> c1_out;
			std::vector<RNSPoly*> ksk_a;
			std::vector<RNSPoly*> ksk_b;
			{
				{
					c0_out.push_back(&c0);
					c1_out.push_back(&c1);
					auto& ksk = cc.GetRotationKey(index, keyID, slots);
					ksk_a.push_back(&ksk.a);
					ksk_b.push_back(&ksk.b);
					index_.push_back(index);
				}
			}

			in1.hoistedRotationFused(index_, c0_out, c1_out, ksk_a, ksk_b, in0, in1);

			if (moddown) {
				// c1.moddown(true, false);
				// c0.moddown(true, false);
				modDown(false);
			}
		}
	} else {
		/*
		RNSPoly& in = cc.getKeySwitchAux();
		in.setLevel(c1.getLevel());
		in.copy(c1);

		RNSPoly& aux = MGPUkeySwitchCore(in, kskRot, moddown);
		if (!moddown) {
			//in.moddown(true, false);
			c1.SetModUp(false);
			c1.generateSpecialLimbs(false);
		}
		c1.automorph(index, 1, &in);
		//c1.moddown(true, false);

		if (!moddown) {
			//aux.moddown(true, false);
			c0.SetModUp(false);
			c0.generateSpecialLimbs(true);
		}
		aux.add(aux, c0);
		//c0.add(aux);
		c0.automorph(index, 1, &aux);
		//c0.moddown();
		*/

		auto& in0 = cc.getKeySwitchAux2();
		auto& in1 = cc.getKeySwitchAux();
		in1.copy(c1);
		in1.modup();
		// c1.modupInto(cc.getKeySwitchAux());
		in0.copy(c0);

		std::vector<int> index_;
		std::vector<RNSPoly*> c0_out;
		std::vector<RNSPoly*> c1_out;
		std::vector<RNSPoly*> ksk_a;
		std::vector<RNSPoly*> ksk_b;
		{
			{
				c0_out.push_back(&c0);
				c1_out.push_back(&c1);
				auto& ksk = cc.GetRotationKey(index, keyID, slots);
				ksk_a.push_back(&ksk.a);
				ksk_b.push_back(&ksk.b);
				index_.push_back(index);
			}
		}

		in1.hoistedRotationFused(index_, c0_out, c1_out, ksk_a, ksk_b, in0, in1);

		if (moddown) {
			modDown(false);
		}
	}
}

void Ciphertext::rotate(const Ciphertext& c, const int index) {
	this->copy(c);
	this->rotate(index, true);
}

void Ciphertext::conjugate(const Ciphertext& c) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::CONJUGATE]++;

	if (0 && cc.GPUid.size() == 1) {
		if constexpr (0) {
			this->copy(c);

			int index = 2 * cc.N - 1;
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			cc.getKeySwitchAux().rotateModupDotKSK(c0, c1, cc.GetRotationKey(index, c.keyID, slots));

			c1.moddown(true, false);
			for (int i = 0; i < (int)c1.GPU.size(); ++i) {
				c1.GPU.at(i).automorph(index, 1, nullptr, c1.isModUp());
			}
			c0.moddown(true, false);
			for (int i = 0; i < (int)c0.GPU.size(); ++i) {
				c0.GPU.at(i).automorph(index, 1, nullptr, c0.isModUp());
			}
		} else {
			int index = 2 * cc.N - 1;
			// auto& in0 = cc.getKeySwitchAux2();
			auto& in1 = cc.getKeySwitchAux();
			in1.copy(c.c1);
			in1.modup();
			// c1.modupInto(cc.getKeySwitchAux());
			// in0.copy(c0);

			std::vector<int> index_;
			std::vector<RNSPoly*> c0_out;
			std::vector<RNSPoly*> c1_out;
			std::vector<RNSPoly*> ksk_a;
			std::vector<RNSPoly*> ksk_b;
			{
				{
					c0_out.push_back(&c0);
					c1_out.push_back(&c1);
					auto& ksk = cc.GetRotationKey(index, c.keyID, slots);
					ksk_a.push_back(&ksk.a);
					ksk_b.push_back(&ksk.b);
					index_.push_back(index);
				}
			}

			c0.dropToLevel(c.c0.getLevel());
			c1.dropToLevel(c.c1.getLevel());
			c0.grow(c.c0.getLevel());
			c1.grow(c.c1.getLevel());

			in1.hoistedRotationFused(index_, c0_out, c1_out, ksk_a, ksk_b, c.c0, in1);

			if (1) {
				modDown();
			}

			this->copyMetadata(c);
		}
	} else {
		/*this->copy(c);

		RNSPoly& in = cc.getKeySwitchAux();
		in.setLevel(c1.getLevel());
		in.copyShallow(c1);
*/
		/*
		RNSPoly& aux = MGPUkeySwitchCore(in, cc.GetRotationKey(index, c.keyID, slots), true);
#pragma omp parallel for num_threads(cc.GPUid.size())
		for (int i = 0; i < (int)cc.GPUid.size(); ++i) {
			assert(omp_get_num_threads() == (int)cc.GPUid.size());
			c1.GPU.at(i).automorph(index, 1, &in.GPU.at(i), c1.isModUp());
		}
		c0.add(aux);
#pragma omp parallel for num_threads(cc.GPUid.size())
		for (int i = 0; i < (int)cc.GPUid.size(); ++i) {
			assert(omp_get_num_threads() == (int)cc.GPUid.size());
			c0.GPU.at(i).automorph(index, 1, nullptr, c0.isModUp());
		}

  */
		int index = 2 * cc.N - 1;
		// auto& in0 = cc.getKeySwitchAux2();
		auto& in1 = cc.getKeySwitchAux();
		in1.copy(c.c1);
		in1.modup();
		// c1.modupInto(cc.getKeySwitchAux());
		// in0.copy(c0);

		std::vector<int> index_;
		std::vector<RNSPoly*> c0_out;
		std::vector<RNSPoly*> c1_out;
		std::vector<RNSPoly*> ksk_a;
		std::vector<RNSPoly*> ksk_b;
		{
			{
				c0_out.push_back(&c0);
				c1_out.push_back(&c1);
				auto& ksk = cc.GetRotationKey(index, c.keyID, slots);
				ksk_a.push_back(&ksk.a);
				ksk_b.push_back(&ksk.b);
				index_.push_back(index);
			}
		}

		c0.dropToLevel(c.c0.getLevel());
		c1.dropToLevel(c.c1.getLevel());
		c0.grow(c.c0.getLevel());
		c1.grow(c.c1.getLevel());

		in1.hoistedRotationFused(index_, c0_out, c1_out, ksk_a, ksk_b, c.c0, in1);

		if (1) {
			modDown(false);
		}

		this->copyMetadata(c);
	}
}

void Ciphertext::rotate_hoisted(const std::vector<int>& indexes_, std::vector<Ciphertext*> results, const bool ext) {
	std::vector<int> indexes;
	for (auto i : indexes_) {
		indexes.push_back(normalyzeIndex(i));
	}

	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::HOISTEDROTATE]++;
	op_count[OPS::HOISTEDROTATEOUTS] += indexes.size();

	constexpr bool PRINT = false;
	assert(indexes.size() == results.size());

	bool grow_full = true;
	for (auto& i : results) {
		i->growToLevel(grow_full ? cc.L : this->c0.getLevel());
		i->dropToLevel(this->c0.getLevel());
		if (ext)
			i->c0.generateSpecialLimbs(false, false);
		if (ext)
			i->c1.generateSpecialLimbs(false, false);

		// i->c0.setLevel(c0.getLevel());
		// i->c1.setLevel(c1.getLevel());
	}

	if (!hoistRotateFused) {
		if (cc.GPUid.size() == 1) {
			cc.getKeySwitchAux().setLevel(c1.getLevel());
			c1.modupInto(cc.getKeySwitchAux());

			if constexpr (PRINT) {
				{
					cudaDeviceSynchronize();
					std::cout << "GPU: " << 0 << "Out ModUp after NTT ";
					for (size_t j = 0; j < cc.getKeySwitchAux().GPU[0].DIGITlimb.size(); ++j) {
						for (size_t i = 0; i < cc.getKeySwitchAux().GPU[0].DIGITlimb[j].size(); ++i) {
							std::cout << cc.getKeySwitchAux().GPU[0].DIGITmeta[j][i].id;
							SWITCH(cc.getKeySwitchAux().GPU[0].DIGITlimb[j][i], printThisLimb(2));
						}
						std::cout << std::endl;
					}
					std::cout << std::endl;
					cudaDeviceSynchronize();
				}
			}

			for (size_t i = 0; i < indexes.size(); ++i) {
				if (indexes[i] == 0) {
					results[i]->copy(*this);
					if (ext) {
						results[i]->extend();
					}
				} else {
					RNSPoly& aux0 = results[i]->c1.dotKSKInPlaceFrom(cc.getKeySwitchAux(), cc.GetRotationKey(indexes[i], keyID, slots), &c1);
					// results[i]->c0.dropToLevel(getLevel());
					// results[i]->c1.dropToLevel(getLevel());
					if (!ext)
						results[i]->c1.moddown(true, false, 0);
					results[i]->c1.automorph(indexes[i], 1);

					if (!ext)
						aux0.moddown(true, false, 1);

					// results[i]->c0.generateSpecialLimbs(true);
					results[i]->c0.add(c0, aux0);
					results[i]->c0.automorph(indexes[i], 1);
					// if (!ext)
					//     results[i]->c0.moddown(true, false);

					results[i]->keyID		= keyID;
					results[i]->NoiseLevel	= NoiseLevel;
					results[i]->NoiseFactor = NoiseFactor;
				}
			}
		} else {

			RNSPoly& in = cc.getKeySwitchAux();
			in.setLevel(c1.getLevel());
			in.copy(c1);
			in.modup();

			for (size_t i = 0; i < indexes.size(); ++i) {
				if (indexes[i] == 0) {
					results[i]->copy(*this);
					if (ext) {
						results[i]->extend();
					}
				} else {
					RNSPoly& aux0 = in.dotKSKInPlace(cc.GetRotationKey(indexes[i], keyID, slots), &c1);
					// results[i]->c0.dropToLevel(getLevel());
					// results[i]->c1.dropToLevel(getLevel());
					if (!ext) {
						in.moddown(true, false, 0);
					}
					// std::cout << "in ismodup: " << in.isModUp() << std::endl;
					results[i]->c1.automorph(indexes[i], 1, &in);
					// std::cout << "results[i] c1 ismodup: " << results[i]->c1.isModUp() << std::endl;
					if (!ext) {
						aux0.moddown(true, false, 1);
					}
					// std::cout << "aux0 ismodup: " << aux0.isModUp() << std::endl;
					// std::cout << "c0 ismodup: " << c0.isModUp() << std::endl;
					results[i]->c0.add(c0, aux0);
					// std::cout << "results[i] c0 ismodup: " << results[i]->c0.isModUp() << std::endl;
					results[i]->c0.automorph(indexes[i], 1, nullptr);
					// std::cout << "results[i] c0 ismodup: " << results[i]->c0.isModUp() << std::endl;

					results[i]->copyMetadata(*this);
				}
			}
		}
	} else {
		RNSPoly& in = cc.getKeySwitchAux();
		if (cc.GPUid.size() == 1) {
			in.setLevel(c1.getLevel());
			c1.modupInto(in);
		} else {
			in.setLevel(c1.getLevel());
			in.copy(c1);
			in.modup();
		}

		if constexpr (PRINT) {
			{
				cudaDeviceSynchronize();
				std::cout << "GPU: " << 0 << "Out ModUp after NTT ";
				for (size_t j = 0; j < cc.getKeySwitchAux().GPU[0].DIGITlimb.size(); ++j) {
					for (size_t i = 0; i < cc.getKeySwitchAux().GPU[0].DIGITlimb[j].size(); ++i) {
						std::cout << cc.getKeySwitchAux().GPU[0].DIGITmeta[j][i].id;
						SWITCH(cc.getKeySwitchAux().GPU[0].DIGITlimb[j][i], printThisLimb(2));
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
				cudaDeviceSynchronize();
			}
		}

		std::vector<int> index;
		std::vector<RNSPoly*> c0_out;
		std::vector<RNSPoly*> c1_out;
		std::vector<RNSPoly*> ksk_a;
		std::vector<RNSPoly*> ksk_b;
		for (size_t i = 0; i < indexes.size(); ++i) {
			if (indexes[i] == 0) {
				results[i]->copy(*this);
				if (ext) {
					results[i]->extend();
				}
			} else {
				c0_out.push_back(&results[i]->c0);
				c1_out.push_back(&results[i]->c1);
				auto& ksk = cc.GetRotationKey(indexes[i], keyID, slots);
				ksk_a.push_back(&ksk.a);
				ksk_b.push_back(&ksk.b);
				index.push_back(indexes[i]);

				results[i]->copyMetadata(*this);
			}
		}

		in.hoistedRotationFused(index, c0_out, c1_out, ksk_a, ksk_b, c0, c1);

		if (!ext) {
			for (size_t i = 0; i < indexes.size(); ++i) {
				if (indexes[i] != 0) {
					results[i]->modDown();
				}
			}
		}
	}
}

void Ciphertext::mult(const Ciphertext& b, const Ciphertext& c, bool rescale) {
	if (this == &b && this == &c) {
		this->square(rescale);
	} else if (this == &b) {
		this->mult(c, rescale);
	} else if (this == &c) {
		this->mult(b, rescale);
	} else {
		if (b.getLevel() <= c.getLevel()) {
			this->copy(b);
			this->mult(c, rescale);
		} else {
			this->copy(c);
			this->mult(b, rescale);
		}
	}
}

void Ciphertext::square(const Ciphertext& src, bool rescale) {
	if (this == &src) {
		this->square(rescale);
	} else {
		this->copy(src);
		this->square(rescale);
	}
}

void Ciphertext::dropToLevel(int level) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	c0.dropToLevel(level);
	c1.dropToLevel(level);
}

int Ciphertext::getLevel() const {
	assert(c0.getLevel() == c1.getLevel());
	return c0.getLevel();
}

void Ciphertext::multScalar(const Ciphertext& b, const double c, bool rescale) {
	this->copy(b);
	this->multScalar(c, rescale);
}

void Ciphertext::evalLinearWSumMutable(uint32_t n, const std::vector<Ciphertext*>& ctxs, std::vector<double> weights) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::WSUM]++;
	op_count[OPS::WSUMINPUTS] += n;

	// TODO adjujst level of inputs for correct precission

	if constexpr (1) {
		if (this->getLevel() == -1) {
			this->c0.grow(ctxs[0]->getLevel());
			this->c1.grow(ctxs[0]->getLevel());
			this->NoiseLevel = 1;
		}

		for (size_t i = 0; i < n; ++i) {
			if (cc.rescaleTechnique == FIXEDMANUAL) {
				assert(ctxs[i]->NoiseLevel == 1);
				assert(getLevel() <= ctxs[i]->getLevel());
			} else {
				assert(ctxs[i]->NoiseLevel == 1);

				// assert(getLevel() == ctxs[i]->getLevel()); TODO uncomment and fix FIXEDAUTO bug
			}
			assert(ctxs[0]->keyID == ctxs[i]->keyID);
		}

		std::vector<uint64_t> elem(MAXP * n);

		// #pragma omp parallel for
		double scalingFactor;
		for (size_t i = 0; i < n; ++i) {
			auto aux = cc.ElemForEvalMult(c0.getLevel(), weights[i], ctxs[i]->getLevel());
			for (size_t j = 0; j < aux.size(); ++j)
				elem[i * MAXP + j] = aux[j];
		}

		std::vector<const RNSPoly*> c0s(n), c1s(n);

		for (size_t i = 0; i < n; ++i) {
			c0s[i] = &ctxs[i]->c0;
			c1s[i] = &ctxs[i]->c1;
		}
		c0.evalLinearWSum(n, c0s, elem);
		c1.evalLinearWSum(n, c1s, elem);

		// this->copyMetadata(*ctxs[0]);
		this->slots = ctxs[0]->slots;
		this->keyID = ctxs[0]->keyID;
		for (int i = 1; i < n; ++i) {
			assert(this->keyID == ctxs[i]->keyID);
			slots = std::max(slots, ctxs[i]->slots);
		}
		this->NoiseLevel  = 2;
		this->NoiseFactor = cc.param.ScalingFactorReal.at(getLevel()) * cc.param.ScalingFactorReal.at(getLevel());
	} else {
		this->multScalar(*ctxs[0], weights[0], false);
		for (int i = 1; i < n; ++i) {
			assert(getLevel() <= ctxs[i]->getLevel());
		}
		for (int i = 1; i < n; ++i) {
			this->addMultScalar(*ctxs[i], weights[i]);
		}
	}
}

void Ciphertext::addMultScalar(const Ciphertext& b, double d) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::MULTSCALAR]++;
	op_count[OPS::COPY]++;
	op_count[OPS::ADD]++;

	assert(NoiseLevel == 2);
	assert(b.NoiseLevel == 1);
	assert(b.getLevel() >= getLevel());
	assert(keyID == b.keyID);
	auto elem = cc.ElemForEvalMult(c0.getLevel(), d);

	RNSPoly aux0(cc);
	RNSPoly aux1(cc);
	aux0.copy(b.c0);
	aux0.multScalar(elem);
	c0.add(aux0);
	aux1.copy(b.c1);
	aux1.multScalar(elem);
	c1.add(aux1);
}

void Ciphertext::addScalar(const Ciphertext& b, double c) {
	this->copy(b);
	this->addScalar(c);
}

void Ciphertext::add(const Ciphertext& b, const Ciphertext& c) {
	assert(NoiseLevel <= 2);
	if (this == &b && this == &c) {
		this->add(c);
	} else if (this == &b) {
		this->add(c);
	} else if (this == &c) {
		this->add(b);
	} else {
		if (b.getLevel() <= c.getLevel()) {
			this->copy(b);
			this->add(c);
		} else {
			this->copy(c);
			this->add(b);
		}
	}
}

void Ciphertext::growToLevel(int level) {

	c0.grow(level);
	c1.grow(level);
	if (c0.isModUp())
		c0.generateSpecialLimbs(false, false);
	if (c1.isModUp())
		c1.generateSpecialLimbs(false, false);
}

void Ciphertext::copy(const Ciphertext& ciphertext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	assert(this != &ciphertext);
	op_count[OPS::COPY]++;
	c0.copy(ciphertext.c0);
	c1.copy(ciphertext.c1);
	this->copyMetadata(ciphertext);
}

void Ciphertext::multPt(const Ciphertext& c, const Plaintext& b, bool rescale) {
	this->copy(c);
	multPt(b, rescale);
}

void Ciphertext::addMultPt(const Ciphertext& c, const Plaintext& b, bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::ADDMULTPT]++;

	assert(NoiseLevel == 2);
	assert(c.NoiseLevel == 1);
	assert(b.NoiseLevel == 1);

	c0.addMult(c.c0, b.c0);
	c1.addMult(c.c1, b.c0);

	if (rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
		this->rescale();
	}
}

void Ciphertext::addPt(const Ciphertext& ciphertext, const Plaintext& plaintext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	this->copy(ciphertext);
	this->addPt(plaintext);
}

void Ciphertext::reinterpretContext(const Ciphertext& ciphertext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	assert(cc_.get() != ciphertext.cc_.get());
	assert(!ciphertext.c0.isModUp());
	assert(!ciphertext.c1.isModUp());
	assert(ciphertext.getLevel() <= cc.L);

	for (int i = 0; i <= ciphertext.getLevel(); i++) {
		assert(ciphertext.cc.prime.at(i).p == cc.prime.at(i).p);
		assert(ciphertext.cc.prime.at(i).type == cc.prime.at(i).type);
	}

	CKKS::SetCurrentContext(cc_);
	this->copy(ciphertext);
}

void Ciphertext::keySwitch(const KeySwitchingKey& ksk) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	assert(ksk.keyID == this->keyID);

	RNSPoly& aux = cc.getKeySwitchAux();
	aux.copy(c1); // This is to save on memory allocations for keyswitching, not best performance but not expected for relinearize or rotation

	RNSPoly& aux0 = MGPUkeySwitchCore(aux, ksk, true); // c1.dotKSKInPlaceFrom(aux, ksk, &aux);

	c1.copy(aux);
	c0.add(aux0);
}

void Ciphertext::sub(const Ciphertext& ciphertext, const Ciphertext& ciphertext1) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	assert(ciphertext.getLevel() <= ciphertext1.getLevel());
	this->copy(ciphertext);
	this->sub(ciphertext1);
}

bool Ciphertext::adjustForAddOrSub(const Ciphertext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	/*
	if (cc.rescaleTechnique == FIXEDMANUAL) {
		if (b.NoiseLevel > NoiseLevel || (b.getLevel() < getLevel()))
			return false;
		else
			return true;
	} else
	*/
	if (cc.rescaleTechnique == FIXEDMANUAL || cc.rescaleTechnique == FIXEDAUTO) {
		if (getLevel() - NoiseLevel > b.getLevel() - b.NoiseLevel) {
			if (b.NoiseLevel == 1 && NoiseLevel == 2) {
				// this->dropToLevel(b.getLevel() + 1);
				rescale();
			} else if (b.NoiseLevel == 2 && NoiseLevel == 1) {
				this->multScalar(1.0);
				// this->dropToLevel(b.getLevel());
			}
			return true;
		} else if (b.NoiseLevel == 1 && NoiseLevel == 2) {
			rescale();
			return true;
		} else if (NoiseLevel == 1 && b.NoiseLevel == 2) {
			return false;
		} else {
			return true;
		}
	} else if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		usint c1lvl	  = getLevel();
		usint c2lvl	  = b.getLevel();
		usint c1depth = this->NoiseLevel;
		usint c2depth = b.NoiseLevel;
		auto sizeQl1  = c1lvl + 1;
		// auto sizeQl2 = c2lvl + 1;

		if (c1lvl > c2lvl) {
			if (c1depth == 2) {
				if (c2depth == 2) {
					double scf1 = NoiseFactor;
					double scf2 = b.NoiseFactor;
					double scf	= cc.param.ScalingFactorReal[c1lvl];	 // cryptoParams->GetScalingFactorReal(c1lvl);
					double q1	= cc.param.ModReduceFactor[sizeQl1 - 1]; // cryptoParams->GetModReduceFactor(sizeQl1 - 1);
					multScalarNoPrecheck(scf2 / scf1 * q1 / scf);
					rescale();
					if (getLevel() > b.getLevel()) {
						this->dropToLevel(b.getLevel());
					}

					assert(std::abs((NoiseFactor * scf2 / scf1 * q1 / scf - b.NoiseFactor) / b.NoiseFactor) < 0.001);
					NoiseFactor = b.NoiseFactor;
					/*
					rescale();
					double scf1 = NoiseFactor;
					double scf2 = b.NoiseFactor;
					double scf = cc.param.ScalingFactorReal[c1lvl];  // cryptoParams->GetScalingFactorReal(c1lvl);
					multScalarNoPrecheck(scf2 / scf1 / scf);
					this->dropToLevel(c2lvl);
					//LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl);
					NoiseFactor = scf2;
*/
				} else {
					if (c1lvl - 1 == c2lvl) {
						rescale();
					} else {
						double scf1 = NoiseFactor;
						double scf2 = cc.param.ScalingFactorRealBig[c2lvl + 1]; // cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
						double scf	= cc.param.ScalingFactorReal[c1lvl];		// cryptoParams->GetScalingFactorReal(c1lvl);
						double q1	= cc.param.ModReduceFactor[sizeQl1 - 1];	// cryptoParams->GetModReduceFactor(sizeQl1 - 1);
						multScalarNoPrecheck(scf2 / scf1 * q1 / scf);
						rescale();
						if (getLevel() - 1 > b.getLevel()) {
							this->dropToLevel(b.getLevel() + 1);
							// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 2);
						}
						rescale();
						assert(std::abs((NoiseFactor * scf2 / scf1 * q1 / scf - b.NoiseFactor) / b.NoiseFactor) < 0.001);

						NoiseFactor = b.NoiseFactor;
					}
				}
			} else {
				if (c2depth == 2) {
					double scf1 = NoiseFactor;
					double scf2 = b.NoiseFactor;
					double scf	= cc.param.ScalingFactorReal[c1lvl]; // cryptoParams->GetScalingFactorReal(c1lvl);
					multScalarNoPrecheck(scf2 / scf1 / scf);
					this->dropToLevel(c2lvl);
					// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl);
					assert(std::abs((NoiseFactor * scf2 / scf1 / scf - b.NoiseFactor) / b.NoiseFactor) < 0.001);
					NoiseFactor = scf2;
				} else {
					double scf1 = NoiseFactor;
					double scf2 = cc.param.ScalingFactorRealBig[c2lvl + 1]; // cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
					double scf	= cc.param.ScalingFactorReal[c1lvl];		// cryptoParams->GetScalingFactorReal(c1lvl);
					multScalarNoPrecheck(scf2 / scf1 / scf);
					if (c1lvl - 1 > c2lvl) {
						this->dropToLevel(c2lvl + 1);
						// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 1);
					}
					rescale();
					assert(std::abs((NoiseFactor * scf2 / scf1 / scf - b.NoiseFactor) / b.NoiseFactor) < 0.001);
					NoiseFactor = b.NoiseFactor;
				}
			}
			return true;
		} else if (c1lvl < c2lvl) {
			return false;
		} else {
			if (c1depth < c2depth) {
				multScalar(1.0, false);
			} else if (c2depth < c1depth) {
				return false;
			}
			return true;
		}
	}
	assert("This never happens" == nullptr);
	return false;
}

bool Ciphertext::adjustForMult(const Ciphertext& ciphertext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	if (adjustForAddOrSub(ciphertext)) {
		if (NoiseLevel == 2)
			rescale();
		if (ciphertext.NoiseLevel == 2)
			return false;
		else
			return true;
	} else {
		if (NoiseLevel == 2)
			rescale();
		return false;
	}
}

bool Ciphertext::hasSameScalingFactor(const Plaintext& b) const {
	return NoiseFactor > b.NoiseFactor * (1 - 1e-9) && NoiseFactor < b.NoiseFactor * (1 + 1e-9);
}

void Ciphertext::clearOpRecord() {
	op_count.clear();
}

void Ciphertext::dotProductPt(Ciphertext* ciphertexts, Plaintext* plaintexts, const int n, const bool ext) {

	std::vector<Plaintext*> pts(n);
	for (int i = 0; i < n; ++i) {
		pts[i] = &plaintexts[i];
	}
	dotProductPt(ciphertexts, pts.data(), n, ext);
}

void Ciphertext::dotProductPt(Ciphertext* ciphertexts, Plaintext** plaintexts, const int n, const bool ext) {
	std::vector<Ciphertext*> pts(n);
	for (int i = 0; i < n; ++i) {
		pts[i] = &ciphertexts[i];
	}
	dotProductPt(pts.data(), plaintexts, n, ext);
}

void Ciphertext::dotProductPt(Ciphertext** ciphertexts, Plaintext** plaintexts, const int n, const bool ext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	std::vector<const RNSPoly*> c0s(n, nullptr), c1s(n, nullptr), pts(n, nullptr);

	for (int i = 0; i < n; ++i) {
		c0s[i] = &(ciphertexts[i]->c0);
		c1s[i] = &(ciphertexts[i]->c1);
		pts[i] = &(plaintexts[i]->c0);
		assert(getLevel() <= ciphertexts[i]->getLevel());
		assert(getLevel() <= plaintexts[i]->c0.getLevel());
		if (ext) {
			assert(ciphertexts[i]->c0.isModUp());
			assert(plaintexts[i]->c0.isModUp());
		}
		assert(ciphertexts[0]->keyID == ciphertexts[i]->keyID);
	}
	c0.dotProductPt(c1, c0s, c1s, pts, ext);

	// Manage metadata
	this->multMetadata(*ciphertexts[0], *plaintexts[0]);
	for (int i = 1; i < n; ++i) {
		this->slots = std::max(this->slots, ciphertexts[i]->slots);
		this->slots = std::max(this->slots, plaintexts[i]->slots);
	}
}

void Ciphertext::dotProduct(const std::vector<Ciphertext*>& a, const std::vector<Ciphertext*>& b, const bool ext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	assert(a.size() == b.size());
	assert(a.size() > 0);
	assert(a[0]->c0.isModUp() == b[0]->c1.isModUp());
	bool in_ext = a[0]->c0.isModUp();

	this->growToLevel(a[0]->getLevel());

	std::vector<const RNSPoly*> c0s(a.size(), nullptr), c1s(a.size(), nullptr), d0s(a.size(), nullptr), d1s(a.size(), nullptr);

	for (size_t i = 0; i < a.size(); ++i) {
		assert(this->cc_ == a[i]->cc_);
		assert(this->cc_ == b[i]->cc_);
		assert(a[0]->NoiseFactor == a[i]->NoiseFactor);
		assert(a[0]->NoiseFactor == b[i]->NoiseFactor);
		assert(a[0]->keyID == a[i]->keyID);
		assert(a[0]->keyID == b[i]->keyID);
		assert(a[i]->NoiseLevel == 1);
		assert(b[i]->NoiseLevel == 1);
		assert(a[0]->getLevel() == a[i]->getLevel());
		assert(a[0]->getLevel() == b[i]->getLevel());

		assert(a[0]->c0.isModUp() == a[i]->c0.isModUp());
		assert(a[0]->c1.isModUp() == a[i]->c1.isModUp());
		assert(a[0]->c0.isModUp() == b[i]->c0.isModUp());
		assert(a[0]->c1.isModUp() == b[i]->c1.isModUp());
		c0s[i] = &(a[i]->c0);
		c1s[i] = &(a[i]->c1);
		d0s[i] = &(b[i]->c0);
		d1s[i] = &(b[i]->c1);
	}

	c0.dotProduct(c1, cc.GetEvalKey(a[0]->keyID).a, cc.GetEvalKey(a[0]->keyID).b, c0s, c1s, d0s, d1s, in_ext, ext);

	this->multMetadata(*a[0], *b[0]);
	for (size_t i = 1; i < a.size(); ++i) {
		this->slots = std::max(this->slots, a[i]->slots);
		this->slots = std::max(this->slots, b[i]->slots);
	}
}

void Ciphertext::multMonomial(/*Ciphertext& ctxt,*/ int power) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	CKKS::SetCurrentContext(cc_);

	if (!cc.precom.monomialCache.contains(power) || cc.precom.monomialCache.find(power)->second.getLevel() != this->getLevel()) {
		// TODO compute fully as a GPU function.
		RNSPoly monomial(cc.getAuxilarPoly());
		monomial.grow(c0.getLevel());
		monomial.dropToLevel(c0.getLevel());
		std::vector<uint64_t> coefs(cc.N, 0);

		if (power < cc.N) {
			coefs[power] = 1;

			for (auto& g : monomial.GPU) {
				cudaSetDevice(g.device);
				int limb_size = g.getLimbSize(monomial.getLevel());
				for (int i = 0; i < limb_size; ++i) {
					SWITCH(g.limb[i], load(coefs));
					g.s.wait(STREAM(g.limb[i]));
				}

				// for (auto& l : g.limb) {
				//     SWITCH(l, load(coefs));
				//     g.s.wait(STREAM(l));
				// }

				for (int i = 0; i < limb_size; i += cc.batch) {
					STREAM(g.limb[i]).wait(g.s);
				}
			}
		} else {
			for (auto& g : monomial.GPU) {
				cudaSetDevice(g.device);
				int limb_size = g.getLimbSize(monomial.getLevel());
				for (int i = 0; i < limb_size; ++i) {
					coefs[power % cc.N] = (cc.prime[PRIMEID(g.limb[i])].p - 1) /*% ctxt.cc.prime[PRIMEID(l)].p*/;
					SWITCH(g.limb[i], load(coefs));
					g.s.wait(STREAM(g.limb[i]));
				}
				// for (auto& l : g.limb) {
				//     coefs[power % cc.N] = (cc.prime[PRIMEID(l)].p - 1) /*% ctxt.cc.prime[PRIMEID(l)].p*/;
				//     SWITCH(l, load(coefs));
				//     g.s.wait(STREAM(l));
				// }
				for (int i = 0; i < limb_size; i += cc.batch) {
					STREAM(g.limb[i]).wait(g.s);
				}
			}
		}

		// cudaDeviceSynchronize();
		monomial.NTT(cc.batch, true);
		// cudaDeviceSynchronize();

		cc.precom.monomialCache.erase(power);
		cc.precom.monomialCache.emplace(power, std::move(monomial));
	}

	RNSPoly& monomial = cc.precom.monomialCache.find(power)->second;

	c0.multElement(monomial);
	c1.multElement(monomial);

	/* Based on this (OpenFHE):
std::vector<DCRTPoly>& cv = ciphertext->GetElements();
const auto elemParams     = cv[0].GetParams();
auto paramsNative         = elemParams->GetParams()[0];
usint N                   = elemParams->GetRingDimension();
usint M                   = 2 * N;

	NativePoly monomial(paramsNative, Format::COEFFICIENT, true);

	usint powerReduced = power % M;
	usint index        = power % N;
	monomial[index]    = powerReduced < N ? NativeInteger(1) : paramsNative->GetModulus() - NativeInteger(1);

	DCRTPoly monomialDCRT(elemParams, Format::COEFFICIENT, true);
	monomialDCRT = monomial;
	monomialDCRT.SetFormat(Format::EVALUATION);

	for (usint i = 0; i < ciphertext->NumberCiphertextElements(); i++) {
		cv[i] *= monomialDCRT;
	}
	*/
}

void Ciphertext::printOpRecord() {
	std::cout << "|-------------- OP COUNT --------------|\n";
	for (const auto& [op, c] : op_count) {
		std::cout << opstr[op] << c << "\n";
	}
	std::cout << "|--------------------------------------|" << std::endl;
}

} // namespace FIDESlib::CKKS
