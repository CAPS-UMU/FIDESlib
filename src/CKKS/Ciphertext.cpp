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
using sc                  = std::experimental::source_location;
constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc                  = std::source_location;
constexpr int PREFIX_SIZE = 23;
#endif

namespace FIDESlib::CKKS {

bool hoistRotateFused         = true;
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
	: my_range(std::move(ct_moved.my_range)), keyID(std::move(ct_moved.keyID)), cc_(std::move(ct_moved.cc_)), cc(ct_moved.cc), c0(std::move(ct_moved.c0)),
	  c1(std::move(ct_moved.c1)),
	  NoiseFactor(ct_moved.NoiseFactor), NoiseLevel(ct_moved.NoiseLevel), slots(ct_moved.slots) {
}

Ciphertext::Ciphertext(const Context& cc)
	: my_range(loc, LIFETIME), cc_((assert(cc != nullptr), CudaNvtxStart(std::string{ sc::current().function_name() }.substr()), cc)), cc(*cc_),
	  c0(cc->getAuxilarPoly()), c1(cc->getAuxilarPoly()) {
	c0.dropToLevel(-1);
	c1.dropToLevel(-1);
	c0.SetModUp(false);
	c1.SetModUp(false);
	CudaNvtxStop();
}

Ciphertext::Ciphertext(const Context& cc, const RawCipherText& rawct)
	: Ciphertext(cc) {
	this->load(rawct);
}

Ciphertext::~Ciphertext() {
	if (!c1.GPU.empty())
		cc_->returnAuxilarPoly(std::move(c1));
	if (!c0.GPU.empty())
		cc_->returnAuxilarPoly(std::move(c0));
}

void Ciphertext::copyMetadata(const Ciphertext& a) {
	assert(this->getLevel() == a.getLevel());
	this->slots       = a.slots;
	this->keyID       = a.keyID;
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
	// assert(a.NoiseLevel == b.NoiseLevel);
	this->NoiseLevel  = a.NoiseLevel + b.NoiseLevel;
	this->NoiseFactor = a.NoiseFactor * b.NoiseFactor;
}

void Ciphertext::multMetadata(const Ciphertext& a, const Plaintext& b) {
	assert(this->getLevel() == a.getLevel());
	if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
		// assert(a.getLevel() == b.c0.getLevel());
		//  assert(a.NoiseFactor == b.NoiseFactor);
	}
	this->slots = std::max(a.slots, b.slots);
	this->keyID = a.keyID;
	// assert(a.NoiseLevel == b.NoiseLevel);
	this->NoiseLevel = a.NoiseLevel + b.NoiseLevel;

	this->NoiseFactor = a.NoiseFactor * b.NoiseFactor;
}

int Ciphertext::normalyzeIndex(int index) const {

	return FIDESlib::CKKS::normalyzeIndex(index, slots, cc.N);
}

void Ciphertext::add(const Ciphertext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	assert(keyID == b.keyID);
	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp() || b.c1.isModUp()) {
			assert(getLevel() == b.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}

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

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp() || b.c1.isModUp()) {
			assert(getLevel() == b.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}
		assert(this->getLevel() <= b.getLevel());
		dropToLevel(b.getLevel());
	}
	op_count[OPS::ADD]++;

	c0.add(b.c0);
	c1.add(b.c1);

	this->addMetadata(*this, b);
}

void Ciphertext::addMutable(Ciphertext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	assert(keyID == b.keyID);
	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp() || b.c1.isModUp()) {
			assert(getLevel() == b.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}

		if (!adjustForAddOrSub(b)) {
			if (b.adjustForAddOrSub(*this))
				add(b);
			else
				assert(false);
			return;
		}
	}

	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		assert(this->getLevel() == b.getLevel());
	} else if (getLevel() > b.getLevel()) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp() || b.c1.isModUp()) {
			assert(getLevel() == b.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}
		assert(this->getLevel() <= b.getLevel());
		dropToLevel(b.getLevel());
	}
	add(b);
}

void Ciphertext::sub(const Ciphertext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	assert(keyID == b.keyID);
	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp() || b.c1.isModUp()) {
			assert(getLevel() == b.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}

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

void Ciphertext::subMutable(Ciphertext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	assert(keyID == b.keyID);
	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp() || b.c1.isModUp()) {
			assert(getLevel() == b.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}

		if (!adjustForAddOrSub(b)) {
			if (b.adjustForAddOrSub(*this))
				sub(b);
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
	sub(b);
}

void Ciphertext::addPt(const Plaintext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
			assert(getLevel() == b.c0.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}

		if (b.NoiseLevel == 1 && NoiseLevel == 2 && b.c0.getLevel() == getLevel() - 1) {
			this->rescaleInternal();
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

void Ciphertext::addPtMutable(Plaintext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
			assert(getLevel() == b.c0.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}

		if (b.NoiseLevel == 1 && NoiseLevel == 2 && b.c0.getLevel() == getLevel() - 1) {
			this->rescaleInternal();
		}

		if (b.c0.getLevel() != this->getLevel()) {
			if (!b.adjustPlaintextToCiphertext(b, *this)) {
				assert(false);
			} else {
				addPt(b);
			}
			return;
		}
	}
	assert(NoiseLevel == b.NoiseLevel);
	addPt(b);
}

void Ciphertext::subPt(const Plaintext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
			assert(getLevel() == b.c0.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
			assert(NoiseLevel == 1);
		}

		if (b.NoiseLevel == 1 && NoiseLevel == 2 && b.c0.getLevel() == getLevel() - 1) {
			this->rescaleInternal();
		}

		if (b.c0.getLevel() != this->getLevel()) {
			Plaintext b_(cc_);
			if (!b_.adjustPlaintextToCiphertext(b, *this)) {
				assert(false);
			} else {
				subPt(b_);
			}
			return;
		}
	}
	assert(NoiseLevel == b.NoiseLevel);
	op_count[OPS::ADDPT]++;

	c0.sub(b.c0);

	this->addMetadata(*this, b);
}

void Ciphertext::subPtMutable(Plaintext& b) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
			assert(getLevel() == b.c0.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
			assert(NoiseLevel == 1);
		}

		if (b.NoiseLevel == 1 && NoiseLevel == 2 && b.c0.getLevel() == getLevel() - 1) {
			this->rescaleInternal();
		}

		if (b.c0.getLevel() != this->getLevel()) {
			if (!b.adjustPlaintextToCiphertext(b, *this)) {
				assert(false);
			} else {
				subPt(b);
			}
			return;
		}
	}
	assert(NoiseLevel == b.NoiseLevel);
	subPt(b);
}

void Ciphertext::load(const RawCipherText& rawct) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	keyID = rawct.keyid;
	c0.load(rawct.sub_0, rawct.moduli);
	c1.load(rawct.sub_1, rawct.moduli);

	NoiseLevel  = rawct.NoiseLevel;
	NoiseFactor = rawct.Noise;
	slots       = rawct.slots;
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
	rawct.Noise      = NoiseFactor;
	rawct.keyid      = keyID;
	rawct.slots      = slots; // TODO store other interesting metadata
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

void Ciphertext::multPt(const Plaintext& b, bool rescale, bool ignore_scale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	constexpr bool PRINT = false;

	if (!ignore_scale) {
		if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

			if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
				assert(getLevel() == b.c0.getLevel());
				assert(NoiseLevel == b.NoiseLevel);
				assert(NoiseLevel == 1);
			}

			if constexpr (PRINT)
				std::cout << "multPt: Rescale input ciphertext" << std::endl;
			if (NoiseLevel == 2)
				this->rescaleInternal();
		}

		if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

			if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
				assert(getLevel() == b.c0.getLevel());
				assert(NoiseLevel == b.NoiseLevel);
				assert(NoiseLevel == 1);
			}

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
						this->rescaleInternal();
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
	}
	op_count[OPS::MULTPT]++;

	c0.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL);
	c1.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL);

	this->multMetadata(*this, b);
	if (rescale && cc.rescaleTechnique == CKKS::FIXEDMANUAL) {
		NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
		NoiseLevel -= 1;
	}
}

void Ciphertext::multPtMutable(Plaintext& b, bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	constexpr bool PRINT = false;

	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
			assert(getLevel() == b.c0.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
			assert(NoiseLevel == 1);
		}

		if constexpr (PRINT)
			std::cout << "multPt: Rescale input ciphertext" << std::endl;
		if (NoiseLevel == 2)
			this->rescaleInternal();
	}

	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
			assert(getLevel() == b.c0.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
			assert(NoiseLevel == 1);
		}

		if (b.c0.getLevel() != this->getLevel() || b.NoiseLevel == 2 /*!hasSameScalingFactor(b)*/) {
			if constexpr (PRINT)
				std::cout << "multPt: adjust input plaintext" << std::endl;

			// if (!this->adju)
			if (!b.adjustPlaintextToCiphertext(b, *this)) {
				if constexpr (PRINT)
					std::cout << "multPt: FAILED!" << std::endl;
				assert(false);
			} else {
				if (NoiseLevel == 2)
					this->rescaleInternal();
				if (b.NoiseLevel == 2) {
					if constexpr (PRINT)
						std::cout << "multPt: Rescale input plaintext" << std::endl;
					b.rescale();
				}
				multPt(b, rescale);
			}
			return;
		}
	}

	assert(NoiseLevel < 2);
	assert(b.NoiseLevel < 2);
	multPt(b, rescale);
}

void Ciphertext::rescaleInternal() {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	// assert(this->NoiseLevel == 2);
	if (cc.rescaleTechnique != FIXEDMANUAL) {
		// this wouldn't do anything in OpenFHE
	}
	op_count[OPS::RESCALE]++;

	NoiseFactor /= (getLevel() == cc.L + 1) ? cc.specialPrime[0].p : cc.param.ModReduceFactor.at(c0.getLevel());

	if constexpr (RESCALE_DOUBLE) {
		c0.rescaleDouble(c1);
	} else {
		c0.rescale();
		c1.rescale();
	}

	// Manage metadata
	NoiseLevel -= 1;
	assert(NoiseFactor == (NoiseLevel == 1 ? cc.param.ScalingFactorReal[this->getLevel()] : cc.param.ScalingFactorRealBig[this->getLevel()]));
}


void Ciphertext::rescale() {
	if (cc.rescaleTechnique != FIXEDMANUAL)
		return;

	rescaleInternal();
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

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
			assert(getLevel() == b.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}

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
	if ((cc.rescaleTechnique == FIXEDMANUAL || cc.rescaleTechnique == FIXEDAUTO) && getLevel() > b.getLevel()) {
		dropToLevel(b.getLevel(), true);
	}
	assert(NoiseLevel == 1);
	assert(NoiseLevel == b.NoiseLevel);
	op_count[OPS::MULT]++;
	// assert(c0.getLevel() <= b.c0.getLevel());
	// assert(c1.getLevel() <= b.c1.getLevel());
	Out(KEYSWITCH, " start ");
	assert(this->NoiseLevel == 1);
	assert(b.NoiseLevel == 1);

	KeySwitchingKey& kskEval = cc.GetEvalKey(keyID);

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
		this->rescaleInternal();
	}
	if constexpr (PRINT)
		std::cout << "End mult" << std::endl;
	if constexpr (PRINT)
		CudaCheckErrorMod;

	Out(KEYSWITCH, " finish ");
}

void Ciphertext::multMutable(Ciphertext& b, bool rescale, const bool moddown) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	assert(keyID == b.keyID);
	if (cc.rescaleTechnique == FIXEDAUTO || cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {

		if (c0.isModUp() || c1.isModUp() || b.c0.isModUp()) {
			assert(getLevel() == b.getLevel());
			assert(NoiseLevel == b.NoiseLevel);
		}

		if (!adjustForMult(b)) {
			if (b.adjustForMult(*this))
				mult(b, rescale, moddown);
			else
				assert(false);
			return;
		}
	} else if (cc.rescaleTechnique == FIXEDMANUAL && getLevel() > b.getLevel()) {
		dropToLevel(b.getLevel(), true);
	}
	assert(NoiseLevel == 1);
	assert(NoiseLevel == b.NoiseLevel);
	mult(b, rescale, moddown);
}

void Ciphertext::square(bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {

		if (c0.isModUp() || c1.isModUp()) {
			assert(NoiseLevel == 1);
		}

		if (NoiseLevel == 2)
			this->rescaleInternal();
	}
	assert(this->NoiseLevel == 1);
	op_count[OPS::SQUARE]++;

	// TODO: optimize
	if (cc.GPUid.size() == 1) {
		this->mult(*this, rescale);
	} else {
		this->mult(*this, rescale);
	}
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
	NoiseFactor *= getLevel() == cc.L + 1 ? cc.specialPrime[0].p : cc.param.ScalingFactorReal.at(c0.getLevel());
	if (rescale && cc.rescaleTechnique == FIXEDAUTO) {
		this->rescaleInternal();
	}
}

void Ciphertext::multScalar(const double c, bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT || cc.rescaleTechnique == FIXEDAUTO) {

		if (c0.isModUp() || c1.isModUp()) {
			assert(NoiseLevel == 1);
		}

		if (NoiseLevel == 2)
			this->rescaleInternal();
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

void Ciphertext::automorph(const int index, const int br) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	auto& aux0 = cc.getModdownAux(0);
	auto& aux1 = cc.getModdownAux(1);
	aux0.copy(c0);
	aux1.copy(c1);
	c0.automorph(index, br, &aux0);
	c1.automorph(index, br, &aux1);
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

void Ciphertext::rotate(const int index__, const bool moddown) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::ROTATE]++;
	int index = normalyzeIndex(index__);

	assert(index != 0);
	//constexpr bool PRINT = false;
	assert(!c1.isModUp());
	{
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
				int32_t actual_index;
				auto& ksk = cc.GetRotationKey(index, keyID, slots, actual_index);
				ksk_a.push_back(&ksk.a);
				ksk_b.push_back(&ksk.b);
				index_.push_back(actual_index);
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
			int actual_index;
			auto& ksk = cc.GetRotationKey(index, c.keyID, slots, actual_index);
			ksk_a.push_back(&ksk.a);
			ksk_b.push_back(&ksk.b);
			index_.push_back(actual_index);
		}
	}

	dropToLevel(c.getLevel(), true);
	c0.grow(c.c0.getLevel());
	c1.grow(c.c1.getLevel());

	in1.hoistedRotationFused(index_, c0_out, c1_out, ksk_a, ksk_b, c.c0, in1);

	if (1) {
		modDown(false);
	}

	this->copyMetadata(c);
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

	bool grow_full = false;
	for (auto& i : results) {
		i->growToLevel(grow_full ? cc.L : this->c0.getLevel());
		i->dropToLevel(getLevel(), true);
		if (ext)
			i->extend(false);

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
					int actual_index;
					RNSPoly& aux0 = results[i]->c1.dotKSKInPlaceFrom(cc.getKeySwitchAux(), cc.GetRotationKey(indexes[i], keyID, slots, actual_index), &c1);
					// results[i]->c0.dropToLevel(getLevel());
					// results[i]->c1.dropToLevel(getLevel());
					if (!ext)
						results[i]->c1.moddown(true, false, 0);
					results[i]->c1.automorph(actual_index, 1);

					if (!ext)
						aux0.moddown(true, false, 1);

					// results[i]->c0.generateSpecialLimbs(true);
					results[i]->c0.add(c0, aux0);
					results[i]->c0.automorph(actual_index, 1);
					// if (!ext)
					//     results[i]->c0.moddown(true, false);

					results[i]->keyID       = keyID;
					results[i]->NoiseLevel  = NoiseLevel;
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
					int actual_index;
					RNSPoly& aux0 = in.dotKSKInPlace(cc.GetRotationKey(indexes[i], keyID, slots, actual_index), &c1);
					// results[i]->c0.dropToLevel(getLevel());
					// results[i]->c1.dropToLevel(getLevel());
					if (!ext) {
						in.moddown(true, false, 0);
					}
					// std::cout << "in ismodup: " << in.isModUp() << std::endl;
					results[i]->c1.automorph(actual_index, 1, &in);
					// std::cout << "results[i] c1 ismodup: " << results[i]->c1.isModUp() << std::endl;
					if (!ext) {
						aux0.moddown(true, false, 1);
					}
					// std::cout << "aux0 ismodup: " << aux0.isModUp() << std::endl;
					// std::cout << "c0 ismodup: " << c0.isModUp() << std::endl;
					results[i]->c0.add(c0, aux0);
					// std::cout << "results[i] c0 ismodup: " << results[i]->c0.isModUp() << std::endl;
					results[i]->c0.automorph(actual_index, 1, nullptr);
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
				int actual_index;
				auto& ksk = cc.GetRotationKey(indexes[i], keyID, slots, actual_index);
				ksk_a.push_back(&ksk.a);
				ksk_b.push_back(&ksk.b);
				index.push_back(actual_index);

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

void Ciphertext::multMutable(Ciphertext& b, Ciphertext& c, bool rescale) {
	if (this == &b && this == &c) {
		this->square(rescale);
	} else if (this == &b) {
		this->multMutable(c, rescale);
	} else if (this == &c) {
		this->multMutable(b, rescale);
	} else {
		if (b.getLevel() <= c.getLevel()) {
			this->copy(b);
			this->multMutable(c, rescale);
		} else {
			this->copy(c);
			this->multMutable(b, rescale);
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

void Ciphertext::dropToLevel(const int level, bool skip_adjust) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	if (c0.getLevel() > level) {
		assert(c1.getLevel() > level);
		if (!skip_adjust && (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT)) {
			assert(NoiseLevel == 1 || NoiseLevel == 2);
			bool ok = adjustScaleAndLevel(this->NoiseLevel,
			                              level,
			                              this->NoiseLevel == 1 ? cc.param.ScalingFactorReal[level] : cc.param.ScalingFactorRealBig[level]);
			assert(ok);
			(void)ok;
		} else {
			c0.dropToLevel(level);
			c1.dropToLevel(level);
		}
	}
}

int32_t Ciphertext::getLevel() const {
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

	if constexpr (1) {
		if (static_cast<int32_t>(this->getLevel()) == -1) {
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
			}
			assert(ctxs[0]->keyID == ctxs[i]->keyID);
		}

		std::vector<uint64_t> elem(MAXP * n);

		// #pragma omp parallel for
		// double scalingFactor;
		for (size_t i = 0; i < n; ++i) {
			auto aux = cc.ElemForEvalMult(c0.getLevel(), weights[i], ctxs[i]->getLevel());
			for (size_t j          = 0; j < aux.size(); ++j)
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
		for (uint32_t i = 1; i < n; ++i) {
			assert(this->keyID == ctxs[i]->keyID);
			slots = std::max(slots, ctxs[i]->slots);
		}
		this->NoiseLevel  = 2;
		this->NoiseFactor = cc.param.ScalingFactorRealBig.at(getLevel());
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

void Ciphertext::evalPartialLinearWSumWithBias(const std::vector<Ciphertext*>& ctxs,
                                               std::vector<double> weights,
                                               double bias,
                                               int limit,
                                               int target_level,
                                               int rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::WSUM]++;

	if (0 == limit)
		limit = ctxs.size();
	op_count[OPS::WSUMINPUTS] += limit;

	if constexpr (1) {
		assert(target_level != -1);
		this->growToLevel(target_level);
		this->dropToLevel(target_level, true);

		for (int i = 0; i < limit; ++i) {
			if (cc.rescaleTechnique == FIXEDMANUAL) {
				assert(ctxs[i]->NoiseLevel == 1);
				assert(getLevel() <= ctxs[i]->getLevel());
			} else {
				assert(ctxs[i]->NoiseLevel == 1);
			}
			assert(ctxs[0]->keyID == ctxs[i]->keyID);
		}

		std::vector<uint64_t> elem;
		std::vector<Ciphertext*> ctxs_trim;
		// #pragma omp parallel for
		// double scalingFactor;
		int idx = 0;
		for (int i = 1; i <= limit; ++i) {
			if (lbcrypto::IsNotEqualZero(weights[i])) {
				auto aux = cc.ElemForEvalMult(target_level, weights[i], ctxs[i - 1]->getLevel());
				elem.resize(elem.size() + MAXP);

				for (size_t j            = 0; j < aux.size(); ++j)
					elem[idx * MAXP + j] = aux[j];
				idx += 1;
				ctxs_trim.push_back(ctxs[i - 1]);
			}
		}
		bool with_bias;

		if ((with_bias = lbcrypto::IsNotEqualZero(bias))) {
			auto aux = cc.ElemForEvalAddOrSub(target_level, std::abs(bias), 2);
			if (bias < 0.0) {
				for (auto i = 0u; i < aux.size(); ++i) {
					aux[i] = cc.prime[i].p - aux[i];
				}
			}
			elem.resize(elem.size() + MAXP);
			for (size_t j            = 0; j < aux.size(); ++j)
				elem[idx * MAXP + j] = aux[j];
			idx += 1;
		}

		std::vector<const RNSPoly*> c0s(ctxs_trim.size()), c1s(ctxs_trim.size());

		for (size_t i = 0; i < ctxs_trim.size(); ++i) {
			c0s[i] = &ctxs_trim[i]->c0;
			c1s[i] = &ctxs_trim[i]->c1;
		}
		c0.evalLinearWSum(ctxs_trim.size(), c0s, elem, with_bias);
		c1.evalLinearWSum(ctxs_trim.size(), c1s, elem);

		// this->copyMetadata(*ctxs[0]);

		this->slots = ctxs[0]->slots;
		this->keyID = ctxs[0]->keyID;
		for (uint32_t i = 0; i < ctxs_trim.size(); ++i) {
			assert(this->keyID == ctxs[i]->keyID);
			slots = std::max(slots, ctxs[i]->slots);
		}
		this->NoiseLevel  = 2;
		this->NoiseFactor = cc.param.ScalingFactorRealBig.at(getLevel());

		if (rescale)
			this->rescale();

	}
}

struct Objective {
	int n, target_level;
	std::vector<double> weights;
	Ciphertext* ctxt;
};

void evalMultiplePartialLinearWSumWithBiasInternal(const std::vector<Ciphertext*>& ctxs, const std::vector<Objective>& out_ctxt) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	Context cc_ = ctxs[0]->cc_;
	CKKS::SetCurrentContext(cc_);

	int max_n = 0;
	for (auto j = 0u; j < out_ctxt.size(); ++j) {
		assert(out_ctxt[j].target_level != -1);
		max_n = std::max(max_n, out_ctxt[j].n);
	}

	for (auto i = 0u; i < ctxs.size(); ++i) {
		for (auto j = 0u; j < out_ctxt.size(); ++j) {
			if (cc_->rescaleTechnique == FIXEDMANUAL) {
				assert(ctxs[i]->NoiseLevel == 1);
				assert(out_ctxt[j].ctxt->getLevel() <= ctxs[i]->getLevel());
			} else {
				assert(ctxs[i]->NoiseLevel == 1);
			}
			assert(ctxs[0]->keyID == ctxs[i]->keyID);
		}
	}

	std::vector<uint64_t> elem;
	std::vector<Ciphertext*> ctxs_trim;
	// #pragma omp parallel for
	// double scalingFactor;
	for (int i = 1; i <= max_n; ++i) {
		ctxs_trim.push_back(ctxs[i - 1]);
	}
	int idx = 0;
	for (auto j = 0u; j < out_ctxt.size(); ++j) {
		for (int i = 1; i <= max_n; ++i) {
			elem.resize(elem.size() + MAXP);
			if ((int)out_ctxt[j].weights.size() > i) {
				if (lbcrypto::IsNotEqualZero(out_ctxt[j].weights[i])) {
					auto aux = cc_->ElemForEvalMult(out_ctxt[j].target_level, out_ctxt[j].weights[i], ctxs[i - 1]->getLevel());
					for (size_t j            = 0; j < aux.size(); ++j)
						elem[idx * MAXP + j] = aux[j];
				}
			}
			idx += 1;
		}

		elem.resize(elem.size() + MAXP);
		if (lbcrypto::IsNotEqualZero(out_ctxt[j].weights[0])) {
			auto aux = cc_->ElemForEvalAddOrSub(out_ctxt[j].target_level, std::abs(out_ctxt[j].weights[0]), 2);
			if (out_ctxt[j].weights[0] < 0.0) {
				for (auto i = 0u; i < aux.size(); ++i) {
					aux[i] = cc_->prime[i].p - aux[i];
				}
			}
			for (size_t j            = 0; j < aux.size(); ++j)
				elem[idx * MAXP + j] = aux[j];
		}
		idx += 1;
	}

	std::vector<const RNSPoly*> c0s(ctxs_trim.size()), c1s(ctxs_trim.size());
	std::vector<RNSPoly*> c0s_out(out_ctxt.size()), c1s_out(out_ctxt.size());

	for (size_t i = 0; i < ctxs_trim.size(); ++i) {
		c0s[i] = &ctxs_trim[i]->c0;
		c1s[i] = &ctxs_trim[i]->c1;
	}
	for (size_t i = 0; i < out_ctxt.size(); ++i) {
		c0s_out[i] = &out_ctxt[i].ctxt->c0;
		c1s_out[i] = &out_ctxt[i].ctxt->c1;
	}
	RNSPoly::evalLinearWSumMultiple(max_n, c0s, c0s_out, elem, false);
	RNSPoly::evalLinearWSumMultiple(max_n, c1s, c1s_out, elem, true);

	// this->copyMetadata(*ctxs[0]);

	for (auto j = 0u; j < out_ctxt.size(); ++j) {
		Ciphertext& c = *out_ctxt[j].ctxt;
		c.slots       = ctxs[0]->slots;
		c.keyID       = ctxs[0]->keyID;
		for (uint32_t i = 0; i < ctxs_trim.size(); ++i) {
			assert(c.keyID == ctxs[i]->keyID);
			c.slots = std::max(c.slots, ctxs[i]->slots);
		}
		c.NoiseLevel  = 2;
		c.NoiseFactor = cc_->param.ScalingFactorRealBig.at(c.getLevel());
	}
}

std::vector<Ciphertext> Ciphertext::evalMultiplePartialLinearWSumWithBias(const std::vector<Ciphertext*>& ctxs,
                                                                          std::vector<std::vector<double>> weights,
                                                                          std::vector<int> target_level,
                                                                          int rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	Context cc_ = ctxs[0]->cc_;
	CKKS::SetCurrentContext(cc_);
	op_count[OPS::WSUM]++;

	int n_out = weights.size();
	assert(weights.size() == target_level.size());
	std::vector<Ciphertext> ctxs_result;
	for (int i = 0; i < n_out; ++i) {
		ctxs_result.emplace_back(cc_);
		ctxs_result.back().growToLevel(target_level[i]);
	}

	std::vector<Objective> out_ctxt(n_out);
	for (int i = 0; i < n_out; ++i) {
		auto weights_trimmed = weights[i];
		while (!lbcrypto::IsNotEqualZero(weights_trimmed.back()))
			weights_trimmed.pop_back();

		op_count[OPS::WSUMINPUTS] += weights_trimmed.size();
		out_ctxt[i] = { .n = (int)weights_trimmed.size() - 1, .target_level = target_level[i], .weights = weights_trimmed, .ctxt = &ctxs_result[i] };
	}

	std::sort(out_ctxt.begin(),
	          out_ctxt.end(),
	          [](const Objective& lhs, const Objective& rhs) { return lhs.n < rhs.n || (lhs.n == rhs.n && lhs.ctxt < rhs.ctxt); });

	constexpr int MAX_OUTPUTS = 8;

	int groups = (n_out + MAX_OUTPUTS - 1) / MAX_OUTPUTS;
	while (n_out > 0) {
		std::vector<Objective> out_ctxt_group;
		int group_size = (n_out + groups - 1) / groups;
		groups -= 1;
		n_out -= group_size;

		for (int i = 0; i < group_size; ++i) {
			out_ctxt_group.emplace_back(out_ctxt.back());
			out_ctxt.pop_back();
		}

		evalMultiplePartialLinearWSumWithBiasInternal(ctxs, out_ctxt_group);
	}

	if (rescale) {
		for (auto& i : ctxs_result)
			i.rescale();
	}

	return ctxs_result;
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

void Ciphertext::addMutable(Ciphertext& b, Ciphertext& c) {
	if (this == &b && this == &c) {
		this->addMutable(c);
	} else if (this == &b) {
		this->addMutable(c);
	} else if (this == &c) {
		this->addMutable(b);
	} else {
		if (b.getLevel() <= c.getLevel()) {
			this->copy(b);
			this->addMutable(c);
		} else {
			this->copy(c);
			this->addMutable(b);
		}
	}
}

void Ciphertext::addMutable(const Ciphertext& b, Ciphertext& c) {
	if (this == &b && this == &c) {
		this->addMutable(*this);
	} else if (this == &b) {
		this->addMutable(c);
	} else if (this == &c) {
		this->add(b);
	} else {
		if (b.getLevel() <= c.getLevel()) {
			this->copy(b);
			this->addMutable(c);
		} else {
			this->copy(c);
			this->add(b);
		}
	}
}

void Ciphertext::addMutable(Ciphertext& b, const Ciphertext& c) {
	addMutable(c, b);
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
	if (this == &ciphertext) {
		return;
	}
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

void Ciphertext::multPtMutable(const Ciphertext& c, Plaintext& b, bool rescale) {
	this->copy(c);
	multPtMutable(b, rescale);
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
		this->rescaleInternal();
	}
}

void Ciphertext::addPt(const Ciphertext& ciphertext, const Plaintext& plaintext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	this->copy(ciphertext);
	this->addPt(plaintext);
}

void Ciphertext::addPtMutable(const Ciphertext& ciphertext, Plaintext& plaintext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	this->copy(ciphertext);
	this->addPtMutable(plaintext);
}

void Ciphertext::reinterpretContext(const Ciphertext& ciphertext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	assert(cc_.get() != ciphertext.cc_.get());
	assert(!ciphertext.c0.isModUp());
	assert(!ciphertext.c1.isModUp());
	assert(ciphertext.getLevel() <= cc.L);

	for (int32_t i = 0; i <= ciphertext.getLevel(); i++) {
		assert(ciphertext.cc.prime.at(i).p == cc.prime.at(i).p);
		assert(ciphertext.cc.prime.at(i).type == cc.prime.at(i).type);
	}

	CKKS::SetCurrentContext(cc_);
	this->copy(ciphertext);

	assert(ciphertext.NoiseLevel <= 2);
	{
		// This ensures the different computation made for scaling factors in FLEXIBLE modes does not break the code
		this->NoiseFactor = this->NoiseLevel == 1 ?
			cc.param.ScalingFactorReal[ciphertext.getLevel()] :
			cc.param.ScalingFactorRealBig[ciphertext.getLevel()];
	}
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

void Ciphertext::subMutable(const Ciphertext& ciphertext, Ciphertext& ciphertext1) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	assert(ciphertext.getLevel() <= ciphertext1.getLevel());
	this->copy(ciphertext);
	this->subMutable(ciphertext1);
}

bool Ciphertext::adjustScaleAndLevel(const int scaleDegree, const int level, const double scaling_factor) {
	assert(scaleDegree == 2 ? std::abs(cc.param.ScalingFactorReal[level - 1] * cc.param.ModReduceFactor[level] - cc.param.ScalingFactorRealBig[level]) /
		(cc.param.ScalingFactorReal[level - 1] * cc.param.ModReduceFactor[level] + cc.param.ScalingFactorRealBig[level]) <
		1e-11 :
		true);
	assert(scaleDegree < 3 ? scaling_factor == (scaleDegree == 1 ? cc.param.ScalingFactorReal[level] : (cc.param.ScalingFactorRealBig[level])) : true);

	uint32_t c1lvl   = getLevel();
	uint32_t c2lvl   = level;
	uint32_t c1depth = this->NoiseLevel;
	uint32_t c2depth = scaleDegree;
	auto sizeQl1     = c1lvl + 1;
	// auto sizeQl2 = c2lvl + 1;

	if (c1lvl > c2lvl) {
		if (c1depth == 2) {
			if (c2depth == 2) {
				// FIDESlib bit-compat: mirror OpenFHE AdjustLevelsAndDepthInPlace exactly --
				// same floating-point evaluation order for the adjustment factor, the
				// mod-reduce factor of the CURRENT top level, and rescale before dropping.
				double scf1 = NoiseFactor;
				double scf2 = scaling_factor;
				double scf  = cc.param.ScalingFactorReal[c1lvl]; // cryptoParams->GetScalingFactorReal(c1lvl);
				double q1   = cc.param.ModReduceFactor[c1lvl];   // cryptoParams->GetModReduceFactor(sizeQl1 - 1);
				multScalarNoPrecheck(scf2 / scf1 * q1 / scf);
				NoiseFactor = cc.param.ScalingFactorRealBig[getLevel() - 1] * cc.param.ModReduceFactor[getLevel()];
				rescaleInternal();
				if (getLevel() > static_cast<int32_t>(c2lvl)) {
					this->dropToLevel(c2lvl, true);
				}
				NoiseFactor = scaling_factor;
			} else {
				if (c1lvl - 1 == c2lvl) {
					rescaleInternal();
				} else {
					double scf1 = NoiseFactor;
					double scf2 = cc.param.ScalingFactorRealBig[c2lvl + 1]; // cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
					double scf  = cc.param.ScalingFactorReal[c1lvl];        // cryptoParams->GetScalingFactorReal(c1lvl);
					double q1   = cc.param.ModReduceFactor[sizeQl1 - 1];    // cryptoParams->GetModReduceFactor(sizeQl1 - 1);
					multScalarNoPrecheck(scf2 / scf1 * q1 / scf);
					NoiseFactor = cc.param.ScalingFactorRealBig[this->getLevel() - 1] * cc.param.ModReduceFactor[c1lvl];
					rescaleInternal();
					if (getLevel() - 1 > static_cast<int32_t>(c2lvl)) {
						this->dropToLevel(c2lvl + 1, true);
						// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 2);
					}
					NoiseFactor = cc.param.ScalingFactorRealBig[this->getLevel()];
					// NoiseFactor *= scf2 / scf1 * q1 / scf;
					rescaleInternal();
					// assert(std::abs((NoiseFactor * scf2 / scf1 * q1 / scf - scaling_factor) / scaling_factor) < 0.001);

					NoiseFactor = scaling_factor;
				}
			}
		} else {
			if (c2depth == 2) {
				double scf1 = NoiseFactor;
				double scf2 = scaling_factor;
				double scf  = cc.param.ScalingFactorReal[c1lvl]; // cryptoParams->GetScalingFactorReal(c1lvl);
				multScalarNoPrecheck(scf2 / scf1 / scf);
				this->dropToLevel(c2lvl, true);
				// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl);
				assert(std::abs((NoiseFactor * scf2 / scf1 / scf - scaling_factor) / scaling_factor) < 0.001);
				NoiseFactor = scf2;
			} else {
				double scf1 = NoiseFactor;
				double scf2 = cc.param.ScalingFactorRealBig[c2lvl + 1]; // cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
				double scf  = cc.param.ScalingFactorReal[c1lvl];        // cryptoParams->GetScalingFactorReal(c1lvl);
				multScalarNoPrecheck(scf2 / scf1 / scf);
				if (c1lvl - 1 > c2lvl) {
					this->dropToLevel(c2lvl + 1, true);
					// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 1);
				}
				NoiseFactor *= scf2 / scf1 / scf;
				rescaleInternal();
				// assert(std::abs((NoiseFactor * scf2 / scf1 / scf - scaling_factor) / scaling_factor) < 0.001);
				NoiseFactor = scaling_factor;
			}
		}
		assert(
			scaleDegree < 3 ? this->NoiseFactor == (NoiseLevel == 1 ? cc.param.ScalingFactorReal[level] : (cc.param.ScalingFactorRealBig[level])) :
			true);
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
				rescaleInternal();
			} else if (b.NoiseLevel == 2 && NoiseLevel == 1) {
				this->multScalar(1.0);
			}
			return true;
		} else if (b.NoiseLevel == 1 && NoiseLevel == 2) {
			rescaleInternal();
			return true;
		} else if (NoiseLevel == 1 && b.NoiseLevel == 2) {
			return false;
		} else {
			return true;
		}
	} else if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		return adjustScaleAndLevel(b.NoiseLevel, b.getLevel(), b.NoiseFactor);
	}
	assert("This never happens" == nullptr);
	return false;
}

bool Ciphertext::adjustForMult(const Ciphertext& ciphertext) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	if (adjustForAddOrSub(ciphertext)) {
		if (NoiseLevel == 2)
			rescaleInternal();
		if (ciphertext.NoiseLevel == 2)
			return false;
		else
			return true;
	} else {
		if (NoiseLevel == 2)
			rescaleInternal();
		return false;
	}
}

bool Ciphertext::hasSameScalingFactor(const Plaintext& b) const {
	return NoiseFactor > b.NoiseFactor * (1 - 1e-9) && NoiseFactor < b.NoiseFactor * (1 + 1e-9);
}

void Ciphertext::negate() {
	std::vector<uint64_t> negOne(getLevel() + 1);
	for (size_t i = 0; i < negOne.size(); ++i)
		negOne[i] = cc.prime[i].p - 1;
	c0.multScalar(negOne);
	c1.multScalar(negOne);
}

void Ciphertext::negate(const Ciphertext& ciphertext) {
	this->copy(ciphertext);
	negate();
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
	this->dropToLevel(a[0]->getLevel(), true);

	std::vector<const RNSPoly*> c0s(a.size(), nullptr), c1s(a.size(), nullptr), d0s(a.size(), nullptr), d1s(a.size(), nullptr);

	for (size_t i = 0; i < a.size(); ++i) {
		assert(this->cc_ == a[i]->cc_);
		assert(this->cc_ == b[i]->cc_);
		// assert(a[0]->NoiseFactor == a[i]->NoiseFactor);
		// assert(a[0]->NoiseFactor == b[i]->NoiseFactor);
		assert(a[0]->keyID == a[i]->keyID);
		assert(a[0]->keyID == b[i]->keyID);
		// assert(a[i]->NoiseLevel == 1);
		// assert(b[i]->NoiseLevel == 1);
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

	RNSPoly& c2 = c0.dotProduct(c1, cc.GetEvalKey(a[0]->keyID).a, cc.GetEvalKey(a[0]->keyID).b, c0s, c1s, d0s, d1s, in_ext, ext);

	if (c2.isModUp()) {
		c2.moddown(true, false, 0);
	}
	RNSPoly& aux = MGPUkeySwitchCore(c2, cc.GetEvalKey(a[0]->keyID), !(ext || in_ext));

	c0.add(aux);
	c1.add(c2);

	if (!ext && in_ext) {
		modDown();
	}

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
uint32_t N                   = elemParams->GetRingDimension();
uint32_t M                   = 2 * N;

	NativePoly monomial(paramsNative, Format::COEFFICIENT, true);

	uint32_t powerReduced = power % M;
	uint32_t index        = power % N;
	monomial[index]    = powerReduced < N ? NativeInteger(1) : paramsNative->GetModulus() - NativeInteger(1);

	DCRTPoly monomialDCRT(elemParams, Format::COEFFICIENT, true);
	monomialDCRT = monomial;
	monomialDCRT.SetFormat(Format::EVALUATION);

	for (uint32_t i = 0; i < ciphertext->NumberCiphertextElements(); i++) {
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
