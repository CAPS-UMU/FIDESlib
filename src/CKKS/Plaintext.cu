//
// Created by carlosad on 25/04/24.
//

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Plaintext.cuh"
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
// constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc = std::source_location;
// constexpr int PREFIX_SIZE = 23;
#endif

namespace FIDESlib::CKKS {

void Plaintext::copyMetadata(const Plaintext& a) {
	assert(this->c0.getLevel() == a.c0.getLevel());
	this->slots		  = a.slots;
	this->NoiseLevel  = a.NoiseLevel;
	this->NoiseFactor = a.NoiseFactor;
}

void Plaintext::addMetadata(const Plaintext& a, const Plaintext& b) {
	assert(this->c0.getLevel() == a.c0.getLevel());
	if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
		assert(a.c0.getLevel() == b.c0.getLevel());
		assert(a.NoiseFactor == b.NoiseFactor);
	}
	this->slots = std::max(a.slots, b.slots);
	assert(a.NoiseLevel == b.NoiseLevel);
	this->NoiseLevel  = a.NoiseLevel;
	this->NoiseFactor = a.NoiseFactor;
}

void Plaintext::multMetadata(const Plaintext& a, const Plaintext& b) {
	assert(this->c0.getLevel() == a.c0.getLevel());
	if (RESCALE_TECHNIQUE::FLEXIBLEAUTO == cc.rescaleTechnique || RESCALE_TECHNIQUE::FLEXIBLEAUTOEXT == cc.rescaleTechnique) {
		assert(a.c0.getLevel() == b.c0.getLevel());
		assert(a.NoiseFactor == b.NoiseFactor);
	}
	this->slots = std::max(a.slots, b.slots);
	assert(a.NoiseLevel == b.NoiseLevel);
	this->NoiseLevel  = a.NoiseLevel + b.NoiseLevel;
	this->NoiseFactor = a.NoiseFactor * b.NoiseFactor;
}

Plaintext::Plaintext(Context& cc)
: my_range(loc, LIFETIME), cc_((assert(cc != nullptr), CudaNvtxStart(std::string{ sc::current().function_name() }.substr()), cc)), cc(*cc_),
  c0(this->cc, -1, false, true) {
	CudaNvtxStop();
}

Plaintext::Plaintext(Context& cc, const RawPlainText& raw)
: my_range(loc, LIFETIME), cc_((assert(cc != nullptr), CudaNvtxStart(std::string{ sc::current().function_name() }.substr()), cc)), cc(*cc_),
  c0(this->cc, -1, false, true) {
	load(raw);
	CudaNvtxStop();
}

void Plaintext::load(const RawPlainText& raw) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	c0.loadConstant(raw.sub_0, raw.moduli);

	/*
	cudaDeviceSynchronize();
	for (auto& i : c0.GPU) {
		for (auto& j : i.limb) {
			SWITCH(j, printThisLimb(1));
		}
	}
	std::cout << std::endl;
	cudaDeviceSynchronize();
	*/

	NoiseFactor = raw.Noise;
	NoiseLevel	= raw.NoiseLevel;
	slots		= raw.slots;
}

void Plaintext::store(RawPlainText& raw) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	cudaDeviceSynchronize();

	raw.numRes = c0.getLevel() + 1;
	raw.sub_0.resize(raw.numRes);
	c0.store(raw.sub_0);
	raw.N = cc.N;
	c0.sync();

	raw.Noise	   = NoiseFactor;
	raw.NoiseLevel = NoiseLevel;
	raw.slots	   = slots;
	cudaDeviceSynchronize();
}

void Plaintext::moddown() {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	c0.moddown(true, true);
}

bool Plaintext::adjustScaleAndLevel(const int scaleDegree, const int level, const double scaling_factor) {
	assert(scaleDegree == 2 ? cc.param.ScalingFactorReal[level - 1] * cc.param.ModReduceFactor[level] == cc.param.ScalingFactorRealBig[level] : true);
	assert(scaleDegree < 3 ? scaling_factor == (scaleDegree == 1 ? cc.param.ScalingFactorReal[level] : (cc.param.ScalingFactorRealBig[level])) : true);

	usint c1lvl	  = c0.getLevel();
	usint c2lvl	  = level;
	usint c1depth = this->NoiseLevel;
	usint c2depth = scaleDegree;
	auto sizeQl1  = c1lvl + 1;
	// auto sizeQl2 = c2lvl + 1;

	if (c1lvl > c2lvl) {
		if (c1depth == 2) {
			if (c2depth == 2) {
				double scf1 = NoiseFactor;
				double scf2 = scaling_factor;
				double scf	= cc.param.ScalingFactorReal[c1lvl];	 // cryptoParams->GetScalingFactorReal(c1lvl);
				double q1	= cc.param.ModReduceFactor[sizeQl1 - 1]; // cryptoParams->GetModReduceFactor(sizeQl1 - 1);

				multScalar(scf2 / scf1 * q1 / scf);
				rescale();
				if (c0.getLevel() > static_cast<int32_t>(c2lvl)) {
					this->dropToLevel(c2lvl, true);
				}

				assert(std::abs((NoiseFactor * scf2 / scf1 * q1 / scf - scaling_factor) / scaling_factor) < 0.001);
				NoiseFactor = scaling_factor;
			} else {
				if (c1lvl - 1 == c2lvl) {
					rescale();
				} else {
					double scf1 = NoiseFactor;
					double scf2 = cc.param.ScalingFactorRealBig[c2lvl + 1]; // cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
					double scf	= cc.param.ScalingFactorReal[c1lvl];		// cryptoParams->GetScalingFactorReal(c1lvl);
					double q1	= cc.param.ModReduceFactor[sizeQl1 - 1];	// cryptoParams->GetModReduceFactor(sizeQl1 - 1);
					multScalar(scf2 / scf1 * q1 / scf);
					// NoiseFactor *= scf2 / scf1 * q1 / scf;
					// NoiseFactor /= cc.param.ScalingFactorReal[this->c0.getLevel()];
					rescale();
					if (c0.getLevel() - 1 > static_cast<int32_t>(c2lvl)) {
						this->dropToLevel(c2lvl + 1, true);
						// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 2);
					}
					rescale();
					assert(std::abs((NoiseFactor * scf2 / scf1 * q1 / scf - scaling_factor) / scaling_factor) < 0.001);

					// assert(NoiseFactor == scaling_factor);
					NoiseFactor = scaling_factor;
				}
			}
		} else {
			if (c2depth == 2) {
				double scf1 = NoiseFactor;
				double scf2 = scaling_factor;
				double scf	= cc.param.ScalingFactorReal[c1lvl]; // cryptoParams->GetScalingFactorReal(c1lvl);
				multScalar(scf2 / scf1 / scf);
				this->dropToLevel(c2lvl, true);
				// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl);
				assert(std::abs((NoiseFactor * scf2 / scf1 / scf - scaling_factor) / scaling_factor) < 0.001);
				NoiseFactor = scf2;
			} else {
				double scf1 = NoiseFactor;
				double scf2 = cc.param.ScalingFactorRealBig[c2lvl + 1]; // cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
				double scf	= cc.param.ScalingFactorReal[c1lvl];		// cryptoParams->GetScalingFactorReal(c1lvl);
				multScalar(scf2 / scf1 / scf);
				if (c1lvl - 1 > c2lvl) {
					this->dropToLevel(c2lvl + 1, true);
					// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 1);
				}
				rescale();
				assert(std::abs((NoiseFactor * scf2 / scf1 / scf - scaling_factor) / scaling_factor) < 0.001);
				NoiseFactor = scaling_factor;
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

bool Plaintext::adjustPlaintextToCiphertext(const Plaintext& p, const Ciphertext& c) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	// constexpr bool PRINT = false;

	if (cc.rescaleTechnique == FIXEDAUTO) {
		if (p.c0.getLevel() - p.NoiseLevel > c.getLevel() - c.NoiseLevel) {
			this->copy(p);
			if (c.NoiseLevel == 1 && NoiseLevel == 2) {
				this->c0.dropToLevel(c.getLevel() + 1);
				rescale();
			} else {
				this->c0.dropToLevel(c.getLevel());
			}
			return true;
		} else if (c.NoiseLevel == 1 && p.NoiseLevel == 2) {
			this->copy(p);
			rescale();
			return true;
		} else if (p.NoiseLevel == 1 && c.NoiseLevel == 2) {
			return false;
		} else {
			this->copy(p);
			return true;
		}
	}
	if (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT) {
		if (p.c0.getLevel() < c.getLevel()) {
			return false;
		}
		this->copy(p);
		return adjustScaleAndLevel(c.NoiseLevel, c.getLevel(), c.NoiseFactor);
		/* TODO remove if above code verified
		usint c1lvl	  = p.c0.getLevel();
		usint c2lvl	  = c.getLevel();
		usint c1depth = p.NoiseLevel;
		usint c2depth = c.NoiseLevel;
		auto sizeQl1  = c1lvl + 1;
		auto sizeQl2  = c2lvl + 1;

		if (c1lvl > c2lvl) {
			this->copy(p);
			if (c1depth == 2) {
				if (c2depth == 2) {
					double scf1 = NoiseFactor;
					double scf2 = c.NoiseFactor;
					double scf	= cc.param.ScalingFactorReal[c1lvl];	 // cryptoParams->GetScalingFactorReal(c1lvl);
					double q1	= cc.param.ModReduceFactor[sizeQl1 - 1]; // cryptoParams->GetModReduceFactor(sizeQl1 - 1);
					multScalar(scf2 / scf1 * q1 / scf, false);
					rescale();
					if (c1lvl > c2lvl) {
						this->c0.dropToLevel(c2lvl);
						// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 1);
					}
					NoiseFactor *= scf2 / scf1 * q1 / scf;
					assert(NoiseFactor == c.NoiseFactor);
					NoiseFactor = c.NoiseFactor;
				} else {
					if (c1lvl - 1 == c2lvl) {
						rescale();
					} else {
						double scf1 = NoiseFactor;
						double scf2 = cc.param.ScalingFactorRealBig[c2lvl + 1]; // cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
						double scf	= cc.param.ScalingFactorReal[c1lvl];		// cryptoParams->GetScalingFactorReal(c1lvl);
						double q1	= cc.param.ModReduceFactor[sizeQl1 - 1];	// cryptoParams->GetModReduceFactor(sizeQl1 - 1);
						multScalar(scf2 / scf1 * q1 / scf, false);
						rescale();
						if (c1lvl - 2 > c2lvl) {
							this->c0.dropToLevel(c2lvl + 1);
							// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 2);
						}
						rescale();

						NoiseFactor = c.NoiseFactor;
					}
				}
			} else {
				if (c2depth == 2) {
					double scf1 = NoiseFactor;
					double scf2 = c.NoiseFactor;
					double scf	= cc.param.ScalingFactorReal[c1lvl]; // cryptoParams->GetScalingFactorReal(c1lvl);
					multScalar(scf2 / scf1 / scf, false);
					this->c0.dropToLevel(c2lvl);
					// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl);
					NoiseFactor = scf2;
				} else {
					if constexpr (PRINT)
						std::cout << "Adjusting plaintext with noiseDegree 1" << std::endl;
					double scf1 = NoiseFactor;
					double scf2 = cc.param.ScalingFactorRealBig[c2lvl + 1]; // cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
					double scf	= cc.param.ScalingFactorReal[c1lvl];		// cryptoParams->GetScalingFactorReal(c1lvl);
					if constexpr (PRINT)
						std::cout << "Scale adjustment: " << scf << std::endl;

					multScalar(scf2 / scf1 / scf, false);
					if (c1lvl - 1 > c2lvl) {
						if constexpr (PRINT)
							std::cout << "Dropping levels: " << c1lvl - c2lvl - 1 << std::endl;
						this->c0.dropToLevel(c2lvl + 1);
						// LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 1);
					}
					rescale();
					NoiseFactor = c.NoiseFactor;
				}
			}
			return true;
		} else if (c1lvl < c2lvl) {
			return false;
		} else {
			this->copy(p);
			if (c1depth < c2depth) {
				multScalar(1.0, false);
			} else if (c2depth < c1depth) {
				rescale();
			}
			return true;
		}
		*/
	}
	return false;
}

void Plaintext::copy(const Plaintext& p) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	this->c0.copy(p.c0);
	this->copyMetadata(p);
}

void Plaintext::multScalar(double c, bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	/*
	if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT ||
		cc.rescaleTechnique == Context::FIXEDAUTO) {
		if (NoiseLevel == 2)
			this->rescale();
	}
	assert(this->NoiseLevel == 1);
	*/
	auto elem = cc.ElemForEvalMult(c0.getLevel(), c);
	/*
	for (int i = 0; i < elem.size(); i++) {
		std::cout << elem[i] << " ";
	}
	std::cout << std::endl;
*/
	c0.multScalar(elem);

	if (rescale) {
		c0.rescale();
	}
	// Manage metadata
	NoiseLevel += 1;
	NoiseFactor *= cc.param.ScalingFactorReal.at(c0.getLevel() + rescale);
	if (rescale) {
		NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + rescale);
		NoiseLevel -= 1;
	}
}

void Plaintext::rotate_hoisted(const std::vector<int>& indexes, std::vector<Plaintext*>& results) {
	assert(indexes.size() == results.size() && "rotate_hoisted: mismatched indexes and results sizes");
	CKKS::SetCurrentContext(cc_);

	for (size_t i = 0; i < indexes.size(); ++i) {
		int index = indexes[i];
		if (index == 0) {
			results[i]->copy(*this);
		} else {

			// Copy and rotate

			results[i]->c0.grow(this->c0.getLevel());
			results[i]->c0.dropToLevel(this->c0.getLevel());
			results[i]->copyMetadata(*this);
			results[i]->c0.automorph(index, 1, &this->c0);
			// results[i]->copy(*this);
			// results[i]->automorph(index);
		}
	}
}

#if false
void Plaintext::multPt(const Plaintext& b, bool rescale) {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if (NoiseLevel == 2)
            this->rescale();
    }

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        // if (b.c0.getLevel() != this.getLevel() || b.NoiseLevel == 2 /*!hasSameScalingFactor(b)*/) {
        if (!hasSameScalingFactor(b)) {
            Plaintext b_(cc);
            if (NoiseLevel == 2)
                this->rescale();
            if (b_.NoiseLevel == 2)
                b_.rescale();
            multPt(b_, rescale);
            return;
        }
    }

    assert(NoiseLevel < 2);
    assert(b.NoiseLevel < 2);
    c0.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL);

    // Manage metadata
    NoiseLevel += b.NoiseLevel;
    NoiseFactor *= b.NoiseFactor;
    if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
        NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
        NoiseLevel -= 1;
    }
}

void Plaintext::addPt(const Plaintext& c) {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());

    // assert(NoiseLevel == b.NoiseLevel);
    c0.add(c.c0);
}

#endif

void Plaintext::rescale() {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);

	assert(this->NoiseLevel >= 2);
	/*
	std::cout << "Rescale plaintext, level" << c0.getLevel() << std::endl;
	for (auto& i : c0.GPU) {
		std::cout << i.limb.size() << " ";
	}
	std::cout << std::endl;
*/
	c0.rescale();

	// Manage metadata
	NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
	NoiseLevel -= 1;
}

void Plaintext::automorph(const int index) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	/*
	if (c0.isModUp()) {
		std::cout << "isModup plaintext automorph not implemented" << std::endl;
	}
	*/
	if (index != 0) {
		auto& aux = cc.getModdownAux(0);
		aux.setLevel(c0.getLevel());
		aux.automorph(index, 1, &c0);
		c0.copy(aux);
	}
}

void Plaintext::dropToLevel(const int level, bool skip_adjust) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc_);
	if (c0.getLevel() <= level)
		return;
	if (!skip_adjust && (cc.rescaleTechnique == FLEXIBLEAUTO || cc.rescaleTechnique == FLEXIBLEAUTOEXT)) {
		assert(NoiseLevel == 1 || NoiseLevel == 2);
		bool ok = adjustScaleAndLevel(
		  this->NoiseLevel, level, this->NoiseLevel == 1 ? cc.param.ScalingFactorReal[level] : cc.param.ScalingFactorReal[level] * cc.param.ScalingFactorReal[level]);
		assert(ok);
	} else {
		c0.dropToLevel(level);
	}
}

void Plaintext::multPt(const Plaintext& b1, const Plaintext& b, bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());

	this->copy(b1);
	this->dropToLevel(std::min(b.c0.getLevel(), b1.c0.getLevel()), false);
	multPt(b, rescale);
	if (rescale && NoiseLevel > 1) {
		this->rescale();
	}
}

void Plaintext::multPt(const Plaintext& b, bool rescale) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());

	constexpr bool PRINT = false;
	if (cc.rescaleTechnique == CKKS::FIXEDAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
		if constexpr (PRINT)
			std::cout << "multPt: Rescale input ciphertext" << std::endl;
		if (NoiseLevel > 1)
			this->rescale();
	}
	if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
		if constexpr (PRINT)
			std::cout << "multPt: Rescale input ciphertext" << std::endl;

		if (b.c0.getLevel() > c0.getLevel()) {
			Plaintext b_(cc_);
			b_.copy(b);
			if (b_.NoiseLevel == 2)
				b_.rescale();
			if (b_.c0.getLevel() > c0.getLevel()) {
				b_.dropToLevel(c0.getLevel(), false);
			}
			multPt(b_, rescale);
			return;
		}
	}

	assert(NoiseLevel < 2);
	assert(b.NoiseLevel < 2);
	// op_count[OPS::MULTPT]++;

	c0.multPt(b.c0, false);

	this->multMetadata(*this, b);

	if (rescale && NoiseLevel > 1) {
		this->rescale();
	}
}

void Plaintext::subPt(const Plaintext& c) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());

	assert(NoiseLevel == c.NoiseLevel);
	c0.sub(c.c0);
	this->addMetadata(*this, c);
}

} // namespace FIDESlib::CKKS