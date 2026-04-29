//
// Created by carlosad on 25/04/24.
//

#ifndef FIDESLIB_CKKS_PLAINTEXT_CUH
#define FIDESLIB_CKKS_PLAINTEXT_CUH

#include "RNSPoly.cuh"
#include "openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib::CKKS {

/**
 * @brief Represents a plaintext in the CKKS scheme.
 *
 * Stores the underlying polynomial and associated metadata such as noise factor,
 * noise level, and slot count. Provides operations for rescaling, scalar multiplication,
 * rotation, level dropping, and metadata handling.
 *
 * @note Most methods assume the corresponding @ref Ciphertext context.
 */
class Plaintext {
	static constexpr const char* loc{ "Plaintext" };
	CudaNvtxRange my_range;

  public:
	Context& cc_;
	ContextData& cc;
	RNSPoly c0;
	double NoiseFactor = 0;
	int NoiseLevel	   = 1;
	int slots		   = 0;

	/**
	 * @brief Defaulted move constructor.
	 *
	 * Moves resources from another Plaintext instance. The default implementation
	 * is sufficient because the class only contains trivially movable members.
	 */
	Plaintext(Plaintext&& pt) = default;
	/**
	 * @brief Copy metadata from another plaintext.
	 *
	 * Copies noise factor, noise level, and slot information from @p a.
	 *
	 * @param a Source plaintext whose metadata will be copied.
	 */
	void copyMetadata(const Plaintext& a);
	/**
	 * @brief Add metadata from two plaintexts.
	 *
	 * Combines metadata of @p a and @p b (e.g., noise levels) and stores the
	 * result in this instance.
	 *
	 * @param a First plaintext source.
	 * @param b Second plaintext source.
	 */
	void addMetadata(const Plaintext& a, const Plaintext& b);
	/**
	 * @brief Multiply metadata from two plaintexts.
	 *
	 * Multiplies (or otherwise combines) the metadata of @p a and @p b and
	 * stores the result in this instance. Behaviour depends on the specific
	 * scheme implementation.
	 *
	 * @param a First plaintext source.
	 * @param b Second plaintext source.
	 */
	void multMetadata(const Plaintext& a, const Plaintext& b);
	/**
	 * @brief Construct a new plaintext bound to a CKKS context.
	 *
	 * @param cc Reference to the CKKS context that will be used for operations.
	 */
	explicit Plaintext(Context& cc);
	/**
	 * @brief Construct a plaintext from a raw ciphertext representation.
	 *
	 * @param cc   Reference to the CKKS context.
	 * @param raw  Raw plaintext data to initialize from.
	 *
	 * @todo Verify that the raw layout matches the expectations of the
	 *       underlying OpenFHE library. Behaviour may be unclear for
	 *       exceptional raw formats.
	 *
	 * @note **Analysis**: `RawPlainText` (see `openfhe-interface/RawCiphertext.cuh`) holds the coefficient vector,
	 *       the scaling factor, and level information. The constructor should copy these fields into the
	 *       internal `RNSPoly` and set `NoiseFactor`/`NoiseLevel` accordingly. Verify that coefficient ordering
	 *       (NTT vs. standard) matches the library version used.
	 */
	Plaintext(Context& cc, const RawPlainText& raw);
	/**
	 * @brief Load raw plaintext data into this instance.
	 *
	 * @param raw Raw plaintext representation to load.
	 *
	 * @todo Confirm handling of scaling factors during load.
	 *
	 * @note **Analysis**: Loading must transfer both the coefficient data and the scaling factor from `raw`
	 *       into `c0` and `NoiseFactor`. The scaling factor determines the magnitude of encoded values;
	 *       ensure the factor is stored in the same floating‑point precision as the rest of the CKKS pipeline.
	 */
	void load(const RawPlainText& raw);
	/**
	 * @brief Store this plaintext into a raw representation.
	 *
	 * @param raw Destination for the raw plaintext data.
	 *
	 * @todo Ensure that the stored raw format is compatible with downstream
	 *       consumers (e.g., serialization libraries).
	 *
	 * @note **Analysis**: The serialization should write the coefficient vector, current level, and scale in the
	 *       exact order expected by `RawPlainText`. Pay attention to endianness and any padding; mismatches will
	 *       cause deserialization errors in consumers written in other languages (e.g., Python bindings).
	 */
	void store(RawPlainText& raw);
	/**
	 * @brief Deep copy of another plaintext.
	 *
	 * Copies the underlying polynomial and all metadata from @p p.
	 *
	 * @param p Source plaintext to copy from.
	 *
	 * @todo Verify that any internal GPU buffers are correctly duplicated.
	 * @note **Analysis**: The class contains a `RNSPoly` which may allocate device memory. A correct copy must perform a deep copy of that memory (e.g., via
	 * `cudaMemcpy`), otherwise the source and destination will share the same GPU buffers leading to race conditions or double‑free errors when destructors
	 * run. Ensure the implementation clones all GPU resources.
	 */
	void copy(const Plaintext& p);

	/**
	 * @brief Rescale the plaintext to the next level.
	 *
	 * Reduces the scale of the underlying polynomial and updates metadata.
	 *
	 * @todo Clarify exact scaling factor and error propagation.
	 * @note **Analysis**: Rescaling divides the plaintext by the modulus of the current level and updates the internal scaling factor (`NoiseFactor`). The
	 * exact factor equals the ratio of the current modulus to the next modulus in the chain. Each rescale introduces relative error proportional to the
	 * divisor; documenting the expected error growth helps users decide when additional precision is required.
	 */
	void rescale();
	/**
	 * @brief Multiply the plaintext by a scalar.
	 *
	 * @param c        Scalar value to multiply with.
	 * @param rescale  If true, perform a rescaling step after multiplication.
	 *
	 * @todo Determine when rescaling is required for correctness.
	 * @note **Analysis**: Multiplication scales the plaintext by `c`, consequently scaling the internal factor. If the resulting scale exceeds the maximum
	 * representable scale for the current modulus, a subsequent rescale is necessary to bring it back within range. Typically, rescaling is triggered when
	 * `scale * c > maxScale / 2`. The implementation should document the heuristic used.
	 */
	void multScalar(double c, bool rescale = false);

	/**
	 * @brief Rotate the plaintext according to provided indexes.
	 *
	 * This function performs a hoisted rotation, producing a set of rotated
	 * results stored in @p results.
	 *
	 * @param indexes  Vector of rotation offsets.
	 * @param results  Output vector receiving pointers to rotated plaintexts.
	 *
	 * @todo Investigate memory ownership semantics for @p results.
	 */
	void rotate_hoisted(const std::vector<int>& indexes, std::vector<Plaintext*>& results);

	/**
	 * @brief Perform modulus reduction (mod-down) on the plaintext.
	 *
	 * @todo Confirm the precise effect on the polynomial coefficients.
	 * @note **Analysis**: Mod‑down reduces the coefficient modulus to the next level in the RNS chain, effectively dividing each coefficient by the current
	 * modulus and optionally rounding. This operation lowers the noise budget but also decreases the scaling factor. Clarify whether the implementation
	 * performs rounding or truncation, and how it updates `NoiseFactor`.
	 */
	void moddown();
	bool adjustScaleAndLevel(int scaleDegree, int level, double scaling_factor);
	/**
	 * @brief Adjust this plaintext to be compatible with a ciphertext.
	 *
	 * @param p Plaintext to adjust.
	 * @param c Ciphertext that defines the target level/scale.
	 * @return true if adjustment succeeded, false otherwise.
	 *
	 * @todo Clarify the adjustments performed and their impact on noise.
	 * @note **Analysis**: Compatibility usually requires matching the plaintext’s level and scaling factor to those of the ciphertext. This may involve a mod‑down
	 * followed by a rescale. Each operation injects noise; the function should document how it propagates noise and any bounds checked before returning `false`.
	 */
	bool adjustPlaintextToCiphertext(const Plaintext& p, const Ciphertext& c);

	/**
	 * @brief Perform an automorphism operation at the given index.
	 *
	 * @param index Index specifying the automorphism to apply.
	 *
	 * @todo Verify the mathematical meaning of the index parameter.
	 */
	void automorph(const int index);

	/**
	 * @brief Drop the plaintext to a lower level.
	 *
	 * @param level Target level to drop to.
	 * @param skip_adjust
	 */
	void dropToLevel(const int level, bool skip_adjust);

	void addPt(const Plaintext& b);
	void addPt(const Plaintext& b, const Plaintext& b2);
	void subPt(const Plaintext& c);
	void multPt(const Plaintext& b, bool rescale = false);
	void multPt(const Plaintext& b1, const Plaintext& b, bool rescale = false);
};

} // namespace FIDESlib::CKKS
#endif // FIDESLIB_CKKS_PLAINTEXT_CUH
