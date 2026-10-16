//
// Created by carlosad on 24/04/24.
//

#ifndef FIDESLIB_CKKS_CIPHERTEXT_CUH
#define FIDESLIB_CKKS_CIPHERTEXT_CUH

#include "RNSPoly.cuh"
#include "forwardDefs.cuh"
#include "openfhe-interface/RawCiphertext.cuh"
#include <complex>
#include <source_location>

namespace FIDESlib::CKKS {

/**
 * @brief Global flag controlling whether hoisted rotation optimisation is enabled.
 *
 * This can be toggled at runtime to switch between the standard hoisted-rotation
 * implementation and the fused hoisted‑rotation variant.
 */
extern bool hoistRotateFused;

/**
 * @class Ciphertext
 * @brief Represents a ciphertext in the CKKS scheme.
 *
 * The ciphertext holds two RNSPoly components (c0 and c1) together with
 * associated metadata (key identifier, noise level, scaling factor, etc.).
 * It provides arithmetic operations, key‑switching, rotations and
 * rescaling.  Most operations maintain the level information and may
 * automatically perform rescaling or modulus‑down conversion as required
 * by the CKKS protocol.
 */
class Ciphertext {
    /** @brief Identifier string for NVTX profiling. */
    static constexpr const char* loc{ "Ciphertext" };
    /** @brief NVTX range object for profiling this class' methods. */
    CudaNvtxRange my_range;

  public:
    /** @brief Identifier for the key associated with this ciphertext. */
    KeyHash keyID{ "" };
    /** @brief Reference to the CKKS context owning this ciphertext. */
    Context cc_;
    /** @brief Reference to the context data (metadata) for this ciphertext. */
    ContextData& cc;
    /** @brief The two polynomial components of the ciphertext (c0 and c1). */
    RNSPoly c0, c1;
    /** @brief Scaling factor for this ciphertext. */
    double NoiseFactor = 0;
    /** @brief Discrete noise level indicator. */
    int NoiseLevel = { 1 };
    /** @brief Number of slots (vector length) encoded in the ciphertext. */
    int slots = 0;

    /**
     * @brief Move constructor. Transfers ownership of the internal resources
     *        from another Ciphertext instance.
     *
     * @param ct_moved Rvalue reference to the source Ciphertext.
     */
    Ciphertext(Ciphertext&& ct_moved) noexcept;
    Ciphertext& operator=(Ciphertext&& ct_moved) noexcept;

    /**
     * @brief Constructs an empty Ciphertext bound to a given context.
     *
     * The created ciphertext has uninitialised polynomial components and zero noise.
     *
     * @param cc Reference to the CKKS Context the ciphertext belongs to.
     */
    explicit Ciphertext(const Context& cc);

    /**
     * @brief Constructs a Ciphertext from a serialized RawCipherText.
     *
     * This loads polynomial data and metadata from the raw representation.
     *
     * @param cc   Reference to the CKKS Context.
     * @param rawct Serialized representation of a ciphertext.
     */
    Ciphertext(const Context& cc, const RawCipherText& rawct);

    /** @brief Destructor. Releases any allocated resources. */
    ~Ciphertext();

    /**
     * @brief Copies only the metadata fields from another ciphertext.
     *
     * This does **not** copy polynomial data.
     *
     * @param a Source ciphertext to copy metadata from.
     */
    void copyMetadata(const Ciphertext& a);

    /**
     * @brief Merges metadata from two ciphertexts into *this*.
     *
     * Used after operations that combine two ciphertexts.
     *
     * @param a First operand.
     * @param b Second operand.
     */
    void addMetadata(const Ciphertext& a, const Ciphertext& b);

    /**
     * @brief Merges metadata from a ciphertext and a plaintext into *this*.
     *
     * @param a Ciphertext operand.
     * @param b Plaintext operand.
     */
    void addMetadata(const Ciphertext& a, const Plaintext& b);

    /**
     * @brief Merges metadata appropriate for multiplication of two ciphertexts.
     *
     * @param a First operand.
     * @param b Second operand.
     */
    void multMetadata(const Ciphertext& a, const Ciphertext& b);

    /**
     * @brief Merges metadata for multiplication of a ciphertext with a plaintext.
     *
     * @param a Ciphertext operand.
     * @param b Plaintext operand.
     */
    void multMetadata(const Ciphertext& a, const Plaintext& b);

    /**
     * @brief Normalises a rotation index to the valid range for the current slot count.
     *
     * The function makes the index positive, maps it into `[0, slots)` and, if the
     * index exceeds half the slot count, adds `N/2 - slots` to map it into the
     * underlying NTT domain.
     *
     * @param index Desired rotation index (may be negative).
     * @return Normalized index within `[0, slots)`.
     */
    int normalyzeIndex(int index) const;

    /**
     * @brief Loads ciphertext data from a RawCipherText representation.
     *
     * @param rawct Serialized ciphertext to load.
     */
    void load(const RawCipherText& rawct);

    /**
     * @brief Stores the current ciphertext into a RawCipherText representation.
     *
     * @param rawct Destination object that will receive the serialized data.
     */
    void store(RawCipherText& rawct);

    /**
     * @brief Adds another ciphertext to *this* (in‑place).
     *
     * The method ensures both ciphertexts are on the same level, performing
     * rescaling or level adjustments when required by the current
     * `rescaleTechnique`.  It may invoke `adjustForAddOrSub` and, if that
     * fails, create a temporary copy of `b` with aligned scaling before
     * retrying.  After the polynomial addition, metadata is merged via
     * `addMetadata`.  The operation count for `ADD` is incremented.
     *
     * @param b Ciphertext to add.
     */
    void add(const Ciphertext& b);
    void addMutable(Ciphertext& b);

    /**
     * @brief Subtracts another ciphertext from *this* (in‑place).
     *
     * Mirrors the behaviour of `add` but uses subtraction on the polynomial
     * components.  Rescaling, level alignment and metadata handling follow the
     * same logic as in `add`.
     *
     * @param b Ciphertext to subtract.
     */
    void sub(const Ciphertext& b);
    void subMutable(Ciphertext& b);

    /**
     * @brief Adds a plaintext to *this* ciphertext (in‑place).
     *
     * Handles automatic rescaling when the plaintext and ciphertext have
     * mismatched levels or noise.  For flexible rescaling techniques it may
     * create a temporary adjusted plaintext before performing the addition.
     * After the operation, metadata is merged via `addMetadata`.
     *
     * @param b Plaintext operand.
     */
    void addPt(const Plaintext& b);
    void addPtMutable(Plaintext& b);

    /**
     * @brief Subtracts a plaintext from *this* ciphertext (in‑place).
     *
     * The semantics are analogous to `addPt` but perform subtraction on `c0`.
     *
     * @param pt Plaintext operand.
     */
    void subPt(const Plaintext& pt);
    void subPtMutable(Plaintext& b);

    /**
     * @brief Adds a scalar (double) to both polynomial components.
     *
     * The scalar is first converted to the appropriate modulus representation
     * via `ElemForEvalAddOrSub`.  The sign of the scalar is handled by
     * converting the element to its complement when `c < 0`.  No metadata
     * updates are performed.
     *
     * @param c Scalar value to add.
     */
    void addScalar(const double c);

    /**
     * @brief Adds a COMPLEX scalar to element 0 (EvalAdd(ct, complex),
     * ckksrns-leveledshe.cpp L73-106).
     *
     * The re/im magnitudes are converted via `ElemForEvalAddOrSub` and composed into the
     * compressed 1-slot polynomial {re at 0, im at N/2} (sign-handled per component via the
     * modulus complement), NTT'd and added to c0 (the imaginary part never touches c1). No
     * metadata updates are performed, matching the reference.
     *
     * @param c Complex scalar value to add.
     */
    void addScalar(const std::complex<double> c);

    bool adjustCiphertextToPlaintext(const Plaintext& plaintext);
    /**
     * @brief Multiplies *this* ciphertext by a plaintext.
     *
     * If the current rescaling technique requires matching noise levels,
     * the method may rescale the ciphertext or adjust the plaintext
     * (creating a temporary copy) before multiplication.  After the
     * component‑wise multiplication, metadata is merged via `multMetadata`.
     * If `rescale` is true and the technique is `FIXEDMANUAL`, the
     * ciphertext is rescaled and the noise factor/level are updated.
     *
     * @param b Plaintext operand.
     * @param rescale If true, performs rescaling after multiplication.
     * @param ignore_scales
     */
    void multPt(const Plaintext& b, bool rescale = false, bool ignore_scales = false);
    void multPtMutable(Plaintext& b, bool rescale = false);

    /**
     * @brief Multiplies a ciphertext `c` by a plaintext and stores the result in *this*.
     *
     * Equivalent to copying `c` into *this* and then invoking `multPt(b, rescale)`.
     *
     * @param c Ciphertext operand.
     * @param b Plaintext operand.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void multPt(const Ciphertext& c, const Plaintext& b, bool rescale = false);
    void multPtMutable(const Ciphertext& c, Plaintext& b, bool rescale = false);

    /**
     * @brief Computes `*this + (c * b)` in a single fused operation.
     *
     * The method adds ciphertext `c * b` to *this* and optionally rescales.
     * Metadata is merged via `multMetadata`.
     *
     * @param c Ciphertext to add.
     * @param b Plaintext to multiply.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void addMultPt(const Ciphertext& c, const Plaintext& b, bool rescale = false);

    /**
     * @brief Multiplies *this* by another ciphertext.
     *
     * Handles level alignment and possible rescaling according to the current
     * `rescaleTechnique`.  If necessary, the method calls `adjustForMult`
     * (creating a temporary copy of `b` when alignment fails).  The actual
     * multiplication is performed before a key‑switch operation.  Metadata is
     * merged via `multMetadata`.  The operation count for `MULT` is incremented.
     *
     * @param b Ciphertext multiplier.
     * @param rescale Perform rescaling after multiplication if true.
     * @param moddown Perform modulus down‑conversion if true.
     */
    void mult(const Ciphertext& b, bool rescale = false, bool moddown = true);
    void multMutable(Ciphertext& b, bool rescale = false, bool moddown = true);

    /**
     * @brief Multiplies two ciphertexts and stores the result in *this*.
     *
     * This overload forwards to the two‑argument version, using `b` and `c`
     * as the two multiplicands.  It exists for convenience when the two
     * operands are already separate objects.
     *
     * @param b First operand.
     * @param c Second operand.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void mult(const Ciphertext& b, const Ciphertext& c, bool rescale = false);
    void multMutable(Ciphertext& b, Ciphertext& c, bool rescale = false);

    /**
     * @brief Multiplies both polynomial components by a scalar without pre‑checks.
     *
     * The scalar is converted to the appropriate modulus element via
     * `ElemForEvalMult`.  If `rescale` is true and the technique is
     * `FIXEDMANUAL`, the ciphertext is rescaled after multiplication.
     *
     * @param c Scalar multiplier.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void multScalarNoPrecheck(const double c, bool rescale = false);

    /**
     * @brief Multiplies both polynomial components by a scalar, performing necessary checks.
     *
     * For flexible rescaling techniques the method may rescale the ciphertext
     * when the noise level is 2.  The scalar is converted using
     * `ElemForEvalMult` and the multiplication is applied to both components.
     * Metadata (noise level/factor) is updated accordingly; if `rescale`
     * is true and the technique is `FIXEDMANUAL`, a rescaling step follows.
     *
     * @param c Scalar multiplier.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void multScalar(const double c, bool rescale = false);

    /**
     * @brief Multiplies a ciphertext `b` by a scalar and stores the result in *this*.
     *
     * The method copies `b` into *this* and then calls `multScalar(c,
     * rescale)`.  It is useful when the caller wishes to keep `b` unchanged.
     *
     * @param b Ciphertext operand.
     * @param c Scalar multiplier.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void multScalar(const Ciphertext& b, const double c, bool rescale = false);

    /**
     * @brief Multiplies both polynomial components by a COMPLEX scalar
     * (EvalMult(ct, complex), ckksrns-leveledshe.cpp L155-174 / L887-928).
     *
     * ForkBitExact transcription: the re factor = GetElementForEvalMult(re, SF(level)) and
     * the im factor = X^{M/4} * GetElementForEvalMult(im, SF(level)) are composed into the
     * compressed 1-slot polynomial F = Re + X^{N/2}*Im (two values per limb: {fRe, fIm} at
     * coefficient positions {0, N/2}), NTT'd (length-2 negacyclic -> {fRe +/- i*fIm}) and
     * applied in one `multElement(slots=1)` pass (the block-constant read is exactly the
     * NTT representation of the constant message). For non-FIXEDMANUAL the nsd-2 input is
     * pre-reduced (EvalMultInPlace L167-172). Metadata: noise level + 1, SF *= SF(level)
     * (target_level == -1 case); `rescale` follows the real multScalar convention
     * (FIXEDMANUAL-gated rescale).
     *
     * @param c Complex scalar multiplier.
     * @param rescale Perform rescaling after multiplication if true.
     */
    void multScalar(const std::complex<double> c, bool rescale = false);

    /**
     * @brief COMPLEX-scalar multiply that lands the product on `target_level`
     * (EvalMult(ct, complex, target_level), ckksrns-leveledshe.cpp L164-174 + L923-927).
     *
     * The multiply uses the folded scFactor (ElemForEvalMult(target_level, v, level)),
     * only the surviving target_level+1 lowest towers are computed (the dropped towers'
     * products vanish), and the metadata ends at SF = SFRealBig(target_level). Consumed by
     * the complex weighted sums (evalPartialLinearWSumWithBias<std::complex<double>>).
     *
     * @param c Complex scalar multiplier.
     * @param target_level Target level of the product (must be < current level).
     */
    void multScalar(const std::complex<double> c, int target_level);

    /**
     * @brief Multiplies a ciphertext `b` by a COMPLEX scalar and stores the result in *this*.
     *
     * Copies `b` into *this*, then calls `multScalar(c, rescale)`.
     */
    void multScalar(const Ciphertext& b, const std::complex<double> c, bool rescale = false);

    /**
     * @brief Multiplies a ciphertext `b` by a COMPLEX scalar with a target level and stores
     * the result in *this*. Copies `b` into *this*, then calls `multScalar(c, target_level)`.
     */
    void multScalar(const Ciphertext& b, const std::complex<double> c, int target_level);

    /**
     * @brief Squares the ciphertext (i.e., multiplies it by itself).
     *
     * After the multiplication the metadata (noise level/factor) is updated.
     * If `rescale` is true and the technique is `FIXEDMANUAL`, a rescaling
     * step follows.
     *
     * @param rescale Perform rescaling after squaring if true.
     */
    void square(bool rescale = false);

    /**
     * @brief Squares the source ciphertext `src` and stores the result in *this*.
     *
     * Equivalent to copying `src` into *this* and then calling `square(rescale)`.
     *
     * @param src Source ciphertext to square.
     * @param rescale Perform rescaling after squaring if true.
     */
    void square(const Ciphertext& src, bool rescale = false);

    /**
     * @brief Rotates the slots of the ciphertext by `index` positions.
     *
     * The index is normalised using `normalyzeIndex`.  Depending on the
     * `moddown` flag, a modulus‑down conversion may be performed after the
     * rotation.
     *
     * @param index Number of slots to rotate (positive = left).
     * @param moddown Perform modulus down‑conversion after rotation if true.
     */
    void rotate(const int index, bool moddown = true);

    /**
     * @brief Rotates ciphertext `c` by `index` slots and stores the result in *this*.
     *
     * This overload copies `c` into *this* and then calls the single‑argument
     * `rotate(index, true)`.
     *
     * @param c Source ciphertext.
     * @param index Number of slots to rotate.
     */
    void rotate(const Ciphertext& c, const int index);

    /**
     * @brief Computes the complex conjugate of ciphertext `c` and stores it in *this*.
     *
     * The operation is performed component‑wise and updates the operation
     * count for `CONJUGATE`.
     *
     * @param c Source ciphertext.
     */
    void conjugate(const Ciphertext& c);

    /**
     * @brief Performs modulus down‑conversion on the ciphertext.
     *
     * Both polynomial components are down‑converted; when `free` is true the
     * temporary special limbs are released.
     *
     * @param free If true, frees intermediate buffers after conversion.
     */
    void modDown(bool free = false);

    /** @brief Increases the modulus level of the ciphertext (level up). */
    void modUp();

    /** @brief Rescales the ciphertext to the next level, adjusting scaling factor. */
    void rescaleInternal();
    void rescale();

    /**
     * @brief OpenFHE 4a31f521 raiseDegree (AdjustLevelsAndDepthInPlace FIXEDAUTO degree-mismatch
     * branch): bring a degree-1 ciphertext to degree-2 WITHOUT a rescale by multiplying it by the
     * last-tower prime q_l of `targetLevel` (the upcoming rescale divides the factor out exactly).
     */
    void raiseDegreeToLevel(int targetLevel);

    void addMutable(const Ciphertext& b, Ciphertext& c);
    void addMutable(Ciphertext& b, const Ciphertext& c);
    /**
     * @brief Grows the ciphertext to at least `level` by performing necessary modulus upgrades.
     *
     * The method ensures both components reach the requested level, generating
     * special limbs if required.
     *
     * @param level Desired minimum level.
     */
    void growToLevel(int level);

    /** @brief Drops the ciphertext to a lower level, discarding higher moduli.

        @todo adjust the scale on FLEXIBLE modes
    */
    void dropToLevel(int level, bool skip_adjust = false);

    /**
     * @brief Returns the current modulus level of the ciphertext.
     *
     * @return Current level index (both components share the same level).
     */
    [[nodiscard]] int32_t getLevel() const;

    /**
     * @brief Performs an automorphism on the ciphertext.
     *
     * The automorphism is applied to both polynomial components; the `br`
     * argument is passed to the underlying `RNSPoly::automorph` method and
     * is implementation‑specific.
     *
     * @param index Automorphism index.
     * @param br    Additional parameter used by the underlying implementation.
     */
    void automorph(const int index, const int br);

    /**
     * @brief Performs a batch automorphism on multiple ciphertexts.
     *
     * Equivalent to calling `automorph` on each component; the `br` argument
     * is forwarded unchanged.
     *
     * @param index Automorphism index.
     * @param br    Additional parameter used by the underlying implementation.
     */
    void automorph_multi(const int index, const int br);

    /**
     * @brief Extends the ciphertext to a higher level if needed.
     *
     * If `init` is true, the method also (re)initialises internal structures
     * and, when necessary, scales the components by the prime modulus.
     *
     * @param init If true, also (re)initialises internal structures.
     */
    void extend(bool init = true);

    /**
     * @brief Performs hoisted rotations for a set of indexes.
     *
     * The method normalises each index, allocates temporary results if they do
     * not already exist, and optionally extends the ciphertexts before rotation.
     *
     * @param indexes Vector of rotation indexes.
     * @param results Vector to receive resulting ciphertext pointers.
     * @param ext     If true, extends ciphertexts before rotation.
     */
    void rotate_hoisted(const std::vector<int>& indexes, std::vector<Ciphertext*> results, bool ext);

    /**
     * @brief Evaluates a linear weighted sum (mutable version) over `n` ciphertexts.
     *
     * The method grows the result to the appropriate level, multiplies each
     * ciphertext by the corresponding weight (converted via `ElemForEvalMult`),
     * and accumulates the sum.  Metadata (noise level/factor) is updated to
     * reflect the combined operation.
     *
     * @param n      Number of ciphertexts.
     * @param ctxs   Vector of ciphertext pointers.
     * @param weights Vector of weights applied to each ciphertext.
     */
    void evalLinearWSumMutable(uint32_t n, const std::vector<Ciphertext*>& ctxs, std::vector<double> weights);

    /**
     * @brief Evaluates a linear weighted sum with a constant bias term
     * (EvalPartialLinearWSumWithBias, ckksrns-advancedshe.cpp L264-341), templated on the
     * weight/bias scalar type.
     *
     * double: the FIDESlib fused path — per-limb factors via ElemForEvalMult (target-level
     * folded) run through the batched evalLinearWSum kernels; metadata = nsd 2, SFBig(level),
     * optional FIXEDMANUAL-gated rescale.
     *
     * std::complex<double>: the FIDESlib fused path — each term's complex constant factor
     * (the same BuildConstant2 {re +/- i*im} poly multScalar(complex) uses, ElemForEvalMult
     * target-level folded, nsd-2 operands pre-reduced) is applied in ONE
     * eval_linear_w_sum_complex_ pass instead of the per-term multScalar(complex)+add; the
     * complex constant bias (addScalar(complex)) and the same metadata convention follow.
     * Mod-p addition is associative, so this is bit-identical to the reference's term-by-term
     * transcription.
     *
     * @param ctxs        Ciphertext operands (NoiseLevel 1, or 2 for the nsd-2 pre-reduce).
     * @param weights     weights[1]..weights[limit] applied to ctxs[0]..ctxs[limit-1].
     * @param bias        constant term added when nonzero.
     * @param limit       number of terms (default: ctxs.size()).
     * @param target_level landing level of the sum (must not be -1).
     * @param rescale     apply ModReduceInPlace at the end if true.
     */
    template <typename T>
    void evalPartialLinearWSumWithBias(const std::vector<Ciphertext*>& ctxs, std::vector<T> weights, T bias, int limit = 0, int target_level = -1, int rescale = true);

    static std::vector<Ciphertext>
    evalMultiplePartialLinearWSumWithBias(const std::vector<Ciphertext*>& ctxs, std::vector<std::vector<double>> weights, std::vector<int> target_level, int rescale);

    /// @brief Fused complex batched WSum: the std::complex<double> counterpart of the double
    /// overload above. Every output j = sum_i weights[j][i]*ctxs[i-1] + weights[j][0] is one
    /// evalLinearWSumComplexMultiple kernel pass; the complex bias is folded into the packed
    /// factor buffer (an extra operand pair at i == max_n) and added to element 0 only.
    static std::vector<Ciphertext>
    evalMultiplePartialLinearWSumWithBias(const std::vector<Ciphertext*>& ctxs,
        std::vector<std::vector<std::complex<double>>> weights, std::vector<int> target_level, int rescale);
    /**
     * @brief Adds a scaled version of `ciphertext` (multiplied by `d`) to *this*.
     *
     * The scalar multiplication is performed on a temporary copy of the
     * operand, which is then added component‑wise.  Metadata is merged via
     * `addMetadata`.
     *
     * @param ciphertext Ciphertext to be scaled and added.
     * @param d          Scalar multiplier.
     */
    void addMultScalar(const Ciphertext& ciphertext, double d);

    /**
     * @brief Adds scalar `c` to ciphertext `b` and stores result in *this*.
     *
     * Equivalent to copying `b` into *this* and invoking `addScalar(c)`.
     *
     * @param b Ciphertext operand.
     * @param c Scalar to add.
     */
    void addScalar(const Ciphertext& b, double c);

    /**
     * @brief Adds COMPLEX scalar `c` to ciphertext `b` and stores result in *this*.
     *
     * Equivalent to copying `b` into *this* and invoking `addScalar(c)` (the complex
     * overload — the compressed {re@0, im@N/2} constant added to element 0).
     *
     * @param b Ciphertext operand.
     * @param c Complex scalar to add.
     */
    void addScalar(const Ciphertext& b, const std::complex<double> c);

    /**
     * @brief Adds two ciphertexts and stores the result in *this*.
     *
     * Handles self‑addition and aliasing cases; the method may copy the
     * appropriate operand before performing the addition.
     *
     * @param ciphertext   First operand.
     * @param ciphertext1  Second operand.
     */
    void add(const Ciphertext& ciphertext, const Ciphertext& ciphertext1);
    void addMutable(Ciphertext& b, Ciphertext& c);

    /**
     * @brief Subtracts the second ciphertext from the first and stores the result in *this*.
     *
     * Handles self‑subtraction and aliasing cases similarly to `add`.
     *
     * @param ciphertext   Minuend.
     * @param ciphertext1  Subtrahend.
     */
    void sub(const Ciphertext& ciphertext, const Ciphertext& ciphertext1);
    void subMutable(const Ciphertext& ciphertext, Ciphertext& ciphertext1);
    bool adjustScaleAndLevel(int scaleDegree, int level, double scaling_factor);

    /**
     * @brief Copies the full content (polynomials and metadata) from another ciphertext.
     *
     * @param ciphertext Source ciphertext.
     */
    void copy(const Ciphertext& ciphertext);

    /**
     * @brief Adds a plaintext to a ciphertext and stores the result in *this*.
     *
     * @param ciphertext Ciphertext operand.
     * @param plaintext  Plaintext operand.
     */
    void addPt(const Ciphertext& ciphertext, const Plaintext& plaintext);
    void addPtMutable(const Ciphertext& ciphertext, Plaintext& plaintext);

    /**
     * @brief Reinterprets the internal data of `ciphertext` under the current context.
     *
     * The method checks that the source ciphertext belongs to a compatible
     * context (same primes) and that its level does not exceed the current
     * context's maximum.  It then copies the data.
     *
     * @param ciphertext Source ciphertext.
     */
    void reinterpretContext(const Ciphertext& ciphertext);

    /**
     * @brief Performs a key‑switch operation using the provided switching key.
     *
     * @param ksk KeySwitchingKey to apply.
     */
    void keySwitch(const KeySwitchingKey& ksk);

    /**
     * Sparse secret-encapsulation key switch (OpenFHE FHECKKSRNS::KeySwitchSparse). Operates on a
     * single-limb (level-0) ciphertext and a sparse (GHS) key over Ql*P'. Composed as
     *   c1Ext = exact Ql -> Ql*P' extension of c1 (centred lift)
     *   c0 = c0_bottom + exactModDown(c1Ext * b);  c1 = exactModDown(c1Ext * a)
     * where the mod-down adds the alpha*P' overflow correction. Single-GPU only.
     * @param ksk the sparse switching key (a/b over Ql*P'), see KeySwitchingKey::InitializeSparse.
     */
    void keySwitchSparse(const KeySwitchingKey& ksk);

    /**
     * @brief Adjusts internal scaling factors to make addition/subtraction possible.
     *
     * The method may rescale or multiply by 1.0 depending on the current
     * `rescaleTechnique`.  It returns `true` when the adjustment succeeds.
     *
     * @param ciphertext Operand that may have a different scaling factor.
     * @return `true` on successful adjustment.
     */
    bool adjustForAddOrSub(const Ciphertext& ciphertext);

    /**
     * @brief Adjusts scaling factors to enable multiplication with `ciphertext`.
     *
     * The method first calls `adjustForAddOrSub`; if the noise level of the
     * operand is higher, the current ciphertext may be rescaled.  It returns
     * `true` when the multiplication can proceed without further adjustment.
     *
     * @param ciphertext Operand to align scaling with.
     * @return `true` on successful adjustment.
     */
    bool adjustForMult(const Ciphertext& ciphertext);

    /**
     * @brief Checks whether the ciphertext and a plaintext share the same scaling factor.
     *
     * The check accounts for a small tolerance (`1e-9` relative error).
     *
     * @param b Plaintext to compare against.
     * @return `true` if scaling factors match within tolerance.
     */
    [[nodiscard]] bool hasSameScalingFactor(const Plaintext& b) const;
    void negate();
    void negate(const Ciphertext& ciphertext);

    /** @brief Clears the operation record used for debugging/tracing. */
    static void clearOpRecord();

    /** @brief Prints the operation record to standard output. */
    static void printOpRecord();

    /**
     * @brief Computes the dot product between arrays of ciphertexts and plaintexts.
     *
     * The operation multiplies corresponding components and accumulates the result
     * in *this*.  When `ext` is true the method expects the inputs to have
     * already performed modulus‑up conversion.
     *
     * @param ciphertexts Pointer to array of ciphertexts.
     * @param plaintexts  Pointer to array of plaintexts.
     * @param n Number of elements.
     * @param ext If true, performs the operation on the extension libs aswell.
     */
    void dotProductPt(Ciphertext* ciphertexts, Plaintext* plaintexts, const int n, bool ext);

    /**
     * @brief Computes the dot product between ciphertexts and an array of plaintext pointers.
     *
     * @param ciphertexts Pointer to array of ciphertexts.
     * @param plaintexts Pointer to array of pointers to plaintexts.
     * @param n Number of elements.
     * @param ext If true, performs the operation on the extension libs aswell.
     */
    void dotProductPt(Ciphertext* ciphertexts, Plaintext** plaintexts, const int n, bool ext);

    /**
     * @brief Computes the dot product between arrays of ciphertext pointers and plaintext pointers.
     *
     * @param ciphertexts Pointer to array of pointers to ciphertexts.
     * @param plaintexts Pointer to array of pointers to plaintexts.
     * @param n Number of elements.
     * @param ext If true, performs the operation on the extension libs aswell.
     */
    void dotProductPt(Ciphertext** ciphertexts, Plaintext** plaintexts, const int n, bool ext);
    void dotProduct(const std::vector<Ciphertext*>& a, const std::vector<Ciphertext*>& b, bool ext);

    /**
     * @brief Multiplies the ciphertext by a monomial of the given `power`.
     *
     * The monomial is cached in the context; if not present, it is generated.
     *
     * @param power Exponent of the monomial.
     */
    void multMonomial(int power);

    /** Total GPU bytes held by this ciphertext (its c0 + c1 polynomials). */
    uint64_t getMemoryUsage() const;
};

} // namespace FIDESlib::CKKS

#endif // FIDESLIB_CKKS_CIPHERTEXT_CUH
