#ifndef API_ENGINE_HPP
#define API_ENGINE_HPP

#include "Ciphertext.hpp"
#include "Plaintext.hpp"
#include "PublicKey.hpp"
#include "engine/Backend.hpp"

#include <any>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fideslib {

template <typename T> class CryptoContextImpl;

/// @brief Per-backend arguments for the host-side OpenFHE EvalBootstrapSetup call, which the facade
/// performs. The engine only supplies this policy; it does no bootstrap-setup work itself.
struct BootstrapSetupPolicy {
	bool precompute;
	bool btSlotsEncoding;
	int32_t modEvalLevels;
};

/// @brief One implementation per backend. Stateless: each method receives the
/// owning context. A backend's method bodies live entirely in its own
/// translation unit (engine/cpu for the CPU/OpenFHE backend, engine/cuda for the
/// CUDA backend), so CPU and CUDA code are separated by file rather than
/// interleaved per function.
class Engine {
  protected:
	/// @brief Backing for the default implementations: throws "<op> is not implemented by the
	/// <name()> backend".
	[[noreturn]] void notImplemented(const char* op) const;

  public:
	// ---- Lifecycle ----
	virtual ~Engine()				 = default;
	virtual const char* name() const = 0;
	/// @brief Which backend this engine is (for explicit serialization; never inferred from devices).
	virtual Backend backend() const = 0;

	// ---- Negation ----
	virtual Ciphertext<DCRTPoly> evalNegate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct);
	virtual void evalNegateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct);

	// ---- Addition ----
	virtual Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	virtual void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	virtual Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	virtual Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar);
	virtual void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	virtual void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar);
	virtual Ciphertext<DCRTPoly> evalAddMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& ciphertexts);
	virtual void evalAddManyInPlace(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Ciphertext<DCRTPoly>>& ciphertexts);

	// ---- Subtraction ----
	virtual Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	virtual Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	virtual Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	virtual Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar);
	virtual Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, double scalar, const Ciphertext<DCRTPoly>& ct);
	virtual void evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	virtual void evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar);
	virtual void evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, double scalar, Ciphertext<DCRTPoly>& ct1);

	// ---- Multiplication ----
	virtual Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	virtual Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	virtual Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, double scalar);
	virtual void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	virtual void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar);
	virtual void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	// ---- Square ----
	virtual Ciphertext<DCRTPoly> evalSquare(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct);
	virtual void evalSquareInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct);

	// ---- Rotation ----
	virtual Ciphertext<DCRTPoly> evalRotate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, int32_t index);
	virtual void evalRotateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, int32_t index);
	virtual std::shared_ptr<void> evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct);
	virtual Ciphertext<DCRTPoly>
	evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp);
	virtual Ciphertext<DCRTPoly>
	evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst);
	virtual std::vector<Ciphertext<DCRTPoly>>
	evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const uint32_t m, const std::shared_ptr<void>& precomp);
	virtual std::vector<Ciphertext<DCRTPoly>>
	evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst);

	// ---- Chebyshev series ----
	virtual Ciphertext<DCRTPoly> evalChebyshevSeries(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	virtual void evalChebyshevSeriesInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);

	// ---- Rescale ----
	virtual Ciphertext<DCRTPoly> rescale(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext);
	virtual void rescaleInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext);

	// ---- Accumulate (sum reduction) ----
	virtual Ciphertext<DCRTPoly> accumulateSum(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, int slots, int stride);
	virtual void accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride);
	virtual void accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride, int start);

	// ---- Bootstrapping ----
	/// @brief Per-backend arguments for the facade's host EvalBootstrapSetup call (the setup itself is a
	/// host computation and lives on the facade, not here).
	virtual BootstrapSetupPolicy bootstrapSetupPolicy(bool precompute, bool btsfirstboot, int32_t modEvalLevels) const;
	virtual void evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>& ctx, const PrivateKey<DCRTPoly>& secretKey, uint32_t slots);
	virtual Ciphertext<DCRTPoly>
	evalBootstrap(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled);
	virtual void evalBootstrapInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled);

	// ---- Convolution transform (BSGS linear transform) ----
	virtual void convolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
	  Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  const std::vector<int>& indexes,
	  int stride,
	  int rowSize);
	virtual void specialConvolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
	  Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  Plaintext& mask,
	  const std::vector<int>& indexes,
	  int stride,
	  int maskRotationStride,
	  int rowSize);

	// ---- Host readback ----
	// Refresh ct->host with the backend's current value, syncing it back from the device if resident, so a
	// caller can recover the underlying lbcrypto::Ciphertext without decrypting. This does NOT evict —
	// ct->device stays resident for further ops. CPU: no-op (already on host). CUDA: store() +
	// GetOpenFHECipherText, growing the host container to the device limb count when needed (e.g. after a
	// level-raising ModRaise).
	virtual void recoverHostCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct);

	// ---- Device residency (CUDA-only; the CPU backend implements these as no-ops) ----
	virtual void loadContext(CryptoContextImpl<DCRTPoly>& ctx, const PublicKey<DCRTPoly>& publicKey);
	virtual void loadPlaintext(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt);
	virtual void loadCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct);

	// ---- Ciphertext backend hooks ----
	// The neutral value type holds only its canonical host value plus an opaque `device` slot; all
	// backend manipulation of that slot happens here. The CPU backend reads the host value; the CUDA
	// backend reads/writes ct.device. The value types reach these through thin facade methods.
	// Clone the backend-resident payload of `src` into a fresh `device` slot (empty any == not resident).
	virtual std::any cloneCiphertextBackend(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& src);
	virtual size_t ciphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct);
	virtual size_t ciphertextNoiseScaleDeg(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct);
	virtual void setCiphertextSlots(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t slots);
	virtual void setCiphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t level);

	// ---- Context backend state (no ctx argument; each context owns its engine) ----
	virtual bool isContextLoaded() const;
	virtual void synchronize() const;
	/// @brief Default is a no-op, not a throw: the context destructor calls this unconditionally.
	virtual void teardown();
	virtual void setDevices(const std::vector<int>& devices);
	virtual std::vector<int> devices() const;
};

} // namespace fideslib

#endif
