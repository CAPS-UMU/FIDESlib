#ifndef API_CUDAENGINE_HPP
#define API_CUDAENGINE_HPP

#include "engine/Engine.hpp"

#include "CKKS/forwardDefs.cuh" // FIDESlib::CKKS::Context (= shared_ptr<ContextData>)

#include <memory>
#include <vector>

namespace fideslib {

/// @brief CUDA backend. Every operation runs FIDESlib's CUDA CKKS layer.
/// All CUDA operation code lives in CudaEngine.cpp.
class CudaEngine final : public Engine {
  public:
	const char* name() const override {
		return "FIDESlib (CUDA)";
	}

	Backend backend() const override {
		return Backend::CUDA;
	}

	Ciphertext<DCRTPoly> evalNegate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) override;
	void evalNegateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt) override;
	Ciphertext<DCRTPoly> evalAdd(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar) override;
	void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) override;
	void evalAddInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) override;
	Ciphertext<DCRTPoly> evalAddMany(CryptoContextImpl<DCRTPoly>& ctx, const std::vector<Ciphertext<DCRTPoly>>& ciphertexts) override;
	void evalAddManyInPlace(CryptoContextImpl<DCRTPoly>& ctx, std::vector<Ciphertext<DCRTPoly>>& ciphertexts) override;
	Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, Plaintext& pt) override;
	Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt, const Ciphertext<DCRTPoly>& ct) override;
	Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, double scalar) override;
	Ciphertext<DCRTPoly> evalSub(CryptoContextImpl<DCRTPoly>& ctx, double scalar, const Ciphertext<DCRTPoly>& ct) override;
	void evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	void evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) override;
	void evalSubInPlace(CryptoContextImpl<DCRTPoly>& ctx, double scalar, Ciphertext<DCRTPoly>& ct1) override;
	Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2) override;
	Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, Plaintext& pt) override;
	Ciphertext<DCRTPoly> evalMult(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct1, double scalar) override;
	void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Plaintext& pt) override;
	void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, double scalar) override;
	void evalMultInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2) override;
	Ciphertext<DCRTPoly> evalSquare(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) override;
	void evalSquareInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	Ciphertext<DCRTPoly> evalRotate(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, int32_t index) override;
	void evalRotateInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, int32_t index) override;
	Ciphertext<DCRTPoly>
	evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const uint32_t m, const std::shared_ptr<void>& precomp) override;
	Ciphertext<DCRTPoly>
	evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const int32_t index, const std::shared_ptr<void>& digits, bool addFirst) override;
	std::vector<Ciphertext<DCRTPoly>> evalFastRotation(CryptoContextImpl<DCRTPoly>& ctx,
	  const Ciphertext<DCRTPoly>& ct,
	  const std::vector<int32_t>& indices,
	  const uint32_t m,
	  const std::shared_ptr<void>& precomp) override;
	std::vector<Ciphertext<DCRTPoly>>
	evalFastRotationExt(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst) override;
	Ciphertext<DCRTPoly> evalChebyshevSeries(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) override;
	void evalChebyshevSeriesInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b) override;
	Ciphertext<DCRTPoly> rescale(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext) override;
	void rescaleInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext) override;
	Ciphertext<DCRTPoly> accumulateSum(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct, int slots, int stride) override;
	void accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride) override;
	void accumulateSumInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct, int slots, int stride, int start) override;
	BootstrapSetupPolicy bootstrapSetupPolicy(bool precompute, bool btsfirstboot, int32_t modEvalLevels) const override;
	void evalBootstrapKeyGen(CryptoContextImpl<DCRTPoly>& ctx, const PrivateKey<DCRTPoly>& secretKey, uint32_t slots) override;
	Ciphertext<DCRTPoly>
	evalBootstrap(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) override;
	void evalBootstrapInPlace(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations, uint32_t precision, bool prescaled) override;
	void recoverHostCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	void convolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
	  Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  const std::vector<int>& indexes,
	  int stride,
	  int rowSize) override;
	void specialConvolutionTransformInPlace(CryptoContextImpl<DCRTPoly>& ctx,
	  Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  Plaintext& mask,
	  const std::vector<int>& indexes,
	  int stride,
	  int maskRotationStride,
	  int rowSize) override;
	void loadContext(CryptoContextImpl<DCRTPoly>& ctx, const PublicKey<DCRTPoly>& publicKey) override;
	void loadPlaintext(CryptoContextImpl<DCRTPoly>& ctx, Plaintext& pt) override;
	void loadCiphertext(CryptoContextImpl<DCRTPoly>& ctx, Ciphertext<DCRTPoly>& ct) override;
	std::shared_ptr<void> evalFastRotationPrecompute(CryptoContextImpl<DCRTPoly>& ctx, const Ciphertext<DCRTPoly>& ct) override;

	std::any cloneCiphertextBackend(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& src) override;
	size_t ciphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	size_t ciphertextNoiseScaleDeg(CryptoContextImpl<DCRTPoly>& ctx, const CiphertextImpl<DCRTPoly>& ct) override;
	void setCiphertextSlots(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t slots) override;
	void setCiphertextLevel(CryptoContextImpl<DCRTPoly>& ctx, CiphertextImpl<DCRTPoly>& ct, size_t level) override;

	bool isContextLoaded() const override;
	void synchronize() const override;
	void teardown() override;
	void setDevices(const std::vector<int>& devices) override;
	std::vector<int> devices() const override;

	// ---- CUDA-owned per-context state ----
  private:
	/// @brief CUDA CKKS context; null == not loaded. Owned by this engine (one engine per context).
	std::unique_ptr<FIDESlib::CKKS::Context> context_;
	/// @brief Devices this context is loaded on (default: device 0).
	std::vector<int> devices_ = { 0 };

  public:
	~CudaEngine() override;
};

} // namespace fideslib

#endif
