#ifndef API_CRYPTOCONTEXT_HPP
#define API_CRYPTOCONTEXT_HPP

#include <any>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <complex>

#include "CCParams.hpp"
#include "Ciphertext.hpp"
#include "Definitions.hpp"
#include "KeyPair.hpp"
#include "Plaintext.hpp"
#include "PublicKey.hpp"
#include "Serialize.hpp"

namespace fideslib {

/// @brief Specialization of CryptoContext for the DCRTPoly representation.
template <> class CryptoContextImpl<DCRTPoly> {

  public:
	CryptoContextImpl() = default;
	~CryptoContextImpl();

	// ---- Copy ----

	CryptoContextImpl(const CryptoContextImpl&)			   = delete;
	CryptoContextImpl& operator=(const CryptoContextImpl&) = delete;

	// ---- Move ----

	CryptoContextImpl(CryptoContextImpl&&)			  = default;
	CryptoContextImpl& operator=(CryptoContextImpl&&) = delete;

	// ---- Context Setup ----

	/// @brief Enable a particular feature in the context.
	void Enable(PKESchemeFeature feature);
	void Enable(uint32_t featureMask);

	// ---- Getters ----

	uint32_t GetCyclotomicOrder() const;
	uint32_t GetRingDimension() const;
	double GetPreScaleFactor(uint32_t slots);

	// ---- Setters ----
	void SetAutoLoadPlaintexts(bool autoload);
	void SetAutoLoadCiphertexts(bool autoload);
	void SetDevices(const std::vector<int>& devices);

	// ---- Load to devices ----

	/// @brief Load the context to the devices.
	void LoadContext(const PublicKey<DCRTPoly>& publicKey);
	/// @brief Load a plaintext to the devices.
	/// @param pt Plaintext to load.
	void LoadPlaintext(Plaintext& pt);
	/// @brief Load a ciphertext to the devices.
	/// @param ct Ciphertext to load.
	void LoadCiphertext(Ciphertext<DCRTPoly>& ct);

	// ---- Key Generation ----

	/// @brief Generate a public/private key pair.
	KeyPair<DCRTPoly> KeyGen();
	/// @brief Generate the evaluation multiplication keys.
	void EvalMultKeyGen(const PrivateKey<DCRTPoly>& sk);
	/// @brief Generate the evaluation rotation keys for the given steps.
	void EvalRotateKeyGen(const PrivateKey<DCRTPoly>& sk, const std::vector<int32_t>& steps);

	// ---- Bootstrapping ----

	/// @brief Generate bootstrap precomputation data.
	void EvalBootstrapSetup(const std::vector<uint32_t>& levelBudget, std::vector<uint32_t> dim1, uint32_t slots, uint32_t correctionFactor);
	/// @brief Generate the evaluation bootstrap keys.
	void EvalBootstrapKeyGen(const PrivateKey<DCRTPoly>& sk, uint32_t slots);

	// ---- Serialization ----
	static bool SerializeEvalMultKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag = "");
	static bool SerializeEvalAutomorphismKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag = "");

	// ---- Deserialization ----
	bool DeserializeEvalMultKey(std::istream& ser, const SerType& sertype) const;
	bool DeserializeEvalAutomorphismKey(std::istream& ser, const SerType& sertype) const;

	// ---- Encoding ----

	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, std::shared_ptr<void> params = nullptr, uint32_t slots = 0);
	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, std::shared_ptr<void> params = nullptr, uint32_t slots = 0);

	// ---- Encryption ----

	Ciphertext<DCRTPoly> Encrypt(Plaintext& pt, const PublicKey<DCRTPoly>& pk);
	Ciphertext<DCRTPoly> Encrypt(const PublicKey<DCRTPoly>& pk, Plaintext& pt);
	Ciphertext<DCRTPoly> Encrypt(Plaintext& pt, const PrivateKey<DCRTPoly>& sk);
	Ciphertext<DCRTPoly> Encrypt(const PrivateKey<DCRTPoly>& sk, Plaintext& pt);
	DecryptResult Decrypt(Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, Plaintext* pt);
	DecryptResult Decrypt(const PrivateKey<DCRTPoly>& sk, Ciphertext<DCRTPoly>& ct, Plaintext* pt);

	// ---- Operations ----

	Ciphertext<DCRTPoly> EvalNegate(const Ciphertext<DCRTPoly>& ct);
	void EvalNegateInPlace(Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalAdd(Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, double scalar);
	Ciphertext<DCRTPoly> EvalAdd(double scalar, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, std::complex<double> scalar);
	Ciphertext<DCRTPoly> EvalAdd(std::complex<double> scalar, const Ciphertext<DCRTPoly>& ct);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	void EvalAddInPlace(Plaintext& pt, Ciphertext<DCRTPoly>& ct1);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalAddInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, std::complex<double> scalar);
	void EvalAddInPlace(std::complex<double> scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalAddMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalAddMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalAddMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct);
	void EvalAddMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalAddMany(const std::vector<Ciphertext<DCRTPoly>>& ciphertexts);
	void EvalAddManyInPlace(std::vector<Ciphertext<DCRTPoly>>& ciphertexts);

	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalSub(Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct, double scalar);
	Ciphertext<DCRTPoly> EvalSub(double scalar, const Ciphertext<DCRTPoly>& ct);
	void EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	void EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalSubInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalSubMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalSubMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalSubMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct);
	void EvalSubMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalMult(Plaintext& pt, const Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, double scalar);
	Ciphertext<DCRTPoly> EvalMult(double scalar, const Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, std::complex<double> scalar);
	Ciphertext<DCRTPoly> EvalMult(std::complex<double> scalar, const Ciphertext<DCRTPoly>& ct1);
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalMultInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, std::complex<double> scalar);
	void EvalMultInPlace(std::complex<double> scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalMultMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct1);
	void EvalMultMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalSquare(const Ciphertext<DCRTPoly>& ct);
	void EvalSquareInPlace(Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalSquareMutable(Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalRotate(const Ciphertext<DCRTPoly>& ciphertext, int32_t index);
	void EvalRotateInPlace(Ciphertext<DCRTPoly>& ciphertext, int32_t index);

	std::shared_ptr<void> EvalFastRotationPrecompute(const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalFastRotation(const Ciphertext<DCRTPoly>& ct, int32_t index, uint32_t m, const std::shared_ptr<void>& precomp);
	Ciphertext<DCRTPoly> EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, int32_t index, const std::shared_ptr<void>& digits, bool addFirst);
	std::vector<Ciphertext<DCRTPoly>> EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, uint32_t m, const std::shared_ptr<void>& precomp);
	std::vector<Ciphertext<DCRTPoly>>
	EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst);

	Ciphertext<DCRTPoly> EvalChebyshevSeries(const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	void EvalChebyshevSeriesInPlace(Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	static std::vector<double> GetChebyshevCoefficients(std::function<double(double)>& func, double a, double b, size_t degree);

	Ciphertext<DCRTPoly> Rescale(const Ciphertext<DCRTPoly>& ciphertext);
	void RescaleInPlace(Ciphertext<DCRTPoly>& ciphertext);

	static void SetLevel(Ciphertext<DCRTPoly>& ct, size_t level);

	Ciphertext<DCRTPoly> EvalBootstrap(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);
	void EvalBootstrapInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);

	Ciphertext<DCRTPoly> AccumulateSum(const Ciphertext<DCRTPoly>& ct, int slots, int stride = 1);
	void AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride = 1);

	void ConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct, int gStep, int bStep, const std::vector<Plaintext>& pts, const std::vector<int>& indexes, int stride = 1, int rowSize = 0);

	void SpecialConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  Plaintext& mask,
	  const std::vector<int>& indexes,
	  int stride			 = 1,
	  int maskRotationStride = 1,
	  int rowSize			 = 0);

  public:
	// ---- Internal State ----

	std::any cpu;
	std::any gpu;
	/// @brief Whether the context has been loaded to the devices.
	bool loaded = false;
	/// @brief List of devices the context is loaded on.
	std::vector<int> devices = { 0 };
	/// @brief Whether plaintexts should be automatically loaded to the device upon encryption.
	bool auto_load_plaintexts = false;
	/// @brief Whether ciphertexts should be automatically loaded to the device upon creation.
	bool auto_load_ciphertexts = true;
	/// @brief Self reference to enable shared_from_this-like behavior.
	std::weak_ptr<CryptoContextImpl<DCRTPoly>> self_reference;
	/// @brief Multiplicative depth of the context.
	uint32_t multiplicative_depth = 0;
	/// @brief Rotation indexes for which rotation keys are available.
	std::vector<int32_t> rotation_indexes;
	/// @brief Secret key distribution.
	SecretKeyDist keyDist = UNIFORM_TERNARY;

	// ---- Copy helpers ----

	uint32_t CopyDeviceCiphertext(const CiphertextImpl<DCRTPoly>& ct);

	// --- Map Handling ----

	/// @brief  Registry of plaintexts stored on the GPU (opaque types).
	std::unordered_map<uint32_t, std::shared_ptr<void>> device_plaintexts;
	std::unique_ptr<std::shared_mutex> device_plaintexts_mutex;
	/// @brief  Registry of ciphertexts stored on the GPU (opaque types).
	std::unordered_map<uint32_t, std::shared_ptr<void>> device_ciphertexts;
	std::unique_ptr<std::shared_mutex> device_ciphertexts_mutex;
	/// @brief Next available handle for GPU objects. Zero is reserved as a null handle.
	uint32_t next_gpu_handle = 1;

	uint32_t RegisterDevicePlaintext(std::shared_ptr<void>&& p);
	uint32_t RegisterDeviceCiphertext(std::shared_ptr<void>&& c);
	std::shared_ptr<void>& GetDevicePlaintext(uint32_t handle);
	std::shared_ptr<void>& GetDeviceCiphertext(uint32_t handle);
	bool EvictDevicePlaintext(uint32_t handle);
	bool EvictDeviceCiphertext(uint32_t handle);

	void Synchronize() const;

	static std::vector<int> GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep);
};

} // namespace fideslib

#endif // API_CRYPTOCONTEXT_HPP