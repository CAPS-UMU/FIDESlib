
#ifndef __RAW_CIPHER_TEXT__
#define __RAW_CIPHER_TEXT__

#include "CKKS/forwardDefs.cuh"
#include <cinttypes>
#include <vector>
// #include "CKKS/BootstrapPrecomputation.cuh"
#include <openfhe.h>

namespace FIDESlib::CKKS {

constexpr bool REVERSE = false;

/*
 * The rawCipherText Class contains the basic information needed to hold a ciphertext, with pointers to data that can be contained on the GPU or main memory
 * This is stored in RNS/DCRT format
 */
struct RawCipherText {
	// lbcrypto::CryptoContext<lbcrypto::DCRTPoly> & cc; // Original CryptoContext object from OpenFHE;
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> originalCipherText; // Original Ciphertext object from OpenFHE;
	std::vector<std::vector<uint64_t>> sub_0;
	// uint64_t* sub_0; // pointer to sub-ciphertext 0
	std::vector<std::vector<uint64_t>> sub_1; // pointer to sub-ciphertext 1
	uint64_t* sub_2;						  // pointer to sub-ciphertext 1
	std::vector<uint64_t> moduli;			  // moduli for each limb
	int numRes;								  // number of residues of ciphertext, length of moduli array and first dimension of sub-ciphertexts
	int N;									  // length of each polynomial
	Format format;							  // current format of ciphertext, either coefficient or evaluation
	double Noise;
	int NoiseLevel;
	int slots;
	std::string keyid;
	// GPUCKKS::Event e;
};

struct RawPlainText {
	//  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> & cc; // Original CryptoContext object from OpenFHE;
	lbcrypto::Plaintext originalPlainText; // Original Ciphertext object from OpenFHE;
	std::vector<std::vector<uint64_t>> sub_0;
	std::vector<uint64_t> moduli; // moduli for each limb
	int numRes;					  // number of residues of ciphertext, length of moduli array and first dimension of sub-ciphertexts
	int N;						  // length of each polynomial
	Format format;				  // current format of ciphertext, either coefficient or evaluation
	double Noise;
	int slots;
	int NoiseLevel;
};

struct RawParams {
	int N;
	int L;
	int K;
	int logN;
	lbcrypto::ScalingTechnique scalingTechnique;
	std::vector<uint64_t> moduli;
	std::vector<uint64_t> root_of_unity;
	std::vector<uint64_t> cyclotomic_order;
	std::vector<uint64_t> SPECIALmoduli;
	std::vector<uint64_t> SPECIALroot_of_unity;
	std::vector<uint64_t> SPECIALcyclotomic_order;
	std::map<int, std::vector<uint64_t>> psi;
	std::map<int, std::vector<uint64_t>> psi_inv;
	std::vector<uint64_t> N_inv;
	std::vector<double> ModReduceFactor;
	std::vector<std::vector<uint64_t>> m_QlQlInvModqlDivqlModq;

	int dnum;
	std::vector<std::vector<uint64_t>> PARTITIONmoduli;
	std::vector<std::vector<std::vector<uint64_t>>> PartQlHatInvModq;
	std::vector<std::vector<std::vector<std::vector<uint64_t>>>> PartQlHatModp;
	std::vector<uint64_t> PHatInvModp;
	std::vector<std::vector<uint64_t>> PHatModq;
	std::vector<uint64_t> PInvModq;

	std::vector<double> ScalingFactorReal;
	std::vector<double> ScalingFactorRealBig;

	/** Bootstrapping */
	std::vector<double> coefficientsCheby;
	int doubleAngleIts{ 0 };
	uint32_t bootK;
	uint32_t correctionFactor;
	bool sparse_encaps{ false };
	int p;
};

struct RawKeySwitchKey {
	std::vector<std::vector<std::vector<uint64_t>>> r_key_moduli;
	std::vector<std::vector<std::vector<std::vector<uint64_t>>>> r_key;
	std::vector<std::vector<std::vector<uint64_t>>> dcrt_keys;
	std::string keyid;
};

std::vector<std::vector<uint64_t>> GetRawArray(std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>> polys);

RawCipherText GetRawCipherText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ct, int REV = 1);

void GetOpenFHECipherText(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result, RawCipherText raw, int REV = 1);

RawPlainText GetRawPlainText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, lbcrypto::Plaintext pt);
RawPlainText GetRawPlainText(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, lbcrypto::ReadOnlyPlaintext pt);

void GetOpenFHEPlaintext(lbcrypto::Plaintext result, RawPlainText raw, int REV = 1);

// RawParams GetRawParams(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc);
RawParams GetRawParams(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, FIDESlib::BOOT_CONFIG boot_conf = UNIFORM);

RawKeySwitchKey GetKeySwitchKey(std::shared_ptr<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>> ksk);

RawKeySwitchKey GetEvalKeySwitchKey(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys);

RawKeySwitchKey GetEvalKeySwitchKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey);

RawKeySwitchKey GetRotationKeySwitchKey(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, int index, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc);

RawKeySwitchKey GetRotationKeySwitchKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int index);

RawKeySwitchKey GetConjugateKeySwitchKey(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey);

void GenAndAddRotationKeys(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, FIDESlib::CKKS::Context& GPUcc, std::vector<int> indexes);

std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>> GenRotationKeys(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, std::vector<int> indexes);

std::shared_ptr<std::map<uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>> GenRotationKeys(const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& keys, std::vector<int> indexes);

void AddRotationKeys(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, FIDESlib::CKKS::Context& GPUcc, std::vector<int> indexes);

/** Used in AddBootstrapPrecomputation */
void AddBootstrapPlaintexts(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, int slots, FIDESlib::CKKS::Context& GPUcc_, FIDESlib::CKKS::BootstrapPrecomputation& result);

/** Used in GenBootstrapKeys and AddBootstrapKeys */
std::vector<int> GetBootstrapIndexes(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, int slots, FIDESlib::CKKS::BootstrapPrecomputation* result_);

/** Server-side bootstrap setup */
void AddBootstrapPrecomputation(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int slots, FIDESlib::CKKS::Context& GPUcc_);

/** DEPRECATED: split into GenBootstrapKeys and AddBootstrapPrecomputation overload with public key*/
void AddBootstrapPrecomputation(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, int slots, FIDESlib::CKKS::Context& GPUcc);

/**
 * Used in AddBootstrapPrecomputation
 */
void AddBootstrapKeys(const lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int slots, FIDESlib::CKKS::Context& GPUcc_);

/**
 * Client-side bootstrap setup: Keys for FIDESlib Bootstrap do not match OpenFHE's, be careful if you intend to run Bootstrap on CPU
 *
 * The code can be used to implement SSE and regular Boot on a regular non-FIDESlib OpenFHE installation with server-side FIDESlib.
 */
void GenBootstrapKeys(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, int slots, bool SSE);
void GenBootstrapKeys(const lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& keys, int slots, bool SSE = false);
} // namespace FIDESlib::CKKS

#endif //__RAW_CIPHER_TEXT__