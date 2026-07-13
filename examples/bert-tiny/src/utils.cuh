#ifndef FIDESLIB_BERT_TINY_UTILS_CUH
#define FIDESLIB_BERT_TINY_UTILS_CUH

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <openfhe.h>
#include <string>

#ifdef duration
#undef duration
#endif

#include <CKKS/Ciphertext.cuh>
#include <CKKS/Context.cuh>
#include <CKKS/KeySwitchingKey.cuh>
#include <CKKS/Plaintext.cuh>
#include <CKKS/forwardDefs.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>

#include "MatMul.cuh"
#include "PolyApprox.cuh"
#include "Transformer.cuh"
#include "Transpose.cuh"

#include <CKKS/AccumulateBroadcast.cuh>
#include <CKKS/ApproxModEval.cuh>
#include <CKKS/Bootstrap.cuh>
#include <CKKS/BootstrapPrecomputation.cuh>
#include <CKKS/CoeffsToSlots.cuh>
#include <CKKS/Parameters.cuh>

extern std::vector<FIDESlib::PrimeRecord> p64;
extern std::vector<FIDESlib::PrimeRecord> sp64;
extern FIDESlib::CKKS::Parameters params;

extern lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;

extern std::vector<int> devices;
extern uint32_t ringDim;
inline const std::string root_dir = "../";

// Initialize devices vector from FIDESLIB_NUM_GPUS environment variable
inline void init_devices_from_env() {
	const char* res = std::getenv("FIDESLIB_NUM_GPUS");
	if (!res || std::strlen(res) == 0) {
		res = std::getenv("FIDESLIB_USE_NUM_GPUS"); // fallback to old name
	}
	if (res && std::strlen(res) > 0) {
		int num_dev = std::atoi(res);
		if (num_dev > 0) {
			devices.clear();
			for (int i = 0; i < num_dev; ++i) {
				devices.push_back(i);
			}
		}
	}
}

inline void read_ring_dim() {
	char* env = getenv("FIDESLIB_RING_DIM");
	if (env && env[0] != '\0') {
		ringDim = std::atoi(env);
	}
	else {
		ringDim = 16;
	}
	std::cout << "Using ring dimension: " << (1 << ringDim) << std::endl;
}

void prepare_gpu_context_bert(FIDESlib::CKKS::Context& cc_gpu, const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, FIDESlib::CKKS::EncoderConfiguration& conf);

void create_cpu_context();

void prepare_cpu_context(FIDESlib::CKKS::Context& cc_gpu, const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, size_t num_slots, size_t blockSize, FIDESlib::CKKS::EncoderConfiguration& conf);

#endif // FIDESLIB_BERT_TINY_UTILS_CUH