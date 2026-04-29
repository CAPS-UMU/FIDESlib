// utils.cu
#include "utils.cuh"
using namespace FIDESlib::CKKS;

std::vector<int> devices{ 0 };

lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = nullptr;

std::vector<FIDESlib::PrimeRecord> p64{ { .p = 2305843009218281473 },
	{ .p = 2251799661248513 },
	{ .p = 2251799661641729 },
	{ .p = 2251799665180673 },
	{ .p = 2251799682088961 },
	{ .p = 2251799678943233 },
	{ .p = 2251799717609473 },
	{ .p = 2251799710138369 },
	{ .p = 2251799708827649 },
	{ .p = 2251799707385857 },
	{ .p = 2251799713677313 },
	{ .p = 2251799712366593 },
	{ .p = 2251799716691969 },
	{ .p = 2251799714856961 },
	{ .p = 2251799726522369 },
	{ .p = 2251799726129153 },
	{ .p = 2251799747493889 },
	{ .p = 2251799741857793 },
	{ .p = 2251799740416001 },
	{ .p = 2251799746707457 },
	{ .p = 2251799756013569 },
	{ .p = 2251799775805441 },
	{ .p = 2251799763091457 },
	{ .p = 2251799767154689 },
	{ .p = 2251799765975041 },
	{ .p = 2251799770562561 },
	{ .p = 2251799769776129 },
	{ .p = 2251799772266497 },
	{ .p = 2251799775281153 },
	{ .p = 2251799774887937 },
	{ .p = 2251799797432321 },
	{ .p = 2251799787995137 },
	{ .p = 2251799787601921 },
	{ .p = 2251799791403009 },
	{ .p = 2251799789568001 },
	{ .p = 2251799795466241 },
	{ .p = 2251799807131649 },
	{ .p = 2251799806345217 },
	{ .p = 2251799805165569 },
	{ .p = 2251799813554177 },
	{ .p = 2251799809884161 },
	{ .p = 2251799810670593 },
	{ .p = 2251799818928129 },
	{ .p = 2251799816568833 },
	{ .p = 2251799815520257 } };

std::vector<FIDESlib::PrimeRecord> sp64{ { .p = 2305843009218936833 },
	{ .p = 2305843009220116481 },
	{ .p = 2305843009221820417 },
	{ .p = 2305843009224179713 },
	{ .p = 2305843009225228289 },
	{ .p = 2305843009227980801 },
	{ .p = 2305843009229160449 },
	{ .p = 2305843009229946881 },
	{ .p = 2305843009231650817 },
	{ .p = 2305843009235189761 },
	{ .p = 2305843009240301569 },
	{ .p = 2305843009242923009 },
	{ .p = 2305843009244889089 },
	{ .p = 2305843009245413377 },
	{ .p = 2305843009247641601 } };

FIDESlib::CKKS::Parameters params{ .logN = 15, .L = 23, .dnum = 1, .primes = p64, .Sprimes = sp64, .batch = 100 };

void prepare_gpu_context_bert(FIDESlib::CKKS::Context& cc_gpu, const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, FIDESlib::CKKS::EncoderConfiguration& conf) {
	if (conf.blockSize * conf.blockSize != conf.numSlots) {
		std::cout << "blockSize: " << conf.blockSize << "; num_slots: " << conf.numSlots << std::endl;
		std::cerr << "Matrix size is different from number of slots" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(cc_gpu);
	eval_key_gpu.Initialize(eval_key);
	(*cc_gpu).AddEvalKey(std::move(eval_key_gpu));

	std::vector<int32_t> rotation_indices = GenerateRotationIndices_GPU(conf.blockSize, conf.bStep, conf.bStepAcc);
	rotation_indices.push_back(30608);
	GenAndAddRotationKeys(cc, keys, cc_gpu, rotation_indices);
}

void create_cpu_context() {
	constexpr uint32_t scale_mod_size	= 52;
	constexpr uint32_t first_mod		= 56;
	constexpr uint32_t num_large_digits = 3;
	constexpr uint32_t depth			= 23;

	const uint32_t ring_dim	 = 1u << 16; // logN = 16 -> N = 65536
	const uint32_t num_slots = 1 << 14;

	lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
	parameters.SetScalingModSize(scale_mod_size);
	parameters.SetFirstModSize(first_mod);
	parameters.SetRingDim(ring_dim);
	parameters.SetBatchSize(num_slots);
	parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
	parameters.SetScalingTechnique(lbcrypto::FLEXIBLEAUTO);
	parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);
	parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
	parameters.SetNumLargeDigits(num_large_digits);
	parameters.SetMultiplicativeDepth(depth);

	if (cc != nullptr) {
		using Impl = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>;
		Impl::ClearEvalAutomorphismKeys();
		Impl::ClearEvalMultKeys();
		Impl::ClearEvalSumKeys();
	}

	cc = lbcrypto::GenCryptoContext(parameters);
	cc->Enable(lbcrypto::FHE);
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::ADVANCEDSHE);
}

void prepare_cpu_context(FIDESlib::CKKS::Context& cc_gpu, const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys, const size_t num_slots, const size_t blockSize, EncoderConfiguration& conf) {
	if (blockSize * blockSize != num_slots) {
		std::cout << "blockSize: " << blockSize << "; num_slots: " << num_slots << std::endl;
		std::cerr << "Matrix size is different from number of slots" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	cc->EvalMultKeyGen(keys.secretKey);

	std::vector<int32_t> rotation_indices = GenerateRotationIndices_GPU(conf.blockSize, conf.bStep, conf.bStepAcc);
	rotation_indices.push_back(30608);
	cc->EvalRotateKeyGen(keys.secretKey, rotation_indices);

	cc->EvalBootstrapSetup(
	  { 3, 3 }, { 16, 16 }, num_slots, 0, true, GetMultiplicativeDepthByCoeffVector((*cc_gpu).GetCoeffsChebyshev(), false) + (*cc_gpu).GetDoubleAngleIts());

	cc->EvalBootstrapKeyGen(keys.secretKey, num_slots);
	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, conf.numSlots, cc_gpu);
}
