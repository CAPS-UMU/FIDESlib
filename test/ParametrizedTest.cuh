//
// Created by oscar on 10/10/24.
//

#ifndef PARAMETRIZEDTEST_CUH
#define PARAMETRIZEDTEST_CUH

#include <cstdint>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <openfhe.h>

#include <CKKS/Context.cuh>
#include <CKKS/Parameters.cuh>
#include <string>

inline const std::string root_dir = "../";

#define __TBB_NO_IMPLICIT_LINKAGE

#define MODES(name)                                                                                                        \
	extern std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> name_fix;     \
	extern std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> name_fixauto; \
	extern std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> name_flex;    \
	extern std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> name_flexext;

inline std::vector<int> devices{ 0 };

namespace FIDESlib::Testing {
/*
extern std::vector<FIDESlib::PrimeRecord> p32;
extern std::vector<FIDESlib::PrimeRecord> p64;
extern std::vector<FIDESlib::PrimeRecord> sp64;
*/
inline std::array<int, 1> batch_configs{ 100 };

inline std::vector<FIDESlib::PrimeRecord> p32{ { .p = 537133057 },
	{ .p = 537591809 },
	{ .p = 537722881 },
	{ .p = 538116097 },
	{ .p = 539754497 },
	{ .p = 540082177 },
	{ .p = 540540929 },
	{ .p = 540672001 },
	{ .p = 541327361 },
	{ .p = 541655041 },
	{ .p = 542310401 },
	{ .p = 543031297 },
	{ .p = 543293441 },
	{ .p = 545062913 },
	{ .p = 546766849 },
	{ .p = 547749889 },
	{ .p = 548012033 },
	{ .p = 548208641 },
	{ .p = 548339713 },
	{ .p = 549388289 },
	{ .p = 549978113 },
	{ .p = 550371329 },
	{ .p = 551288833 },
	{ .p = 552861697 },
	{ .p = 552927233 },
	{ .p = 553254913 },
	{ .p = 554106881 },
	{ .p = 554631169 },
	{ .p = 555220993 },
	{ .p = 555810817 },
	{ .p = 556072961 },
	{ .p = 557187073 },
	{ .p = 557776897 },
	{ .p = 558235649 },
	{ .p = 558432257 },
	{ .p = 559742977 },
	{ .p = 561184769 },
	{ .p = 561774593 },
	{ .p = 562364417 },
	{ .p = 562495489 },
	{ .p = 562757633 },
	{ .p = 563150849 },
	{ .p = 563281921 },
	{ .p = 564658177 },
	{ .p = 565641217 },
	{ .p = 566886401 },
	{ .p = 567869441 },
	{ .p = 568000513 },
	{ .p = 568066049 },
	{ .p = 568262657 },
	{ .p = 568655873 },
	{ .p = 569573377 },
	{ .p = 569638913 },
	{ .p = 570163201 },
	{ .p = 570949633 },
	{ .p = 571146241 },
	{ .p = 571539457 },
	{ .p = 572915713 },
	{ .p = 575078401 },
	{ .p = 575275009 },
	{ .p = 575733761 },
	{ .p = 575864833 },
	{ .p = 576454657 },
	{ .p = 576716801 },
	{ .p = 576913409 },
	{ .p = 577437697 },
	{ .p = 578093057 },
	{ .p = 580190209 },
	{ .p = 580780033 },
	{ .p = 581632001 },
	{ .p = 581959681 },
	{ .p = 582746113 },
	{ .p = 583794689 },
	{ .p = 584384513 },
	{ .p = 584581121 },
	{ .p = 584777729 },
	{ .p = 585695233 },
	{ .p = 586285057 },
	{ .p = 587530241 },
	{ .p = 587661313 },
	{ .p = 590479361 } };

inline std::vector<FIDESlib::PrimeRecord> p64{ { .p = 2305843009218281473 },
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

inline std::vector<FIDESlib::PrimeRecord> sp64{ { .p = 2305843009218936833 },
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

struct GeneralTestParams {
	uint64_t multDepth;
	uint64_t firstModSize = 60;
	uint64_t scaleModSize;
	uint64_t batchSize;
	uint64_t ringDim;
	uint64_t dnum;
	std::vector<int> GPUs;
};

extern std::array<int, 1> batch_configs;

/** Following: https://bu-icsg.github.io/publications/2024/fhe_parallelized_bootstrapping_isca_2024.pdf
 * C. Parameter Set for HEAP
 */
inline FIDESlib::CKKS::Parameters params64_13{ .logN = 13, .L = 5, .dnum = 2, .primes = p64, .Sprimes = sp64 };
// extern GeneralTestParams gparams64_13;

/** Following: https://bu-icsg.github.io/publications/2024/fhe_parallelized_bootstrapping_isca_2024.pdf
 * C. Parameter Set for HEAP
 * Todo: Fix prime sizes.
 */
inline GeneralTestParams gparams64_13{ .multDepth = 5, .scaleModSize = 36, .batchSize = 8, .ringDim = 1 << 13, .dnum = 2, .GPUs = devices };

/** Following: https://bu-icsg.github.io/publications/2024/fhe_parallelized_bootstrapping_isca_2024.pdf
 * C. Parameter Set for HEAP
 */
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_13 = std::tuple(gparams64_13, params64_13);

inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_fix{ tparams64_13,
	lbcrypto::ScalingTechnique::FIXEDMANUAL };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_fixauto{ tparams64_13,
	lbcrypto::ScalingTechnique::FIXEDAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_flex{ tparams64_13,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_flexext{ tparams64_13,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT };

/** Following this: https://eprint.iacr.org/2024/463.pdf
 * Table 5.7 Col.1
 */
inline FIDESlib::CKKS::Parameters params64_14{ .logN = 14, .L = 7, .dnum = 3, .primes = p64, .Sprimes = sp64 };

/** Following this: https://eprint.iacr.org/2024/463.pdf
 * Table 5.7 Col.1
 * Todo: Fix prime sizes.
 */
inline GeneralTestParams gparams64_14{ .multDepth = 7, .scaleModSize = 38, .batchSize = 8, .ringDim = 1 << 14, .dnum = 3, .GPUs = devices };

/** Following this: https://eprint.iacr.org/2024/463.pdf
 * Table 5.7 Col.1
 */
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_14 = std::tuple(gparams64_14, params64_14);

inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_14_fix{ tparams64_14,
	lbcrypto::ScalingTechnique::FIXEDMANUAL };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_14_fixauto{ tparams64_14,
	lbcrypto::ScalingTechnique::FIXEDAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_14_flex{ tparams64_14,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_14_flexext{ tparams64_14,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT };

/** Following this: https://eprint.iacr.org/2024/463.pdf
 * Table 5.7 Col.2
 */
inline FIDESlib::CKKS::Parameters params64_15{ .logN = 15, .L = 9, .dnum = 3, .primes = p64, .Sprimes = sp64 };

/** Following this: https://eprint.iacr.org/2024/463.pdf
 * Table 5.7 Col.2
 * Todo: Fix prime sizes.
 */
inline GeneralTestParams gparams64_15{ .multDepth = 9, .scaleModSize = 41, .batchSize = 8, .ringDim = 1 << 15, .dnum = 3, .GPUs = devices };

/** Following this: https://eprint.iacr.org/2024/463.pdf
 * Table 5.7 Col.2
 */
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_15 = std::tuple(gparams64_15, params64_15);

inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_15_fix{ tparams64_15,
	lbcrypto::ScalingTechnique::FIXEDMANUAL };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_15_fixauto{ tparams64_15,
	lbcrypto::ScalingTechnique::FIXEDAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_15_flex{ tparams64_15,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_15_flexext{ tparams64_15,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT };

/** Following this: https://eprint.iacr.org/2024/463.pdf
 * Table 5.8 Col.1
 */
inline FIDESlib::CKKS::Parameters params64_16{ .logN = 16, .L = 29, .dnum = 4, .primes = p64, .Sprimes = sp64 };

/** Following this: https://eprint.iacr.org/2024/463.pdf
 * Table 5.8 Col.1
 * Todo: Fix prime sizes.
 */
inline GeneralTestParams gparams64_16{ .multDepth = 29, .firstModSize = 60, .scaleModSize = 59 /*35 fails*/, .batchSize = 8, .ringDim = 1 << 16, .dnum = 4, .GPUs = devices };

/** Following this: https://eprint.iacr.org/2024/463.pdf
 * Table 5.8 Col.1
 */
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_16 = std::tuple(gparams64_16, params64_16);

inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_fix{ tparams64_16,
	lbcrypto::ScalingTechnique::FIXEDMANUAL };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_fixauto{ tparams64_16,
	lbcrypto::ScalingTechnique::FIXEDAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_flex{ tparams64_16,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_flexext{ tparams64_16,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT };

inline FIDESlib::CKKS::Parameters params64_16_boot1{ .logN = 16, .L = 23, .dnum = 4, .primes = p64, .Sprimes = sp64 };
inline GeneralTestParams gparams64_16_boot1{ .multDepth = 23, .firstModSize = 52, .scaleModSize = 48 /*35 fails*/, .batchSize = 8, .ringDim = 1 << 16, .dnum = 4, .GPUs = devices };
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_16_boot1 = std::tuple(gparams64_16_boot1, params64_16_boot1);

inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_boot1_fix{ tparams64_16_boot1,
	lbcrypto::ScalingTechnique::FIXEDMANUAL };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_boot1_fixauto{ tparams64_16_boot1,
	lbcrypto::ScalingTechnique::FIXEDAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_boot1_flex{ tparams64_16_boot1,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_boot1_flexext{ tparams64_16_boot1,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT };

inline FIDESlib::CKKS::Parameters params64_16_boot2{ .logN = 16, .L = 34, .dnum = 5, .primes = p64, .Sprimes = sp64 };
inline GeneralTestParams gparams64_16_boot2{ .multDepth = 34, .scaleModSize = 59 /*35 fails*/, .batchSize = 8, .ringDim = 1 << 16, .dnum = 5, .GPUs = devices };
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_16_boot2 = std::tuple(gparams64_16_boot2, params64_16_boot2);
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_boot2_fix{ tparams64_16_boot2,
	lbcrypto::ScalingTechnique::FIXEDMANUAL };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_boot2_fixauto{ tparams64_16_boot2,
	lbcrypto::ScalingTechnique::FIXEDAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_boot2_flex{ tparams64_16_boot2,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_boot2_flexext{ tparams64_16_boot2,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT };

inline FIDESlib::CKKS::Parameters params64_17{ .logN = 17, .L = 44, .dnum = 3, .primes = p64, .Sprimes = sp64 };
inline GeneralTestParams gparams64_17{ .multDepth = 44, .scaleModSize = 59 /*35 fails*/, .batchSize = 8, .ringDim = 1 << 17, .dnum = 3, .GPUs = devices };
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_17 = std::tuple(gparams64_17, params64_17);
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_17_fix{ tparams64_17,
	lbcrypto::ScalingTechnique::FIXEDMANUAL };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_17_fixauto{ tparams64_17,
	lbcrypto::ScalingTechnique::FIXEDAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_17_flex{ tparams64_17,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTO };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_17_flexext{ tparams64_17,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT };

inline FIDESlib::CKKS::Parameters params64_13_LLM{ .logN = 13, .L = 24, .dnum = 1, .primes = p64, .Sprimes = sp64 };
inline GeneralTestParams gparams64_13_LLM{ .multDepth = 24,
	//   .firstScaleModSize = 59,  //  Check this at OpenFHE
	.scaleModSize = 59 /*35 fails*/,
	.batchSize	  = 8,
	.ringDim	  = 1 << 13,
	.dnum		  = 1,
	.GPUs		  = { 0 } };
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_13_LLM = std::tuple(gparams64_13_LLM, params64_13_LLM);
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_LLM_flex{ tparams64_13_LLM,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTO };

inline FIDESlib::CKKS::Parameters params64_15_LLM{ .logN = 15, .L = 26, .dnum = 5, .primes = p64, .Sprimes = sp64 };
inline GeneralTestParams gparams64_15_LLM{ .multDepth = 26, // 28
															//   .firstScaleModSize = 59,  //  Check this at OpenFHE
	.firstModSize = 60,
	.scaleModSize = 55 /*35 fails*/,
	.batchSize	  = 8,
	.ringDim	  = 1 << 15,
	.dnum		  = 5,
	.GPUs		  = { 0 } };
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_15_LLM = std::tuple(gparams64_15_LLM, params64_15_LLM);
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_15_LLM_flex{ tparams64_15_LLM,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTO };

// inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique>
// tparams64_15_LLM_flexext{tparams64_15_LLM, lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT};

inline FIDESlib::CKKS::Parameters params64_16_LLM{ .logN = 16, .L = 26, .dnum = 5, .primes = p64, .Sprimes = sp64 };
inline GeneralTestParams gparams64_16_LLM{ .multDepth = 26, .scaleModSize = 55, .batchSize = 8, .ringDim = 1 << 16, .dnum = 5, .GPUs = { 0 } };
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_16_LLM = std::tuple(gparams64_16_LLM, params64_16_LLM);
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_16_LLM_flex{ tparams64_16_LLM,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTO };

inline FIDESlib::CKKS::Parameters params64_17_LLM{ .logN = 17, .L = 26, .dnum = 5, .primes = p64, .Sprimes = sp64 };
inline GeneralTestParams gparams64_17_LLM{ .multDepth = 26, .scaleModSize = 55, .batchSize = 8, .ringDim = 1 << 17, .dnum = 5, .GPUs = { 0 } };
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_17_LLM = std::tuple(gparams64_17_LLM, params64_17_LLM);
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_17_LLM_flex{ tparams64_17_LLM,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTO };

inline FIDESlib::CKKS::Parameters params32_15{ .logN = 15, .L = 27, .dnum = 4, .primes = p32, .Sprimes = p32 };
inline GeneralTestParams gparams32_15{ .multDepth = 28, .scaleModSize = 28, .batchSize = 8, .ringDim = 1 << 15, .dnum = 4, .GPUs = devices };
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams32_15 = std::tuple(gparams32_15, params32_15);

constexpr int logNboot	   = 16;
constexpr int firstmodboot = 60;
constexpr int scalemodboot = 59;
constexpr int depthboot	   = 23;

inline FIDESlib::CKKS::Parameters params64_13_1{ .logN = logNboot, .L = depthboot, .dnum = 1, .primes = p64, .Sprimes = sp64 };
inline GeneralTestParams gparams64_13_1{ .multDepth = depthboot,
	.firstModSize									= firstmodboot, //  Check this at OpenFHE
	.scaleModSize									= scalemodboot /*35 fails*/,
	.batchSize										= 8,
	.ringDim										= 1 << logNboot,
	.dnum											= 1,
	.GPUs											= { 0 } };
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_13_1 = std::tuple(gparams64_13_1, params64_13_1);
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_1_flex{ tparams64_13_1,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_1_fix{ tparams64_13_1,
	lbcrypto::ScalingTechnique::FIXEDAUTO };

inline FIDESlib::CKKS::Parameters params64_13_2{ .logN = logNboot, .L = depthboot, .dnum = 2, .primes = p64, .Sprimes = sp64 };
inline GeneralTestParams gparams64_13_2{ .multDepth = depthboot,
	.firstModSize									= firstmodboot, //  Check this at OpenFHE
	.scaleModSize									= scalemodboot /*35 fails*/,
	.batchSize										= 8,
	.ringDim										= 1 << logNboot,
	.dnum											= 2,
	.GPUs											= { 0 } };
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_13_2 = std::tuple(gparams64_13_2, params64_13_2);
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_2_flex{ tparams64_13_2,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_2_fix{ tparams64_13_2,
	lbcrypto::ScalingTechnique::FIXEDMANUAL };

inline FIDESlib::CKKS::Parameters params64_13_3{ .logN = logNboot, .L = depthboot, .dnum = 3, .primes = p64, .Sprimes = sp64 };
inline GeneralTestParams gparams64_13_3{ .multDepth = depthboot,
	.firstModSize									= firstmodboot, //  Check this at OpenFHE
	.scaleModSize									= scalemodboot /*35 fails*/,
	.batchSize										= 8,
	.ringDim										= 1 << logNboot,
	.dnum											= 3,
	.GPUs											= { 0 } };
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_13_3 = std::tuple(gparams64_13_3, params64_13_3);
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_3_flex{ tparams64_13_3,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_3_fix{ tparams64_13_3,
	lbcrypto::ScalingTechnique::FIXEDMANUAL };

inline FIDESlib::CKKS::Parameters params64_13_4{ .logN = logNboot, .L = depthboot, .dnum = 4, .primes = p64, .Sprimes = sp64 };
inline GeneralTestParams gparams64_13_4{ .multDepth = depthboot,
	.firstModSize									= firstmodboot, //  Check this at OpenFHE
	.scaleModSize									= scalemodboot /*35 fails*/,
	.batchSize										= 8,
	.ringDim										= 1 << logNboot,
	.dnum											= 4,
	.GPUs											= { 0 } };
inline std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters> tparams64_13_4 = std::tuple(gparams64_13_4, params64_13_4);
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_4_flex{ tparams64_13_4,
	lbcrypto::ScalingTechnique::FLEXIBLEAUTOEXT };
inline std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique> tparams64_13_4_fix{ tparams64_13_4,
	lbcrypto::ScalingTechnique::FIXEDMANUAL };

#define ALL64 params64_13, params64_14, params64_15, params64_16, params64_17, params64_16_boot1, params64_16_boot2
#define TALL64 tparams64_13, tparams64_14, tparams64_15, tparams64_16, tparams64_17, tparams64_16_boot1, tparams64_16_boot2
#define TTALL64 \
	tparams64_13_fix, tparams64_13_fixauto, tparams64_13_flex, tparams64_13_flexext, tparams64_14_fix, tparams64_16_boot1_flex, tparams64_16_boot1_flexext
/**
		tparams64_14_fixauto, tparams64_14_flex, tparams64_14_flexext, tparams64_15_fix, tparams64_15_fixauto,
		tparams64_15_flex, tparams64_15_flexext, tparams64_15_LLM_flex, tparams64_16_fix, tparams64_16_fixauto,
		tparams64_16_flex, tparams64_16_flexext, tparams64_16_boot1_fix, tparams64_16_boot1_fixauto,
		tparams64_16_boot1_flex, tparams64_16_boot1_flexext, tparams64_16_boot2_fix, tparams64_16_boot2_fixauto, tparams64_16_boot2_flex,
		tparams64_16_boot2_flexext, tparams64_17_fix, tparams64_17_fixauto, tparams64_17_flex, tparams64_17_flexext
*/
#define TTALL64BOOT \
	tparams64_13_1_flex, tparams64_13_1_fix, tparams64_13_2_flex, tparams64_13_2_fix, tparams64_13_3_flex, tparams64_13_3_fix, tparams64_13_4_flex, tparams64_13_4_fix

/**
,tparams64_15_LLM_flex,
  tparams64_16_boot1_fix, tparams64_16_boot1_fixauto, tparams64_16_boot1_flex,
   tparams64_16_boot1_flexext, tparams64_16_fix, tparams64_16_fixauto, tparams64_16_flex, tparams64_16_flexext, tparams64_16_boot2_fix,
	tparams64_16_boot2_fixauto, tparams64_16_boot2_flex, tparams64_16_boot2_flexext
*/

class FIDESlibParametrizedTest : public testing::TestWithParam<FIDESlib::CKKS::Parameters> {
  protected:
	FIDESlib::CKKS::Parameters fideslibParams{};

	void SetUp() override {
		fideslibParams = GetParam();
	}
};

inline std::map<uint64_t, std::pair<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>, lbcrypto::KeyPair<lbcrypto::DCRTPoly>>> cached_cc;

class GeneralParametrizedTest : public testing::TestWithParam<std::tuple<std::tuple<GeneralTestParams, FIDESlib::CKKS::Parameters>, lbcrypto::ScalingTechnique>> {
  protected:
	GeneralTestParams generalTestParams;
	FIDESlib::CKKS::Parameters fideslibParams{};

	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc{};
	lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys;
	CKKS::Context GPUcc;

	void SetUp() override {
		auto params		  = GetParam();
		generalTestParams = std::get<0>(std::get<0>(params));
		fideslibParams	  = std::get<1>(std::get<0>(params));

		char* res = getenv("FIDESLIB_USE_NUM_GPUS");

		if (res && !(0 == std::strcmp(res, ""))) {
			int num_dev = atoi(res);
			if (num_dev > 0) {
				std::vector<int> dev;
				for (int i = 0; i < num_dev; ++i) {
					dev.push_back(i);
				}
				devices = dev;
			}
			std::cout << "Devices: " << num_dev << std::endl;
		}

		uint64_t index = generalTestParams.ringDim + (1ul << 20) * generalTestParams.multDepth + std::get<1>(params) + (1ul << 30) * generalTestParams.dnum +
		  (1ul << 40) * generalTestParams.scaleModSize;
		if (cached_cc.contains(index)) {
			cc	 = cached_cc[index].first;
			keys = cached_cc[index].second;
			cc->GetEncodingParams()->SetBatchSize(generalTestParams.batchSize);
		} else {
			lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
			parameters.SetMultiplicativeDepth(generalTestParams.multDepth);
			parameters.SetFirstModSize(generalTestParams.firstModSize);
			parameters.SetScalingModSize(generalTestParams.scaleModSize);
			parameters.SetBatchSize(generalTestParams.batchSize);
			parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
			parameters.SetRingDim(generalTestParams.ringDim);
			parameters.SetNumLargeDigits(generalTestParams.dnum);
			parameters.SetScalingTechnique(std::get<1>(params));
			parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
			parameters.SetPREMode(lbcrypto::INDCPA);

			cc = GenCryptoContext(parameters);
			cc->Enable(lbcrypto::PKE);
			keys			 = cc->KeyGen();
			cached_cc[index] = { cc, keys };
		}
	}

	void TearDown() override {
		// Clean‑up resources that were allocated in SetUp() or during the test.

		if (cc) {
			cc->ClearEvalAutomorphismKeys(); // free evaluation keys, if needed
			if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE))
				std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
		}
		if (GPUcc) {
			GPUcc->clearAuxilarPoly();
			GPUcc->precom.monomialCache.clear();
			GPUcc->clearAutomorphismKeys();
			GPUcc->clearBootPrecomputation();
			GPUcc->clearEvalMultKeys();
			GPUcc->clearParamSwitchKeys();
		}
	}
};

#define ASSERT_EQ_CIPHERTEXT(ct1, ct2)                                                                                                                           \
	do {                                                                                                                                                         \
		ASSERT_EQ(ct1->GetNoiseScaleDeg(), ct2->GetNoiseScaleDeg());                                                                                             \
		ASSERT_EQ(ct1->GetScalingFactor(), ct2->GetScalingFactor());                                                                                             \
		ASSERT_EQ(ct1->GetEncodingType(), ct2->GetEncodingType());                                                                                               \
		for (size_t j = 0; j < 2; ++j) {                                                                                                                         \
			ASSERT_EQ(ct1.get()->GetElements().at(j).GetAllElements().size(), ct2.get()->GetElements().at(j).GetAllElements().size());                           \
			for (size_t i = 0; i < ct1.get()->GetElements().at(j).GetAllElements().size(); ++i) {                                                                \
				std::cout << "(" << j << ", " << i << ") " << std::flush;                                                                                        \
				ASSERT_EQ(ct1.get()->GetElements().at(j).GetAllElements().at(i).GetValues().GetLength(),                                                         \
				  ct2.get()->GetElements().at(j).GetAllElements().at(i).GetValues().GetLength());                                                                \
				ASSERT_EQ(ct1.get()->GetElements().at(j).GetAllElements().at(i).GetValues(), ct2.get()->GetElements().at(j).GetAllElements().at(i).GetValues()); \
			}                                                                                                                                                    \
		}                                                                                                                                                        \
		std::cout << std::endl;                                                                                                                                  \
	} while (0);

#define ASSERT_ERROR_OK(result, resultGPU)                                                                                                 \
	do {                                                                                                                                   \
		double acc = 0.0;                                                                                                                  \
		double Max = 0.0;                                                                                                                  \
		for (size_t i = 0; i < result->GetSlots(); ++i) {                                                                                  \
			double diff = abs(resultGPU->GetRealPackedValue().at(i) - result->GetRealPackedValue().at(i));                                 \
			acc += diff * diff;                                                                                                            \
			Max = std::max(Max, diff);                                                                                                     \
		}                                                                                                                                  \
		acc = std::sqrt(acc / result->GetSlots());                                                                                         \
		std::cout << "Max error: " << Max << " (Expected: " << pow(2.0, -result->GetLogPrecision() + 1) << "), dev: " << acc << std::endl; \
		ASSERT_LE(Max, pow(2.0, -result->GetLogPrecision() + 4));                                                                          \
	} while (0);

#define ASSERT_EQ_DCRTPOLY(ct1, ct2)                                                                                         \
	do {                                                                                                                     \
		for (int j = 0; j < 1; ++j) {                                                                                        \
			ASSERT_EQ(ct1->m_vectors.size(), ct2->m_vectors.size());                                                         \
			for (int i = 0; i < ct1->m_vectors.size(); ++i) {                                                                \
				/*    std::cout << j << " " << i << std::endl;       */                                                      \
				ASSERT_EQ(ct1->m_vectors[i].m_values.get()->m_data.size(), ct2->m_vectors[i].m_values.get()->m_data.size()); \
				ASSERT_EQ(ct1->m_vectors[i].m_values.get()->m_data, ct2->m_vectors[i].m_values.get()->m_data);               \
			}                                                                                                                \
		}                                                                                                                    \
	} while (0);

} // namespace FIDESlib::Testing

#endif // PARAMETRIZEDTEST_CUH