//
// Created by oscar on 11/10/24.
//

#include <vector>

#include "Benchmark.cuh"

namespace FIDESlib::Benchmarks {

std::map<int, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>> context_map;
std::map<int, lbcrypto::KeyPair<lbcrypto::DCRTPoly>> key_map;

std::vector<FIDESlib::PrimeRecord> p32{
    {.p = 537133057}, {.p = 537591809}, {.p = 537722881}, {.p = 538116097}, {.p = 539754497}, {.p = 540082177},
    {.p = 540540929}, {.p = 540672001}, {.p = 541327361}, {.p = 541655041}, {.p = 542310401}, {.p = 543031297},
    {.p = 543293441}, {.p = 545062913}, {.p = 546766849}, {.p = 547749889}, {.p = 548012033}, {.p = 548208641},
    {.p = 548339713}, {.p = 549388289}, {.p = 549978113}, {.p = 550371329}, {.p = 551288833}, {.p = 552861697},
    {.p = 552927233}, {.p = 553254913}, {.p = 554106881}, {.p = 554631169}, {.p = 555220993}, {.p = 555810817},
    {.p = 556072961}, {.p = 557187073}, {.p = 557776897}, {.p = 558235649}, {.p = 558432257}, {.p = 559742977},
    {.p = 561184769}, {.p = 561774593}, {.p = 562364417}, {.p = 562495489}, {.p = 562757633}, {.p = 563150849},
    {.p = 563281921}, {.p = 564658177}, {.p = 565641217}, {.p = 566886401}, {.p = 567869441}, {.p = 568000513},
    {.p = 568066049}, {.p = 568262657}, {.p = 568655873}, {.p = 569573377}, {.p = 569638913}, {.p = 570163201},
    {.p = 570949633}, {.p = 571146241}, {.p = 571539457}, {.p = 572915713}, {.p = 575078401}, {.p = 575275009},
    {.p = 575733761}, {.p = 575864833}, {.p = 576454657}, {.p = 576716801}, {.p = 576913409}, {.p = 577437697},
    {.p = 578093057}, {.p = 580190209}, {.p = 580780033}, {.p = 581632001}, {.p = 581959681}, {.p = 582746113},
    {.p = 583794689}, {.p = 584384513}, {.p = 584581121}, {.p = 584777729}, {.p = 585695233}, {.p = 586285057},
    {.p = 587530241}, {.p = 587661313}, {.p = 590479361}};

std::vector<FIDESlib::PrimeRecord> p64{
    {.p = 2305843009218281473}, {.p = 2251799661248513}, {.p = 2251799661641729}, {.p = 2251799665180673},
    {.p = 2251799682088961},    {.p = 2251799678943233}, {.p = 2251799717609473}, {.p = 2251799710138369},
    {.p = 2251799708827649},    {.p = 2251799707385857}, {.p = 2251799713677313}, {.p = 2251799712366593},
    {.p = 2251799716691969},    {.p = 2251799714856961}, {.p = 2251799726522369}, {.p = 2251799726129153},
    {.p = 2251799747493889},    {.p = 2251799741857793}, {.p = 2251799740416001}, {.p = 2251799746707457},
    {.p = 2251799756013569},    {.p = 2251799775805441}, {.p = 2251799763091457}, {.p = 2251799767154689},
    {.p = 2251799765975041},    {.p = 2251799770562561}, {.p = 2251799769776129}, {.p = 2251799772266497},
    {.p = 2251799775281153},    {.p = 2251799774887937}, {.p = 2251799797432321}, {.p = 2251799787995137},
    {.p = 2251799787601921},    {.p = 2251799791403009}, {.p = 2251799789568001}, {.p = 2251799795466241},
    {.p = 2251799807131649},    {.p = 2251799806345217}, {.p = 2251799805165569}, {.p = 2251799813554177},
    {.p = 2251799809884161},    {.p = 2251799810670593}, {.p = 2251799818928129}, {.p = 2251799816568833},
    {.p = 2251799815520257}};

std::vector<FIDESlib::PrimeRecord> sp64{
    {.p = 2305843009218936833}, {.p = 2305843009220116481}, {.p = 2305843009221820417}, {.p = 2305843009224179713},
    {.p = 2305843009225228289}, {.p = 2305843009227980801}, {.p = 2305843009229160449}, {.p = 2305843009229946881},
    {.p = 2305843009231650817}, {.p = 2305843009235189761}, {.p = 2305843009240301569}, {.p = 2305843009242923009},
    {.p = 2305843009244889089}, {.p = 2305843009245413377}, {.p = 2305843009247641601}};

FIDESlib::CKKS::Parameters params32{.logN = 15, .L = 32, .dnum = 4, .primes = p32, .Sprimes = p32};

FIDESlib::CKKS::Parameters params64_13{.logN = 13, .L = 32, .dnum = 4, .primes = p64, .Sprimes = sp64};
FIDESlib::CKKS::Parameters params64_14{.logN = 14, .L = 32, .dnum = 4, .primes = p64, .Sprimes = sp64};
FIDESlib::CKKS::Parameters params64_15{.logN = 15, .L = 32, .dnum = 4, .primes = p64, .Sprimes = sp64};
FIDESlib::CKKS::Parameters params64_16{.logN = 16, .L = 32, .dnum = 4, .primes = p64, .Sprimes = sp64};

constexpr int msb(int par) {
    int index = 0;
    while (par >>= 1)
        index++;
    return index;
}

GeneralBenchParams gparams64_13{.multDepth = 5,
                                .scaleModSize = 36,
                                .batchSize = 8,
                                .ringDim = 1 << 13,
                                .dnum = 2,
                                .GPUs = {0}};

GeneralBenchParams gparams64_14{.multDepth = 7,
                                .scaleModSize = 38,
                                .batchSize = 8,
                                .ringDim = 1 << 14,
                                .dnum = 3,
                                .GPUs = {0}};

GeneralBenchParams gparams64_15{.multDepth = 9,
                                .scaleModSize = 41,
                                .batchSize = 8,
                                .ringDim = 1 << 15,
                                .dnum = 3,
                                .GPUs = {0}};

GeneralBenchParams gparams64_16{.multDepth = 29,
                                .scaleModSize = 59 /*35*/,
                                .batchSize = 8,
                                .ringDim = 1 << 16,
                                .dnum = 4,
                                .GPUs = {0}};

GeneralBenchParams gparams64_16_boot2{.multDepth = 34,
                                      .scaleModSize = 59 /*35*/,
                                      .batchSize = 8,
                                      .ringDim = 1 << 16,
                                      .dnum = 5,
                                      .GPUs = {0}};

GeneralBenchParams gparams64_16_boot1{.multDepth = 26,
                                      .scaleModSize = 59 /*35*/,
                                      .batchSize = 8,
                                      .ringDim = 1 << 16,
                                      .dnum = 4,
                                      .GPUs = {0}};

GeneralBenchParams gparams32_15{.multDepth = 27,
                                .scaleModSize = 28,
                                .batchSize = 8,
                                .ringDim = 1 << 15,
                                .dnum = 4,
                                .GPUs = {0}};

GeneralBenchParams gparams64_17{.multDepth = 44,
                                .scaleModSize = 59 /*35 fails*/,
                                .batchSize = 8,
                                .ringDim = 1 << 17,
                                .dnum = 3,
                                .GPUs = {0}};

/**  FOR THE ADDITIONAL MODES*/

GeneralBenchParams gparams64_13_auto{.multDepth = 5,
                                     .scaleModSize = 36,
                                     .batchSize = 8,
                                     .ringDim = 1 << 13,
                                     .dnum = 2,
                                     .GPUs = {0},
                                     .tech = lbcrypto::FIXEDAUTO};

GeneralBenchParams gparams64_14_auto{.multDepth = 7,
                                     .scaleModSize = 38,
                                     .batchSize = 8,
                                     .ringDim = 1 << 14,
                                     .dnum = 3,
                                     .GPUs = {0},
                                     .tech = lbcrypto::FIXEDAUTO};

GeneralBenchParams gparams64_15_auto{.multDepth = 9,
                                     .scaleModSize = 41,
                                     .batchSize = 8,
                                     .ringDim = 1 << 15,
                                     .dnum = 3,
                                     .GPUs = {0},
                                     .tech = lbcrypto::FIXEDAUTO};

GeneralBenchParams gparams64_16_auto{.multDepth = 29,
                                     .scaleModSize = 59 /*35*/,
                                     .batchSize = 8,
                                     .ringDim = 1 << 16,
                                     .dnum = 4,
                                     .GPUs = {0},
                                     .tech = lbcrypto::FIXEDAUTO};

GeneralBenchParams gparams64_16_boot2_auto{.multDepth = 34,
                                           .scaleModSize = 59 /*35*/,
                                           .batchSize = 8,
                                           .ringDim = 1 << 16,
                                           .dnum = 5,
                                           .GPUs = {0},
                                           .tech = lbcrypto::FIXEDAUTO};

GeneralBenchParams gparams64_16_boot1_auto{.multDepth = 26,
                                           .scaleModSize = 59 /*35*/,
                                           .batchSize = 8,
                                           .ringDim = 1 << 16,
                                           .dnum = 4,
                                           .GPUs = {0},
                                           .tech = lbcrypto::FIXEDAUTO};

GeneralBenchParams gparams32_15_auto{.multDepth = 27,
                                     .scaleModSize = 28,
                                     .batchSize = 8,
                                     .ringDim = 1 << 15,
                                     .dnum = 4,
                                     .GPUs = {0},
                                     .tech = lbcrypto::FIXEDAUTO};

GeneralBenchParams gparams64_17_auto{.multDepth = 44,
                                     .scaleModSize = 59 /*35 fails*/,
                                     .batchSize = 8,
                                     .ringDim = 1 << 17,
                                     .dnum = 3,
                                     .GPUs = {0},
                                     .tech = lbcrypto::FIXEDAUTO};

GeneralBenchParams gparams64_13_flex{.multDepth = 5,
                                     .scaleModSize = 36,
                                     .batchSize = 8,
                                     .ringDim = 1 << 13,
                                     .dnum = 2,
                                     .GPUs = {0},
                                     .tech = lbcrypto::FLEXIBLEAUTO};

GeneralBenchParams gparams64_14_flex{.multDepth = 7,
                                     .scaleModSize = 38,
                                     .batchSize = 8,
                                     .ringDim = 1 << 14,
                                     .dnum = 3,
                                     .GPUs = {0},
                                     .tech = lbcrypto::FLEXIBLEAUTO};

GeneralBenchParams gparams64_15_flex{.multDepth = 9,
                                     .scaleModSize = 41,
                                     .batchSize = 8,
                                     .ringDim = 1 << 15,
                                     .dnum = 3,
                                     .GPUs = {0},
                                     .tech = lbcrypto::FLEXIBLEAUTO};

GeneralBenchParams gparams64_16_flex{.multDepth = 29,
                                     .scaleModSize = 59 /*35*/,
                                     .batchSize = 8,
                                     .ringDim = 1 << 16,
                                     .dnum = 4,
                                     .GPUs = {0},
                                     .tech = lbcrypto::FLEXIBLEAUTO};

GeneralBenchParams gparams64_16_boot2_flex{.multDepth = 34,
                                           .scaleModSize = 59 /*35*/,
                                           .batchSize = 8,
                                           .ringDim = 1 << 16,
                                           .dnum = 5,
                                           .GPUs = {0},
                                           .tech = lbcrypto::FLEXIBLEAUTO};

GeneralBenchParams gparams64_16_boot1_flex{.multDepth = 26,
                                           .scaleModSize = 59 /*35*/,
                                           .batchSize = 8,
                                           .ringDim = 1 << 16,
                                           .dnum = 4,
                                           .GPUs = {0},
                                           .tech = lbcrypto::FLEXIBLEAUTO};

GeneralBenchParams gparams32_15_flex{.multDepth = 27,
                                     .scaleModSize = 28,
                                     .batchSize = 8,
                                     .ringDim = 1 << 15,
                                     .dnum = 4,
                                     .GPUs = {0},
                                     .tech = lbcrypto::FLEXIBLEAUTO};

GeneralBenchParams gparams64_17_flex{.multDepth = 44,
                                     .scaleModSize = 59 /*35 fails*/,
                                     .batchSize = 8,
                                     .ringDim = 1 << 17,
                                     .dnum = 3,
                                     .GPUs = {0},
                                     .tech = lbcrypto::FLEXIBLEAUTO};

GeneralBenchParams gparams64_13_flexext{.multDepth = 5,
                                        .scaleModSize = 36,
                                        .batchSize = 8,
                                        .ringDim = 1 << 13,
                                        .dnum = 2,
                                        .GPUs = {0},
                                        .tech = lbcrypto::FLEXIBLEAUTOEXT};

GeneralBenchParams gparams64_14_flexext{.multDepth = 7,
                                        .scaleModSize = 38,
                                        .batchSize = 8,
                                        .ringDim = 1 << 14,
                                        .dnum = 3,
                                        .GPUs = {0},
                                        .tech = lbcrypto::FLEXIBLEAUTOEXT};

GeneralBenchParams gparams64_15_flexext{.multDepth = 9,
                                        .scaleModSize = 41,
                                        .batchSize = 8,
                                        .ringDim = 1 << 15,
                                        .dnum = 3,
                                        .GPUs = {0},
                                        .tech = lbcrypto::FLEXIBLEAUTOEXT};

GeneralBenchParams gparams64_16_flexext{.multDepth = 29,
                                        .scaleModSize = 59 /*35*/,
                                        .batchSize = 8,
                                        .ringDim = 1 << 16,
                                        .dnum = 4,
                                        .GPUs = {0},
                                        .tech = lbcrypto::FLEXIBLEAUTOEXT};

GeneralBenchParams gparams64_16_boot2_flexext{.multDepth = 34,
                                              .scaleModSize = 59 /*35*/,
                                              .batchSize = 8,
                                              .ringDim = 1 << 16,
                                              .dnum = 5,
                                              .GPUs = {0},
                                              .tech = lbcrypto::FLEXIBLEAUTOEXT};

GeneralBenchParams gparams64_16_boot1_flexext{.multDepth = 26,
                                              .scaleModSize = 59 /*35*/,
                                              .batchSize = 8,
                                              .ringDim = 1 << 16,
                                              .dnum = 4,
                                              .GPUs = {0},
                                              .tech = lbcrypto::FLEXIBLEAUTOEXT};

GeneralBenchParams gparams32_15_flexext{.multDepth = 27,
                                        .scaleModSize = 28,
                                        .batchSize = 8,
                                        .ringDim = 1 << 15,
                                        .dnum = 4,
                                        .GPUs = {0},
                                        .tech = lbcrypto::FLEXIBLEAUTOEXT};

GeneralBenchParams gparams64_17_flexext{.multDepth = 44,
                                        .scaleModSize = 59 /*35 fails*/,
                                        .batchSize = 8,
                                        .ringDim = 1 << 17,
                                        .dnum = 3,
                                        .GPUs = {0},
                                        .tech = lbcrypto::FLEXIBLEAUTOEXT};

GeneralBenchParams gen_bench_params1 = gparams64_13;
GeneralBenchParams gen_bench_params2 = gparams64_14;
GeneralBenchParams gen_bench_params3 = gparams64_15;
GeneralBenchParams gen_bench_params4 = gparams64_16;
GeneralBenchParams gen_bench_params5 = gparams64_16_boot1;
GeneralBenchParams gen_bench_params6 = gparams64_16_boot2;
GeneralBenchParams gen_bench_params7 = gparams64_17;

FIDESlib::CKKS::Parameters gen_bench_params1_addpted{.logN = msb(static_cast<int>(gen_bench_params1.ringDim)),
                                                     .L = static_cast<int>(gen_bench_params1.multDepth),
                                                     .dnum = static_cast<int>(gen_bench_params1.dnum),
                                                     .primes = p64,
                                                     .Sprimes = sp64};

FIDESlib::CKKS::Parameters gen_bench_params2_addpted{.logN = msb(static_cast<int>(gen_bench_params2.ringDim)),
                                                     .L = static_cast<int>(gen_bench_params2.multDepth),
                                                     .dnum = static_cast<int>(gen_bench_params2.dnum),
                                                     .primes = p64,
                                                     .Sprimes = sp64};

FIDESlib::CKKS::Parameters gen_bench_params3_addpted{.logN = msb(static_cast<int>(gen_bench_params3.ringDim)),
                                                     .L = static_cast<int>(gen_bench_params3.multDepth),
                                                     .dnum = static_cast<int>(gen_bench_params3.dnum),
                                                     .primes = p64,
                                                     .Sprimes = sp64};

FIDESlib::CKKS::Parameters gen_bench_params4_addpted{.logN = msb(static_cast<int>(gen_bench_params4.ringDim)),
                                                     .L = static_cast<int>(gen_bench_params4.multDepth),
                                                     .dnum = static_cast<int>(gen_bench_params4.dnum),
                                                     .primes = p64,
                                                     .Sprimes = sp64};

std::array<FIDESlib::CKKS::Parameters, 9> fideslib_bench_params = {params32,
                                                                   params64_16,
                                                                   gen_bench_params1_addpted,
                                                                   gen_bench_params2_addpted,
                                                                   gen_bench_params3_addpted,
                                                                   gen_bench_params4_addpted,
                                                                   params64_13,
                                                                   params64_14,
                                                                   params64_15};

std::array<GeneralBenchParams, 28> general_bench_params = {
    gen_bench_params1,       gen_bench_params2,          gen_bench_params3,          gen_bench_params4,
    gen_bench_params5,       gen_bench_params6,          gen_bench_params7,          gparams64_13_auto,
    gparams64_14_auto,       gparams64_15_auto,          gparams64_16_auto,          gparams64_16_boot1_auto,
    gparams64_16_boot2_auto, gparams64_17_auto,          gparams64_13_flex,          gparams64_14_flex,
    gparams64_15_flex,       gparams64_16_flex,          gparams64_16_boot1_flex,    gparams64_16_boot2_flex,
    gparams64_17_flex,       gparams64_13_flexext,       gparams64_14_flexext,       gparams64_15_flexext,
    gparams64_16_flexext,    gparams64_16_boot1_flexext, gparams64_16_boot2_flexext, gparams64_17_flexext,
};

}  // namespace FIDESlib::Benchmarks

int main(int argc, char** argv) {

    FIDESlib::Benchmarks::GeneralFixture::SetContext();
    FIDESlib::Benchmarks::FIDESlibFixture::SetContext();
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
}
