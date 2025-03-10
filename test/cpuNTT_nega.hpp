//
// Created by carlosad on 12/09/24.
//

#ifndef GPUCKKS_CPUNTT_NEGA_HPP
#define GPUCKKS_CPUNTT_NEGA_HPP

#include <cinttypes>
#include <vector>

namespace FIDESlib::Testing {
void nega_fft(std::vector<uint64_t>& a, bool invert, const uint64_t* psi, const uint64_t* inv_psi, uint64_t mod,
              int its = 1000);

void nega_fft2(std::vector<uint64_t>& a, bool invert, const uint64_t* psi, const uint64_t* inv_psi, uint64_t mod,
               int its = 1000);

void nega_fft_forPrime(std::vector<uint64_t>& a, bool invert, int primeid, int its = 1000);

void nega_fft2_forPrime(std::vector<uint64_t>& a, bool invert, int primeid, int its = 1000);

template <typename T>
void nega_fft_2d(std::vector<T>& a, int sqrtN, int primeid);
}  // namespace FIDESlib::Testing

#endif  //GPUCKKS_CPUNTT_NEGA_HPP
