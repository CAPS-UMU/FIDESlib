//
// Created by carlosad on 3/05/24.
//

#ifndef FIDESLIB_CPUNTT_HPP
#define FIDESLIB_CPUNTT_HPP

#include <cinttypes>
#include <vector>

namespace FIDESlib::Testing {
void fft(std::vector<uint64_t>& a, bool invert, uint64_t root, uint64_t root_1, uint64_t mod, int its = 1000);

void fft_forPrime(std::vector<uint64_t>& a, bool invert, int primeid, int its = 1000);

template <typename T>
void fft_2d(std::vector<T>& a, int sqrtN, int primeid);
}  // namespace FIDESlib::Testing
#endif  //FIDESLIB_CPUNTT_HPP
