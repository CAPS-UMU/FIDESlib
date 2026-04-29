//
// Created by seyda on 5/19/25.
//
#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "pke/openfhe.h"

namespace FIDESlib::CKKS {
std::vector<double> get_chebyshev_coefficients(const std::function<double(double)>& func, double a, double b, uint32_t degree);

} // namespace FIDESlib::CKKS