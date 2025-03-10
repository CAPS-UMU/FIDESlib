//
// Created by carlosad on 23/03/24.
//
#include <gtest/gtest.h>

#include "CKKS/Context.cuh"
#include "ConstantsGPU.cuh"
#include "Math.cuh"
#include "ParametrizedTest.cuh"

namespace FIDESlib::Testing {
class MathTest : public FIDESlibParametrizedTest {};

TEST_P(MathTest, ModPowTestSmallNumbers) {
    EXPECT_EQ(FIDESlib::modpow(2, 3, 5), 3);
    EXPECT_EQ(FIDESlib::modpow(3, 5, 7), 5);
}

TEST_P(MathTest, ModPowTestLargeNumbers) {
    EXPECT_EQ(FIDESlib::modpow(123456789, 987654321, 1000000007), 652541198);
    EXPECT_EQ(FIDESlib::modpow(987654321, 123456789, 1000000007), 379110096);
}

// Test cases for modinv function
TEST_P(MathTest, ModInvTestSmallNumbers) {
    EXPECT_EQ(FIDESlib::modinv(2, 5), 3);
    EXPECT_EQ(FIDESlib::modinv(3, 7), 5);
}

TEST_P(MathTest, ModInvTestLargeNumbers) {
    EXPECT_EQ(FIDESlib::modinv(123456789, 1000000007), 18633540);
    EXPECT_EQ(FIDESlib::modinv(987654321, 1000000007), 152057246);
}

TEST_P(MathTest, ModInvTestVeryLargeNumbers) {
    EXPECT_EQ(FIDESlib::modinv(123456789ll, ((1ll << 61) - 1)), 2217090678635848435ll);
}

INSTANTIATE_TEST_SUITE_P(MathTests, MathTest, testing::Values(params64_16));
}  // namespace FIDESlib::Testing