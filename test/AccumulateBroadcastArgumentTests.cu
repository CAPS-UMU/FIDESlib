//
// Argument validation of the Accumulate/Broadcast helpers. Host-only, no GPU context needed.
//

#include <gtest/gtest.h>
#include <stdexcept>

#include <CKKS/AccumulateBroadcast.cuh>

TEST(AccumulateBroadcast, RotationIndicesAcceptPowerOfTwoBStep) {
	EXPECT_NO_THROW(FIDESlib::CKKS::GetAccumulateRotationIndices(2, 1, 8));
	EXPECT_NO_THROW(FIDESlib::CKKS::GetAccumulateRotationIndices(8, 1, 64));
	EXPECT_NO_THROW(FIDESlib::CKKS::GetbroadcastRotationIndices(4, 8, 64));
}

TEST(AccumulateBroadcast, RotationIndicesRejectInvalidBStep) {
	// bStep == 1 would make the round loop (s <<= logbStep) never terminate.
	EXPECT_THROW(FIDESlib::CKKS::GetAccumulateRotationIndices(1, 1, 8), std::invalid_argument);
	// A bStep that is not a power of two sums some shifts twice.
	EXPECT_THROW(FIDESlib::CKKS::GetAccumulateRotationIndices(3, 1, 8), std::invalid_argument);
	EXPECT_THROW(FIDESlib::CKKS::GetbroadcastRotationIndices(0, 8, 64), std::invalid_argument);
	EXPECT_THROW(FIDESlib::CKKS::GetbroadcastRotationIndices(6, 8, 64), std::invalid_argument);
}
