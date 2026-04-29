//
// Created by carlosad on 11/3/26.
//
#include "ParametrizedTest.cuh"

class GlobalEnv : public ::testing::Environment {
  public:
	void SetUp() override {
		// global initialization before any test runs
	}

	void TearDown() override {
		FIDESlib::CKKS::DeregisterAllContexts();
	}
};

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	::testing::AddGlobalTestEnvironment(new GlobalEnv());
	return RUN_ALL_TESTS();
}