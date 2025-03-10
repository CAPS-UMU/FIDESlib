//
// Created by oscar on 22/10/24.
//

#include <benchmark/benchmark.h>

#include "Benchmark.cuh"
namespace FIDESlib::Benchmarks {

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyModUp)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_limbs"] = state.range(1);
    for (auto _ : state) {
        if (cc.L <= state.range(2)) {
            state.SkipWithMessage("cc.L <= initial levels");
            break;
        }
        FIDESlib::CKKS::RNSPoly a(cc, state.range(2));

        auto start = std::chrono::high_resolution_clock::now();
        a.modup();
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
        CudaCheckErrorMod;
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyModUpContextLimbCount)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    for (auto _ : state) {
        FIDESlib::CKKS::RNSPoly a(cc, cc.L);

        auto start = std::chrono::high_resolution_clock::now();
        a.modup();
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
        CudaCheckErrorMod;
    }
    CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyModUp)->ArgsProduct({{2}, {true, false}, {0, 1, 8, 16}})->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyModUpContextLimbCount)->ArgsProduct({{2}, {true, false}})->UseManualTime();

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyStandardModDown)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    state.counters["p_limbs"] = state.range(2);
    for (auto _ : state) {
        if (cc.L <= state.range(2)) {
            state.SkipWithMessage("cc.L <= initial levels");
            break;
        }
        FIDESlib::CKKS::RNSPoly a(cc, state.range(2));
        a.modup();
        a.generateSpecialLimbs();
        auto start = std::chrono::high_resolution_clock::now();
        a.moddown<FIDESlib::ALGO_NATIVE>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
        CudaCheckErrorMod;
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyStandardModDownContextLimbCount)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    for (auto _ : state) {
        FIDESlib::CKKS::RNSPoly a(cc, cc.L);
        a.modup();
        a.generateSpecialLimbs();
        auto start = std::chrono::high_resolution_clock::now();
        a.moddown<FIDESlib::ALGO_NATIVE>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
        CudaCheckErrorMod;
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyStandardModUpModDown)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    state.counters["p_limbs"] = state.range(2);
    for (auto _ : state) {
        if (cc.L <= state.range(2)) {
            state.SkipWithMessage("cc.L <= initial levels");
            break;
        }
        FIDESlib::CKKS::RNSPoly a(cc, state.range(2));
        auto start = std::chrono::high_resolution_clock::now();
        a.modup();
        a.generateSpecialLimbs();
        a.moddown<FIDESlib::ALGO_NATIVE>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyStandardModUpModDownContextLimbCount)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    for (auto _ : state) {
        FIDESlib::CKKS::RNSPoly a(cc, cc.L);
        auto start = std::chrono::high_resolution_clock::now();
        a.modup();
        a.generateSpecialLimbs();
        a.moddown<FIDESlib::ALGO_NATIVE>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyStandardModDown)
    ->ArgsProduct({{2}, {true, false}, {0, 1, 8, 16}})
    ->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyStandardModDownContextLimbCount)
    ->ArgsProduct({{2}, {true, false}})
    ->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyStandardModUpModDown)
    ->ArgsProduct({{2}, {true, false}, {0, 1, 8, 16}})
    ->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyStandardModUpModDownContextLimbCount)
    ->ArgsProduct({{2}, {true, false}})
    ->UseManualTime();

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyNoneModDown)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    state.counters["p_limbs"] = state.range(2);
    for (auto _ : state) {
        if (cc.L <= state.range(2)) {
            state.SkipWithMessage("cc.L <= initial levels");
            break;
        }
        FIDESlib::CKKS::RNSPoly a(cc, state.range(2));
        a.modup();
        a.generateSpecialLimbs();
        auto start = std::chrono::high_resolution_clock::now();
        a.moddown<FIDESlib::ALGO_NONE>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyNoneModDownContextLimbCount)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    for (auto _ : state) {
        FIDESlib::CKKS::RNSPoly a(cc, cc.L);
        a.modup();
        a.generateSpecialLimbs();
        auto start = std::chrono::high_resolution_clock::now();
        a.moddown<FIDESlib::ALGO_NONE>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyNoneModUpModDown)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    state.counters["p_limbs"] = state.range(2);
    for (auto _ : state) {
        if (cc.L <= state.range(2)) {
            state.SkipWithMessage("cc.L <= initial levels");
            break;
        }
        FIDESlib::CKKS::RNSPoly a(cc, state.range(2));
        auto start = std::chrono::high_resolution_clock::now();
        a.modup();
        a.generateSpecialLimbs();
        a.moddown<FIDESlib::ALGO_NONE>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyNoneModUpModDownContextLimbCount)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    for (auto _ : state) {
        FIDESlib::CKKS::RNSPoly a(cc, cc.L);
        auto start = std::chrono::high_resolution_clock::now();
        a.modup();
        a.generateSpecialLimbs();
        a.moddown<FIDESlib::ALGO_NONE>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyNoneModDown)
    ->ArgsProduct({{2}, {true, false}, {0, 1, 8, 16}})
    ->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyNoneModDownContextLimbCount)
    ->ArgsProduct({{2}, {true, false}})
    ->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyNoneModUpModDown)
    ->ArgsProduct({{2}, {true, false}, {0, 1, 8, 16}})
    ->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyNoneModUpModDownContextLimbCount)
    ->ArgsProduct({{2}, {true, false}})
    ->UseManualTime();

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyShoupNoRedModDown)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    state.counters["p_limbs"] = state.range(2);
    for (auto _ : state) {
        if (cc.L <= state.range(2)) {
            state.SkipWithMessage("cc.L <= initial levels");
            break;
        }
        FIDESlib::CKKS::RNSPoly a(cc, state.range(2));
        a.modup();
        a.generateSpecialLimbs();
        auto start = std::chrono::high_resolution_clock::now();
        a.moddown<FIDESlib::ALGO_SHOUP>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyShoupNoRedModDownContextLimbCount)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    for (auto _ : state) {
        FIDESlib::CKKS::RNSPoly a(cc, cc.L);
        a.modup();
        a.generateSpecialLimbs();
        auto start = std::chrono::high_resolution_clock::now();
        a.moddown<FIDESlib::ALGO_SHOUP>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyShoupNoRedModUpModDown)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    state.counters["p_limbs"] = state.range(2);
    for (auto _ : state) {
        if (cc.L <= state.range(2)) {
            state.SkipWithMessage("cc.L <= initial levels");
            break;
        }
        FIDESlib::CKKS::RNSPoly a(cc, state.range(2));
        auto start = std::chrono::high_resolution_clock::now();
        a.modup();
        a.generateSpecialLimbs();
        a.moddown<FIDESlib::ALGO_SHOUP>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyShoupNoRedModUpModDownContextLimbCount)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    for (auto _ : state) {
        FIDESlib::CKKS::RNSPoly a(cc, cc.L);
        auto start = std::chrono::high_resolution_clock::now();
        a.modup();
        a.generateSpecialLimbs();
        a.moddown<FIDESlib::ALGO_SHOUP>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyShoupNoRedModDown)
    ->ArgsProduct({{2}, {true, false}, {0, 1, 8, 16}})
    ->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyShoupNoRedModDownContextLimbCount)
    ->ArgsProduct({{2}, {true, false}})
    ->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyShoupNoRedModUpModDown)
    ->ArgsProduct({{2}, {true, false}, {0, 1, 8, 16}})
    ->UseManualTime();
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyShoupNoRedModUpModDownContextLimbCount)
    ->ArgsProduct({{2}, {true, false}})
    ->UseManualTime();

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyBarretModDown)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    state.counters["p_limbs"] = state.range(2);
    for (auto _ : state) {
        if (cc.L <= state.range(2)) {
            state.SkipWithMessage("cc.L <= initial levels");
            break;
        }
        FIDESlib::CKKS::RNSPoly a(cc, state.range(2));
        a.modup();
        a.generateSpecialLimbs();
        auto start = std::chrono::high_resolution_clock::now();
        a.moddown<FIDESlib::ALGO_BARRETT>(state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}

BENCHMARK_DEFINE_F(FIDESlibFixture, RNSPolyBarretModDownContextLimbCount)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    CudaCheckErrorMod;
    state.counters["p_ntt"] = state.range(1);
    for (auto _ : state) {
        FIDESlib::CKKS::RNSPoly a(cc, cc.L);
        a.modup();
        a.generateSpecialLimbs();
        auto start = std::chrono::high_resolution_clock::now();
        a.moddown<FIDESlib::ALGO_BARRETT>(  state.range(1));
        CudaCheckErrorMod;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed.count());
    }
    CudaCheckErrorMod;
}
/*
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyBarretModDown)
    ->ArgsProduct({{2, 3, 4, 5}, {false, true}, {0, 1, 8, 16}})
    ->UseManualTime();
*/
BENCHMARK_REGISTER_F(FIDESlibFixture, RNSPolyBarretModDownContextLimbCount)
    ->ArgsProduct({{2, 3, 4, 5}, {false, true}})
    ->UseManualTime();

}  // namespace FIDESlib::Benchmarks