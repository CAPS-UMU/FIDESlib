//
// Created by oscar on 22/10/24.
//

#include <benchmark/benchmark.h>

#include "Benchmark.cuh"
namespace FIDESlib::Benchmarks {
BENCHMARK_DEFINE_F(FIDESlibFixture, LimbINTT32)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();

    FIDESlib::CKKS::Limb<uint32_t> limb(cc, GPUs[0], s, 0);
    limb.load(v);

    for (auto _ : state) {
        limb.INTT<FIDESlib::ALGO_NATIVE>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbNTT32)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();

    FIDESlib::CKKS::Limb<uint32_t> limb(cc, GPUs[0], s, 0);
    limb.load(v);

    for (auto _ : state) {
        limb.NTT<FIDESlib::ALGO_NATIVE>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbNothingINTT32)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();

    FIDESlib::CKKS::Limb<uint32_t> limb(cc, GPUs[0], s, 0);
    limb.load(v);

    for (auto _ : state) {
        limb.INTT<FIDESlib::ALGO_NONE>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbNothingNTT32)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();

    FIDESlib::CKKS::Limb<uint32_t> limb(cc, GPUs[0], s, 0);
    limb.load(v);

    for (auto _ : state) {
        limb.NTT<FIDESlib::ALGO_NONE>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbShoupINTT32)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();

    FIDESlib::CKKS::Limb<uint32_t> limb(cc, GPUs[0], s, 0);
    limb.load(v);

    for (auto _ : state) {
        limb.INTT<FIDESlib::ALGO_SHOUP>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbShoupNTT32)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();

    FIDESlib::CKKS::Limb<uint32_t> limb(cc, GPUs[0], s, 0);
    limb.load(v);

    for (auto _ : state) {
        limb.NTT<FIDESlib::ALGO_SHOUP>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbBarretINTT32)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();

    FIDESlib::CKKS::Limb<uint32_t> limb(cc, GPUs[0], s, 0);
    limb.load(v);

    for (auto _ : state) {
        limb.INTT<FIDESlib::ALGO_BARRETT>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbBarretNTT32)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();

    FIDESlib::CKKS::Limb<uint32_t> limb(cc, GPUs[0], s, 0);
    limb.load(v);

    for (auto _ : state) {
        limb.NTT<FIDESlib::ALGO_BARRETT>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbFP64Accel53bitINTT32)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();

    FIDESlib::CKKS::Limb<uint32_t> limb(cc, GPUs[0], s, 0);
    limb.load(v);

    for (auto _ : state) {
        limb.INTT<FIDESlib::ALGO_BARRETT_FP64>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbFP64Accel53bitNTT32)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;
    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};
    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();

    FIDESlib::CKKS::Limb<uint32_t> limb(cc, GPUs[0], s, 0);
    limb.load(v);

    for (auto _ : state) {
        limb.NTT<FIDESlib::ALGO_BARRETT_FP64>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_REGISTER_F(FIDESlibFixture, LimbINTT32)->ArgsProduct({{0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbNTT32)->ArgsProduct({{0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbNothingINTT32)->ArgsProduct({{0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbNothingNTT32)->ArgsProduct({{0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbShoupINTT32)->ArgsProduct({{0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbShoupNTT32)->ArgsProduct({{0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbBarretINTT32)->ArgsProduct({{0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbBarretNTT32)->ArgsProduct({{0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbFP64Accel53bitINTT32)->ArgsProduct({{0}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbFP64Accel53bitNTT32)->ArgsProduct({{0}});

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbINTT64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
    limb2.load(v2);

    for (auto _ : state) {
        limb2.INTT<FIDESlib::ALGO_NATIVE>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbNTT64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
    limb2.load(v2);

    for (auto _ : state) {
        limb2.NTT<FIDESlib::ALGO_NATIVE>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbNothingINTT64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
    limb2.load(v2);

    for (auto _ : state) {
        limb2.INTT<FIDESlib::ALGO_NONE>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbNothingNTT64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
    limb2.load(v2);

    for (auto _ : state) {
        limb2.NTT<FIDESlib::ALGO_NONE>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbShoupINTT64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
    limb2.load(v2);

    for (auto _ : state) {
        limb2.INTT<FIDESlib::ALGO_SHOUP>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbShoupNTT64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
    limb2.load(v2);

    for (auto _ : state) {
        limb2.NTT<FIDESlib::ALGO_SHOUP>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbBarretINTT64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
    limb2.load(v2);

    for (auto _ : state) {
        limb2.INTT<FIDESlib::ALGO_BARRETT>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbBarretNTT64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
    limb2.load(v2);

    for (auto _ : state) {
        limb2.NTT<FIDESlib::ALGO_BARRETT>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbFP64Accel53bitINTT64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
    limb2.load(v2);

    for (auto _ : state) {
        limb2.INTT<FIDESlib::ALGO_BARRETT_FP64>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbFP64Accel53bitNTT64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    CudaCheckErrorMod;

    FIDESlib::CKKS::Context cc{fideslibParams, GPUs};

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    std::vector<uint32_t> v(cc.N, 10);
    for (auto& i : v)
        i = rand();
    std::vector<uint64_t> v2(cc.N, 10);
    for (auto& i : v2)
        i = rand();

    FIDESlib::CKKS::Limb<uint64_t> limb2(cc, GPUs[0], s, 0);
    limb2.load(v2);

    for (auto _ : state) {
        limb2.NTT<FIDESlib::ALGO_BARRETT_FP64>();
        CudaCheckErrorMod;
    }
}

BENCHMARK_REGISTER_F(FIDESlibFixture, LimbINTT64)->ArgsProduct({{1}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbNTT64)->ArgsProduct({{1}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbNothingINTT64)->ArgsProduct({{1}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbNothingNTT64)->ArgsProduct({{1}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbShoupINTT64)->ArgsProduct({{1}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbShoupNTT64)->ArgsProduct({{1}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbBarretINTT64)->ArgsProduct({{1}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbBarretNTT64)->ArgsProduct({{1}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbFP64Accel53bitINTT64)->ArgsProduct({{1}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbFP64Accel53bitNTT64)->ArgsProduct({{1}});

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbDeviceBatchINTT64)(benchmark::State& state) {
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters par = fideslibParams;
    par.L = 32;
    FIDESlib::CKKS::Context cc{par, GPUs};

    int n = state.range(1);
    state.counters["p_limbs"] = n;
    state.counters["p_batch"] = state.range(2);

    CudaCheckErrorMod;

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    FIDESlib::CKKS::RNSPoly v[4] = {FIDESlib::CKKS::RNSPoly(cc), FIDESlib::CKKS::RNSPoly(cc),
                                    FIDESlib::CKKS::RNSPoly(cc), FIDESlib::CKKS::RNSPoly(cc)};
    for (int i = 0; i < 4; ++i) {
        v[i].grow(std::max(0, std::min(n - 32 * i, 32)) - 1);
    }

    for (auto _ : state) {
        for (auto& i : v)
            i.INTT(state.range(2));
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbDeviceBatchNTT64)(benchmark::State& state) {
    constexpr auto algo = FIDESlib::ALGO_SHOUP;
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters par = fideslibParams;
    par.L = 32;
    FIDESlib::CKKS::Context cc{par, GPUs};

    int n = state.range(1);
    state.counters["p_limbs"] = n;
    state.counters["p_batch"] = state.range(2);

    CudaCheckErrorMod;

    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    FIDESlib::CKKS::RNSPoly v[4] = {FIDESlib::CKKS::RNSPoly(cc), FIDESlib::CKKS::RNSPoly(cc),
                                    FIDESlib::CKKS::RNSPoly(cc), FIDESlib::CKKS::RNSPoly(cc)};
    for (int i = 0; i < 4; ++i) {
        v[i].grow(std::max(0, std::min(n - 32 * i, 32)) - 1);
    }

    for (auto _ : state) {
        for (auto& i : v)
            i.NTT<algo>(state.range(2));
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbDeviceBatchINTT32)(benchmark::State& state) {
    constexpr auto algo = FIDESlib::ALGO_SHOUP;
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    int n = state.range(1);
    state.counters["p_limbs"] = n;
    state.counters["p_batch"] = state.range(2);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters par = fideslibParams;
    par.L = 32;
    FIDESlib::CKKS::Context cc{par, GPUs};

    CudaCheckErrorMod;
    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    FIDESlib::CKKS::RNSPoly v[4] = {FIDESlib::CKKS::RNSPoly(cc), FIDESlib::CKKS::RNSPoly(cc),
                                    FIDESlib::CKKS::RNSPoly(cc), FIDESlib::CKKS::RNSPoly(cc)};
    for (int i = 0; i < 4; ++i) {
        v[i].grow(std::min(n - 32 * i, 32) - 1);
    }

    CudaCheckErrorMod;

    for (auto _ : state) {
        for (auto& i : v)
            i.INTT<algo>(state.range(2));
        CudaCheckErrorMod;
    }
}

BENCHMARK_DEFINE_F(FIDESlibFixture, LimbDeviceBatchNTT32)(benchmark::State& state) {
    constexpr auto algo = FIDESlib::ALGO_SHOUP;
    int devcount = -1;
    cudaGetDeviceCount(&devcount);

    int n = state.range(1);
    state.counters["p_limbs"] = n;
    state.counters["p_batch"] = state.range(2);

    std::vector<int> GPUs;
    for (int i = 0; i < devcount; ++i)
        GPUs.push_back(i);

    FIDESlib::CKKS::Parameters par = fideslibParams;
    par.L = 32;
    FIDESlib::CKKS::Context cc{par, GPUs};

    CudaCheckErrorMod;
    cudaSetDevice(GPUs[0]);
    FIDESlib::Stream s;
    s.init();

    FIDESlib::CKKS::RNSPoly v[4] = {FIDESlib::CKKS::RNSPoly(cc), FIDESlib::CKKS::RNSPoly(cc),
                                    FIDESlib::CKKS::RNSPoly(cc), FIDESlib::CKKS::RNSPoly(cc)};
    for (int i = 0; i < 4; ++i) {
        v[i].grow(std::min(n - 32 * i, 32) - 1);
    }

    CudaCheckErrorMod;

    for (auto _ : state) {
        for (auto& i : v)
            i.NTT<algo>(state.range(2));
        CudaCheckErrorMod;
    }
}

BENCHMARK_REGISTER_F(FIDESlibFixture, LimbDeviceBatchINTT64)
    ->ArgsProduct({{6, 7, 8, 1}, {0, 1, 8, 16, 27, 32, 64, 128}, {1, -1, 2, 3, 4, 6, 8, 12, 16, 100}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbDeviceBatchNTT64)
    ->ArgsProduct({{6, 7, 8, 1}, {0, 1, 8, 16, 27, 32, 64, 128}, {1, -1, 2, 3, 4, 6, 8, 12, 16, 100}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbDeviceBatchINTT32)
    ->ArgsProduct({{0}, {0, 1, 8, 16, 32, 54, 64, 128}, {1, -1, 2, 3, 4, 6, 8, 12, 16, 100}});
BENCHMARK_REGISTER_F(FIDESlibFixture, LimbDeviceBatchNTT32)
    ->ArgsProduct({{0}, {0, 1, 8, 16, 32, 54, 64, 128}, {1, -1, 2, 3, 4, 6, 8, 12, 16, 100}});

}  // namespace FIDESlib::Benchmarks