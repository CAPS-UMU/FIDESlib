//
// Created by carlosad on 25/03/24.
//

#include "CudaUtils.cuh"
#include <cassert>
#include <cstdio>
#include <list>
#include <set>
#include <string>

#include "nvtx3/nvtx3.hpp"
#include <iostream>

// #include "driver_types.h"
#include "CKKS/Context.cuh"
#define DISABLE_STREAMS false

#include <cuda_runtime.h>

namespace FIDESlib {
extern std::vector<cudaDeviceProp> GPUprop;

struct my_domain {
    static constexpr char const* name{ "FIDESlib" };
};

nvtx3::domain const& D = nvtx3::domain::get<my_domain>();

// The memory-pool ranges live on their own domain so nsight-systems renders them as a separate
// NVTX row ("FIDESlibPool") instead of intermixing them with the FIDESlib stack/lifetime ranges.
struct memory_pool_domain {
    static constexpr char const* name{ "FIDESlibPool" };
};

nvtx3::domain const& D_pool = nvtx3::domain::get<memory_pool_domain>();

namespace {
// A LIFETIME range aggregates every live object of one label: the entry keeps the number of
// live objects and, per named component, per-object getters returning their current memory
// footprint (in bytes), so the range message can render both the live count and the aggregate
// memory (e.g. "124 x Plaintext - 345.67 MB" or "1 x Context - 345.67 MB (Buffers 100.50 MB,
// Precomputed 245.17 MB)"). Re-rendering is deduplicated against the last emitted (count, bytes).
struct LifetimeEntry {
    void reset() {
        range.reset();
        count = 0;
        objects.clear();
        renderedCount = -1;
        renderedBytes = 0;
    }
    std::unique_ptr<nvtx3::unique_range_in<my_domain>> range;
    int count = 0;
    std::map<std::string, std::map<const void*, NvtxLifetimeBytesProvider>> objects; // component -> obj -> getter
    int renderedCount = -1;
    uint64_t renderedBytes = 0;
};
} // namespace

std::map<std::string, LifetimeEntry> lifetimes_map;

std::string CudaNvtxFormatBytes(uint64_t bytes) {
    const double KB = 1024.0;
    const double MB = 1024.0 * 1024.0;
    const double GB = 1024.0 * 1024.0 * 1024.0;
    char buf[64];
    if (bytes >= GB) {
        snprintf(buf, sizeof(buf), "%.2f GB", bytes / GB);
    } else if (bytes >= MB) {
        snprintf(buf, sizeof(buf), "%.2f MB", bytes / MB);
    } else if (bytes >= KB) {
        snprintf(buf, sizeof(buf), "%.2f KB", bytes / KB);
    } else {
        snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(bytes));
    }
    return std::string(buf);
}

static void NvtxLifetimeRender(const std::string& msg, int count) {
    using namespace nvtx3;
    int size = msg.size();
    auto& entry = lifetimes_map[msg];

    // Sum the registered getters per component.
    uint64_t totalBytes = 0;
    std::map<std::string, uint64_t> componentBytes;
    for (const auto& [component, objs] : entry.objects) {
        uint64_t compBytes = 0;
        for (const auto& [obj, getter] : objs)
            compBytes += getter();
        componentBytes[component] = compBytes;
        totalBytes += compBytes;
    }

    // Nothing observable changed and the range is already emitted: skip to avoid churning the
    // timeline with identical updates (count unchanged and total memory unchanged).
    if (entry.range && entry.renderedCount == count && entry.renderedBytes == totalBytes) {
        return;
    }

    std::string m = std::to_string(count) + std::string(" x ") + msg + " - " + CudaNvtxFormatBytes(totalBytes);
    if (entry.objects.size() > 1) {
        m += " (";
        bool first = true;
        for (const auto& [component, compBytes] : componentBytes) {
            if (!first)
                m += ", ";
            m += component + " " + CudaNvtxFormatBytes(compBytes);
            first = false;
        }
        m += ")";
    }
    const event_attributes attr{ m,
        rgb{ (uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]), (uint8_t)(255 - 101 * msg[size * 5 / 6]) },
        payload{ count },
        category{ static_cast<unsigned int>(LIFETIME) } };

    if (!entry.range) {
        entry.range = std::make_unique<unique_range_in<my_domain>>(attr);
    } else {
        *entry.range = unique_range_in<my_domain>(attr);
    }
    entry.renderedCount = count;
    entry.renderedBytes = totalBytes;
}

void CudaNvtxStart(const std::string msg, NVTX_CATEGORIES cat, int val) {
    if (cat == FUNCTION) {
        using namespace nvtx3;
        int size = msg.size();
        const event_attributes attr{ msg,
            rgb{ (uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]), (uint8_t)(255 - 101 * msg[size * 5 / 6]) },
            payload{ val },
            category{ static_cast<unsigned int>(cat) } };

        nvtxDomainRangePushEx_impl_init_v3(D, reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
        // nvtxRangePushEx(reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
    } else if (cat == LIFETIME) {
        auto& entry = lifetimes_map[msg];
        entry.count += 1;
        NvtxLifetimeRender(msg, entry.count);
    }
    // nvtxRangePushA(msg.c_str());
}

void CudaNvtxStop(const std::string msg, NVTX_CATEGORIES cat) {
    if (cat == FUNCTION) {
        nvtxDomainRangePop(D);
    } else if (cat == LIFETIME) {
        auto it = lifetimes_map.find(msg);
        if (it == lifetimes_map.end()) {
            return;
        }
        auto& entry = it->second;
        entry.count -= 1;
        if (entry.count <= 0) {
            entry.reset();
        } else {
            NvtxLifetimeRender(msg, entry.count);
        }
    }
}

void CudaNvtxLifetimeRegisterBytes(const std::string& msg, const void* obj, const NvtxLifetimeBytesProvider& getter, const std::string& component) {
    auto& entry = lifetimes_map[msg];
    entry.objects[component][obj] = getter;
    if (entry.range) {
        NvtxLifetimeRender(msg, entry.count);
    }
}

void CudaNvtxLifetimeUnregisterBytes(const std::string& msg, const void* obj, const std::string& component) {
    auto it = lifetimes_map.find(msg);
    if (it == lifetimes_map.end()) {
        return;
    }
    auto& entry = it->second;
    auto compIt = entry.objects.find(component);
    if (compIt == entry.objects.end()) {
        return;
    }
    compIt->second.erase(obj);
    if (compIt->second.empty()) {
        entry.objects.erase(compIt);
    }
    if (entry.range) {
        NvtxLifetimeRender(msg, entry.count);
    }
}

void CudaNvtxLifetimeRefresh(const std::string& msg) {
    auto it = lifetimes_map.find(msg);
    if (it != lifetimes_map.end() && it->second.range) {
        NvtxLifetimeRender(msg, it->second.count);
    }
}

int getNumDevices() {
    int d;
    cudaGetDeviceCount(&d);
    return d;
};

void CudaHostSync() {
    cudaDeviceSynchronize();
}

namespace {
// NVTX lifetime ranges for the FIDESlib memory pool (see GPUmalloc): one persistent range per
// (device, chunk size) rendered as e.g. "Chunks of 512KB (2048 chunks) = 1GB". Slabs are never
// freed until process exit, so these ranges are created on first allocation and never stopped;
// additional slabs of the same chunk size accumulate into the message.
struct PoolLifetime {
    std::unique_ptr<nvtx3::unique_range_in<memory_pool_domain>> range;
    uint64_t chunks = 0;
    uint64_t totalBytes = 0;
};

std::map<int, std::map<int, PoolLifetime>> pool_lifetimes; // [device][chunkBytes]
std::set<int> pool_devices;                                // devices that allocated pool slabs
std::mutex pool_lifetimes_lock;                            // pool allocs are multi-threaded (per-device mempool locks differ)

std::string NvtxPoolFormatSize(uint64_t bytes) {
    const double KB = 1024.0;
    const double MB = 1024.0 * 1024.0;
    const double GB = 1024.0 * 1024.0 * 1024.0;
    char buf[64];
    if (bytes >= GB) {
        snprintf(buf, sizeof(buf), "%.0fGB", bytes / GB);
    } else if (bytes >= MB) {
        snprintf(buf, sizeof(buf), "%.0fMB", bytes / MB);
    } else if (bytes >= KB) {
        snprintf(buf, sizeof(buf), "%.0fKB", bytes / KB);
    } else {
        snprintf(buf, sizeof(buf), "%lluB", static_cast<unsigned long long>(bytes));
    }
    return std::string(buf);
}

void NvtxPoolRender(int device, int chunkBytes) {
    using namespace nvtx3;
    auto& info = pool_lifetimes[device][chunkBytes];
    std::string m = "Chunks of " + NvtxPoolFormatSize(chunkBytes) + " (" + std::to_string(info.chunks) + " chunks) = " + NvtxPoolFormatSize(info.totalBytes);
    if (pool_devices.size() > 1) {
        m = "GPU " + std::to_string(device) + ": " + m;
    }
    int size = m.size();
    const event_attributes attr{ m,
        rgb{ (uint8_t)(255 - 101 * m[size / 6]), (uint8_t)(255 - 101 * m[size * 3 / 6]), (uint8_t)(255 - 101 * m[size * 5 / 6]) },
        payload{ static_cast<int>(info.chunks) },
        category{ static_cast<unsigned int>(LIFETIME) } };
    if (!info.range) {
        info.range = std::make_unique<unique_range_in<memory_pool_domain>>(attr);
    } else {
        *info.range = unique_range_in<memory_pool_domain>(attr);
    }
}

void NvtxPoolRecordSlab(int device, int chunkBytes, uint64_t slabBytes) {
    std::lock_guard<std::mutex> lock(pool_lifetimes_lock);
    pool_devices.insert(device);
    auto& info = pool_lifetimes[device][chunkBytes];
    info.chunks += slabBytes / chunkBytes;
    info.totalBytes += slabBytes;
    NvtxPoolRender(device, chunkBytes);
}
} // namespace

template <bool capture> void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run) {
    cudaGraph_t graph;
    if constexpr (capture) {
        cudaStreamBeginCapture(s.ptr(), cudaStreamCaptureModeRelaxed);
        CudaCheckErrorModNoSync;
    }
    run();
    if constexpr (capture) {
        cudaStreamEndCapture(s.ptr(), &graph);
        if (!exec) {
            cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
            // cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
            CudaCheckErrorModNoSync;
        } else {
            if (cudaGraphExecUpdate(exec, graph, NULL) != cudaSuccess) {
                CudaCheckErrorModNoSync;
                // only instantiate a new graph if update fails
                cudaGraphExecDestroy(exec);
                cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
                // cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
                CudaCheckErrorModNoSync;
            }
        }
        cudaGraphDestroy(graph);
        cudaGraphLaunch(exec, s.ptr());
    }
}

template void run_in_graph<false>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

template void run_in_graph<true>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

/*
void Stream::wait(const Event &ev) const {
cudaStreamWaitEvent(ptr, ev.ptr());
}
*/
void Stream::capture_begin() {
    CudaCheckErrorMod;
    std::cout << "Hello capture" << std::endl;
    cudaStreamCaptureStatus cap;
    cudaStreamIsCapturing(ptr(), &cap);

    CudaCheckErrorMod;
    if (cap == cudaStreamCaptureStatusNone) {
        std::cout << "None" << std::endl;
        cudaStreamBeginCapture(ptr(), cudaStreamCaptureModeGlobal);
    } else if (cap == cudaStreamCaptureStatusActive) {
        std::cout << "Fail: activo" << std::endl;
    } else if (cap == cudaStreamCaptureStatusInvalidated) {
        std::cout << "Fail: invalidado" << std::endl;
    } else {
        std::cout << "Fail" << std::endl;
    }
    CudaCheckErrorMod;
}

void Stream::capture_end() {
    cudaGraph_t graph;
    cudaStreamEndCapture(ptr(), &graph);
    CudaCheckErrorMod;
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    CudaCheckErrorMod;
    cudaGraphDestroy(graph);
    CudaCheckErrorMod;
    cudaGraphLaunch(graphExec, 0);
    cudaGraphExecDestroy(graphExec);

    cudaStreamSynchronize(0);
}

void Stream::record(bool external) {
    // if (ptr_ == 0)
    //	return;
    CudaCheckErrorModNoSync;
#if !DISABLE_STREAMS
    // cudaEventDestroy(ev);
    // cudaEventCreate(&ev, cudaEventDisableTiming);
    assert(ptr_ != nullptr);
    assert(ev != nullptr);
    cudaEventRecordWithFlags(ev, ptr_, external ? cudaEventRecordExternal : cudaEventRecordDefault);
    updated = true;
#endif
}

void Stream::wait_recorded(const Stream& s) {
    if (ptr_ == 0 || s.ptr_ == 0 || ptr_ == s.ptr_)
        return;
#if !DISABLE_STREAMS
    assert(s.updated);
    cudaStreamWaitEvent(ptr_, s.ev);
    this->updated = false;
#endif
}

void Stream::wait(Stream& s, bool external) {
#if !DISABLE_STREAMS
    // CudaCheckErrorModNoSync;
    if (/*ptr_ == 0 ||*/ s.ptr_ == 0 || ptr_ == s.ptr_) {
        updated = false;
        return;
    }
    // assert(ptr_ != nullptr);
    // assert(s.ptr_ != nullptr);
    assert(s.ev != nullptr);
    if (!s.updated) {
        assert(!external); // Has to be recorded in the origin graph
        CudaCheckErrorModNoSync;
        cudaEventRecordWithFlags(s.ev, s.ptr_, cudaEventRecordDefault);
        s.updated = true;
        CudaCheckErrorModNoSync;
    }
    CudaCheckErrorModNoSync;
    cudaStreamWaitEvent(ptr_, s.ev, external ? cudaEventWaitExternal : cudaEventWaitDefault);
    this->updated = false;
#endif
    CudaCheckErrorModNoSync;
}

void Stream::wait(cudaStream_t s) {
#if !DISABLE_STREAMS
    // CudaCheckErrorModNoSync;
    if (s == 0 || s == ptr_) {
        updated = false;
        return;
    }
    assert(ptr_ != nullptr);
    assert(ev != nullptr);
    CudaCheckErrorModNoSync;
    cudaEventRecordWithFlags(ev, s, cudaEventRecordDefault);
    updated = false;
    CudaCheckErrorModNoSync;

    CudaCheckErrorModNoSync;
    cudaStreamWaitEvent(ptr_, ev, cudaEventWaitDefault);
#endif
    CudaCheckErrorModNoSync;
}

int low = -1;
int high = -1;

constexpr int POOL_SIZE = 37;
cudaStream_t stream_pool[MAXG][POOL_SIZE];

bool initPool() {
    int devs;
    cudaGetDeviceCount(&devs);
    for (int j = 0; j < devs; j++) {
        cudaSetDevice(j);
        for (int i = 0; i < POOL_SIZE; ++i) {
            cudaStreamCreateWithFlags(&stream_pool[j][i], 0);
        }
    }
    return true;
}

#define USEPOOL true
#if USEPOOL
// The pool is created on first use instead of from a static initializer. A static
// initializer issues the first CUDA runtime calls of the process before main(); if one
// of them fails (e.g. cudaGetDeviceCount on a host with a degraded GPU) the runtime
// stays in the error state, the pool remains empty and Stream::init crashes later.
bool poolInitialized() {
    static const bool initialized = initPool();
    return initialized;
}
#else
bool initialized = false;
#endif

void Stream::init(int priority) {
    static int assigner_idx[MAXD] = { 0 };
    if (ptr_) {
        // free[ptr]++;
        cudaEventDestroy(ev);
        // cudaStreamDestroy(ptr_);
        ptr_ = nullptr;
        ev = nullptr;
    }

#if !DISABLE_STREAMS
#if USEPOOL
    poolInitialized();
#endif
    if (high == -1) {
        cudaDeviceGetStreamPriorityRange(&low, &high);
    }

    // int prio = low + priority * ((high - low - 1)) / 100;

#if USEPOOL
    int dev;
    cudaGetDevice(&dev);
    ptr_ = stream_pool[dev][assigner_idx[dev]];
    assigner_idx[dev] += 1;
    if (assigner_idx[dev] >= POOL_SIZE)
        assigner_idx[dev] -= POOL_SIZE;
#else
    cudaStreamCreateWithPriority(&ptr_, 0 /*cudaStreamNonBlocking*/, priority);
    // cudaStreamCreateWithFlags(&ptr, cudaStreamNonBlocking);
#endif

    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventCreate(&ev, cudaEventDisableTiming);
#else
    ptr_ = nullptr;
    ev = nullptr;
#endif
    // free[ptr] = 0;
}

void Stream::initDefault() {
    ptr_ = 0;
    ev = nullptr;
    updated = true;
}

// std::map<void *, int> free;

Stream::~Stream() {
    if (ptr_) {
        // free[ptr]++;
        // cudaStreamDestroy(ptr_);
        ptr_ = nullptr;
    }
    if (ev) {
        cudaEventDestroy(ev);
        ev = nullptr;
    }
}

Stream::Stream() = default;

Stream::Stream(Stream&& s) noexcept : ptr_(s.ptr_), ev(s.ev) {
    s.ptr_ = nullptr;
    s.ev = nullptr;
}

std::vector<cudaDeviceProp> GPUprop;

void initGPUprop() {
    if (GPUprop.empty()) {
        int count = 0;
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) {
            GPUprop.emplace_back();
            cudaGetDeviceProperties(&GPUprop.back(), i);

            std::cout << "GPU " << i << ": " << GPUprop[i].name << "\n SMs: " << GPUprop[i].multiProcessorCount
                      << ", SharedMem: " << GPUprop[i].sharedMemPerMultiprocessor / 1024l << " KB, Blocks/SM: " << GPUprop[i].maxBlocksPerMultiProcessor
                      << ", Threads/SM: " << GPUprop[i].maxThreadsPerMultiProcessor << ", L2 size: " << GPUprop[i].l2CacheSize / (1024l * 1024l)
                      << " MB, Bus: " << (long long)GPUprop[i].memoryBusWidth

                      << "-bit" << std::endl;
        }
    }
}

std::mutex mempool_lock[MAXG];

std::map<int, std::vector<void*>> size_to_memory[MAXG];

FIDESlib::Stream s[MAXG];

#define MEMPOOL true

// void* GPUmalloc(int id, int bytes, cudaStream_t stream, FIDESlib::CKKS::Context& cc) {
void* GPUmalloc(int id, int bytes, cudaStream_t stream, bool cache) {
    void* ptr = nullptr;

    uint64_t MBs = 1024;

    if (bytes < 64 * 1024) {
        int next_pow2 = 1024;
        while (next_pow2 < bytes) {
            next_pow2 *= 2;
        }
        bytes = next_pow2;
        cache = true;
        MBs = bytes / 1024;
    }
#if MEMPOOL
    if (cache && (bytes & (bytes - 1)) == 0) {
        std::vector<void*>& free_limb = size_to_memory[id][bytes];

        if (s[id].ptr() == nullptr) {
            mempool_lock[id].lock();
            if (s[id].ptr() == nullptr) {
                s[id].init();
            }
            mempool_lock[id].unlock();
        }
        CudaCheckErrorModNoSync;
        if (free_limb.empty()) {
            uint64_t* base;
            cudaMallocAsync(&base, MBs * 1024 * 1024, s[id].ptr());

            mempool_lock[id].lock();
            for (uint32_t i = 0; i < MBs * 1024 * 1024; i += bytes) {
                free_limb.emplace_back(((char*)base) + i);
            }
            // Keep the pool NVTX lifetime in sync with the memory actually reserved for this chunk size.
            NvtxPoolRecordSlab(id, bytes, MBs * 1024 * 1024);
            mempool_lock[id].unlock();
        }
        CudaCheckErrorModNoSync;

        // if (stream != nullptr) {
        s[id].record();
        CudaCheckErrorModNoSync;
        cudaStreamWaitEvent(stream, s[id].ev);
        //}

        CudaCheckErrorModNoSync;
        mempool_lock[id].lock();
        ptr = free_limb.back();
        free_limb.pop_back();
        mempool_lock[id].unlock();
        // ptr = free_limb.front();
        // free_limb.pop_front();
        // std::cout << "get " << ptr << std::endl;
        return ptr;
    }

#endif
    // std::cout << bytes << std::endl;
    if (0) {
        cudaMalloc(&ptr, bytes);
    } else if (1) {
        cudaMallocAsync(&ptr, bytes, 0);
    } else {
        if (size_to_memory[id][bytes].empty()) {
            cudaSetDevice(id);
            cudaMallocAsync(&ptr, bytes, stream);
        } else {
            mempool_lock[id].lock();
            if (size_to_memory[id][bytes].empty()) {
                mempool_lock[id].unlock();
                cudaSetDevice(id);
                cudaMallocAsync(&ptr, bytes, stream);
            } else {
                ptr = size_to_memory[id][bytes].back();
                size_to_memory[id][bytes].pop_back();
                mempool_lock[id].unlock();
            }
        }
    }

    return ptr;
}

struct pointerdata {
    void* pointer;
    int id;
    int bytes;
};

void CUDART_CB streamCallback(void* userData) {
    auto* p = reinterpret_cast<pointerdata*>(userData);

    mempool_lock[p->id].lock();
    size_to_memory[p->id][p->bytes].push_back(p->pointer);
    mempool_lock[p->id].unlock();
    delete p;
}

thread_local bool gpufree_presynced = false;

void GPUfree(void* ptr, int id, int bytes, cudaStream_t stream, bool cache) {
    if (bytes < 64 * 1024) {
        int next_pow2 = 1024;
        while (next_pow2 < bytes) {
            next_pow2 *= 2;
        }
        bytes = next_pow2;
        cache = true;
    }

#if MEMPOOL
    if (cache && (bytes & (bytes - 1)) == 0) {
        std::vector<void*>& free_limb = size_to_memory[id][bytes];
        if (s[id].ptr() == nullptr) {
            mempool_lock[id].lock();
            s[id].init();
            mempool_lock[id].unlock();
        }
        CudaCheckErrorModNoSync;
        // cudaDeviceSynchronize();
        // if (stream != nullptr) {
        if (!gpufree_presynced) // redundant while ContextData::clearAuxilarPoly holds the device synchronized
            s[id].wait(stream);
        //}
        CudaCheckErrorModNoSync;
        mempool_lock[id].lock();
        free_limb.emplace_back(ptr);
        mempool_lock[id].unlock();
        // std::cout << "free " << ptr << std::endl;
        return;
    }
#endif

    if (0) {
        cudaFree(ptr);
    } else if (1) {
        cudaFreeAsync(ptr, 0);
    } else {
        auto* p = new pointerdata;
        p->id = id;
        p->bytes = bytes;
        p->pointer = ptr;
        cudaLaunchHostFunc(stream, streamCallback, p);
    }
}

int GetTargetThreads(int id) {
    return GPUprop[id].multiProcessorCount * GPUprop[id].maxThreadsPerMultiProcessor;
}
} // namespace FIDESlib