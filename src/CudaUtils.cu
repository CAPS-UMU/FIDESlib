//
// Created by carlosad on 25/03/24.
//
#include <cassert>
#include <string>
#include "CudaUtils.cuh"

#include <iostream>
#include <nvtx3/nvtx3.hpp>
//#include "driver_types.h"

namespace FIDESlib {

struct my_domain {
    static constexpr char const* name{"FIDESlib"};
};

nvtx3::domain const& D = nvtx3::domain::get<my_domain>();

std::map<std::string, std::pair<nvtx3::unique_range_in<my_domain>*, int>> lifetimes_map;

void CudaNvtxStart(const std::string msg, NVTX_CATEGORIES cat, int val) {

    if (cat == FUNCTION) {
        using namespace nvtx3;
        int size = msg.size();
        const event_attributes attr{msg,
                                    rgb{(uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
                                        (uint8_t)(255 - 101 * msg[size * 5 / 6])},
                                    payload{val}, category{cat}};

        nvtxDomainRangePushEx_impl_init_v3(D, reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
        //nvtxRangePushEx(reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
    } else if (cat == LIFETIME) {

        using namespace nvtx3;
        int size = msg.size();
        auto& [r, i] = lifetimes_map[msg];
        std::string m = std::to_string(i + 1) + std::string(" x ") + msg;
        const event_attributes attr{m,
                                    rgb{(uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
                                        (uint8_t)(255 - 101 * msg[size * 5 / 6])},
                                    payload{i + 1}, category{cat}};
        i = i + 1;
        if (!r) {
            r = new unique_range_in<my_domain>(attr);
        } else {
            *r = unique_range_in<my_domain>(attr);
        }
    }
    //nvtxRangePushA(msg.c_str());
}

void CudaNvtxStop(const std::string msg, NVTX_CATEGORIES cat) {
    if (cat == FUNCTION) {
        nvtxDomainRangePop(D);
    } else if (cat == LIFETIME) {
        using namespace nvtx3;
        int size = msg.size();

        auto& [r, i] = lifetimes_map[msg];
        std::string m = std::to_string(i - 1) + std::string(" x ") + msg;
        const event_attributes attr{m,
                                    rgb{(uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
                                        (uint8_t)(255 - 101 * msg[size * 5 / 6])},
                                    payload{i - 1}, category{cat}};

        i = i - 1;
        if (i <= 0) {
            if (r) {
                delete r;
                r = nullptr;
            }
        } else {
            *r = unique_range_in<my_domain>(attr);
        }

        //nvtxRangePushEx(reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
    }
}

void CudaHostSync() {
    cudaDeviceSynchronize();
}

template <bool capture>
void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run) {
    cudaGraph_t graph;
    if constexpr (capture) {
        cudaStreamBeginCapture(s.ptr, cudaStreamCaptureModeRelaxed);
        CudaCheckErrorModNoSync;
    }
    run();
    if constexpr (capture) {
        cudaStreamEndCapture(s.ptr, &graph);
        if (!exec) {
            cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
            //cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
            CudaCheckErrorModNoSync;
        } else {
            if (cudaGraphExecUpdate(exec, graph, NULL) != cudaSuccess) {
                CudaCheckErrorModNoSync;
                // only instantiate a new graph if update fails
                cudaGraphExecDestroy(exec);
                cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
                //cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
                CudaCheckErrorModNoSync;
            }
        }
        cudaGraphDestroy(graph);
        cudaGraphLaunch(exec, s.ptr);
    }
}

template void run_in_graph<false>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

template void run_in_graph<true>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

/*
    void Stream::wait(const Event &ev) const {
        cudaStreamWaitEvent(ptr, ev.ptr);
    }
*/
void Stream::capture_begin() {
    CudaCheckErrorMod;
    std::cout << "Hello capture" << std::endl;
    cudaStreamCaptureStatus cap;
    cudaStreamIsCapturing(ptr, &cap);

    CudaCheckErrorMod;
    if (cap == cudaStreamCaptureStatusNone) {
        std::cout << "None" << std::endl;
        cudaStreamBeginCapture(ptr, cudaStreamCaptureModeGlobal);
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
    cudaStreamEndCapture(ptr, &graph);
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

void Stream::record() {
    //cudaEventDestroy(ev);
    //cudaEventCreate(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, ptr);
}

void Stream::wait_recorded(const Stream& s) const {
    cudaStreamWaitEvent(ptr, s.ev);
    //this->wait(ev);
}

void Stream::wait(const Stream& s) const {

    assert(ptr != nullptr);
    assert(s.ptr != nullptr);
    assert(ev != nullptr);
    CudaCheckErrorModNoSync;
    cudaEventRecord(ev, (s.ptr));
    cudaStreamWaitEvent(ptr, ev);
}

void breakpoint() {}

void Stream::init(int primeid, int LK) {
    if (ptr) {
        //free[ptr]++;
        cudaEventDestroy(ev);
        cudaStreamDestroy(ptr);
        ptr = nullptr;
        ev = nullptr;
    }
    //int low;
    //int high;
    //cudaDeviceGetStreamPriorityRange(&low, &high);
    //int prio = low + ((high - low -1)* primeid)/ LK;
    //cudaStreamCreateWithPriority(&ptr, cudaStreamNonBlocking, prio);
    cudaStreamCreateWithFlags(&ptr, cudaStreamNonBlocking);
    cudaEventCreate(&ev, cudaEventDisableTiming);

    //free[ptr] = 0;
}

//std::map<void *, int> free;

Stream::~Stream() {
    if (ptr) {
        //free[ptr]++;
        cudaStreamDestroy(ptr);
        ptr = nullptr;
    }
    if (ev) {
        cudaEventDestroy(ev);
        ev = nullptr;
    }
}

Stream::Stream() = default;

Stream::Stream(Stream&& s) noexcept : ptr(s.ptr), ev(s.ev) {
    s.ptr = nullptr;
    s.ev = nullptr;
}
}  // namespace FIDESlib