//
// Created by carlosad on 14/03/24.
//

#ifndef FIDESLIB_CUDAUTILS_CUH
#define FIDESLIB_CUDAUTILS_CUH

#include <cuda_runtime.h>
#include <execinfo.h>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace FIDESlib {

enum NVTX_CATEGORIES { NONE, LIFETIME, FUNCTION };

void CudaNvtxStart(const std::string msg, NVTX_CATEGORIES cat = FUNCTION, int val = 0);
void CudaNvtxStop(const std::string msg = "", NVTX_CATEGORIES cat = FUNCTION);
class CudaNvtxRange {
    const std::string msg;
    const NVTX_CATEGORIES cat;
    bool valid = true;

   public:
    explicit CudaNvtxRange(const std::string msg, NVTX_CATEGORIES cat = FUNCTION, int val = 0) : msg(msg), cat(cat) {
        CudaNvtxStart(msg, cat, val);
    }

    CudaNvtxRange(CudaNvtxRange&& r) noexcept : msg(r.msg), cat(r.cat) {
        this->valid = r.valid;
        r.valid = false;
    }

    ~CudaNvtxRange() {
        if (valid)
            CudaNvtxStop(msg, cat);
    }
};

void CudaHostSync();
void breakpoint();

#define CudaCheckErrorMod                                                                    \
    do {                                                                                     \
        cudaDeviceSynchronize();                                                             \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess) {                                                              \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            FIDESlib::breakpoint();                                                          \
            exit(0);                                                                         \
        }                                                                                    \
    } while (0)

#define CudaCheckErrorModNoSync                                                              \
    do {                                                                                     \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess) {                                                              \
            void* array[10];                                                                 \
            size_t size;                                                                     \
            size = backtrace(array, 10);                                                     \
            backtrace_symbols_fd(array, size, STDERR_FILENO);                                \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            FIDESlib::breakpoint();                                                          \
            exit(0);                                                                         \
        }                                                                                    \
    } while (0)

class Event;

extern std::map<void*, int> free;

class Stream {
   public:
    cudaStream_t ptr = nullptr;
    cudaEvent_t ev = nullptr;
    //Event ev;

    void init(int primeid = 0, int LK = 64);

    //void wait(const Event &ev) const;
    void wait(const Stream& s) const;

    Stream();

    Stream(Stream& s) = delete;

    Stream(const Stream& s) = delete;

    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& s) noexcept;

    ~Stream();

    void record();

    void wait_recorded(const Stream& s) const;

    void capture_begin();

    void capture_end();
};

template <bool capture>
void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

}  // namespace FIDESlib
#endif  //FIDESLIB_CUDAUTILS_CUH
