//
// Created by carlosad on 14/03/24.
//

#ifndef FIDESLIB_CUDAUTILS_CUH
#define FIDESLIB_CUDAUTILS_CUH

#include <cuda.h>
#include <driver_types.h>
#include <execinfo.h>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace FIDESlib {

void initGPUprop();
int GetTargetThreads(int id);

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
		r.valid		= false;
	}

	~CudaNvtxRange() {
		if (valid)
			CudaNvtxStop(msg, cat);
	}
};

int getNumDevices();

void CudaHostSync();

inline void breakpoint() {
}

// TODO: Remove the cudart unloading.
#define CudaCheckErrorMod                                                                    \
	do {                                                                                     \
		cudaDeviceSynchronize();                                                             \
		cudaError_t e = cudaGetLastError();                                                  \
		if (e == cudaErrorCudartUnloading) {                                                 \
			exit(0);                                                                         \
		} else if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {             \
                                                                                             \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			FIDESlib::breakpoint();                                                          \
			exit(0);                                                                         \
		}                                                                                    \
	} while (0)

#define CudaCheckErrorModMGPU                                                                \
	do {                                                                                     \
		cudaStreamSynchronize(0);                                                            \
		cudaError_t e = cudaGetLastError();                                                  \
		if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {                    \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			FIDESlib::breakpoint();                                                          \
			exit(0);                                                                         \
		}                                                                                    \
	} while (0)

// TODO: FIX THE CUDARTUNLOADING ERROR, IT HAPPENS WHEN THE LIBRARY IS BEING UNLOADED, CAN BE IGNORED FOR NOW
#define CudaCheckErrorModNoSync                                                                                          \
	do {                                                                                                                 \
		/*cudaDeviceSynchronize();*/                                                                                     \
		cudaError_t e = cudaGetLastError();                                                                              \
		if (e == cudaErrorCudartUnloading) {                                                                             \
			exit(0);                                                                                                     \
		} else if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled && e != cudaErrorGraphExecUpdateFailure) { \
			void* array[10];                                                                                             \
			size_t size;                                                                                                 \
			size = backtrace(array, 10);                                                                                 \
			backtrace_symbols_fd(array, size, STDERR_FILENO);                                                            \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));                             \
			FIDESlib::breakpoint();                                                                                      \
			exit(0);                                                                                                     \
		}                                                                                                                \
	} while (0)

#define NCCLCHECK(cmd)                                                                              \
	do {                                                                                            \
		ncclResult_t res = cmd;                                                                     \
		if (res != ncclSuccess) {                                                                   \
			printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
			exit(EXIT_FAILURE);                                                                     \
		}                                                                                           \
	} while (0)

class Event;

extern std::map<void*, int> free;

class Stream {
  private:
	cudaStream_t ptr_ = nullptr;

  public:
	cudaEvent_t ev = nullptr;
	bool updated   = false;
	// Event ev;

	void init(int priority = 0);

	cudaStream_t ptr() {
		updated = false;
		return ptr_;
	}

	void initDefault();

	// void wait(const Event &ev) const;
	void wait(Stream& s, bool external = false);
	void wait(cudaStream_t s);

	Stream();

	Stream(Stream& s) = delete;

	Stream(const Stream& s) = delete;

	Stream& operator=(const Stream&) = delete;

	Stream(Stream&& s) noexcept;

	~Stream();

	void record(bool external = false);

	void wait_recorded(const Stream& s);

	void capture_begin();

	void capture_end();
};

template <bool capture> void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

void* GPUmalloc(int id, int bytes, cudaStream_t stream, bool cache = false);
void GPUfree(void* ptr, int id, int bytes, cudaStream_t stream, bool cache = false);

} // namespace FIDESlib
#endif // FIDESLIB_CUDAUTILS_CUH
