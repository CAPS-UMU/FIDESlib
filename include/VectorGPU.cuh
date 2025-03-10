//
// Created by carlos on 6/03/24.
//

#ifndef FIDESLIB_VECTORGPU_CUH
#define FIDESLIB_VECTORGPU_CUH
#include <driver_types.h>
#include <cassert>
#include <type_traits>
#include <vector>
#include "CudaUtils.cuh"
#include "LogUtil.hpp"
#include "cuda.h"

namespace FIDESlib {
template <typename T>
class VectorGPU {
    static_assert(std::is_trivially_copyable_v<T>);
    bool freeing;
    bool managed;

   public:
    T* data;
    const int size;
    const int device;

    VectorGPU(VectorGPU<T>&& v) noexcept;
    /*
        VectorGPU<T> &operator=(VectorGPU<T> && v) noexcept
         {
            freeing = v.freeing;
            managed = v.managed;
            data = v.data;
            size = v.size;
            device = v.device;
            v.freeing = true;
            return *this;
        }
*/
    VectorGPU<T>& operator=(VectorGPU<T>& other) = delete;
    VectorGPU<T>& operator=(const VectorGPU<T>& v) = delete;
    VectorGPU(const VectorGPU<T>& v) = delete;
    VectorGPU(T* data, const int size, const int device, const int offset = 0);
    VectorGPU(const Stream& stream, const int size, const int device, const T* src = nullptr);
    void free(const Stream& stream);
    ~VectorGPU();
};

}  // namespace FIDESlib
#endif  //FIDESLIB_VECTORGPU_CUH
