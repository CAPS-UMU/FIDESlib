#include <cuda_runtime.h>
#include <iostream>
#include <tuple>

int main() {
    int deviceCount, device;
    int gpuDeviceCount = 0;
    cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess)
        deviceCount = 0;
    /* machines with no GPUid can still report one emulation device */
    for (device = 0; device < deviceCount; ++device) {
        std::ignore = cudaGetDeviceProperties(&properties, device);

        if (properties.major != 9999) { /* 9999 means emulation only */
            ++gpuDeviceCount;
            printf("GPU %d: %s\n", device, properties.name);
        }
    }
    printf("%d GPU device(s) found\n", gpuDeviceCount);

    /* don't just return the number of gpus, because other runtime cuda
       errors can also yield non-zero return values */
    if (gpuDeviceCount > 0)
        return 0; /* success */
    else
        return 1; /* failure */
}