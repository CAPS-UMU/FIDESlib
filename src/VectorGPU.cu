//
// Created by carlosad on 2/05/24.
//
#include "VectorGPU.cuh"

namespace FIDESlib {
template <typename T>
VectorGPU<T>::VectorGPU(VectorGPU<T>&& v) noexcept : freeing(v.freeing), managed(v.managed), data(v.data), size(v.size), device(v.device) {
	v.freeing = true;
	v.managed = false;
}

template <typename T>
VectorGPU<T>::VectorGPU(T* data, const int size, const int device, const int offset)
: data(data + offset), size(size), device(device), managed(false), freeing(true) {
	assert(data != nullptr);
	{
		cudaPointerAttributes att{};
		cudaPointerGetAttributes(&att, data);
		assert(att.type == cudaMemoryTypeManaged || att.type == cudaMemoryTypeDevice);
		assert(att.device == this->device);
	}
	CudaCheckErrorModNoSync;
	assert(size > 0);

	Out(MEMORY, "Unmanaged vector construct OK");
}

template <typename T> VectorGPU<T>::~VectorGPU() {
	assert(freeing == true);
	Out(MEMORY, "Vector destruct OK");
}

template <typename T> void VectorGPU<T>::free(Stream& stream) {
	if (!managed) {
		return;
	}
	assert(!freeing);
	// cudaDeviceSynchronize();
	GPUfree(data, device, sizeof(T) * size, stream.ptr(), true);
	// cudaFreeAsync((void*)data, stream.ptr());
	freeing = true;
	Out(MEMORY, "Managed vector free OK");
}

template <typename T>
VectorGPU<T>::VectorGPU(Stream& stream, const int size, const int device, const T* src)
: data(nullptr), size(size), device(device), freeing(false), managed(true) {
	assert(device >= 0);
	{
		int device_count = -1;
		assert(cudaGetDeviceCount(&device_count) == cudaSuccess);
		assert(device < device_count);
		(void)device_count;
	}
	{
		int dev = -1;
		assert(cudaGetDevice(&dev) == cudaSuccess);
		assert(dev == device);
		(void)dev;
		// cudaSetDevice(device);
	}
	int bytes = size * sizeof(T);

	if (size == 0) {
		managed = false;
		freeing = true;
	} else {
		// cudaDeviceSynchronize();
		data = (T*)GPUmalloc(device, bytes, stream.ptr(), true);
		// cudaDeviceSynchronize();
		// cudaMallocAsync(&data, bytes, stream.ptr());

		if (src != nullptr) {
			cudaMemcpyAsync(data, src, bytes, cudaMemcpyHostToDevice, stream.ptr());
		}
	}
	Out(MEMORY, "Managed vector construct OK");
}

template class VectorGPU<int>;
template class VectorGPU<void*>;
template class VectorGPU<void**>;
template class VectorGPU<uint32_t>;
template class VectorGPU<uint64_t>;
} // namespace FIDESlib