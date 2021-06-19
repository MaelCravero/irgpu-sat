#pragma once

#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

namespace host::utils
{
    template <typename T>
    T* malloc(const std::vector<T>& vec)
    {
        T* res;

        auto rc = cudaMalloc(&res, vec.size() * sizeof(T));

        if (rc)
            throw std::bad_alloc();

        return res;
    }

    template <typename T>
    void memcpy(T* dst, const T* src, size_t size, cudaMemcpyKind kind)
    {
        auto rc = cudaMemcpy(dst, src, size, kind);

        if (rc)
            throw std::domain_error("unable to memcpy");
    }

    template <typename T>
    void memcpy(T* dst, const std::vector<T>& src)
    {
        memcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    template <typename T>
    void memcpy(std::vector<T>& dst, T* src)
    {
        memcpy(dst.data(), src, dst.size() * sizeof(T), cudaMemcpyDeviceToHost);
    }

    template <typename T>
    T* init_from(const std::vector<T>& vec)
    {
        auto res = malloc(vec);
        memcpy(res, vec);

        return res;
    }

} // namespace host::utils

namespace device::utils
{
    inline __device__ size_t x_idx()
    {
        return blockDim.x * blockIdx.x + threadIdx.x;
    }
} // namespace device::utils
