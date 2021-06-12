#pragma once

#include <exception>

namespace host
{
    template <typename T>
    class Box
    {
    public:
        Box(T* ptr)
            : device_ptr_(ptr)
        {}

        ~Box()
        {
            auto rc = cudaFree(device_ptr_);
            if (rc)
                throw std::domain_error("could not free");
        }

        const T* get() const
        {
            return device_ptr_;
        }

        T* get()
        {
            return device_ptr_;
        }

        operator const T*() const
        {
            return device_ptr_;
        }

        operator T*()
        {
            return device_ptr_;
        }

    private:
        T* device_ptr_;
    };

} // namespace host
