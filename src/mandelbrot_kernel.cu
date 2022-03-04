
#include "mandelbrot_kernel.cuh"
#include "cuda_exception.cuh"

namespace FastMandelbrot
{
    static constexpr auto THREAD_PER_BLOCK = 64u;

    static __global__ void _mandelbrot_kernel(
        cudaSurfaceObject_t surface, unsigned int width,
        float2 origin, float radius, unsigned int step_count)
    {
        //  Get pixel position in image
        const auto x = blockIdx.x * blockDim.x + threadIdx.x;
        const auto y = blockIdx.y;
        float4 rgba_value{1.f, 0, 0, 1.f};

        if (x < width)
        {
            surf2Dwrite(rgba_value, surface, x * sizeof(float4), y);
        }
    }

    static dim3 _image_grid_dim(unsigned int width, unsigned int height, unsigned int& thread_per_block)
    {
        thread_per_block = std::min<int>(thread_per_block, width);
        const auto horizontal_block_count = static_cast<unsigned int>(std::ceil((float)width / (float)thread_per_block));
        return dim3{horizontal_block_count, height};
    }

    void call_mandelbrot_kernel(
        cudaSurfaceObject_t surface, unsigned int width, unsigned int height,
        float2 origin, float radius, unsigned int step_count)
    {
        unsigned int thread_per_block = THREAD_PER_BLOCK;
        const auto grid_dim = _image_grid_dim(width, height, thread_per_block);

        _mandelbrot_kernel<<<grid_dim, thread_per_block>>>(surface, width, origin, radius, step_count);

        // wait the device and catch kernel errors (such as invalid memory access)
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}