
#include "mandelbrot_kernel.cuh"
#include "cuda_exception.cuh"

namespace FastMandelbrot
{
    static constexpr auto THREAD_PER_BLOCK = 1024u;

    // Reference:
    static __global__ void _mandelbrot_kernel(
        cudaSurfaceObject_t surface, unsigned int width,
        double2 origin, double size, unsigned int step_count)
    {
        //  Get pixel position in image
        const auto x = blockIdx.x * blockDim.x + threadIdx.x;
        const auto y = blockIdx.y;

        const auto unit_per_pixel = size / static_cast<double>(width);

        if (x < width)
        {
            const auto c = double2{
                origin.x + unit_per_pixel * x,
                origin.y + unit_per_pixel * y
            };
            auto sequence = double2{0., 0.};

            int count = 0;
            // while count < step_count && |sequence| < 2.
            while (count < step_count && sequence.x * sequence.x + sequence.y * sequence.y < 4.)
            {
                const auto next_sequence = double2{
                    sequence.x * sequence.x - sequence.y * sequence.y + c.x,
                    2.f * sequence.x * sequence.y + c.y};
                sequence = next_sequence;
                count++;
            }

            const auto value = static_cast<float>(count) / static_cast<float>(step_count);
            const auto rgba = float4{value, value, value, 1.f};
            surf2Dwrite(rgba, surface, x * sizeof(float4), y);
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
        double2 origin, double size, unsigned int step_count)
    {
        unsigned int thread_per_block = THREAD_PER_BLOCK;
        const auto grid_dim = _image_grid_dim(width, height, thread_per_block);

        _mandelbrot_kernel<<<grid_dim, thread_per_block>>>(surface, width, origin, size, step_count);

        // wait the device and catch kernel errors (such as invalid memory access)
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}