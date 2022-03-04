#ifndef MANDELBROT_KERNEL_H_
#define MANDELBROT_KERNEL_H_

#include <cuda.h>

namespace FastMandelbrot
{

    void call_mandelbrot_kernel(
        cudaSurfaceObject_t surface, unsigned int width, unsigned int height,
        float2 origin, float radius, unsigned int step_count);
}

#endif /*MANDELBROT_KERNEL_H_ */
