
#include <iostream>
#include "program.h"

int main()
{
    const unsigned int width = 512;
    const unsigned int height = 1024;

    FastMandelbrot::program program{width, height};
    program.run();

    return 0;
}
