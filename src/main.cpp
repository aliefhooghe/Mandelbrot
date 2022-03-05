
#include <iostream>
#include "program.h"

int main()
{
    const unsigned int width = 2048;
    const unsigned int height = 1024;

    // Reference:
    https://fr.wikipedia.org/wiki/Ensemble_de_Mandelbrot

    FastMandelbrot::program program{width, height};
    program.run();

    return 0;
}
