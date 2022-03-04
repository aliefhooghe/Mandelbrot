cmake_minimum_required (VERSION 3.8)
project(FastMandelbrot CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(fast_mandelbrot
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_exception.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_texture.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_texture.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/mandelbrot_kernel.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/program.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/program.h
)

target_link_libraries(fast_mandelbrot PUBLIC
    SDL2::SDL2
    OpenGL::GL
    OpenGL::GLU
    GLEW::GLEW
)
