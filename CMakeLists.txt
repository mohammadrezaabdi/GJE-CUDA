cmake_minimum_required(VERSION 3.19)
project(GJE CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(GJE main.cu matrix_utils.cpp matrix_utils.h)

set_target_properties(
        GJE
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)