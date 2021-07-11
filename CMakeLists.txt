cmake_minimum_required(VERSION 3.19)
project(GJE CUDA)
set(TARGET_H matrix_utils.h matrix_utils.cuh)
set(TARGET_SRC main.cu matrix_utils.cu matrix_utils.cpp)

set(CMAKE_CUDA_STANDARD 14)
add_executable(GJE main.cu)

set_target_properties(
        GJE
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)