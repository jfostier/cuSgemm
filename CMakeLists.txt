cmake_minimum_required(VERSION 3.18)
project(MatmatCUDA LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(matmat_VERSION_MAJOR 1)
set(matmat_VERSION_MINOR 0)

# Compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Find CUDAToolkit
find_package(CUDAToolkit REQUIRED)

# Include CUDA directories
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Add executable
add_executable(sgemm src/main.cu src/kernel_1.cu src/kernel_2.cu
                                 src/kernel_3.cu src/kernel_4.cu
                                 src/kernel_5.cu src/kernel_6.cu)

set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES "80;86;89")

# Link against cuBLAS
target_link_libraries(sgemm PRIVATE CUDA::cudart CUDA::cublas)
