cmake_minimum_required(VERSION 2.8)
set(PROJECT_NAME fourier_based_circular_convolution)
project(${PROJECT_NAME})

set(BOOST_ROOT "")
find_package(Boost REQUIRED)

# This k52 example required CUDA-package as it has CUDA-depended code and test.
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    add_definitions(-DBUILD_WITH_CUDA)
endif()

find_package(k52 REQUIRED HINTS "${CMAKE_CURRENT_LIST_DIR}/../../install")

# We have to include CUDA variables to include and link directories
include_directories(
    ${k52_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${FFTW3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

link_directories(
    ${k52_LIBRARY_DIRS}
    ${CUDA_LIBRARIES}
)

# Shows all the NVCC compiler output
set(CUDA_VERBOSE_BUILD ON)

# Add C++ complier flags
list(APPEND CMAKE_CXX_FLAGS -O3 -Wall)

# Add CUDA FFT Test executable - similar as for C++ Compiler
# Example is *.cu file, because it consists both of C++ code and CUDA C code
set(CUDA_SOURCE fft_performance_examples.cu)
cuda_add_executable(${PROJECT_NAME} ${CUDA_SOURCE})

# And link all the executables against k52_LIBRARIES which already includes additionally
target_link_libraries(${PROJECT_NAME} libk52_dsp_cuda_fft_lib.a ${k52_LIBRARIES})