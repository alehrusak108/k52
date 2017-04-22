cmake_minimum_required(VERSION 2.8)
set(PROJECT_NAME fast_fourier_transform_example)
project(${PROJECT_NAME})

set(BOOST_ROOT "")
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(k52 REQUIRED HINTS "${CMAKE_CURRENT_LIST_DIR}/../../install")

include_directories(
    ${k52_INCLUDE_DIRS}
    ${FFTW3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)
link_directories(
    ${k52_LIBRARY_DIRS}
    ${CUDA_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME} ${k52_LIBRARIES})