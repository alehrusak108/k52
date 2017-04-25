include(FindCUDA)

find_package(CUDA REQUIRED)

# Shows all the NVCC compiler output
set(CUDA_VERBOSE_BUILD ON)

if(CUDA_FOUND)
    message("${K52_MESSAGE_PREFIX} CUDA Toolkit has been found. Starting CUDA configuration checking... ${K52_MESSAGE_POSTFIX}")
    message("-- CUDA Toolkit version is ${CUDA_VERSION_STRING} ${K52_MESSAGE_POSTFIX}")
    message("-- CUDA Toolkit root directory is: ${CUDA_TOOLKIT_ROOT_DIR} ${K52_MESSAGE_POSTFIX}")
    message("-- CUDA include directory is: ${CUDA_INCLUDE_DIRS} ${K52_MESSAGE_POSTFIX}")
    message("-- CUDA link library (cudart) is: ${CUDA_LIBRARIES} ${K52_MESSAGE_POSTFIX}")
    message("-- CUDA CUFFT library is: ${CUDA_CUFFT_LIBRARIES} ${K52_MESSAGE_POSTFIX}")
    if(CUDA_BUILD_EMULATION STREQUAL "OFF")
        message("-- CUDA device emulation is: ${CUDA_BUILD_EMULATION}. Sources are compiled to be run on a CUDA-Capable NVIDIA Device. ${K52_MESSAGE_POSTFIX}")
    else()
        message("-- CUDA device emulation is: ${CUDA_BUILD_EMULATION}. Sources are compiled to be run on a CPU. ${K52_MESSAGE_POSTFIX}")
    endif()
    add_definitions(-DBUILD_WITH_CUDA)
    message("${K52_MESSAGE_PREFIX} CUDA configuration checking finished. ${K52_MESSAGE_POSTFIX}")
endif()

if(NOT CUDA_FOUND)
    message(WARNING "${K52_MESSAGE_PREFIX} CUDA has NOT been found. CUDA depended code will FAIL TO EXECUTE. ${K52_MESSAGE_POSTFIX}")
endif()

# NVCC Compiler additional flags
# list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_10,code=sm_10) - deprecated
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_11,code=sm_11)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_12,code=sm_12)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
