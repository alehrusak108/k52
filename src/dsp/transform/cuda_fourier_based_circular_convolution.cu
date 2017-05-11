#include <k52/dsp/transform/cuda_fourier_based_circular_convolution.h>
#include <cstdio>
#include <stdexcept>

#ifdef BUILD_WITH_CUDA

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <boost/thread/mutex.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <k52/dsp/transform/util/cuda_utils.h>
#include "../../../../../../../usr/local/cuda/include/cuda_runtime_api.h"
#include "../../../../../../../usr/local/cuda/include/device_launch_parameters.h"
#include "../../../../../../../usr/local/cuda/include/cufftXt.h"

// TODO: DELETE THIS IMPORTS - THEY ARE ONLY FOR CLION COMPILATION PURPOSE

#endif

using ::std::vector;
using ::std::complex;
using ::k52::dsp::CudaFourierBasedCircularConvolution;

#ifdef BUILD_WITH_CUDA

// CUDA kernel function used to multiply two signals in parallel
// NOTE: Result is written instead of first signal
__global__ void MultiplySignals(cufftComplex *first,
                                cufftComplex *second,
                                int signal_size)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < signal_size)
    {
        // Elements of the result of signals multiplication are calculated in parallel
        // using thread_id variable - thread index.
        // Each thread calculates one element of result sequence at first[thread_id] moment.
        cufftComplex result_element;
        result_element.x = first[thread_id].x * second[thread_id].x - first[thread_id].y * second[thread_id].y;
        result_element.y = first[thread_id].x * second[thread_id].y + first[thread_id].y * second[thread_id].x;
        first[thread_id] = result_element;
    }
}

CudaFourierBasedCircularConvolution::CudaFourierBasedCircularConvolution(size_t sequence_size, int batch_size)
{
    cufft_transformer_ = boost::make_shared<CudaFastFourierTransform>(sequence_size, batch_size);
}

vector<complex<double> > CudaFourierBasedCircularConvolution::EvaluateConvolution(
        const vector<complex<double> > &first_signal,
        const vector<complex<double> > &second_signal) const
{
    if (first_signal.size() != second_signal.size())
    {
        throw std::runtime_error("Can evaluate convolution only for sequences of the same size.");
    }

    size_t signal_size = first_signal.size();

    // Create one signal based on input signals to pass it through
    // Two GPUs via cudaLibXt, assuming, that signals are of the same size
    // And sum of sizes is multiplied by 2.
    vector<complex<double> > sum_signal;

    sum_signal.reserve(signal_size * 2);
    copy(first_signal.begin(), first_signal.end(), back_inserter(sum_signal));
    copy(second_signal.begin(), second_signal.end(), back_inserter(sum_signal));

    // Here are used additional CudaFastFourierTransform methods
    // to prevent from useless copying cufftComplex arrays into vector
    cudaLibXtDesc *sum_signal_transform =
            cufft_transformer_->DirectTransformLibXtDesc(sum_signal);

    int signal_memory_size = sizeof(cufftComplex) * signal_size;

    cudaXtDesc *result_descriptor = sum_signal_transform->descriptor;

    // Get FFT-results from each GPU
    cufftComplex *gpu0_result = (cufftComplex*) (result_descriptor->data[0]);
    cufftComplex *gpu1_result = (cufftComplex*) (result_descriptor->data[1]);

    // Copy FFT-results from GPU_1 to GPU_0
    // To calculate multiplication in parallel on one device
    cufftComplex *gpu0_result_from_gpu1;
    cudaMalloc((void**) &gpu0_result_from_gpu1, signal_memory_size);
    cudaMemcpy(gpu0_result_from_gpu1, gpu1_result, signal_memory_size, cudaMemcpyDeviceToDevice);

    cudaSetDevice(0);
    MultiplySignals<<<64, 256>>>(gpu0_result_from_gpu1, gpu0_result, signal_size);

    for (int i = 0; i < signal_size; i++) {
        std::cout << gpu0_result_from_gpu1[i].x << "\t" << gpu0_result_from_gpu1[i].y << std::endl;
    }

    cufftXtFree(sum_signal_transform);
    cudaFree(gpu0_result);

    return cufft_transformer_->InverseTransformFromDevice(gpu0_result_from_gpu1, signal_size);
}

#endif //BUILD_WITH_CUDA
