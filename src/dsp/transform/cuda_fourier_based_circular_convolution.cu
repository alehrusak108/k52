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
__global__ void MultiplySignals(cufftComplex *first, cufftComplex *second, int signal_size)
{
        // Elements of the result of signals multiplication are calculated in parallel
        // using thread_id variable - thread index.
        // Each thread calculates one element of result sequence at first[thread_id] moment.
    const int threads_count = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int id = thread_id; id < signal_size; id += threads_count) {
        cufftComplex result_element;
        result_element.x = first[thread_id].x * second[thread_id].x - first[thread_id].y * second[thread_id].y;
        result_element.y = first[thread_id].x * second[thread_id].y + first[thread_id].y * second[thread_id].x;
        first[id] = result_element;
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

    // Here are used additional CudaFastFourierTransform methods
    // to prevent from useless copying cufftComplex arrays into vector
    cudaLibXtDesc *first_transform =
            cufft_transformer_->DirectTransformLibXtDesc(first_signal);
    cudaLibXtDesc *second_transform =
            cufft_transformer_->DirectTransformLibXtDesc(second_signal);

    // Perform Multiplication on several GPUs
    int available_gpus = cufft_transformer_->GetAvailableGPUs();
    MultiplySignalsOnMultipleGPUs(first_transform, second_transform, signal_size, available_gpus);

    // NOTE: Multiplication results were written instead of first_transform
    vector<complex<double> > convolution =
            cufft_transformer_->InverseTransformLibXtDesc(first_transform, signal_size);

    cufftXtFree(first_transform);
    cufftXtFree(second_transform);

    return convolution;
}

void CudaFourierBasedCircularConvolution::MultiplySignalsOnMultipleGPUs(
        cudaLibXtDesc *first_desc, cudaLibXtDesc *second_desc, int signal_size, int gpu_count) const
{
    int device;
    for (int gpu_index = 0; gpu_index < gpu_count; gpu_index++)
    {
        device = first_desc->descriptor->GPUs[gpu_index];

        // Set GPU
        cudaSetDevice(device);
        // Perform GPU computations
        MultiplySignals<<<32, 256>>>(
                (cufftComplex *) first_desc->descriptor->data[gpu_index],
                (cufftComplex *) second_desc->descriptor->data[gpu_index],
                signal_size
        );
    }

    // Wait for device to finish all operation
    for (int gpu_index = 0; gpu_index < gpu_count; gpu_index++)
    {
        device = first_desc->descriptor->GPUs[gpu_index];
        cudaSetDevice(device);
        cudaDeviceSynchronize();
    }
}

#endif //BUILD_WITH_CUDA
