#include <k52/dsp/transform/cuda_fourier_based_circular_convolution.h>
#include <k52/common/helpers.h>
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
#include "../../../../../../../usr/local/cuda/include/device_launch_parameters.h"
#include "../../../../../../../usr/local/cuda/include/cuda_runtime_api.h"

// TODO: DELETE THIS IMPORTS - THEY ARE ONLY FOR CLION COMPILATION PURPOSE

#endif

using ::std::vector;
using ::std::complex;
using ::k52::dsp::CudaFourierBasedCircularConvolution;
using ::k52::common::Helpers;

#ifdef BUILD_WITH_CUDA

// CUDA kernel function used to multiply two signals in parallel
// NOTE: Result is written instead of first signal
// TODO: Why CUFFT doesn't scale result signal on it's length, but FFTW does?
__global__ void MultiplySignals(cufftComplex *first, cufftComplex *second, int signal_size, float scale)
{
    // Elements of the result of signals multiplication are calculated in parallel
    // using thread_id variable - thread index.
    // Each thread calculates one element of result sequence at a moment.
    const int threads_count = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int id = thread_id; id < signal_size; id += threads_count) {
        cufftComplex result;
        result.x = (first[id].x * second[id].x - first[id].y * second[id].y);
        result.y = (first[id].x * second[id].y + first[id].y * second[id].x);
        first[id].x = result.x * scale;
        first[id].y = result.y * scale;
    }
}

CudaFourierBasedCircularConvolution::CudaFourierBasedCircularConvolution(size_t signal_size, size_t page_size)
{
    cufft_transformer_ = boost::make_shared<CudaFastFourierTransform>(signal_size, page_size_);
    this->page_size_ = page_size;

    signal_memory_size_ = signal_size * sizeof(cufftComplex);
    cudaError cuda_result;

    cuda_result = cudaMalloc((void **) &d_first_signal_, signal_memory_size_);
    CudaUtils::checkErrors(cuda_result, "Convolution: allocation 1 on single GPU");

    cuda_result = cudaMalloc((void **) &d_second_signal_, signal_memory_size_);
    CudaUtils::checkErrors(cuda_result, "Convolution: allocation 2 on single GPU");
}

vector<complex<double> > CudaFourierBasedCircularConvolution::EvaluateConvolution(
        const vector<complex<double> > &first_signal,
        const vector<complex<double> > &second_signal)
{
    if (first_signal.size() != second_signal.size())
    {
        throw std::runtime_error("Can evaluate convolution only for sequences of the same size.");
    }

    size_t signal_size = first_signal.size();

    cufft_transformer_->SetDeviceSignalFromVector(first_signal);
    cufft_transformer_->DirectTransform();
    vector<complex<double> > first_transform = cufft_transformer_->GetTransformResult();
    std::cout << "1" << std::endl;

    cufft_transformer_->SetDeviceSignalFromVector(second_signal);
    cufft_transformer_->DirectTransform();
    vector<complex<double> > second_transform = cufft_transformer_->GetTransformResult();
    std::cout << "2" << std::endl;

    cufftComplex *h_first = CudaUtils::VectorToCufftComplexAlloc(first_transform);
    cufftComplex *h_second = CudaUtils::VectorToCufftComplexAlloc(second_transform);

    cudaError cuda_result;
    cuda_result = cudaMemcpy(d_first_signal_, h_first, signal_memory_size_, cudaMemcpyHostToDevice);
    CudaUtils::checkErrors(cuda_result, "CUFFT SetDeviceSignalFromVector 1 setting signal from vector. Copy from Host to Device");

    cuda_result = cudaMemcpy(d_second_signal_, h_second, signal_memory_size_, cudaMemcpyHostToDevice);
    CudaUtils::checkErrors(cuda_result, "CUFFT SetDeviceSignalFromVector 2 setting signal from vector. Copy from Host to Device");

    float scale = 1.0f / signal_size;
    MultiplySignals<<<256, 512>>>(d_first_signal_, d_second_signal_, signal_size, scale);

    cudaDeviceSynchronize();

    cuda_result = cudaMemcpy(h_first, d_first_signal_, signal_memory_size_, cudaMemcpyDeviceToHost);
    CudaUtils::checkErrors(cuda_result, "CUFFT SetDeviceSignalFromVector 3 setting signal from vector. Copy from Host to Device");

    vector<complex<double> > multiplication = CudaUtils::CufftComplexToVector(h_first, signal_size);

    std::cout << "multiplication" << std::endl;
    cufft_transformer_->SetDeviceSignalFromVector(multiplication);
    cufft_transformer_->InverseTransform();

    return multiplication;
}

#endif //BUILD_WITH_CUDA
