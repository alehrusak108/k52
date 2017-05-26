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

    std::cout << "FIRST" << std::endl;
    cufft_transformer_->SetDeviceSignalFromVector(first_signal);
    cufft_transformer_->DirectTransform();
    vector<complex<double> > first_transform = cufft_transformer_->GetTransformResult();

    std::cout << "SECOND" << std::endl;
    cufft_transformer_->SetDeviceSignalFromVector(second_signal);
    cufft_transformer_->DirectTransform();
    vector<complex<double> > second_transform = cufft_transformer_->GetTransformResult();

    cufftComplex *first = CudaUtils::VectorToCufftComplexAlloc(first_transform);
    cufftComplex *second = CudaUtils::VectorToCufftComplexAlloc(second_transform);

    float scale = 1.0f / signal_size;
    MultiplySignals<<<256, 512>>>(first, second, signal_size, scale);

    vector<complex<double> > multiplication = CudaUtils::CufftComplexToVector(first, signal_size);

    std::cout << "THIRD" << std::endl;
    cufft_transformer_->SetDeviceSignalFromVector(multiplication);
    cufft_transformer_->InverseTransform();

    return cufft_transformer_->GetTransformResult();
}

#endif //BUILD_WITH_CUDA
