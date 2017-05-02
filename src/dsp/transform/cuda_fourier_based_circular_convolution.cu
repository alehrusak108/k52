#include <k52/dsp/transform/cuda_fourier_based_circular_convolution.h>
#include <cstdio>
#include <stdexcept>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "TemplateArgumentsIssues"

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

// TODO: DELETE THIS IMPORTS - THEY ARE ONLY FOR CLION COMPILATION PURPOSE

#endif

using ::std::vector;
using ::std::complex;
using ::k52::dsp::CudaFourierBasedCircularConvolution;

#ifdef BUILD_WITH_CUDA

__global__ void MultiplySignals(const cufftComplex *first,
                                const cufftComplex *second,
                                cufftComplex *result,
                                int signal_size)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < signal_size) {
        // Elements of the result of signals multiplication are calculated in parallel
        // using thread_id variable - thread index.
        // Each thread calculates one element of result sequence at first[thread_id] moment.
        cufftComplex result_element;
        result_element.x = first[thread_id].x * second[thread_id].x - first[thread_id].y * second[thread_id].y;
        result_element.y = first[thread_id].x * second[thread_id].y + first[thread_id].y * second[thread_id].x;
        result[thread_id] = result_element;
    }
}

CudaFourierBasedCircularConvolution::CudaFourierBasedCircularConvolution(size_t sequence_size, int batch_size)
{
    cufft_transformer_ = boost::make_shared<CudaFastFourierTransform>(sequence_size, batch_size);
}

vector<complex<double> > CudaFourierBasedCircularConvolution::EvaluateConvolution(
        const vector<complex<double> > &first_sequence,
        const vector<complex<double> > &second_sequence) const
{
    if (first_sequence.size() != second_sequence.size())
    {
        throw std::runtime_error("Can evaluate convolution only for sequences of the same size.");
    }

    size_t signal_size = first_sequence.size();

    vector<complex<double> > first_sequence_transform =
            cufft_transformer_->DirectTransform(first_sequence);
    vector<complex<double> > second_sequence_transform =
            cufft_transformer_->DirectTransform(second_sequence);

    int signal_memory_size = sizeof(cufftComplex) * signal_size;

    cufftComplex* d_first;
    cudaMalloc((void**) &d_first, signal_memory_size);
    cudaMemcpy(d_first, first_sequence_transform.data(), signal_memory_size, cudaMemcpyHostToDevice);

    cufftComplex* d_second;
    cudaMalloc((void**) &d_second, signal_memory_size);
    cudaMemcpy(d_second, second_sequence_transform.data(), signal_memory_size, cudaMemcpyHostToDevice);

    cufftComplex* d_multiplication;
    cudaMalloc((void**) &d_multiplication, signal_memory_size);

    MultiplySignals<<<64, 128>>>(d_first, d_second, d_multiplication, signal_size);

    vector<complex<double> > convolution = CudaUtils::CufftComplexToVector(d_multiplication, signal_size);

    return cufft_transformer_->InverseTransform(convolution);
}

#endif //BUILD_WITH_CUDA

#pragma clang diagnostic pop