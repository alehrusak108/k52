#include <k52/dsp/transform/cuda_fast_fourier_transform.h>
#include <k52/dsp/transform/util/cuda_utils.h>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#ifdef BUILD_WITH_CUDA

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime_api.h>
#include <boost/thread/mutex.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include "../../../../../../../usr/local/cuda/include/cufft.h"
#include "../../../../../../../usr/local/cuda/include/cufftXt.h"
#include "../../../../../../../usr/local/cuda/include/cuda_runtime_api.h"

// TODO: DELETE THIS IMPORTS!

#endif

using ::std::vector;
using ::std::complex;
using ::std::invalid_argument;
using ::std::runtime_error;

namespace k52
{
namespace dsp
{

#ifdef BUILD_WITH_CUDA

// Using pImpl approach to hide CUFFT from outside use
// NOTE: Prefix "device_" means that variable is allocated in CUDA Device Memory
//       Prefix "host_" means that variable is allocated in RAM (Host)

class CudaFastFourierTransform::CudaFastFourierTransformImpl
{

public:
    CudaFastFourierTransformImpl(vector<complex<double> > signal, size_t page_size)
            : signal_(signal), page_size_(page_size) {

        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);

        if (signal_.size() % page_size != 0)
        {
            throw std::invalid_argument("CUDA FFT FATAL: Modulo of sequence_size with page_size should be 0.");
        }

        signal_size_ = signal_.size();
        signal_memory_size_ = sizeof(cufftComplex) * signal_size_;
        total_pages = signal_size_ / page_size_;

        cufftResult cufftResult;
        cufftResult = cufftPlan1d(&cufft_execution_plan_, page_size_, CUFFT_C2C, BATCH_COUNT_);
        CudaUtils::checkCufftErrors(cufftResult, "CUFFT Create Plan");

        cudaError cuda_result;
        cuda_result = cudaMalloc((void **) &device_signal_, signal_memory_size_);
        CudaUtils::checkErrors(cuda_result, "CUFFT FORWARD allocation on single GPU");

        // Copy the whole signal to Device
        host_signal_ = CudaUtils::VectorToCufftComplex(signal_);
        cuda_result = cudaMemcpy(device_signal_, host_signal_, signal_size_, cudaMemcpyHostToDevice);
        CudaUtils::checkErrors(cuda_result, "CUFFT FORWARD memory copying from Host to Device");
    }

    ~CudaFastFourierTransformImpl() {

        std::cout << "Destroying CUFFT Context..." << std::endl << std::endl;

        cufftResult cufft_result = cufftDestroy(cufft_execution_plan_);
        CudaUtils::checkCufftErrors(cufft_result, "CUFFT Execution Plan destructor");

        cudaError cuda_result = cudaFree(device_signal_);
        CudaUtils::checkErrors(cuda_result, "CUFFT Execution Plan destructor");

        free(host_signal_);

        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);

        std::cout << "CUFFT Context Destroyed" << std::endl << std::endl;
    }

    void DirectTransform()
    {
        Transform(CUFFT_FORWARD);
    }

    void InverseTransform()
    {
        Transform(CUFFT_INVERSE);
    }

    void Transform(int transform_direction)
    {
        cufftResult cufft_result;

        for (unsigned int page_number = 0; page_number < total_pages; page_number++) {
            size_t start_index = page_size_ * page_number;
            size_t end_index = start_index + page_size_;
            vector<complex<double> >::const_iterator page_start = signal_.begin() + start_index;
            vector<complex<double> >::const_iterator page_end = signal_.begin() + end_index;
            vector<complex<double> > signal_page(page_start, page_end);

            // NOTE: Transformed signal will be written instead of source signal to escape memory wasting
            clock_t execution_time = clock();
            cufft_result = cufftExecC2C(
                    cufft_execution_plan_,
                    (device_signal_ + start_index),
                    (device_signal_ + end_index),
                    transform_direction
            );
            std::cout << std::endl << "CUFFT FORWARD Transformation finished in: " << (float) (clock() - execution_time) / CLOCKS_PER_SEC << " seconds " << std::endl;
            CudaUtils::checkCufftErrors(cufft_result, "CUFFT FORWARD C2C execution");
        }
    }

    vector<complex<double> > GetTransformResult() const
    {
        // Copy whole device memory (FFT calculation results - device_signal) to Host memory (RAM)
        cudaError cuda_result;
        cuda_result = cudaMemcpy(host_signal_, device_signal_, page_size_, cudaMemcpyDeviceToHost);
        CudaUtils::checkErrors(cuda_result, "CUFFT FORWARD C2C Copying execution results from Device to Host");

        return CudaUtils::CufftComplexToVector(host_signal_, signal_size_);
    }

private:

    // static fields and initializers
    static boost::mutex cuda_mutex_;
    static const int BATCH_COUNT_ = 1;

    // instance fields and initializers
    vector<complex<double> > signal_;
    size_t signal_size_;
    size_t page_size_;
    int total_pages;
    int signal_memory_size_;

    cufftComplex *device_signal_;
    cufftComplex *host_signal_;
    cufftHandle cufft_execution_plan_;
};

boost::mutex CudaFastFourierTransform::CudaFastFourierTransformImpl::cuda_mutex_;

CudaFastFourierTransform::CudaFastFourierTransform(vector<complex<double> > signal, size_t page_size)
{
    cuda_fast_fourier_transform_impl_ =
            boost::make_shared<CudaFastFourierTransformImpl>(signal, page_size);
}

CudaFastFourierTransform::~CudaFastFourierTransform()
{
}

void CudaFastFourierTransform::DirectTransform()
{
    cuda_fast_fourier_transform_impl_->DirectTransform();
}

void CudaFastFourierTransform::InverseTransform()
{
    cuda_fast_fourier_transform_impl_->InverseTransform();
}

vector<complex<double> > CudaFastFourierTransform::GetTransformResult() const
{
    return cuda_fast_fourier_transform_impl_->GetTransformResult();
}

#endif //BUILD_WITH_CUDA

} // namespace dsp
} // namespace k52