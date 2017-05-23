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
#include <k52/common/helpers.h>
#include "../../../../../../../usr/local/cuda/include/cufft.h"
#include "../../../../../../../usr/local/cuda/include/cufftXt.h"
#include "../../../../../../../usr/local/cuda/include/cuda_runtime_api.h"
#include "../../../../../../../usr/local/cuda/include/device_launch_parameters.h"

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
            : signal_(signal), page_size_(page_size)
    {

        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);

        if (signal_.size() % page_size != 0)
        {
            throw std::invalid_argument("CUDA FFT FATAL: Modulo of sequence_size with page_size should be 0.");
        }

        signal_size_ = signal_.size();
        total_pages_ = signal_size_ / page_size_;

        cudaSetDevice(0);

        std::cout << std::endl << "Constructing the CUFFT Context with the following parameters: " << std::endl
                  << "Signal Size: " << signal_size_ << std::endl
                  << "Page Size: " << page_size_ << std::endl
                  << "Total Pages: " << total_pages_ << std::endl << std::endl;

        cufftResult cufftResult;
        cufftResult = cufftPlan1d(&cufft_execution_plan_, page_size_, CUFFT_C2C, BATCH_COUNT_);
        CudaUtils::checkCufftErrors(cufftResult, "CUFFT Create Plan");

        device_signal_pages_ = (cufftComplex **) malloc(total_pages_ * sizeof(cufftComplex*));
        host_signal_page_ = (cufftComplex *) malloc(page_size_ * sizeof(cufftComplex));
        for (unsigned int page_number = 0; page_number < total_pages_; page_number++)
        {
            size_t start_index = page_size_ * page_number;
            size_t end_index = start_index + page_size_;
            vector<complex<double> >::const_iterator page_start = signal_.begin() + start_index;
            vector<complex<double> >::const_iterator page_end = signal_.begin() + end_index;
            vector<complex<double> > signal_page(page_start, page_end);

            cudaError cuda_result;
            cuda_result = cudaMalloc((void **) &device_signal_pages_[page_number], sizeof(cufftComplex) * page_size_);
            CudaUtils::checkErrors(cuda_result, "CUFFT FORWARD allocation on single GPU");

            // Copy the whole signal to Device
            CudaUtils::VectorToCufftComplex(signal_page, host_signal_page_);
            cuda_result = cudaMemcpy(device_signal_pages_[page_number], host_signal_page_, page_size_, cudaMemcpyHostToDevice);
            CudaUtils::checkErrors(cuda_result, "CUFFT FORWARD memory copying from Host to Device");
        }
    }

    ~CudaFastFourierTransformImpl() {

        std::cout << "Destroying CUFFT Context..." << std::endl << std::endl;

        cudaSetDevice(0);

        cufftResult cufft_result = cufftDestroy(cufft_execution_plan_);
        CudaUtils::checkCufftErrors(cufft_result, "CUFFT Execution Plan destructor");

        for (size_t page_number = 0; page_number < total_pages_; page_number++)
        {
            cudaError cuda_result = cudaFree(device_signal_pages_[page_number]);
            CudaUtils::checkErrors(cuda_result, "CUFFT cudaFree");
        }

        free(host_signal_page_);

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
        cudaSetDevice(0);
        for (size_t page_number = 0; page_number < total_pages_; page_number++)
        {
            cufftResult cufft_result = cufftExecC2C(
                    cufft_execution_plan_,
                    device_signal_pages_[page_number],
                    device_signal_pages_[page_number],
                    transform_direction
            );
            CudaUtils::checkCufftErrors(cufft_result, "CUFFT FORWARD C2C execution");
        }
    }

    vector<complex<double> > GetTransformResult()
    {
        cudaSetDevice(0);
        vector<complex<double> > result;
        result.reserve(signal_size_);
        for (size_t page_number = 0; page_number < total_pages_; page_number++)
        {
            cudaError cuda_result = cudaMemcpy(host_signal_page_, device_signal_pages_[page_number], page_size_, cudaMemcpyDeviceToHost);
            CudaUtils::checkErrors(cuda_result, "CUFFT FORWARD C2C Copying execution results from Device to Host");
            vector<complex<double> > page_vector = CudaUtils::CufftComplexToVector(host_signal_page_, page_size_);
            k52::common::Helpers::PrintComplexVector(page_vector);
            result.insert(result.end(), page_vector.begin(), page_vector.end());
        }
        return result;
    }

private:

    // static fields and initializers
    static boost::mutex cuda_mutex_;
    static const int BATCH_COUNT_ = 1;

    // instance fields and initializers
    vector<complex<double> > signal_;
    size_t signal_size_;
    size_t page_size_;
    int total_pages_;

    cufftComplex **device_signal_pages_;
    cufftComplex *host_signal_page_;
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

vector<complex<double> > CudaFastFourierTransform::GetTransformResult()
{
    return cuda_fast_fourier_transform_impl_->GetTransformResult();
}

#endif //BUILD_WITH_CUDA

} // namespace dsp
} // namespace k52