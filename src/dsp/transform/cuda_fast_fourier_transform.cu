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

using ::std::ios;
using ::std::ofstream;
using ::std::vector;
using ::std::complex;
using ::std::invalid_argument;
using ::std::runtime_error;

namespace k52
{
namespace dsp
{

#ifdef BUILD_WITH_CUDA

// Initializes given pointer to signal page with signal data using "begin" and "end" indexes
__global__ void InitializeSignalPage(cufftComplex *page, cufftComplex *signal, int page_size, int from_index)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < page_size)
    {
        page[thread_id].x = signal[from_index + thread_id].x;
        page[thread_id].y = signal[from_index + thread_id].y;
    }
}

// Copies given pointer to signal page into signal using "begin" and "end" indexes
__global__ void CopyPageToSignal(cufftComplex *signal, cufftComplex *page, int page_size, int from_index)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < page_size)
    {
        signal[from_index + thread_id].x = page[thread_id].x;
        signal[from_index + thread_id].y = page[thread_id].y;
    }
}

// Using pImpl approach to hide CUFFT from outside use
// NOTE: Prefix "device_" means that variable is allocated in CUDA Device Memory
//       Prefix "host_" means that variable is allocated in RAM (Host)
class CudaFastFourierTransform::CudaFastFourierTransformImpl
{

public:
    CudaFastFourierTransformImpl(size_t signal_size, size_t page_size)
            : signal_size_(signal_size), page_size_(page_size)
    {

        ofstream test_output;
        test_output.open("fast_fourier_transform_test.txt", ios::out | ios::app);

        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);

        if (signal_size <= 0)
        {
            throw std::invalid_argument("CUDA FFT FATAL: Modulo of sequence_size with page_size should be 0.");
        }

        total_pages_ = signal_size_ / page_size_;

        cudaSetDevice(0);

        test_output << std::endl << "Constructing the CUFFT Context with the following parameters: " << std::endl
                  << "Signal Size: " << signal_size_ << std::endl
                  << "Page Size: " << page_size_ << std::endl
                  << "Total Pages: " << total_pages_ << std::endl << std::endl;

        cufftResult cufftResult;
        cufftResult = cufftPlan1d(&cufft_execution_plan_, page_size_, CUFFT_C2C, BATCH_COUNT_);
        CudaUtils::checkCufftErrors(cufftResult, "CUFFT Create Plan");

        cudaError cuda_result;
        cuda_result = cudaMalloc((void **) &device_signal_, sizeof(cufftComplex) * signal_size_);
        CudaUtils::checkErrors(cuda_result, "CUFFT FORWARD allocation on single GPU");

        cuda_result = cudaMalloc((void **) &device_signal_page_, sizeof(cufftComplex) * page_size_);
        CudaUtils::checkErrors(cuda_result, "CUFFT FORWARD allocation memory for a signal page");

        host_signal_ = (cufftComplex *) malloc(sizeof(cufftComplex) * signal_size_);
    }

    ~CudaFastFourierTransformImpl() {

        ofstream test_output;
        test_output.open("fast_fourier_transform_test.txt", ios::out | ios::app);

        test_output << "Destroying CUFFT Context..." << std::endl << std::endl;

        cufftResult cufft_result = cufftDestroy(cufft_execution_plan_);
        CudaUtils::checkCufftErrors(cufft_result, "CUFFT Execution Plan destructor");

        cudaError cuda_result;
        cuda_result = cudaFree(device_signal_);
        CudaUtils::checkErrors(cuda_result, "CUFFT cudaFree for device_signal_");

        cuda_result = cudaFree(device_signal_page_);
        CudaUtils::checkErrors(cuda_result, "CUFFT cudaFree for device_signal_page_");

        free(host_signal_);

        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);

        test_output << "CUFFT Context Destroyed" << std::endl << std::endl;
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
        // MAKE device_signal 1D and copy arrays in __global__ function
        for (size_t page_number = 0; page_number < total_pages_; page_number++)
        {
            size_t from_index = page_size_ * page_number;
            InitializeSignalPage<<<256, 512>>>(device_signal_page_, device_signal_, page_size_, from_index);

            cufftResult cufft_result = cufftExecC2C(
                    cufft_execution_plan_,
                    device_signal_page_,
                    device_signal_page_,
                    transform_direction
            );
            CudaUtils::checkCufftErrors(cufft_result, "CUFFT FORWARD C2C execution");

            CopyPageToSignal<<<256, 512>>>(device_signal_, device_signal_page_, page_size_, from_index);
        }
    }

    void SetDeviceSignal(vector<complex<double> > signal)
    {
        // Copy the whole signal to Device
        CudaUtils::VectorToCufftComplex(signal, host_signal_);
        cudaError cuda_result = cudaMemcpy(device_signal_, host_signal_, sizeof(cufftComplex) * signal_size_, cudaMemcpyHostToDevice);
        CudaUtils::checkErrors(cuda_result, "CUFFT SetDeviceSignal setting other signal. Copy from Host to Device");
    }

    void SetDeviceSignal(cufftComplex *signal)
    {
        // Copy the whole signal to Device
        cudaError cuda_result = cudaMemcpy(device_signal_, signal, sizeof(cufftComplex) * signal_size_, cudaMemcpyHostToDevice);
        CudaUtils::checkErrors(cuda_result, "CUFFT SetDeviceSignal setting other signal. Copy from Host to Device");
    }

    vector<complex<double> > GetTransformResult()
    {
        cudaError cuda_result = cudaMemcpy(host_signal_, device_signal_, signal_size_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        for (int i = 0; i < signal_size_; i++)
        {
            std::cout << host_signal_[i].x << "\t" << host_signal_[i].y << std::endl;
        }
        CudaUtils::checkErrors(cuda_result, "CUFFT GetTransformResult Copying execution results from Device to Host");
        return CudaUtils::CufftComplexToVector(host_signal_, signal_size_);
    }

    cufftComplex* GetTransformResultArray()
    {
        cudaError cuda_result = cudaMemcpy(host_signal_, device_signal_, signal_size_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        CudaUtils::checkErrors(cuda_result, "CUFFT GetTransformResultArray Copying execution results from Device to Host");
        return host_signal_;
    }

private:

    // static fields and initializers
    static boost::mutex cuda_mutex_;
    static const int BATCH_COUNT_ = 1;

    // instance fields and initializers
    size_t signal_size_;
    size_t page_size_;
    int total_pages_;

    cufftComplex *device_signal_;
    cufftComplex *host_signal_;
    cufftComplex *device_signal_page_;
    cufftHandle cufft_execution_plan_;
};

boost::mutex CudaFastFourierTransform::CudaFastFourierTransformImpl::cuda_mutex_;

CudaFastFourierTransform::CudaFastFourierTransform(size_t signal_size, size_t page_size)
{
    cuda_fast_fourier_transform_impl_ =
            boost::make_shared<CudaFastFourierTransformImpl>(signal_size, page_size);
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

void CudaFastFourierTransform::SetDeviceSignal(cufftComplex *signal)
{
    cuda_fast_fourier_transform_impl_->SetDeviceSignal(signal);
}

void CudaFastFourierTransform::SetDeviceSignal(vector<complex<double> > signal)
{
    cuda_fast_fourier_transform_impl_->SetDeviceSignal(signal);
}

vector<complex<double> > CudaFastFourierTransform::GetTransformResult()
{
    return cuda_fast_fourier_transform_impl_->GetTransformResult();
}

cufftComplex *CudaFastFourierTransform::GetTransformResultArray()
{
    return cuda_fast_fourier_transform_impl_->GetTransformResultArray();
}

#endif //BUILD_WITH_CUDA

} // namespace dsp
} // namespace k52