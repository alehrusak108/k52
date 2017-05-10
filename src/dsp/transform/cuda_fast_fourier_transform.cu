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
    CudaFastFourierTransformImpl(size_t sequence_size, int transforms_count)
            : signal_size_(sequence_size), transforms_count_(transforms_count) {

        std::ofstream test_output;
        test_output.open("test_output.txt");

        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);

        if (sequence_size <= 0) {
            throw std::invalid_argument("sequence_size <= 0");
        }

        signal_memory_size_ = sizeof(cufftComplex) * signal_size_;

        // Use only 2 GPUs if even more available
        int available_gpus;
        cudaGetDeviceCount(&available_gpus);
        int gpu_to_use = available_gpus > 2 ? 2 : available_gpus;
        int *gpu_array = GetAvailableGPUArray(gpu_to_use);

        cufft_work_size_ = (size_t *) malloc (sizeof(size_t) * gpu_to_use);
        cufftCreate(&cufft_execution_plan_);
        cufftResult set_gpus_result = cufftXtSetGPUs(cufft_execution_plan_, available_gpus, gpu_array);
        test_output << std::endl << "CUFFT Set GPUs result: " << set_gpus_result << std::endl;
        cufftResult plan_prepare_result = cufftMakePlan1d(
                cufft_execution_plan_,
                signal_size_,
                CUFFT_C2C,
                transforms_count_,
                cufft_work_size_
        );
        test_output << std::endl << "CUFFT Execution Plan prepared: " << plan_prepare_result << std::endl;
    }

    ~CudaFastFourierTransformImpl() {

        std::ofstream test_output;
        test_output.open("test_output.txt");
        test_output << "Destroying CUFFT Context..." << std::endl;

        // Destroy CUFFT Execution Plan
        cufftResult destructor_result = cufftDestroy(cufft_execution_plan_);
        std::cout << "CUFFT Execution Plan destructor returned: " << destructor_result << std::endl << std::endl;

        free(cufft_work_size_);

        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);
        test_output.close();
    }

    vector<complex<double> > DirectTransform(const vector<complex<double> > &sequence)
    {
        return Transform(sequence, CUFFT_FORWARD);
    }

    vector<complex<double> > InverseTransform(const vector<complex<double> > &sequence)
    {
        return Transform(sequence, CUFFT_INVERSE);
    }

    vector<complex<double> > Transform(const vector<complex<double> > &sequence, int transform_direction) const
    {

        std::ofstream test_output;
        test_output.open("test_output.txt");
        
        if (signal_size_ != sequence.size()) {
            throw std::invalid_argument(
                    "CudaFastFourierTransform can transform only data of the same size as was specified on construction.");
        }

        cufftComplex *host_signal = CudaUtils::VectorToCufftComplex(sequence);

        cudaLibXtDesc *device_signal;
        cufftXtMalloc(cufft_execution_plan_, &device_signal, CUFFT_XT_FORMAT_INPLACE);
        cufftXtMemcpy(cufft_execution_plan_, device_signal, host_signal, CUFFT_COPY_HOST_TO_DEVICE);

        test_output << std::endl << "Signal memory allocated: " << signal_memory_size_ << " bytes." << std::endl;

        // NOTE: Transformed signal will be written instead of source signal to escape memory wasting
        clock_t execution_time = clock();
        cufftResult execution_result = cufftXtExecDescriptorC2C(
                cufft_execution_plan_,
                device_signal,
                device_signal,
                CUFFT_FORWARD
        );
        test_output << std::endl << "CUFFT Transformation finished in: " << (float) (clock() - execution_time) / CLOCKS_PER_SEC << " seconds " << std::endl;
        test_output << std::endl << "CUFFT C2C (float) Execution result: " << execution_result << std::endl;

        // Copy Device memory (FFT calculation results - d_signal_output_) to Host memory (RAM)
        cufftXtMemcpy(cufft_execution_plan_, host_signal, device_signal, CUFFT_COPY_DEVICE_TO_HOST);

        vector<complex<double> > result_vector = CudaUtils::CufftComplexToVector(host_signal, signal_size_);

        cufftXtFree(device_signal);
        cudaFree(host_signal);

        test_output.close();
        return result_vector;
    }

private:

    // static fields and initializers
    static boost::mutex cuda_mutex_;

    // instance fields and initializers
    size_t signal_size_;
    int transforms_count_;
    int signal_memory_size_;

    size_t *cufft_work_size_;
    cufftHandle cufft_execution_plan_;

    int* GetAvailableGPUArray(int gpu_count)
    {
        int *gpu_array = (int*) malloc(sizeof(int) * gpu_count);
        for (unsigned int index = 0; index < gpu_count; index++)
        {
            gpu_array[index] = index;
        }
        return gpu_array;
    }
};

boost::mutex CudaFastFourierTransform::CudaFastFourierTransformImpl::cuda_mutex_;

CudaFastFourierTransform::CudaFastFourierTransform(size_t sequence_size, int planned_executions)
{
    cuda_fast_fourier_transform_impl_ =
            boost::make_shared<CudaFastFourierTransformImpl>(sequence_size, planned_executions);
}

CudaFastFourierTransform::~CudaFastFourierTransform()
{
}

vector<complex<double> > CudaFastFourierTransform::DirectTransform(
        const vector<complex<double> > &sequence) const
{
    return cuda_fast_fourier_transform_impl_->DirectTransform(sequence);
}

vector<complex<double> > CudaFastFourierTransform::InverseTransform(
        const vector<complex<double> > &sequence) const
{
    return cuda_fast_fourier_transform_impl_->InverseTransform(sequence);
}

#endif //BUILD_WITH_CUDA

} // namespace dsp
} // namespace k52