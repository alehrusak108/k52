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

        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);

        if (sequence_size <= 0)
        {
            throw std::invalid_argument("sequence_size <= 0");
        }

        signal_memory_size_ = sizeof(cufftComplex) * signal_size_;

        // Use only 2 GPUs if even more available
        cudaError error = cudaGetDeviceCount(&available_gpus);
        CudaUtils::checkErrors(error, "CUDA Get Device Count");

        int *gpu_array = GetAvailableGPUArray();

        cufft_work_size_ = (size_t *) malloc (sizeof(size_t) * available_gpus);
        cufftResult result;
        result = cufftCreate(&cufft_execution_plan_);
        CudaUtils::checkCufftErrors(result, "CUFFT Create Plan");

        result = cufftXtSetGPUs(cufft_execution_plan_, available_gpus, gpu_array);
        CudaUtils::checkCufftErrors(result, "CUFFT Set GPUs");

        result = cufftMakePlan1d(
                cufft_execution_plan_,
                signal_size_,
                CUFFT_C2C,
                transforms_count_,
                cufft_work_size_
        );
        CudaUtils::checkCufftErrors(result, "CUFFT Execution Plan preparing");
    }

    ~CudaFastFourierTransformImpl() {

        std::cout << "Destroying CUFFT Context..." << std::endl << std::endl;

        cufftResult result = cufftDestroy(cufft_execution_plan_);
        CudaUtils::checkCufftErrors(result, "CUFFT Execution Plan destructor");

        free(cufft_work_size_);

        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);

        std::cout << "CUFFT Context Destroyed" << std::endl << std::endl;
    }

    vector<complex<double> > DirectTransform(const vector<complex<double> > &sequence)
    {
        return Transform(sequence, CUFFT_FORWARD);
    }

    vector<complex<double> > InverseTransform(const vector<complex<double> > &sequence)
    {
        return Transform(sequence, CUFFT_INVERSE);
    }

    vector<complex<double> > Transform(const vector<complex<double> > &sequence, int transform_direction)
    {

        if (signal_size_ != sequence.size())
        {
            throw std::invalid_argument(
                    "CudaFastFourierTransform can transform only data of the same size as was specified on construction.");
        }

        cufftComplex *host_signal = CudaUtils::VectorToCufftComplex(sequence);

        cufftResult result;

        cudaLibXtDesc *device_signal;
        result = cufftXtMalloc(cufft_execution_plan_, &device_signal, CUFFT_XT_FORMAT_INPLACE);
        CudaUtils::checkCufftErrors(result, "CUFFT FORWARD allocation across GPUs");

        result = cufftXtMemcpy(cufft_execution_plan_, device_signal, host_signal, CUFFT_COPY_HOST_TO_DEVICE);
        CudaUtils::checkCufftErrors(result, "CUFFT FORWARD memory copying from Host to Device");

        std::cout << std::endl << "CUFFT FORWARD memory allocated across GPUs: " << signal_memory_size_ << " bytes." << std::endl;

        // NOTE: Transformed signal will be written instead of source signal to escape memory wasting
        clock_t execution_time = clock();
        result = cufftXtExecDescriptorC2C(
                cufft_execution_plan_,
                device_signal,
                device_signal,
                transform_direction
        );
        std::cout << std::endl << "CUFFT FORWARD Transformation finished in: " << (float) (clock() - execution_time) / CLOCKS_PER_SEC << " seconds " << std::endl;
        CudaUtils::checkCufftErrors(result, "CUFFT FORWARD C2C execution");

        // Copy Device memory (FFT calculation results - device_signal) to Host memory (RAM)
        result = cufftXtMemcpy(cufft_execution_plan_, host_signal, device_signal, CUFFT_COPY_DEVICE_TO_HOST);
        CudaUtils::checkCufftErrors(result, "CUFFT FORWARD C2C Copying execution results from Device to Host");

        vector<complex<double> > result_vector = CudaUtils::CufftComplexToVector(host_signal, signal_size_);

        cufftXtFree(device_signal);
        free(host_signal);

        return result_vector;
    }

    cudaLibXtDesc* DirectTransformLibXtDesc(const vector<complex<double> > &sequence)
    {
        if (signal_size_ != sequence.size())
        {
            throw std::invalid_argument(
                    "CudaFastFourierTransform LibXtDesc can transform only data of doubled size of a signal size.");
        }

        cufftResult result;

        cufftComplex *host_signal = CudaUtils::VectorToCufftComplex(sequence);

        cudaLibXtDesc *device_transform;
        result = cufftXtMalloc(cufft_execution_plan_, &device_transform, CUFFT_XT_FORMAT_INPLACE);
        CudaUtils::checkCufftErrors(result, "CUFFT FORWARD LibXtDesc allocation across GPUs");

        result = cufftXtMemcpy(cufft_execution_plan_, device_transform, host_signal, CUFFT_COPY_HOST_TO_DEVICE);
        CudaUtils::checkCufftErrors(result, "CUFFT FORWARD LibXtDesc memory copying from Host to Device");

        // NOTE: Transformed signal will be written instead of source signal to escape memory wasting
        clock_t execution_time = clock();
        result = cufftXtExecDescriptorC2C(
                cufft_execution_plan_,
                device_transform,
                device_transform,
                CUFFT_FORWARD
        );
        std::cout << std::endl << "CUFFT FORWARD LibXtDesc Transformation finished in: " << (float) (clock() - execution_time) / CLOCKS_PER_SEC << " seconds " << std::endl;
        CudaUtils::checkCufftErrors(result, "CUFFT FORWARD LibXtDesc C2C execution");

        // Copy the data to natural order on GPUs
        cudaLibXtDesc *natural_ordered_transform;
        cufftXtMalloc(cufft_execution_plan_, &natural_ordered_transform, CUFFT_XT_FORMAT_INPLACE);
        CudaUtils::checkCufftErrors(result, "CUFFT FORWARD LibXtDesc C2C allocation memory for result");

        cufftXtMemcpy(cufft_execution_plan_, natural_ordered_transform, device_transform, CUFFT_COPY_DEVICE_TO_DEVICE);
        CudaUtils::checkCufftErrors(result, "CUFFT FORWARD LibXtDesc C2C memory copying from Device to Host");

        cufftComplex *f = (cufftComplex *) malloc (natural_ordered_transform->descriptor->size[0]);
        cufftXtMemcpy(cufft_execution_plan_, (void **) &f, natural_ordered_transform->descriptor->data[0], CUFFT_COPY_DEVICE_TO_HOST);
        int size = (int) natural_ordered_transform->descriptor->size[0] / sizeof(cufftComplex);
        std::cout << std::endl << "SIZE 1: " << size << std::endl << std::endl;
        for (int i = 0; i < size; i++) {
            std::cout << f[i].x << "\t" << f[i].y << std::endl;
        }

        cufftComplex *s = (cufftComplex *) malloc (natural_ordered_transform->descriptor->size[1]);
        cufftXtMemcpy(cufft_execution_plan_, (void **) &s, natural_ordered_transform->descriptor->data[1], CUFFT_COPY_DEVICE_TO_HOST);
        size = (int) natural_ordered_transform->descriptor->size[1] / sizeof(cufftComplex);
        std::cout << std::endl << "SIZE 2: " << size << std::endl << std::endl;
        for (int i = 0; i < size; i++) {
            std::cout << s[i].x << "\t" << s[i].y << std::endl;
        }

        cufftXtFree(device_transform);
        return natural_ordered_transform;
    }

    // For this method it is assumed, that input_signal is already in GPU memory
    vector<complex<double> > InverseTransformLibXtDesc(cudaLibXtDesc *device_signal, int signal_size)
    {
        cufftResult result;

        clock_t execution_time = clock();

        // NOTE: Transformed signal will be written instead of source signal to escape memory wasting
        result = cufftXtExecDescriptorC2C(
                cufft_execution_plan_,
                device_signal,
                device_signal,
                CUFFT_INVERSE
        );
        std::cout << std::endl << "CUFFT INVERSE Transformation finished in: " << (float) (clock() - execution_time) / CLOCKS_PER_SEC << " seconds " << std::endl;
        CudaUtils::checkCufftErrors(result, "CUFFT INVERSE LibXtDesc C2C execution");

        cufftComplex *host_transformed = (cufftComplex *) malloc (signal_memory_size_);
        result = cufftXtMemcpy(cufft_execution_plan_, host_transformed, device_signal, CUFFT_COPY_DEVICE_TO_HOST);
        CudaUtils::checkCufftErrors(result, "CUFFT INVERSE LibXtDesc C2C Copying results from Device to Host");

        vector<complex<double> > result_vector = CudaUtils::CufftComplexToVector(host_transformed, signal_size_);

        cufftXtFree(device_signal);
        free(host_transformed);

        return result_vector;
    }

    int GetAvailableGPUs()
    {
        return available_gpus;
    }

private:

    // static fields and initializers
    static boost::mutex cuda_mutex_;

    // instance fields and initializers
    size_t signal_size_;
    int transforms_count_;
    int signal_memory_size_;

    int available_gpus;
    size_t *cufft_work_size_;
    cufftHandle cufft_execution_plan_;

    int* GetAvailableGPUArray()
    {
        int *gpu_array = (int*) malloc(sizeof(int) * available_gpus);
        for (unsigned int index = 0; index < available_gpus; index++)
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

cudaLibXtDesc* CudaFastFourierTransform::DirectTransformLibXtDesc(
        const vector<complex<double> > &sequence) const
{
    return cuda_fast_fourier_transform_impl_->DirectTransformLibXtDesc(sequence);
}

vector<complex<double> > CudaFastFourierTransform::InverseTransformLibXtDesc(
        cudaLibXtDesc *device_signal, int signal_size) const
{
    return cuda_fast_fourier_transform_impl_->InverseTransformLibXtDesc(device_signal, signal_size);
}

int CudaFastFourierTransform::GetAvailableGPUs() const {
    return cuda_fast_fourier_transform_impl_->GetAvailableGPUs();
}

#endif //BUILD_WITH_CUDA

} // namespace dsp
} // namespace k52