#include <k52/dsp/transform/cuda_fast_fourier_transform.h>
#include <k52/dsp/transform/util/cuda_utils.h>
#include <cstdio>
#include <stdexcept>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "TemplateArgumentsIssues"

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

namespace k52 {
namespace dsp {

#ifdef BUILD_WITH_CUDA

// Using pImpl approach to hide CUFFT for outside use
// NOTE: Prefix "d_" means that variable is allocated in CUDA Device Memory
//       Prefix "h_" means that variable is allocated in RAM (Host)
class CudaFastFourierTransform::CudaFastFourierTransformImpl {

public:
    CudaFastFourierTransformImpl(size_t sequence_size, int executions_planned)
            : signal_size_(sequence_size), executions_planned_(executions_planned) {

        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);

        if (sequence_size <= 0) {
            throw std::invalid_argument("sequence_size <= 0");
        }

        signal_memory_size_ = sizeof(cufftComplex) * signal_size_;

        // Planned Execitions (batch) other than 1 for cufftPlan1d() have been deprecated.
        // Here used cufftPlanMany() for multiple execution.

        /*int dimensions = 1; // 1D FFTs
        int ranks_array[] = { signal_size_ }; // Sizes of arrays of each dimension
        int istride = executions_planned_; // Distance between two successive input elements
        int ostride = executions_planned_; // Same for the output elements
        int idist = 1; // Distance between batches
        int odist = 1; // Same for the output elements
        int inembed[] = { 0 }; // Input size with pitch (ignored for 1D transforms)
        int onembed[] = { 0 }; // Output size with pitch (ignored for 1D transforms)

        // Single-Dimensional FFT execution plan configuration
        cufftResult plan_prepare_result = cufftPlanMany(
                &cufft_execution_plan_,
                dimensions,
                ranks_array,
                inembed, istride, idist,
                onembed, ostride, odist,
                CUFFT_C2C,
                executions_planned_
        );*/

        cudaMalloc((void**) &d_signal_, signal_memory_size_);
        std::cout << "Signal memory allocated: " << signal_memory_size_ << " bytes." << std::endl;

        cufftResult plan_prepare_result = cufftPlan1d(&cufft_execution_plan_, signal_size_, CUFFT_C2C, 1);
        std::cout << "CUFFT Execution Plan prepared: " << plan_prepare_result << std::endl;
    }

    ~CudaFastFourierTransformImpl() {

        std::cout << "Destroying CUFFT Context:" << std::endl;

        // Destroy CUFFT Execution Plan
        cufftResult destructor_result = cufftDestroy(cufft_execution_plan_);
        std::cout << "CUFFT Execution Plan destructor returned: " << destructor_result << std::endl;

        cudaFree(d_signal_);
        std::cout << "Signal memory cleared. " << std::endl;

        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);
    }

    vector<complex<double> > DirectTransform(const vector<complex<double> > &sequence)
    {

        if (signal_size_ != sequence.size()) {
            throw std::invalid_argument(
                    "CudaFastFourierTransform can transform only data of the same size as was specified on construction.");
        }

        // Allocate host memory for the signal
        cufftComplex* h_signal = CudaUtils::VectorToCufftComplex(sequence);

        // Copy signal host memory to device
        cudaMemcpy(d_signal_, h_signal, signal_memory_size_, cudaMemcpyHostToDevice);

        // NOTE: Transformed signal will be written instead of source signal to escape memory wasting
        cufftResult execution_result = cufftExecC2C(cufft_execution_plan_, d_signal_, d_signal_, CUFFT_FORWARD);

        std::cout << "CUFFT DIRECT C2C Execution result: " << execution_result << std::endl;

        // Copy Device memory (FFT calculation results - d_signal_output_) to Host memory (RAM)
        cufftComplex* h_result = (cufftComplex *) malloc(signal_memory_size_);
        cudaMemcpy(h_result, d_signal_, signal_memory_size_, cudaMemcpyDeviceToHost);

        //return vector<complex<double> >();
        return CudaUtils::CufftComplexToVector(h_result, signal_size_);
    }

    vector<complex<double> > InverseTransform(const vector<complex<double> > &sequence)
    {

        if (signal_size_ != sequence.size()) {
            throw std::invalid_argument(
                    "CudaFastFourierTransform can transform only data of the same size as was specified on construction.");
        }

        // Allocate host memory for the signal
        cufftComplex* h_signal = CudaUtils::VectorToCufftComplex(sequence);

        // Copy host signal memory to device
        cudaMemcpy(d_signal_, h_signal, signal_memory_size_, cudaMemcpyHostToDevice);

        // NOTE: Transformed signal will be written instead of source signal to escape memory wasting
        cufftResult execution_result = cufftExecC2C(cufft_execution_plan_, d_signal_, d_signal_, CUFFT_INVERSE);

        std::cout << "CUFFT INVERSE C2C Execution result: " << execution_result << std::endl;

        // Copy Device memory (FFT calculation results - d_signal_output_) to Host memory (RAM)
        cufftComplex* h_result = (cufftComplex *) malloc(signal_memory_size_);
        cudaMemcpy(h_result, d_signal_, signal_memory_size_, cudaMemcpyDeviceToHost);

        return CudaUtils::CufftComplexToVector(h_result, signal_size_);
    }

private:

    // static fields and initializers
    static boost::mutex cuda_mutex_;

    // instance fields and initializers
    size_t signal_size_;
    int executions_planned_;
    int signal_memory_size_;

    cufftComplex *d_signal_;

    cufftHandle cufft_execution_plan_;
};

boost::mutex CudaFastFourierTransform::CudaFastFourierTransformImpl::cuda_mutex_;

CudaFastFourierTransform::CudaFastFourierTransform(size_t sequence_size, int planned_executions) {
    cuda_fast_fourier_transform_impl_ =
            boost::make_shared<CudaFastFourierTransformImpl>(sequence_size, planned_executions);
}

CudaFastFourierTransform::~CudaFastFourierTransform() {
}

vector<complex<double> > CudaFastFourierTransform::DirectTransform(
        const vector<complex<double> > &sequence) const {
    return cuda_fast_fourier_transform_impl_->DirectTransform(sequence);
}

vector<complex<double> > CudaFastFourierTransform::InverseTransform(
        const vector<complex<double> > &sequence) const {
    return cuda_fast_fourier_transform_impl_->InverseTransform(sequence);
}

#endif //BUILD_WITH_CUDA

} // namespace dsp
} // namespace k52

#pragma clang diagnostic pop