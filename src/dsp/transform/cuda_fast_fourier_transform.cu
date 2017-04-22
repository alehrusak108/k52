#include <k52/dsp/transform/cuda_fast_fourier_transform.h>
#include <stdexcept>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "TemplateArgumentsIssues"

#ifdef BUILD_WITH_CUFFT

#include <cufft.h>
#include <boost/thread/mutex.hpp>
#include <cuda_runtime_api.h>
#include <boost/smart_ptr/make_shared.hpp>

#endif

using ::std::vector;
using ::std::complex;
using ::std::invalid_argument;
using ::std::runtime_error;

namespace k52 {
namespace dsp {

#ifdef BUILD_WITH_CUFFT

// Using pImpl approach to hide CUFFT for outside use
// NOTE: Prefix "d_" means that variable is allocated in CUDA Device Memory
//       Prefix "h_" means that variable is allocated in RAM (Host)
class CudaFastFourierTransform::CudaFastFourierTransformImpl {
public:
    CudaFastFourierTransformImpl(size_t sequence_size)
            : sequence_size_(sequence_size) {
        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);

        if (sequence_size <= 0) {
            throw std::invalid_argument("sequence_size <= 0");
        }

        sequence_memory_size_ = sizeof(cufftComplex) * sequence_size_;

        // Allocate device memory for input (source) signal and output (after FFT) signals
        cudaMalloc((void**) &d_signal_input_, sequence_memory_size_);
        cudaMalloc((void**) &d_signal_output_, sequence_memory_size_);

        cufftPlan1d(&cufft_execution_plan_, sequence_size_, CUFFT_C2C, 1);
    }

    ~CudaFastFourierTransformImpl() {
        boost::mutex::scoped_lock scoped_lock(cuda_mutex_);

        cufftDestroy(cufft_execution_plan_);
        cudaFree(d_signal_input_);
        cudaFree(d_signal_output_);
    }

    std::vector<std::complex<double> > Transform(
            const std::vector<std::complex<double> > &sequence) {
        if (sequence_size_ != sequence.size()) {
            throw std::invalid_argument(
                    "CudaFastFourierTransform can transform only data of the same size as was specified on construction.");
        }

        // Transfer data from vector to cufftComplex* array
        cufftComplex* h_signal_input = VectorToCufftComplex(sequence);
        // And copy retrieved from vector data to a Device
        cudaMemcpy(d_signal_input_, h_signal_input, sequence_memory_size_, cudaMemcpyHostToDevice);

        //Actual computations
        cufftExecC2C(cufft_execution_plan_, d_signal_input_, d_signal_output_, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        // Copy Device memory (FFT calculation results - d_signal_output_) to Host
        cufftComplex* h_signal_output = (cufftComplex *) malloc(sequence_memory_size_);
        cudaMemcpy(h_signal_output, d_signal_output_, sequence_memory_size_, cudaMemcpyDeviceToHost);

        // Then, transfer data from cufftComplex* array into sdt::vector
        return CufftComplexToVector(h_signal_output);
    }

private:
    size_t sequence_size_;

    int sequence_memory_size_;

    cufftComplex *d_signal_input_;
    cufftComplex *d_signal_output_;
    cufftHandle cufft_execution_plan_;

    static boost::mutex cuda_mutex_;

    cufftComplex* VectorToCufftComplex(const std::vector<std::complex<double> > &);

    std::vector<std::complex<double> > CufftComplexToVector(cufftComplex *);
};

cufftComplex* CudaFastFourierTransform::CudaFastFourierTransformImpl::VectorToCufftComplex(
        const std::vector<std::complex<double> > &sequence)  {
    // Allocate host memory for the signal
    cufftComplex* h_signal_input = (cufftComplex *) malloc(sequence_memory_size_);
    // Transfer data from sequence sdt::vector to h_signal_input of a cufftComplex* type
    for (size_t n = 0; n < sequence_size_; ++n) {
        h_signal_input[n].x = sequence[n].real();
        h_signal_input[n].y = sequence[n].imag();
    }
    return h_signal_input;
}

std::vector<std::complex<double> >
CudaFastFourierTransform::CudaFastFourierTransformImpl::CufftComplexToVector(cufftComplex *sequence) {
    vector<complex<double> > transformation_result(sequence_size_);

    for (size_t n = 0; n < sequence_size_; ++n) {
        transformation_result[n].real(sequence[n].x);
        transformation_result[n].imag(sequence[n].y);
    }

    return transformation_result;
}

boost::mutex CudaFastFourierTransform::CudaFastFourierTransformImpl::cuda_mutex_;

CudaFastFourierTransform::CudaFastFourierTransform(size_t sequence_size) {
    cuda_fast_fourier_transform_impl_ = boost::make_shared<CudaFastFourierTransformImpl>(sequence_size);
}

CudaFastFourierTransform::~CudaFastFourierTransform() {
}

vector<complex<double> > CudaFastFourierTransform::Transform(
        const vector<complex<double> > &sequence) const {
    return cuda_fast_fourier_transform_impl_->Transform(sequence);
}

#endif //BUILD_WITH_CUFFT

} // namespace dsp
} // namespace k52
#pragma clang diagnostic pop