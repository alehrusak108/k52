#ifndef K52_CUDA_FOURIER_BASED_CIRCULAR_CONVOLUTION_H
#define K52_CUDA_FOURIER_BASED_CIRCULAR_CONVOLUTION_H

#include <k52/dsp/transform/cuda_fast_fourier_transform.h>
#include <k52/dsp/transform/i_circular_convolution.h>

#ifdef BUILD_WITH_CUDA

#include <cufft.h>
#include "../../../../../../../../usr/local/cuda/include/cufft.h"
#include "../../../../../../../../usr/local/cuda/include/cudalibxt.h"

#endif

using ::std::vector;
using ::std::complex;

namespace k52
{
namespace dsp
{

class CudaFourierBasedCircularConvolution : public ICircularConvolution
{

public:

    CudaFourierBasedCircularConvolution(size_t signal_size, size_t page_size);

    vector<complex<double> > EvaluateConvolution(
            const vector<complex<double> > &first_signal,
            const vector<complex<double> > &second_signal
    );

private:

    // Here used implementation instead of an interface,
    // because CudaFastFourierTransform provides
    // two more methods to perform FFT, that are very useful for convolution
    boost::shared_ptr<CudaFastFourierTransform> cufft_transformer_;

    size_t signal_memory_size_;
    size_t page_size_;

    cufftComplex *d_first_signal_;
    cufftComplex *d_second_signal_;
};

} // namespace dsp
} // namespace k52

#endif //K52_CUDA_FOURIER_BASED_CIRCULAR_CONVOLUTION_H
