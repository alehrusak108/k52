#ifndef K52_CUDA_FOURIER_BASED_CIRCULAR_CONVOLUTION_H
#define K52_CUDA_FOURIER_BASED_CIRCULAR_CONVOLUTION_H

#include <k52/dsp/transform/i_fourier_transform.h>
#include <k52/dsp/transform/cuda_fast_fourier_transform.h>
#include <k52/dsp/transform/fast_fourier_transform.h>
#include <k52/dsp/transform/i_circular_convolution.h>
#include <k52/dsp/transform/inverse_fourier_transform.h>

namespace k52
{
namespace dsp
{

class CudaFourierBasedCircularConvolution : public ICircularConvolution
{

public:

    CudaFourierBasedCircularConvolution(size_t sequence_size, int executions_planned);

    std::vector<std::complex<double> > EvaluateConvolution(
            const std::vector<std::complex<double> > &first_signal,
            const std::vector<std::complex<double> > &second_signal
    ) const;

private:

    // Here used implementation instead of an interface,
    // because CudaFastFourierTransform provides
    // two more methods to perform FFT, that are very useful for convolution
    boost::shared_ptr<CudaFastFourierTransform> cufft_transformer_;
};

} // namespace dsp
} // namespace k52

#endif //K52_CUDA_FOURIER_BASED_CIRCULAR_CONVOLUTION_H
