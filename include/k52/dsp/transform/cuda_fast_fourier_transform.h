#ifndef K52_CUDA_FAST_FOURIER_TRANSFORM_H
#define K52_CUDA_FAST_FOURIER_TRANSFORM_H

#include <k52/dsp/transform/i_fourier_transform.h>

namespace k52
{
namespace dsp
{

class CudaFastFourierTransform : public IFourierTransform
{

public:
    CudaFastFourierTransform(size_t sequence_size);
    ~CudaFastFourierTransform();

    virtual std::vector< std::complex< double > > Transform(
            const std::vector< std::complex< double > > &sequence) const;

private:
    class CudaFastFourierTransformImpl;
    boost::shared_ptr<CudaFastFourierTransformImpl> cuda_fast_fourier_transform_impl_;
};

} // namespace dsp
} // namespace k52


#endif //K52_CUDA_FAST_FOURIER_TRANSFORM_H
