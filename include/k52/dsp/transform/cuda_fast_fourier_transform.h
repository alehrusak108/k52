#pragma clang diagnostic push
#pragma ide diagnostic ignored "TemplateArgumentsIssues"
#ifndef K52_CUDA_FAST_FOURIER_TRANSFORM_H
#define K52_CUDA_FAST_FOURIER_TRANSFORM_H

#include <k52/dsp/transform/i_fourier_transform.h>

using ::std::vector;
using ::std::complex;

namespace k52
{
namespace dsp
{

class CudaFastFourierTransform : public IFourierTransform
{

public:

    // IMPORTANT: Be ware of "executions_planned" number as far as
    // CUFFT performs memory allocations for Planning
    // and it strictly depends on number of planned CUFFT executions
    CudaFastFourierTransform(size_t sequence_size, int executions_planned);
    ~CudaFastFourierTransform();

    // Direct FFT - inherited from IFourierTransform
    virtual vector<complex<double> > DirectTransform(
            const vector<complex<double> > &sequence) const;

    // Inverse FFT - NOT inherited.
    // TODO: Probably a bad design
    // TODO: Suppose, that IFourierTransform should have both DirectTransform and InverseTransform methods.
    virtual vector<complex<double> > InverseTransform(
            const vector<complex<double> > &sequence) const;

private:
    class CudaFastFourierTransformImpl;
    boost::shared_ptr<CudaFastFourierTransformImpl> cuda_fast_fourier_transform_impl_;
};

} // namespace dsp
} // namespace k52


#endif //K52_CUDA_FAST_FOURIER_TRANSFORM_H

#pragma clang diagnostic pop