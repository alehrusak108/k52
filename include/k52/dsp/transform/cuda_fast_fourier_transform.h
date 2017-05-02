#ifndef K52_CUDA_FAST_FOURIER_TRANSFORM_H
#define K52_CUDA_FAST_FOURIER_TRANSFORM_H

#include <k52/dsp/transform/i_fourier_transform.h>

#ifdef BUILD_WITH_CUDA

#include <cufft.h>
#include "../../../../../../../../usr/local/cuda/include/cufft.h"

#endif

using ::std::vector;
using ::std::complex;

namespace k52
{
namespace dsp
{

#ifdef BUILD_WITH_CUDA

class CudaFastFourierTransform : public IFourierTransform
{

public:

    // IMPORTANT: Be ware of "executions_planned" number as far as
    // CUFFT performs memory allocations for Planning
    // and it strictly depends on number of planned CUFFT executions
    CudaFastFourierTransform(size_t sequence_size, int executions_planned);

    ~CudaFastFourierTransform();

    virtual vector<complex<double> > DirectTransform(
            const vector<complex<double> > &sequence) const;

    virtual vector<complex<double> > InverseTransform(
            const vector<complex<double> > &sequence) const;

private:

    class CudaFastFourierTransformImpl;
    boost::shared_ptr<CudaFastFourierTransformImpl> cuda_fast_fourier_transform_impl_;
};

#endif

} // namespace dsp
} // namespace k52


#endif //K52_CUDA_FAST_FOURIER_TRANSFORM_H