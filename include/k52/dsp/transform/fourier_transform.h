#ifndef K52_FOURIER_TRANSFORM_H
#define K52_FOURIER_TRANSFORM_H

#include <k52/dsp/transform/i_fourier_transform.h>

namespace k52
{
namespace dsp
{

class FourierTransform : public IFourierTransform
{
public:
    virtual std::vector< std::complex< double > > DirectTransform(const std::vector<std::complex<double> > &sequence) const;
};

} // namespace dsp
} // namespace k52

#endif //K52_FOURIER_TRANSFORM_H