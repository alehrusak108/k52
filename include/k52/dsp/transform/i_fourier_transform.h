#ifndef K52_IFOURIER_TRANSFORM_H
#define K52_IFOURIER_TRANSFORM_H

#include <boost/smart_ptr/shared_ptr.hpp>

#include <complex>
#include <vector>

#include <k52/common/disallow_copy_and_assign.h>

namespace k52
{
namespace dsp
{

class IFourierTransform
{
public:
    typedef boost::shared_ptr<IFourierTransform> shared_ptr;

    ///Default constructor should be explicitly defined if DISALLOW_COPY_AND_ASSIGN used
    IFourierTransform() {}

    virtual ~IFourierTransform() {};

    virtual std::vector< std::complex< double > > DirectTransform(
            const std::vector<std::complex<double> > &sequence) const = 0;

    virtual std::vector< std::complex< double > > InverseTransform(
            const std::vector<std::complex<double> > &sequence) const = 0;

private:
    DISALLOW_COPY_AND_ASSIGN(IFourierTransform);

};

} // namespace dsp
} // namespace k52

#endif //K52_IFOURIER_TRANSFORM_H
