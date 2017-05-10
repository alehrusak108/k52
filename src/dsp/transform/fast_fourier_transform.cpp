#include <k52/dsp/transform/fast_fourier_transform.h>
#include <stdexcept>

#ifdef BUILD_WITH_FFTW3

#include <fftw3.h>
#include <boost/thread/mutex.hpp>
#include <fstream>

#endif

using ::std::vector;
using ::std::complex;
using ::std::invalid_argument;
using ::std::runtime_error;

namespace k52
{
namespace dsp
{

#ifdef BUILD_WITH_FFTW3

//Using pImpl approach to hide fftw3 for outside use
class FastFourierTransform::FastFourierTransformImpl
{
public:
    FastFourierTransformImpl(size_t sequence_size)
        : signal_size_(sequence_size)
    {
        boost::mutex::scoped_lock scoped_lock(fftw_mutex_);

        if (sequence_size <= 0)
        {
            throw std::invalid_argument("sequence_size <= 0");
        }

        in_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * signal_size_);
        out_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * signal_size_);
        plan_ = fftw_plan_dft_1d(signal_size_, in_, out_, FFTW_FORWARD, FFTW_PATIENT);
    }

    ~FastFourierTransformImpl()
    {
        boost::mutex::scoped_lock scoped_lock(fftw_mutex_);

        fftw_destroy_plan(plan_);
        fftw_free(in_);
        fftw_free(out_);
    }

    vector< complex< double > > DirectTransform(
            const vector< complex< double > > &sequence)
    {
        if (signal_size_ != sequence.size())
        {
            throw std::invalid_argument(
                    "FastFourierTransform can transform only data of the same size as was specified on construction.");
        }

        for (size_t n = 0; n < signal_size_; ++n)
        {
            in_[n][0] = sequence[n].real();
            in_[n][1] = sequence[n].imag();
        }

        // Actual computations
        std::ofstream test_output;
        test_output.open("test_output.txt");
        clock_t execution_time = clock();
        fftw_execute(plan_);
        test_output << std::endl << "FFTW3 Transformation finished in: " << (float) (clock() - execution_time) / CLOCKS_PER_SEC << " seconds " << std::endl;
        test_output.close();

        vector< complex< double > > result(signal_size_);
        for (size_t n = 0; n < signal_size_; ++n)
        {
            result[n].real( out_[n][0] );
            result[n].imag( out_[n][1] );
        }
        return result;
    }

    vector< complex< double > > InverseTransform(
            const vector< complex< double > > &sequence)
    {
        // TODO: Stubbed. Implementation needed.
        return vector< complex< double > >();
    }

private:
    size_t signal_size_;

    fftw_complex* in_;
    fftw_complex* out_;
    fftw_plan plan_;

    static boost::mutex fftw_mutex_;
};

boost::mutex FastFourierTransform::FastFourierTransformImpl::fftw_mutex_;

FastFourierTransform::FastFourierTransform(size_t sequence_size)
{
    fast_fourier_transform_impl_ = new FastFourierTransformImpl(sequence_size);
}

FastFourierTransform::~FastFourierTransform()
{
    delete fast_fourier_transform_impl_;
}

vector< complex< double > > FastFourierTransform::DirectTransform(
        const vector< complex< double > > &sequence) const
{
    return fast_fourier_transform_impl_->DirectTransform(sequence);
}


vector< complex<double> > FastFourierTransform::InverseTransform(
        const vector< complex< double > > &sequence) const
{
    return fast_fourier_transform_impl_->InverseTransform(sequence);
}

#else

class FastFourierTransform::FastFourierTransformImpl{};

FastFourierTransform::FastFourierTransform(size_t sequence_size){}

FastFourierTransform::~FastFourierTransform(){}

vector< complex< double > > FastFourierTransform::Transform(
        const vector< complex< double > > &sequence) const
{
    throw runtime_error("The k52 library must be compiled with fftw3 to use FastFourierTransform class");
}

#endif //BUILD_WITH_FFTW3

} // namespace dsp
} // namespace k52

