#include <k52/dsp/akima_wavelet_function.h>
#include <stdexcept>

#include <k52/dsp/spline.h>
#include <k52/dsp/akima_spline.h>

#ifdef BUILD_WITH_ALGLIB
#include <interpolation.h>

namespace k52
{
namespace dsp
{
class AkimaSpline : public k52::dsp::Spline
{
public:
    typedef boost::shared_ptr<AkimaSpline> shared_ptr;

    static AkimaSpline::shared_ptr CreateAkimaSpline()
    {
         return AkimaSpline::shared_ptr(new AkimaSpline);
    }

    virtual void Initialize(const std::vector<double> &points, const std::vector<double> &values)
    try
    {
        alglib::real_1d_array spline_points_x;
        spline_points_x.setlength(points.size());
        std::copy(points.begin(), points.end(), spline_points_x.getcontent());

        alglib::real_1d_array spline_points_y;
        spline_points_y.setlength(values.size());
        std::copy(values.begin(), values.end(), spline_points_y.getcontent());

        alglib::spline1dbuildakima(spline_points_x, spline_points_y, interpolant_);
    }
    catch(const alglib::ap_error& error)
    {
        std::cerr << "alglib error: " << error.msg << std::endl;
    }
    catch(...)
    {
        std::cerr << __FUNCTION__ << ", an error :" << std::endl;
        throw;
    }

    virtual double Value(double x)
    {
        return alglib::spline1dcalc(interpolant_, x);
    }

protected:
    AkimaSpline() {}

private:
    alglib::spline1dinterpolant interpolant_;
};
} // namespace dsp
} // namespace k52

#else
#   include <k52/dsp/akima_spline.h>
#endif // BUILD_WITH_ALGLIB

namespace k52
{
namespace dsp
{

AkimaWavelet::shared_ptr AkimaWavelet::CreateAkimaWavelet(const std::vector<double>& real,
                                                          const std::vector<double>& imaj)
{
    AkimaWavelet::shared_ptr akima_wavelet(new AkimaWavelet);
    akima_wavelet->spline(k52::dsp::AkimaSpline::Create());
    akima_wavelet->Init(real, imaj);
    return akima_wavelet;
}

AkimaWavelet::AkimaWavelet()
{
}

} // namespace dsp
} // namespace k52
