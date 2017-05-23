#include <k52/dsp/transform/i_fourier_transform.h>
#include <k52/dsp/transform/fast_fourier_transform.h>
#include <k52/dsp/transform/fourier_based_circular_convolution.h>
#include <k52/dsp/transform/circular_convolution.h>

#include <k52/common/helpers.h>

#include <boost/smart_ptr/shared_ptr.hpp>

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <fstream>

#ifdef BUILD_WITH_CUDA

#include <k52/dsp/transform/cuda_fast_fourier_transform.h>
#include <k52/dsp/transform/cuda_fourier_based_circular_convolution.h>

#endif

#define CUFFT_EXECUTIONS_PLANNED 1

using namespace std;

using namespace k52::dsp;

/*void CircularConvolutionTest() {

    cout << endl << "[ CONVOLUTION TEST STARTED ]" << endl;

    size_t signal_size = 33554432;
    cout << endl << "Signal Length is: " << signal_size << endl;
    vector<complex<double> > first_signal = Helpers::GenerateComplexSignal(signal_size);
    vector<complex<double> > second_signal = Helpers::GenerateComplexSignal(signal_size);

    CudaFourierBasedCircularConvolution cufft_convolutor(signal_size, CUFFT_EXECUTIONS_PLANNED);

    clock_t cufft_time = clock();
    vector<complex<double> > cufft_result = cufft_convolutor.EvaluateConvolution(first_signal, second_signal);
    cout << endl << "CUFFT CONVOLUTION TIME: " << (double) (clock() - cufft_time) / CLOCKS_PER_SEC << " seconds" << endl << endl;

    FourierBasedCircularConvolution *fftw_convolutor =
            new FourierBasedCircularConvolution(
                    IFourierTransform::shared_ptr(new FastFourierTransform(signal_size))
            );

    clock_t fftw_time = clock();
    vector<complex<double> > fftw_result = fftw_convolutor->EvaluateConvolution(first_signal, second_signal);
    cout << endl << "FFTW CONVOLUTION TIME: " << (double) (clock() - fftw_time) / CLOCKS_PER_SEC << " seconds" << endl << endl;

    cout << endl << "[ CONVOLUTION TEST FINISHED ]" << endl;
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    //CircularConvolutionTest();
    vector<complex<double> > v;
    FFTWPerformanceTest(v);
}*/
