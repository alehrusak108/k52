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

#define PAGE_SIZE 262144

using namespace std;

using namespace k52::dsp;

using namespace k52::common;

void CudaConvolutionTest(vector<complex<double> > first_signal, vector<complex<double> > second_signal)
{
    ofstream test_output;
    test_output.open("convolution_test.txt", ios::out | ios::app);

    size_t signal_size = first_signal.size();

    CudaFourierBasedCircularConvolution cufft_convolutor(signal_size, PAGE_SIZE);

    clock_t cufft_time = clock();
    vector<complex<double> > output = cufft_convolutor.EvaluateConvolution(first_signal, second_signal);
    //Helpers::PrintComplexVector(output);
    test_output << endl << "CUFFT CONVOLUTION TIME: " << (double) (clock() - cufft_time) / CLOCKS_PER_SEC << " seconds" << endl << endl;
}

void FFTWConvolutionTest(vector<complex<double> > first_signal, vector<complex<double> > second_signal)
{
    ofstream test_output;
    test_output.open("convolution_test.txt", ios::out | ios::app);

    size_t signal_size = first_signal.size();

    FourierBasedCircularConvolution *fftw_convolutor =
            new FourierBasedCircularConvolution(
                    IFourierTransform::shared_ptr(new FastFourierTransform(PAGE_SIZE))
            );

    int total_pages = signal_size / PAGE_SIZE;

    clock_t execution_time = clock();
    for (unsigned int page_number = 0; page_number < total_pages; page_number++)
    {
        size_t start_index = PAGE_SIZE * page_number;
        size_t end_index = start_index + PAGE_SIZE;
        vector<complex<double> >::const_iterator page_start = first_signal.begin() + start_index;
        vector<complex<double> >::const_iterator page_end = first_signal.begin() + end_index;
        vector<complex<double> > first_signal_page(page_start, page_end);

        page_start = second_signal.begin() + start_index;
        page_end = second_signal.begin() + end_index;
        vector<complex<double> > second_signal_page(page_start, page_end);

        vector<complex<double> > output = fftw_convolutor->EvaluateConvolution(first_signal_page, second_signal_page);

        //Helpers::PrintComplexVector(output);
    }

    test_output << endl << "FFTW CONVOLUTION TIME: " << (double) (clock() - execution_time) / CLOCKS_PER_SEC << " seconds" << endl << endl;
}

int main(int argc, char* argv[])
{
    srand(time(NULL));

    ofstream test_output;
    test_output.open("convolution_test.txt", ios::out | ios::app);
    test_output << endl << "CONVOLUTION PERFORMANCE TEST (FFTW vs CUDA)" << endl << endl;
    size_t signal_size = 2097152;
    for (int test_index = 1; test_index <= 7; test_index++) {
        test_output << endl << "TEST #" << test_index << "\t" << "Signal Length is: " << signal_size << endl;
        vector<complex<double> > first_signal = Helpers::GenerateComplexSignal(signal_size);
        vector<complex<double> > second_signal = Helpers::GenerateComplexSignal(signal_size);
        CudaConvolutionTest(first_signal, second_signal);
        test_output << "-----------------------------------------------------------------------" << endl << endl;
        FFTWConvolutionTest(first_signal, second_signal);
        test_output << "===============================================================================" << endl << endl;
        signal_size *= 2;
    }
    test_output.close();
}
