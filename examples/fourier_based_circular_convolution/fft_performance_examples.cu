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
#include <k52/dsp/transform/util/cuda_utils.h>

#endif

#define CUFFT_EXECUTIONS_PLANNED 1

#define PAGE_SIZE 8

using namespace std;

using namespace k52::dsp;
using namespace k52::common;

void CUFFTPerformanceTest(vector<complex<double> > input_signal)
{

    ofstream test_output;
    test_output.open("fast_fourier_transform_test.txt", ios::out | ios::app);
    test_output << endl << "[ CUFFT Performance TEST ] STARTED." << endl;

    clock_t planning_time = clock();
    CudaFastFourierTransform cufftTransformer(input_signal.size(), PAGE_SIZE);
    cufftTransformer.SetDeviceSignal(CudaUtils::VectorToCufftComplexAlloc(input_signal));
    test_output << "CUFFT Data Transfer and Execution Plan prepared in: " << (float) (clock() - planning_time) / CLOCKS_PER_SEC << " seconds" << endl;

    clock_t execution_time = clock();

    cufftTransformer.DirectTransform();
    vector<complex<double> > output = cufftTransformer.GetTransformResult();

    //Helpers::PrintComplexVector(output);

    clock_t finish = clock() - execution_time;
    test_output << endl << "Time elapsed for CUFFT Transform Test: " << (double) (clock() - execution_time) / CLOCKS_PER_SEC << " seconds " << endl << endl;
    test_output << "[ CUFFT Performance TEST ] FINISHED." << endl << endl;
    test_output.close();
}

void FFTWPerformanceTest(vector<complex<double> > input_signal)
{

    ofstream test_output;
    test_output.open("fast_fourier_transform_test.txt", ios::out | ios::app);
    test_output << "[ FFTW3 Performance TEST ] STARTED." << endl;

    int total_pages = input_signal.size() / PAGE_SIZE;

    clock_t planning_time = clock();
    FastFourierTransform fftwTransformer(PAGE_SIZE);
    test_output << endl << "FFTW3 Execution Plan prepared in: " << (float) (clock() - planning_time) / CLOCKS_PER_SEC << " seconds" << endl;

    clock_t execution_time = clock();
    for (size_t page_number = 0; page_number < total_pages; page_number++)
    {
        size_t start_index = PAGE_SIZE * page_number;
        size_t end_index = start_index + PAGE_SIZE;
        vector<complex<double> >::const_iterator page_start = input_signal.begin() + start_index;
        vector<complex<double> >::const_iterator page_end = input_signal.begin() + end_index;
        vector<complex<double> > signal_page(page_start, page_end);

        vector<complex<double> > output = fftwTransformer.DirectTransform(signal_page);
    }

    test_output << endl << endl << "Time elapsed for FFTW3 Transform Test: " << (double) (clock() - execution_time) / CLOCKS_PER_SEC << " seconds " << endl << endl;

    test_output << "[ FFTW3 Performance TEST ] FINISHED." << endl << endl;
    test_output.close();
}

int main(int argc, char* argv[])
{
    srand(time(NULL));

    ofstream test_output;
    test_output.open("fast_fourier_transform_test.txt", ios::out | ios::app);
    test_output << endl << "FFT PERFORMANCE TEST (FFTW vs CUDA)" << endl << endl;
    int signal_size = 64;
    //for (int test_index = 1; test_index <= 7; test_index++) {
        vector<complex<double> > input_signal = Helpers::GenerateComplexSignal(signal_size);
        //test_output << endl << "TEST #" << test_index << "\t" << "Signal Length is: " << signal_size << endl;
        CUFFTPerformanceTest(input_signal);
        test_output << "-----------------------------------------------------------------------" << endl << endl;
        //FFTWPerformanceTest(input_signal);
        test_output << "===============================================================================" << endl << endl;
        signal_size *= 2;
    //}
    test_output.close();
}