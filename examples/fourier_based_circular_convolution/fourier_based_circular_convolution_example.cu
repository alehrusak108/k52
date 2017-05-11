#include <k52/dsp/transform/fast_fourier_transform.h>

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <fstream>

#ifdef BUILD_WITH_CUDA

#include <k52/dsp/transform/cuda_fast_fourier_transform.h>

#endif

#define CUFFT_EXECUTIONS_PLANNED 1

using namespace std;

using namespace k52::dsp;

double CUFFTPerformanceTest(vector<complex<double> > &input_signal) {

    ofstream test_output;
    test_output.open("test_output.txt", ios::out | ios::app);
    test_output << endl << "[ CUFFT Performance TEST ] STARTED." << endl;

    clock_t planning_time = clock();

    CudaFastFourierTransform cufftTransformer(input_signal.size(), CUFFT_EXECUTIONS_PLANNED);

    test_output << "CUFFT Execution Plan prepared in: " << (float) (clock() - planning_time) / CLOCKS_PER_SEC << " seconds" << endl;

    clock_t execution_time = clock();

    // In this test, we don't care about transformation result
    vector<complex<double> > output = cufftTransformer.DirectTransform(input_signal);

    /*cout << endl << "CUFFT OUTPUT" << endl;
    for (int i = 0; i < input_signal.size(); i++) {
        cout << output[i].real() << "\t\t" << output[i].imag() << endl;
    }*/

    clock_t finish = clock() - execution_time;
    test_output << endl << "Time elapsed for CUFFT Transform Test: " << (double) finish / CLOCKS_PER_SEC << " seconds " << endl << endl;
    test_output << "[ CUFFT Performance TEST ] FINISHED." << endl << endl;
    test_output.close();
    return (double) finish / CLOCKS_PER_SEC;
}

double FFTWPerformanceTest(vector<complex<double> > &input_signal) {

    ofstream test_output;
    test_output.open("test_output.txt", ios::out | ios::app);
    test_output << "[ FFTW3 Performance TEST ] STARTED." << endl;

    clock_t planning_time = clock();

    FastFourierTransform fftw3Transformer(input_signal.size());

    test_output << endl << "FFTW3 Execution Plan prepared in: " << (float) (clock() - planning_time) / CLOCKS_PER_SEC << " seconds" << endl;

    clock_t execution_time = clock();

    // In this test, we don't care about transformation result
    vector<complex<double> > output = fftw3Transformer.DirectTransform(input_signal);

    /*cout << endl << "FFTW OUTPUT" << endl;
    for (int i = 0; i < input_signal.size(); i++) {
        cout << output[i].real() << "\t" << output[i].imag() << endl;
    }*/

    clock_t finish = clock() - execution_time;
    test_output << endl << "Time elapsed for FFTW3 Transform Test: " << (double) finish / CLOCKS_PER_SEC << " seconds " << endl << endl;
    test_output << "[ FFTW3 Performance TEST ] FINISHED." << endl << endl;
    test_output.close();
    return (double) finish / CLOCKS_PER_SEC;
}

vector<complex<double> > PrepareTestSignal(size_t signal_size) {
    vector<complex<double> > input_signal(signal_size);
    for (size_t index = 0; index < signal_size; index++) {
        //input_signal[index].real(index);
        //input_signal[index].imag(0);
        input_signal[index].real(-5 + rand() % 15);
        input_signal[index].imag(-5 + rand() % 15);
    }
    /*for (int i = 0; i < signal_size; i++) {
        cout << input_signal[i].real() << "\t" << input_signal[i].imag() << endl;
    }*/
    return input_signal;
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    ofstream test_output;
    test_output.open("test_output.txt", ios::out | ios::app);
    int signal_size = 33554432;
    int window_size = 1024;
    int windows_count = 33554432 / 1024;
    vector<complex<double> > input_signal = PrepareTestSignal(signal_size);
    double cufft_summary = 0.0;
    double fftw_summary = 0.0;
    for (int index = 0; index < windows_count - 1; index++) {
        vector<complex<double> >::const_iterator start = input_signal.begin() + index * window_size;
        vector<complex<double> >::const_iterator end = input_signal.begin() + (index + 1) * window_size;
        vector<complex<double> > window(start, end);
        //test_output << endl << "TEST #" << test_number << "\t" << "Signal Length is: " << signal_size << endl;
        cufft_summary += CUFFTPerformanceTest(window);
        //test_output << "---------------------------------------------" << endl << endl;
        fftw_summary += FFTWPerformanceTest(window);
        //test_output << "===============================================================================" << endl << endl;
        //signal_size *= 2;
    }
    test_output << endl << endl << "CUFFT SUMMARY TIME: " << cufft_summary << endl << endl;
    test_output << endl << endl << "FFTW3 SUMMARY TIME: " << fftw_summary << endl << endl;
    test_output.close();
}
