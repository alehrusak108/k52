#include <k52/dsp/transform/fast_fourier_transform.h>
#include <k52/dsp/transform/fourier_based_circular_convolution.h>

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

double CUFFTPerformanceTest(vector<complex<double> > &input_signal) {

    ofstream test_output;
    test_output.open("test_output.txt", ios::out | ios::app);
    test_output << endl << "[ CUFFT Performance TEST ] STARTED." << endl;

    clock_t planning_time = clock();

    CudaFastFourierTransform cufftTransformer(input_signal.size(), CUFFT_EXECUTIONS_PLANNED);

    test_output << "CUFFT Execution Plan prepared in: " << (float) (clock() - planning_time) / CLOCKS_PER_SEC << " seconds" << endl;

    clock_t execution_time = clock();

    vector<complex<double> > output = cufftTransformer.DirectTransform(input_signal);

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

    vector<complex<double> > output = fftw3Transformer.DirectTransform(input_signal);

    clock_t finish = clock() - execution_time;
    test_output << endl << "Time elapsed for FFTW3 Transform Test: " << (double) finish / CLOCKS_PER_SEC << " seconds " << endl << endl;
    test_output << "[ FFTW3 Performance TEST ] FINISHED." << endl << endl;
    test_output.close();
    return (double) finish / CLOCKS_PER_SEC;
}

vector<complex<double> > PrepareTestSignal(size_t signal_size) {
    vector<complex<double> > input_signal(signal_size);
    for (size_t index = 0; index < signal_size; index++) {
        input_signal[index].real(index);
        input_signal[index].imag(0);
        /*input_signal[index].real(-5 + rand() % 15);
        input_signal[index].imag(-5 + rand() % 15);*/
    }
    printComplexVector(input_signal);
    return input_signal;
}

void FastFourierTransformTest() {
    ofstream test_output;
    test_output.open("fast_fourier_transform_test.txt", ios::out | ios::app);
    int signal_size = 262144;
    for (int test_index = 1; test_index <= 8; test_index++) {
        vector<complex<double> > input_signal = PrepareTestSignal(signal_size);
        test_output << endl << "TEST #" << test_number << "\t" << "Signal Length is: " << signal_size << endl;
        CUFFTPerformanceTest(input_signal);
        test_output << "-----------------------------------------------------------------------" << endl << endl;
        FFTWPerformanceTest(input_signal);
        test_output << "===============================================================================" << endl << endl;
        signal_size *= 2;
    }
    test_output.close();
}

void CircularConvolutionTest() {
    ofstream test_output;
    test_output.open("convolution_test.txt", ios::out | ios::app);

    int signal_size = 32;
    test_output << endl << "Signal Length is: " << signal_size << endl;
    vector<complex<double> > first_signal = PrepareTestSignal(signal_size);
    vector<complex<double> > second_signal = PrepareTestSignal(signal_size);

    CudaFourierBasedCircularConvolution cufftConvolutor(input_signal.size(), CUFFT_EXECUTIONS_PLANNED);
    vector<complex<double> > cufft_result = cufftConvolutor.EvaluateConvolution(first_signal, second_signal);
    printComplexVector(cufft_result);

    test_output << endl << "-----------------------------------------------------------------------" << endl << endl;
    test_output.close();
}

void printComplexVector(ofstream &output_file, vector<complex<double> > &vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        output_file << vec[i].real() << "\t" << vec[i].imag() << endl;
    }
}

void printComplexVector(vector<complex<double> > &vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        cout << vec[i].real() << "\t" << vec[i].imag() << endl;
    }
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    FastFourierTransformTest();
    CircularConvolutionTest();
}
