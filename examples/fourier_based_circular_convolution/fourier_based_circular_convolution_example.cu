#include <k52/dsp/transform/fast_fourier_transform.h>

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>

#ifdef BUILD_WITH_CUDA

#include <k52/dsp/transform/cuda_fast_fourier_transform.h>

#endif

#define CUFFT_EXECUTIONS_PLANNED 1

using namespace std;

using namespace k52::dsp;

void CUFFTPerformanceTest(vector<complex<double> > &input_signal) {

    cout << endl << "[ CUFFT Performance TEST ] STARTED." << endl;

    clock_t planning_time = clock();

    CudaFastFourierTransform cufftTransformer(input_signal.size(), CUFFT_EXECUTIONS_PLANNED);

    cout << "CUFFT Execution Plan prepared in: " << (float) (clock() - planning_time) / CLOCKS_PER_SEC << " seconds" << endl;

    clock_t execution_time = clock();

    // In this test, we don't care about transformation result
    vector<complex<double> > output = cufftTransformer.DirectTransform(input_signal);

    /*cout << endl << "CUFFT OUTPUT" << endl;
    for (int i = 0; i < input_signal.size(); i++) {
        cout << output[i].real() << "\t\t" << output[i].imag() << endl;
    }*/

    cout << endl << "Time elapsed for CUFFT Transform Test: " << (float) (clock() - execution_time) / CLOCKS_PER_SEC << " seconds " << endl << endl;
    cout << "[ CUFFT Performance TEST ] FINISHED." << endl << endl;
}

void FFTWPerformanceTest(vector<complex<double> > &input_signal) {

    cout << "[ FFTW3 Performance TEST ] STARTED." << endl;

    clock_t planning_time = clock();

    FastFourierTransform fftw3Transformer(input_signal.size());

    cout << endl << "FFTW3 Execution Plan prepared in: " << (float) (clock() - planning_time) / CLOCKS_PER_SEC << " seconds" << endl;

    clock_t execution_time = clock();

    // In this test, we don't care about transformation result
    vector<complex<double> > output = fftw3Transformer.DirectTransform(input_signal);

    /*cout << endl << "FFTW OUTPUT" << endl;
    for (int i = 0; i < input_signal.size(); i++) {
        cout << output[i].real() << "\t" << output[i].imag() << endl;
    }*/

    cout << endl << "Time elapsed for FFTW3 Transform Test: " << (float) (clock() - execution_time) / CLOCKS_PER_SEC << " seconds " << endl << endl;
    cout << "[ FFTW3 Performance TEST ] FINISHED." << endl << endl;
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

    int signal_size = 16777216;
    //for (int test_number = 1; test_number <= 10; test_number++) {
        vector<complex<double> > input_signal = PrepareTestSignal(signal_size);
        //cout << endl << "TEST #" << test_number << endl;
        cout << endl << "Test signal size is: " << signal_size << endl;
        CUFFTPerformanceTest(input_signal);
        cout << "---------------------------------------------" << endl << endl;
        FFTWPerformanceTest(input_signal);
        cout << "===============================================================================" << endl << endl;
        signal_size *= 2;
    //}
}
