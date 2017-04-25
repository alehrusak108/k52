#include <k52/dsp/transform/fast_fourier_transform.h>

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>

#ifdef BUILD_WITH_CUDA

#include <k52/dsp/transform/cuda_fast_fourier_transform.h>

#endif

#define SIGNAL_SIZE 10000000
#define CUFFT_EXECUTIONS_PLANNED 1

using namespace std;

using namespace k52::dsp;

int main(int argc, char* argv[])
{
    srand(time(NULL));

    vector<complex<double> > input_signal(SIGNAL_SIZE);
    for (size_t index = 0; index < SIGNAL_SIZE; index++) {
        input_signal[index].real(1 + rand() % 10);
        input_signal[index].imag(-5 + rand() % 11);
    }

    cout << endl << "Test signal size is: " << SIGNAL_SIZE << endl;
    
    clock_t cufft_start_time = clock();
    cout << endl << "[ CUFFT Performance TEST ] STARTED." << endl << endl;

    CudaFastFourierTransform cufftTransformer(SIGNAL_SIZE, CUFFT_EXECUTIONS_PLANNED);

    // In this test, we don't care about transformation result
    cufftTransformer.DirectTransform(input_signal);

    /*for (int i = 0; i < SIGNAL_SIZE; i++) {
        cout << output[i].real() << "\t" << output[i].imag() << endl;
    }*/

    cout << endl << "Time elapsed for CUFFT Transform Test: " << (float) (clock() - cufft_start_time) / CLOCKS_PER_SEC << " seconds " << endl << endl;
    cout << "[ CUFFT Performance TEST ] FINISHED." << endl << endl;

    cout << "===============================================================================" << endl << endl;

    cout << "[ FFTW3 Performance TEST ] STARTED." << endl;
    clock_t fftw3_start_time = clock();

    FastFourierTransform fftw3Transformer(SIGNAL_SIZE);

    cout << endl << "Time elapsed to prepare FFTW3 Execution Plan: " << (float) (clock() - fftw3_start_time) / CLOCKS_PER_SEC << " seconds " << endl;

    // In this test, we don't care about transformation result
    fftw3Transformer.DirectTransform(input_signal);

    cout << endl << "Time elapsed for FFTW3 Transform Test: " << (float) (clock() - fftw3_start_time) / CLOCKS_PER_SEC << " seconds " << endl << endl;
    cout << "[ FFTW3 Performance TEST ] FINISHED." << endl << endl;
}
