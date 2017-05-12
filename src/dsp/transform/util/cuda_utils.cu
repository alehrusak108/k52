#include <k52/dsp/transform/util/cuda_utils.h>

#ifdef BUILD_WITH_CUDA

#include <cufft.h>
#include <iostream>
#include <cstring>

#endif

using ::std::vector;
using ::std::complex;
using ::std::string;

#ifdef BUILD_WITH_CUDA

cufftComplex *CudaUtils::VectorToCufftComplex(const vector<complex<double> > &sequence)
{
    int signal_size = sequence.size();
    cufftComplex* cufft_sequence = (cufftComplex *) malloc(sizeof(cufftComplex) * signal_size);
    for (size_t n = 0; n < signal_size; ++n)
    {
        cufft_sequence[n].x = sequence[n].real();
        cufft_sequence[n].y = sequence[n].imag();
    }
    return cufft_sequence;
}

vector<complex<double> > CudaUtils::CufftComplexToVector(cufftComplex *cufft_sequence, int sequence_size)
{
    vector<complex<double> > result_vector(sequence_size);
    for (size_t n = 0; n < sequence_size; ++n)
    {
        result_vector[n].real(cufft_sequence[n].x);
        result_vector[n].imag(cufft_sequence[n].y);
    }
    return result_vector;
}

void CudaUtils::checkCufftErrors(cufftResult result, const string &failure_message)
{
    if (result != CUFFT_SUCCESS)
    {
        std::cout << std::endl << "FATAL ERROR! " << failure_message << " returned " << (int) result << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CudaUtils::checkErrors(cudaError error, const string &failure_message) {
    if (error != CUFFT_SUCCESS)
    {
        std::cout << std::endl << "FATAL ERROR! " << failure_message << " returned " << (int) error << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif