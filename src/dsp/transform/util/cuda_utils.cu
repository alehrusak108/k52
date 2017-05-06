#include <k52/dsp/transform/util/cuda_utils.h>

#ifdef BUILD_WITH_CUDA

#include <cufft.h>

#endif

using ::std::vector;
using ::std::complex;

#ifdef BUILD_WITH_CUDA

cufftComplex *CudaUtils::VectorToCufftComplex(const vector<complex<double> > &sequence)
{
    int signal_size = sequence.size();
    cufftComplex* cufft_sequence = (cufftComplex *) malloc(sizeof(cufftComplex) * signal_size);
    for (size_t n = 0; n < signal_size; ++n) {
        cufft_sequence[n].x = sequence[n].real();
        cufft_sequence[n].y = sequence[n].imag();
    }
    return cufft_sequence;
}

vector<complex<double> > CudaUtils::CufftComplexToVector(cufftComplex *cufft_sequence, int sequence_size)
{
    vector<complex<double> > result_vector(sequence_size);
    for (size_t n = 0; n < sequence_size; ++n) {
        result_vector[n].real(cufft_sequence[n].x);
        result_vector[n].imag(cufft_sequence[n].y);
    }
    return result_vector;
}

#endif