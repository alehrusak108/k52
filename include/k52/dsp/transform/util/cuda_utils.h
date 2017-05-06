#ifndef K52_CUDAUTILS_H
#define K52_CUDAUTILS_H

#include <cstdlib>
#include <vector>
#include <complex>

#ifdef BUILD_WITH_CUDA

#include <cufft.h>
#include <cuda_runtime_api.h>
#include "../../../../../../../../../usr/local/cuda/include/cufft.h"

#endif

using ::std::vector;
using ::std::complex;

#ifdef BUILD_WITH_CUDA

class CudaUtils
{

public:

    static cufftComplex* VectorToCufftComplex(const vector<complex<double> > &sequence);

    static vector<complex<double> > CufftComplexToVector(cufftComplex *cufft_complex_sequence, int sequence_size);
};

#endif //BUILD_WITH_CUDA
#endif //K52_CUDAUTILS_H