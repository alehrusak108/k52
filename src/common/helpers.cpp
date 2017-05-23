#include <k52/common/helpers.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

using ::std::complex;
using ::std::vector;
using ::std::ofstream;
using ::std::cout;
using ::std::endl;

namespace k52
{
namespace common
{

vector< complex< double > > Helpers::Conjugate(
        const vector< complex< double > > &sequence)
{
    vector< complex < double> > conjugate(sequence.size());

    for (std::size_t i = 0; i < sequence.size(); ++i)
    {
        conjugate[i] = std::conj(sequence[i]);
    }

    return conjugate;
}

void Helpers::PrintVector(const std::vector<double>& values)
{
    cout.precision(15);

    cout << "[ ";
    for(size_t i = 0; i < values.size(); i++)
    {
        cout << values[i];
        if(i != values.size() - 1)
        {
            cout << "; ";
        }
    }
    cout << " ]" << endl;
}

void Helpers::PrintComplexVector(ofstream &output_file, vector<complex<double> > &vec)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        output_file << vec[i].real() << "\t" << vec[i].imag() << endl;
    }
}

void Helpers::PrintComplexVector(vector<complex<double> > &vec)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        cout << vec[i].real() << "\t" << vec[i].imag() << endl;
    }
}

vector<complex<double> > Helpers::GenerateComplexSignal(size_t signal_size)
{
    vector<complex<double> > input_signal(signal_size);
    int var = 0;
    for (size_t index = 0; index < signal_size; index++)
    {
        if (index % 8 == 0) {
            var = 0;
        } else {
            var++;
        }
        input_signal[index].real(var);
        input_signal[index].imag(0);
    }
    return input_signal;
}

}/* namespace common */
}/* namespace k52 */

