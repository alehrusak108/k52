#ifndef K52_HELPERS_H
#define K52_HELPERS_H

#include <complex>
#include <vector>
#include <fstream>

namespace k52
{
namespace common
{

class Helpers
{
public:

    static std::vector <std::complex< double> > Conjugate(const std::vector <std::complex< double> > &sequence);

    static void PrintComplexVector(std::ofstream &output_file, std::vector <std::complex< double> > &vec);

    static void PrintComplexVector(std::vector <std::complex< double> > &vec);

    static void PrintVector(const std::vector<double>& values);

    static std::vector <std::complex< double> > GenerateComplexSignal(size_t size);
};

}/* namespace common */
}/* namespace k52 */

#endif //K52_HELPERS_H
