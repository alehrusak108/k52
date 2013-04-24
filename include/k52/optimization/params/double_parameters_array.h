#ifndef DOUBLEPARAMETERSARRAY_H_
#define DOUBLEPARAMETERSARRAY_H_

#include <k52/optimization/params/discrete_parameters.h>
#include <k52/optimization/params/double_parameter.h>
#include <k52/optimization/params/composite_discrete_parameters.h>
#include <k52/optimization/params/const_chromosome_size_paremeters.h>

namespace k52
{
namespace optimization
{

class DoubleParametersArray : public DiscreteParameters
{
public:
    typedef boost::shared_ptr<DoubleParametersArray> shared_ptr;

    DoubleParametersArray(double min_value, double max_value, double desired_precision, size_t number_of_parameters);

    DoubleParametersArray* Clone() const;

    bool CheckConstraints() const;

    size_t GetChromosomeSize() const;

    void SetChromosome(std::vector<bool>::iterator from, std::vector<bool>::iterator to) const;

    void SetFromChromosome(std::vector<bool>::const_iterator from, std::vector<bool>::const_iterator to);

    std::vector<double> GetValues() const;

    double get_max_value() const;
    double get_min_value() const;
    size_t get_number_of_parameters() const;
    double get_actual_precision() const;

private:
    DoubleParametersArray();
    DoubleParameter::shared_ptr sample_parameter_;
    std::vector<double> values_;
    double min_value_;
    double max_value_;
};

}/* namespace optimization */
}/* namespace k52 */

#endif /* DOUBLEPARAMETERSARRAY_H_ */
