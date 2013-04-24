#include <k52/optimization/params/double_parameters_array.h>

#include <stdexcept>

namespace k52
{
namespace optimization
{

DoubleParametersArray::DoubleParametersArray(double min_value, double max_value, double desired_precision, size_t number_of_parameters)
    : sample_parameter_ (new DoubleParameter(min_value, min_value, max_value, desired_precision)), values_(number_of_parameters)
{
    min_value_ = min_value;
    max_value_ = max_value;
}

DoubleParametersArray* DoubleParametersArray::Clone() const
{
    DoubleParametersArray* clone = new DoubleParametersArray();
    clone->max_value_ = max_value_;
    clone->min_value_ = min_value_;
    clone->values_ = values_;

    clone->sample_parameter_ = sample_parameter_ != NULL ?
        DoubleParameter::shared_ptr( sample_parameter_->Clone() ) : 
        DoubleParameter::shared_ptr();

    return clone;
}

bool DoubleParametersArray::CheckConstraints() const
{
    for(size_t i=0; i<values_.size(); i++)
    {
        //TODO
        //copypaste from DoubleParameter
        if(values_[i] < min_value_ || values_[i] > max_value_)
        {
            return false;
        }
    }
    return true;
}

size_t DoubleParametersArray::GetChromosomeSize() const
{
    return sample_parameter_->GetChromosomeSize() * values_.size();
}

void DoubleParametersArray::SetChromosome(std::vector<bool>::iterator from, std::vector<bool>::iterator to) const
{
    //TODO
    //copypaste from CompositeDiscreteParameters
    size_t chromosome_size = to - from;

    //TODO fix, implemetation needed
    //this->CheckForConstChromosomeSize(chromosome_size);

    std::vector<bool>::iterator current_from = from;
    size_t parameter_chromosome_size = sample_parameter_->GetChromosomeSize();

    for(size_t i = 0; i < values_.size(); i++)
    {
        std::vector<bool>::iterator current_to = current_from + parameter_chromosome_size;

        sample_parameter_->SetValue(values_[i]);

        sample_parameter_->SetChromosome(current_from, current_to);

        current_from = current_to;
    }
}

void DoubleParametersArray::SetFromChromosome(std::vector<bool>::const_iterator from, std::vector<bool>::const_iterator to)
{
    //TODO
    //copypaste from CompositeDiscreteParameters
    size_t chromosome_size = to - from;
    //TODO fix, implemetation needed
    //this->CheckForConstChromosomeSize(chromosome_size);

    std::vector<bool>::const_iterator current_from = from;
    size_t parameter_chromosome_size = sample_parameter_->GetChromosomeSize();

    for(size_t i = 0; i < values_.size(); i++)
    {
        std::vector<bool>::const_iterator current_to = current_from + parameter_chromosome_size;

        sample_parameter_->SetFromChromosome(current_from, current_to);

        values_[i] = sample_parameter_->GetValue();

        current_from = current_to;
    }
}

std::vector<double> DoubleParametersArray::GetValues() const
{
    return values_;
}

double DoubleParametersArray::get_max_value() const
{
    return max_value_;
}

double DoubleParametersArray::get_min_value() const
{
    return min_value_;
}

size_t DoubleParametersArray::get_number_of_parameters() const
{
    return values_.size();
}

double DoubleParametersArray::get_actual_precision() const
{
    return sample_parameter_->get_actual_precision();
}

DoubleParametersArray::DoubleParametersArray() {}

}/* namespace optimization */
}/* namespace k52 */
