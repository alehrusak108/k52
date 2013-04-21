#include <k52/optimization/bounded_nelder_mead.h>

#include <math.h>
#include <stdexcept>
#include <algorithm>
#include <fstream>

#include <k52/optimization/params/i_continuous_parameters.h>

#include "random.h"
#include "objective_function_counter.h"

using ::std::vector;

namespace k52
{
namespace optimization
{

BoundedNelderMead::BoundedNelderMead(double l, double precision, double lower_bound, double upper_bound)
{
    lower_bound_ = lower_bound;
    upper_bound_ = upper_bound;
    precision_ = precision;
    l_ = l;
}

void BoundedNelderMead::Optimize(const IObjectiveFunction &function_to_optimize, IParameters* parametrs_to_optimize)
{
    IContinuousParameters* continuous_parameters = dynamic_cast<IContinuousParameters*> (parametrs_to_optimize);
    if(continuous_parameters == NULL)
    {
        throw std::invalid_argument("parametrsToOptimize must be of type IContinuousParameters for BoundedNelderMead");
    }
    vector<double> initial_parameters = continuous_parameters->GetValues();

    //Size of task
    size_t n = initial_parameters.size();

    //Iteration index
    int r = 0;

    //For simplex points count
    double square_root_from_2 = sqrt(2.);
    double r1 = l_ * ( sqrt((double)(n+1)) + n - 1 ) / ( n * square_root_from_2 );
    double r2 = l_ * ( sqrt((double)(n+1)) - 1 ) / ( n * square_root_from_2 );

    //Random init
    vector<double> base_point(n);
    for(int i = 0; i < n; i++)
    {
        base_point[i] = Random::Instance().GetContinuousRandomQuantity(lower_bound_, upper_bound_);
    }
    //Build simplex
    vector< vector<double> > polygon = GetRegularSimplex(base_point, l_);

    //count values
    vector<double> function_values = CountObjectiveFunctionValues(polygon, continuous_parameters, function_to_optimize);	

    do
    {
        OutputPolygon(polygon);
        r++;
        int first_max_index = 0, secondMaxIndex = 0, minIndex = 0;
        //determine maximums and minimum
        GetIndexes(function_values, &first_max_index, &secondMaxIndex, &minIndex);

        double highest_value = function_values[first_max_index];
        double second_highest_value = function_values[secondMaxIndex];
        double lowest_value = function_values[minIndex];

        //determine center of mass
        vector<double> center_of_mass = GetCenterOfMass(polygon, first_max_index); 

        //Reflect max point - we seek for minimum
        vector<double> reflected_point = Reflexion(center_of_mass, polygon[first_max_index]);
        CorrectByProjectingToBounds(&reflected_point);
        double reflected_point_value = CountObjectiveFunctionValue(reflected_point, continuous_parameters, function_to_optimize);

        if(reflected_point_value < lowest_value)
        {
            vector<double> expanded_point = Expansion(center_of_mass, reflected_point);
            CorrectByProjectingToBounds(&expanded_point);
            double expanded_point_value = CountObjectiveFunctionValue(expanded_point, continuous_parameters, function_to_optimize);

            if(expanded_point_value < reflected_point_value )
            {
                //Replace max with expanded
                polygon[first_max_index] = expanded_point;
                function_values[first_max_index] = expanded_point_value;
                continue;
            }
            else
            {
                //Replace max with reflected
                polygon[first_max_index] = reflected_point;
                function_values[first_max_index] = reflected_point_value;
                continue;
            }
        }
        else
        {
            if(reflected_point_value <= second_highest_value)
            {
                //Replace max with reflected
                polygon[first_max_index] = reflected_point;
                function_values[first_max_index] = reflected_point_value;
                continue;
            }
            else
            {
                if(reflected_point_value < highest_value)
                {
                    //Replace max with reflected
                    polygon[first_max_index] = reflected_point;
                    function_values[first_max_index] = reflected_point_value;
                    highest_value = reflected_point_value;
                }

                vector<double> contraction_point = Contraction(center_of_mass, polygon[first_max_index]);
                double contraction_point_value = CountObjectiveFunctionValue(contraction_point, continuous_parameters, function_to_optimize);

                if(contraction_point_value > highest_value)
                {
                    Reduction(&polygon, minIndex);
                    continue;
                }
                else
                {
                    //Replace max with contracted
                    polygon[first_max_index] = contraction_point;
                    function_values[first_max_index] = contraction_point_value;
                    continue;
                }
            }
        }
    }while( CountDifferance(function_values) > precision_ );


    size_t best_index = std::distance(function_values.begin(), std::max_element(function_values.begin(), function_values.end()));
    continuous_parameters->SetValues( polygon[best_index] );
}

void BoundedNelderMead::CorrectByProjectingToBounds(vector<double>* point)
{
    for(size_t i = 0; i < point->size(); i++)
    {
        if( (*point)[i] > upper_bound_)
        {
            (*point)[i] = upper_bound_;
        }
        else if( (*point)[i] < lower_bound_)
        {
            (*point)[i] = lower_bound_;
        }
    }
}

vector<double> BoundedNelderMead::CountObjectiveFunctionValues(
    const vector< vector<double> >& parameters_values,
    IContinuousParameters* base_parameters,
    const IObjectiveFunction & function_to_optimize)
{
    size_t N = parameters_values.size();    
    vector<double> counted_values(N);

    for(size_t i = 0; i < N; i++)
    {
        base_parameters->SetValues(parameters_values[i]);
        IContinuousParameters::shared_ptr parameters_clone( base_parameters->Clone() );
        counted_values[i] = function_to_optimize(parameters_clone.get());
    }

    return counted_values;
}

double BoundedNelderMead::CountObjectiveFunctionValue(
    const vector<double>& parameters,
    IContinuousParameters* base_parameters,
    const IObjectiveFunction & function_to_optimize)
{
    vector< vector<double> > cover(1);
    cover[0] = parameters;
    return CountObjectiveFunctionValues(cover, base_parameters, function_to_optimize)[0];
}

void BoundedNelderMead::GetIndexes(const vector<double>& values, int* first_max_index, int* secound_max_index, int* min_index)
{
    if(values.size() < 2)
    {
        throw std::invalid_argument("values must have at least 2 elements");
    }

    if( values[0] > values[1] )
    {
        *first_max_index = 0;
        *secound_max_index = 1;
        *min_index = 1;
    }
    else
    {
        *first_max_index = 1;
        *secound_max_index = 0;
        *min_index = 0;
    }

    for (int i = 2; i<values.size(); i++) 
    {
        if (values[i] > values[*first_max_index]) 
        {
            *secound_max_index = *first_max_index;
            *first_max_index = i;
        } 
        else if (values[i]  > values[*secound_max_index]) 
        {
            *secound_max_index = i;
        }

        if( values[i] < values[*min_index])
        {
            *min_index = i;
        }
    }
}

vector< vector<double> > BoundedNelderMead::GetRegularSimplex(const vector<double>& base_point, double l)
{
    //Size of task
    size_t n = base_point.size();

    //For simplex points count
    double square_root_from_2 = sqrt(2.);
    double r1 = l * ( sqrt((double)(n+1)) + n - 1 ) / ( n * square_root_from_2 );
    double r2 = l * ( sqrt((double)(n+1)) - 1 ) / ( n * square_root_from_2 );

    vector< vector<double> > regular_simplex( n + 1 );
    regular_simplex[0] = base_point;

    for(int i = 1; i < n + 1; i++)
    {
        regular_simplex[i] = vector<double>(n);

        for(int j = 0; j < n; j++)
        {
            if( j == i - 1 )
            {
                regular_simplex[i][j] = base_point[j] + r1;
            }
            else
            {
                regular_simplex[i][j] = base_point[j] + r2;
            }
        }
    }
    return regular_simplex;
}

vector<double> BoundedNelderMead::Reflexion(const vector<double>& center_of_mass, const vector<double>& target_point)
{
    double reflection_coefficient = 1;
    vector<double> new_point (target_point.size());

    //Reflect
    for(int i = 0; i < target_point.size(); i++)
    {
        new_point[i] = center_of_mass[i] + reflection_coefficient * (center_of_mass[i] - target_point[i]);
    }

    return new_point;
}

vector<double> BoundedNelderMead::Expansion(const vector<double>& center_of_mass, const vector<double>& target_point)
{
    double expansion_coefficient = 2;
    vector<double> new_point (target_point.size());

    //Expand
    for(int i = 0; i < target_point.size(); i++)
    {
        new_point[i] = center_of_mass[i] + expansion_coefficient * (target_point[i] - center_of_mass[i]);
    }

    return new_point;
}

vector<double> BoundedNelderMead::Contraction(const vector<double>& center_of_mass, const vector<double>& target_point)
{
    double contraction_coefficient = 0.5;
    vector<double> new_point (target_point.size());

    //Contract
    for(int i = 0; i < target_point.size(); i++)
    {
        new_point[i] = center_of_mass[i] + contraction_coefficient * (target_point[i] - center_of_mass[i]);
    }

    return new_point;
}

void BoundedNelderMead::Reduction(vector< vector<double> >* polygon, int point_index)
{
    if(point_index<0 || point_index>= polygon->size())
    {
        throw std::invalid_argument("Incorrect pointIndex");
    }

    double reduction_coefficient = 0.5;
    size_t n = (*polygon)[0].size();

    for(int i = 0; i < n +1; i++)
    {
        if( i != point_index )
        {
            for(int j = 0; j < n; j++)
            {
                (*polygon)[i][j] = (*polygon)[point_index][j] + reduction_coefficient * ( (*polygon)[i][j]  - (*polygon)[point_index][j] );
            }
        }
    }
}

double BoundedNelderMead::CountDifferance(const vector<double>& values)
{
    double summ = 0;
    for(size_t i = 0; i < values.size(); i++)
    {
        summ += values[i];
    }
    double averadge = summ / values.size();

    double square_summ = 0;
    for(size_t i = 0; i < values.size(); i++)
    {
        double diff = values[i] - averadge;
        square_summ += diff*diff;
    }
    return sqrt( square_summ / values.size());
}

vector<double> BoundedNelderMead::GetCenterOfMass(const vector< vector<double> >& polygon, int point_index)
{
    if(point_index<0 || point_index>= polygon.size())
    {
        throw std::invalid_argument("Incorrect point_index");
    }
    size_t n = polygon[0].size();
    vector<double> center_of_mass(n);

    //TODO implement with valarray
    for(int i = 0; i < n; i++)
    {
        center_of_mass[i] = 0;

        for(int j = 0; j < n+1; j++)
        {
            if( j != point_index )
            {
                center_of_mass[i] += polygon[j][i];
            }
        }

        center_of_mass[i] /= n;
    }

    return center_of_mass;
}

void BoundedNelderMead::OutputPolygon(const vector< vector<double> >& polygon)
{
    std::ofstream polygon_output("polygon.plot", std::ofstream::app);
    polygon_output<<std::endl;

    for(size_t i = 0; i < polygon.size(); i++)
    {
        for(size_t j = 0; j < polygon[i].size(); j++)
        {
            polygon_output << polygon[i][j] << " ";
        }
        polygon_output<<std::endl;
    }
}

}/* namespace optimization */
}/* namespace k52 */
