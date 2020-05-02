#include "Metrics.hpp"
#include "NeuralWeb.hpp" //Do not include it in the hpp file to avoid circular include with NeuralWeb


void Metrics::accumulatePopulationFitness( const std::vector<NeuralWebSP>& population ) //cannot be inlined due to consequent circular include with NeuralWeb
{ 
    fitness = 0.0; 
    for( uint p = 0; p < population.size(); p++ ) 
        fitness += population[p]->calculateFitness(); 
} 