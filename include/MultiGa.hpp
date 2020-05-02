#ifndef MULTI_GA_HPP
#define MULTI_GA_HPP

#include "defines.hpp"
#include "DistributionCombi.hpp" //distributions
#include "RandomnessHandler.hpp" //constructor
#include "NeuralWeb.hpp" //call functions from bestNet
#include "HistoricalTrack.hpp" //HistoricalTrack historicalTrack
#include "GeneticAlgorithm.hpp" //std::vector<GeneticAlgorithm*> gas
#include "Dataset.hpp" //trainAndTrack()
#include "Parser.hpp" //constructor, MultiGaParams constructor

#include <vector> //std::vector<GeneticAlgorithm*> gas
#include <memory> //std::vector<GeneticAlgorithmSP> gas, NeuralWebSP bestNet


class NeuralWeb;
class GeneticAlgorithm;

///multipopulation GA that includes a vector of simple GA and migration functionality
class MultiGa
{
    public:
        ///params that are exclusive of multipopulation GA 
        struct MultiGaParams
        {
            double mixFraction; //fraction of population exchanged in multi population migration events
            uint mixNets; //mixFraction * population size. Precomputed for efficiency

            MultiGaParams( const Parser& parser ) : mixFraction( parser.getRealParam( "mixFraction" ) ), mixNets( mixFraction * parser.getUintParam( "popSize" ) ) {;}
        };

        MultiGa( std::vector<GeneticAlgorithmSP> gas, const Parser& parser, RandomnessHandler& randomnessHandler );
        virtual ~MultiGa() {}
    
    //---get
        inline const std::vector<GeneticAlgorithmSP>& getGas() const { return gas; }
        NeuralWebSP getBestNet(); //get best net of current generation in training set (in lossW)
        //get historical best net in the given set and metric
        inline NeuralWebSP getHistoricalBestNet( uint& bestGeneration, uint minGeneration = HTRACK_MIN_GENERATION, uint setIndex = INDEX_SET_VAL, uint metricIndex = INDEX_METRIC_LOSS_W ) { return historicalTrack.getHistoricalBestNet( bestGeneration, minGeneration, setIndex, metricIndex ); }
    //---set
        inline void setGas( const std::vector<GeneticAlgorithmSP>& xGas ) { gas = xGas; }
        inline void addGas( GeneticAlgorithmSP newGa ) { gas.push_back( newGa ); }

    //---API
        void train( const Dataset* dataset, uint generationsPerMix = DEFAULT_MGA_GENERATIONS_PER_MIX, uint mixNum = DEFAULT_MGA_MIXNUM ); //train a given number of mix rounds with the given dataset (with training and val splits). No historical track
        void trainAndTrack( const Dataset* dataset, uint generationsPerMix = DEFAULT_MGA_GENERATIONS_PER_MIX, uint mixNum = DEFAULT_MGA_MIXNUM ); //same as train() but saving historical info in historicalTrack. Slower but required for further selection of the historical best net in val set
        inline void evaluateWholePopulation( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights ) { for( uint g = 0; g < gas.size(); g++ ) gas[g]->evaluateWholePopulation( inputs, outputs, instanceWeights ); }


        
    private:
        MultiGaParams params;
        HistoricalTrack historicalTrack; //registry of the historical evolution of the population (best net per generation and metrics )
        std::vector<GeneticAlgorithmSP> gas; //simple genetic algorithms that make the different populations

        std::vector<DistributionCombi> distributions; //distributions used for migration

    //---multiGa algorithm steps
        void mix();
};

#endif //MULTI_GA_HPP
