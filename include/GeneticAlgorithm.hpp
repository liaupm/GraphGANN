#ifndef GENETIC_ALGORITHM_HPP
#define GENETIC_ALGORITHM_HPP

#include "defines.hpp"
#include "DistributionCombi.hpp" //distributions
#include "RandomnessHandler.hpp" //constructor
#include "NeuralWebBase.hpp" //Metrics totalMetrics
#include "NeuralWeb.hpp" //population, selctedParents, children, bestNet...
#include "Parser.hpp" //constructor, MmxParams constructor, GaParams constructor

#include <vector> //population, inputs, outputs and weights,  std::vector<float> rouletteCumulatedProbs, std::vector<NeuralWeb*> selectedParents, std::vector<NeuralWeb*> children, distributions
#include <map> //intParams and realParams for constructor
#include <memory> //std::vector<NeuralWebSP> currentPopulation, std::vector<NeuralWebSP> selectedParents, std::vector<NeuralWebSP> children


class NeuralWeb;

///real-coded steady-state genetic algoritm for training neural networks. Roulette selection, MMX crossover with mutation and deterministic replacement
class GeneticAlgorithm
{
    public:
        ///params of the MMX crossover operator
        struct MmxParams
        {
            double a; //negative and close to 0
            double b; //negative and bigger than a in abs
            double c; //positive
            double d; //positive and smaller than 0.5
            double tempCalculation1; //slope of the first line tract
            double tempCalculation2; //slope of the second line tract

            MmxParams( const Parser& parser ) : a( parser.getRealParam( "a" ) ), b( parser.getRealParam( "b" ) ), c( parser.getRealParam( "c" ) ), d( parser.getRealParam( "d" ) )  
            , tempCalculation1( ( b - a ) / c ), tempCalculation2( d / ( 1.0 - c ) ) {;}
        };

        ///rest of GA params
        struct GaParams
        {
            uint crossNum; //number of crosses per generation. Currently only supported 1
            uint parentNum; //number of selected parents per cross
            uint outspringNum; //number of children per cross. Currently only supported 2
            uint deathNum; //number of worst net replaced by children = crossNum * outspringNum to keep const population size

            float mutationProbWeights; //prob of weight mutation per cross 
            float mutationAmountWeights; //max amount of weight change due to mutation per cross. Min = 0.0
            float mutationProbActivation; //prob of scale mutation per cross 
            float mutationAmountScales; //max amount of scale change due to mutation per cross. Min = 0.0


            GaParams( const Parser& parser ) 
            : crossNum( parser.getIntParam( "crossNum" ) ), parentNum( parser.getIntParam( "parentNum" ) ), outspringNum( parser.getIntParam( "outspringNum" ) ), deathNum( parser.getIntParam( "crossNum" ) * parser.getIntParam( "outspringNum" ) )
            , mutationProbWeights( parser.getRealParam( "mutationProbWeights" ) ), mutationAmountWeights( parser.getRealParam( "mutationAmountWeights" ) ), mutationProbActivation( parser.getRealParam( "mutationProbActvation" ) ), mutationAmountScales( parser.getRealParam( "mutationAmountScales" ) ) {;}
        };


        GeneticAlgorithm( const std::vector<NeuralWebSP>& iniPopulation, const Parser& parser, RandomnessHandler& randomnessHandler );
        virtual ~GeneticAlgorithm() {}

    //---get
        inline const std::vector<NeuralWebSP>& getCurrentPopulation() const { return currentPopulation; }
        inline uint getPopSize() const { return currentPopulation.size(); }
        inline const Metrics& getTotalMetrics() const { return totalMetrics; }

        NeuralWebSP getBestNet(); //return the net with lowest fitness (training metrics)

    //---set
        inline void addNet( NeuralWebSP newNet ) { currentPopulation.push_back( newNet ); } //used by migration operator of MultiGa
        NeuralWebSP popNet( uint netIndex ); //draw a new from population without moving the whole vector of nets. //used by death() and migration operator of MultiGa (requires returning it for substracting their fitness from the total )
    
    //---API
        //train for a given number of generations with the given instances
        void train( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights, uint generationNum = DEFAULT_GA_GENERATION_NUM, bool evaluateAll = DEFAULT_GA_EVALUATEALL );
        void evaluateWholePopulation( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights ); //update train metrics of all the nets in the population with the given instances
        

    private:
        MmxParams mmxParams; //params of MMX crossover algorithm
        GaParams gaParams; //rest of GA params
        double scaleMin; //lbound of arc scales. Used in mmxCrossActivationParam()
        double scaleMax; //ubound of arc scales. Used in mmxCrossActivationParam()

        std::vector<DistributionCombi> distributions; //all the distributions used for selection, cross and mutation

        std::vector<NeuralWebSP> currentPopulation; //whole population
        std::vector<float> rouletteCumulatedProbs; //selection prob of the whole population based on fitness
        std::vector<NeuralWebSP> selectedParents; //selected parent for crossing. Updated every generation
        std::vector<NeuralWebSP> children; //newly generated nets. Updated every generation

        Metrics totalMetrics; //sum of metrics of the whole population. Used for calculating selection probs from fitness

    //---GA steps: called in order every generation by train()
        void selectParents(); //roulette selection of parents for cross into selectedParents vector
        void mmxCross( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights ); //crossover of both scales and weights to generate children vector
        void mmxCrossScales(); //MMX crossover of node scales using the selected parents. Includes mutation
        void mmxCrossWeights(); //MMX crossover of arc weights using the selected parents. Includes mutation
        void death(); //deterministic replacement of the worst nets in population vector by nets in children vector
};

#endif //GENETIC_ALGORITHM_HPP
