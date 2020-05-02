#ifndef POPULATION_CREATOR_HPP
#define POPULATION_CREATOR_HPP

#include "defines.hpp"
#include "DistributionCombi.hpp" //iniDistributionWeights, iniDistributionScales, iniDistributionSigns
#include "RandomnessHandler.hpp" //init

#include <vector> //std::vector<NeuralWeb*> createPopulation
#include <memory> //NeuralWebSP baseNet, std::vector<NeuralWebSP> createPopulation()


class NeuralWeb;

///creator of random initial populations based on a reference net 
class PopulationCreator
{
    public:
        PopulationCreator() : baseNet(nullptr) {;}
        virtual ~PopulationCreator() = default;

    //---API
        //initialization separated from constructor to allow for creating the object before some of the required args exist
        void init( NeuralWebSP xBaseNet, RandomnessHandler& randomnessHandler, float xScaleMax = DEFAULT_POPCREATOR_SCALE_MAX, float xScaleMin = DEFAULT_POPCREATOR_SCALE_MIN );
        //void init( const NeuralWeb* xBaseNet, float xScaleMax = DEFAULT_POPCREATOR_SCALE_MAX, float xScaleMin = DEFAULT_POPCREATOR_SCALE_MIN );
        void reseed( RandomnessHandler& randomnessHandler ); //give new seeds to all the distributions' random engines
        std::vector<NeuralWebSP> createPopulation( uint populationSize = DEFAULT_POPCREATOR_POPSIZE ); //generate populationSize nets with random params (scales and weights)

        inline float sampleIniDistributionWeights( uint nodeIndex ) { return iniDistributionWeights[nodeIndex].sample(); }
        inline float sampleIniDistributionScales( uint nodeIndex ) { return iniDistributionScales[nodeIndex].sample(); }
        inline float sampleIniDistributionSigns( uint arcIndex ) { return iniDistributionSigns[arcIndex].sample(); }


    private:
        NeuralWebSP baseNet; //reference net structure
        //distributions: each node or arc is given its own distribution to ensure that the initial values actually follow the desired distribution. Node or arc id matches distribution index
        std::vector<DistributionCombi> iniDistributionWeights; //arc weights distribution: node-wise because all the incoming weights to a node are randomized together
        std::vector<DistributionCombi> iniDistributionScales; //node scales distributions
        std::vector<DistributionCombi> iniDistributionSigns; //sign distributions for arcs with ANY sign (unrestricted)
};

#endif //POPULATION_CREATOR_HPP
