#ifndef RANDOMNESS_HANDLER_HPP
#define RANDOMNESS_HANDLER_HPP

#include "defines.hpp"
#include "DistributionCombi.hpp" //std::vector<DistributionCombi> datasetDistributions

#include <vector> //std::vector<uint> seeds in constructor, std::vector<RandomEngine2*> mainREs, std::vector<DistributionCombi> datasetDistributions
#include <set> //std::set<uint> usedSeeds
#include <random> //main RandomEngines
#include <memory> //std::vector<RandomEngine2SP> mainREs


///generates appropriate seeds for all the generators in the app by using a different type or random generator and checking for repeated seeds
///also holds the dataset-related distributions because they must be shared by all the dataset instances rather than having a copy in each one
class RandomnessHandler
{
    public:
        RandomnessHandler( const std::vector<uint>& seeds ) 
        {  
            for( uint s = 0; s < seeds.size(); s++ )
                mainREs.emplace_back( new RandomEngine2( seeds[s] ) );

            datasetDistributions.emplace_back( DISTRIBUTIONTYPE_UNIFORM, std::vector<double>( { 0.0, 1.0 } ), getValidSeed( INDEX_RANDOMNESS_MAINRE_DATA ) ); 
        }

        virtual ~RandomnessHandler() {}

    //---get
        inline RandomEngineSP getDataDistributionRE( uint index ) { return datasetDistributions[index].getRandomEngine(); }

    //---API
        inline uint getValidSeed( uint reIndex ) { uint seed; do { seed = ( *mainREs[reIndex] )();  } while( ! usedSeeds.insert( seed ).second ); return seed; } //produces a not-used-yet seed from the given main random engine
        inline std::vector<uint> getValidSeeds( uint reIndex, uint seedNum = 1 ) { std::vector<uint> seeds; for( uint s = 0; s < seedNum; s++ ) seeds.push_back( getValidSeed( reIndex ) ); return seeds; } //produces n not-used-yet seeds from the given main random engine


    private:
        std::vector<RandomEngine2SP> mainREs; //main random engines
        std::vector<DistributionCombi> datasetDistributions;
        std::set<uint> usedSeeds;
};

#endif //RANDOMNESS_HANDLER_HPP
