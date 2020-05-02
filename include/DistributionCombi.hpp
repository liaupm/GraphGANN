#ifndef DISTRIBUTION_COMBI_HPP
#define DISTRIBUTION_COMBI_HPP

#include "defines.hpp"
#include "DistributionInterface.hpp" //DistributionInterface* distribution

#include <memory> //RandomEngineSP randomEngine, DistributionInterfaceSP distribution


///combination of a STL distribution and a STL RandomEngine
class DistributionCombi
{
    public:
        inline DistributionCombi( uint distributionType = DISTRIBUTIONTYPE_UNIFORM, const std::vector<double>& distributionParams = { 0.0, 1.0 }, uint seed = 0 )
        : randomEngine( new RandomEngine( seed ) )
        , distribution ( DistributionInterface::createSubobject( distributionParams, distributionType ) ) {;}

        virtual ~DistributionCombi() {}

    //---get
        inline RandomEngineSP getRandomEngine() const { return randomEngine; }
        inline DistributionInterfaceSP getDistribution() const { return distribution; }

    //---set
        inline void setSeed( uint xSeed ) { randomEngine->seed( xSeed); }

    //---API
        inline double sample() { return distribution->sample( *randomEngine ); } //sample the distribution once by getting a new random number from the random engine
        inline double sampleScaled( double lBound, double uBound ) { return lBound + ( uBound - lBound ) * distribution->sample( *randomEngine ); } //sample and scale. Used with uniform distribution when the bounds change between samples

    private:
        RandomEngineSP randomEngine;
        DistributionInterfaceSP distribution;
};

#endif //CG_DISTRIBUTION_COMBI_HPP
