#ifndef DISTRIBUTION_INTERFACE_HPP
#define DISTRIBUTION_INTERFACE_HPP

#include "defines.hpp"

#include <random> //RandomEngine


///////////////////////////////////////////////////////////* CONTINUOUS DISTRIBUTIONS *////////////////////////////////////////////////////////////////////////
///common interface for STL continuous distributions
class DistributionInterface 
{
    public:
        //static
        static DistributionInterface* createSubobject( const std::vector<double>& distributionParams, uint distributionType = DISTRIBUTIONTYPE_UNIFORM ); //factory. Params meaning depends on the distribution

        DistributionInterface() {;}
        virtual ~DistributionInterface() {}

        //API
        virtual double sample( RandomEngine& randomEngine ) = 0; //sample once the distribution
};


//========================================== UNIFORM ==========================================
class UniformDistribution : public std::uniform_real_distribution<double>, public DistributionInterface
{
    public:
        UniformDistribution( const std::vector<double>& params ) : std::uniform_real_distribution<double>( params[0], params[1] ), DistributionInterface() {;}
        virtual ~UniformDistribution() {;}

        //API
        double sample( RandomEngine& randomEngine ) override { return std::uniform_real_distribution<double>::operator()( randomEngine ); }
};


//========================================== NORMAL ==========================================
class NormalDistribution : public std::normal_distribution<double>, public DistributionInterface
{
    public:
        NormalDistribution( const std::vector<double>& params ) : std::normal_distribution<double>( params[0], params[1] ), DistributionInterface() {;}
        virtual ~NormalDistribution() {;}

        //API
        double sample( RandomEngine& randomEngine ) override { return std::normal_distribution<double>::operator()( randomEngine ); }
};

#endif //DISTRIBUTION_INTERFACE_HPP
