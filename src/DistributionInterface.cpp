#include "DistributionInterface.hpp"

///////////////////////////////////////// DISTRIBUTION INTERFACE /////////////////////////////////////////////////////////////////////////
//static
DistributionInterface* DistributionInterface::createSubobject( const std::vector<double>& distributionParams, uint distributionType )
{
    switch( distributionType )
    {
        case DISTRIBUTIONTYPE_NORMAL: 
            return new NormalDistribution( distributionParams );
        default:
           return new UniformDistribution( distributionParams );
    }
}