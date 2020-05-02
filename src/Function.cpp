#include "Function.hpp"

///////////////////////////////////////// FUNCTION BASE /////////////////////////////////////////////////////////////////////////
//static
FunctionBase* FunctionBase::createSubobject( FunctionType functionType, const std::vector<double>& params )
{
	switch( functionType )
    {
        case FunctionType::SAT_EXPONENTIAL:
            return new SatExponential();
        case FunctionType::SIGMOID:
            return new Sigmoid();
        default:
            return nullptr;
    }
}
