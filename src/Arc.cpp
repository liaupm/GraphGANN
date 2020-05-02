#include "Arc.hpp"
#include "Node.hpp" //parent forwardProp(). Do not include it in the hpp file to avoid circular include with Node

#include <math.h> //std::pow in forwardProp()


double Arc::forwardProp() //cannot be inlined due to consequent circular include with Node
{
	double result = weight; //if the arc is a bias, it returns weight
	for( uint p = 0; p < parents.size(); p++ ) //the arc is a product term of parent nodes, each with an exponent. In the simplest case, a single parent node with exponent = 1
	{
		if( parents[p] != nullptr )
			result *= std::pow( parents[p]->forwardProp(), exponents[p] );
	}
	return result;
}