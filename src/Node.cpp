#include "Node.hpp"


bool Node::createBias( std::vector<ArcSP>& arcs )
{
    if( parents.size() == 0 ) //if input layer, do not create bias and return false
        return false;

    arcs.push_back( std::make_shared<Arc>( arcs.size(), Arc::Sign::ANY, nullptr, this ) ); //add bias to the NeuralWeb total set of arcs passed by ref. Biases are not restricted in sign and have no parent node
    parents.push_back( arcs.back().get() );  //add bias to this node's parent arcs
        
    return true;
}