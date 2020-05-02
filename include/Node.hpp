#ifndef NODE_HPP
#define NODE_HPP

#include "defines.hpp"
#include "Function.hpp" //activationFunction. Required for calling FunctionBase::createSubobject() in constructors 
#include "Arc.hpp" //children and parents. Required for calling parents[p]->forwardProp() in forwardProp()

#include <vector> //children, parents, scales
#include <string> //name
#include <memory> //FunctionBaseSP activationFunction, std::make_shared<Arc> in createBias()


class FunctionBase;
class Arc;

///each node or neuron in the neural web. Holds pointers to parent and child arcs, activation function, trainable scale and current value
class Node
{
    public:
        Node( uint id, const std::string& name, FunctionBase::FunctionType activationFunctionType, const std::vector<double>& activationFunctionParams = {} ) //creation constructor
        : id(id), name(name)
		, scales( std::vector<double>( 1, INI_NODE_SCALE ) ), activationFunction( FunctionBase::createSubobject( activationFunctionType, activationFunctionParams ) )
		, bTrainableScale(true), value(INI_NODE_VALUE), done(false) {;} //no parent and child links are created at this point. They are added afterwards

        Node( const Node* originalNode ) //fake deep copy constructor
		: id( originalNode->id ), name( originalNode->name )
		, scales(originalNode->scales), activationFunction( FunctionBase::createSubobject( originalNode->activationFunction->getFunctionType(), originalNode->activationFunction->getParams() ) ) //shallow copy would be ok too
        , bTrainableScale(originalNode->bTrainableScale), value(INI_NODE_VALUE), done(false) {;} //no parent and child links are copied at this point because the new aarcs and have to be created at NeuralWeb level before

        virtual ~Node() {}

    //----get
        inline uint getId() const { return id; }
        inline const std::string& getName() const { return name; }

        inline const std::vector<Arc*>& getChildren() const { return children; }
        inline const std::vector<Arc*>& getParents() const { return parents; }
        inline std::vector<Arc*>& getChildrenEditable() { return children; }
        inline std::vector<Arc*>& getParentsEditable() { return parents; }
        
        inline const std::vector<double>& getScales() const { return scales; }
        inline FunctionBaseSP getActivationFunction() { return activationFunction; }

        inline bool getBTrainableScale() const { return bTrainableScale; }
        inline double getValue() const { return value; }
        inline bool getDone() const { return done; }

    //---set
        inline void setId( uint xId ) { id = xId; }
        inline void setChildren( const std::vector<Arc*>& xChildren ) { children = xChildren; }
        inline void setParents( const std::vector<Arc*>& xParents ) { parents = xParents; }
        inline void addChild( Arc* newChild ) { children.push_back( newChild ); }
        inline void addParent( Arc* newParent ) { parents.push_back( newParent ); }
        inline void setParentWeights( const std::vector<double>& xWeights ) { for( uint p = 0; p < parents.size(); p++ )  parents[p]->setWeight( xWeights[p] ); }

        inline void setScales( const std::vector<double>& xScales ) { scales = xScales; }
        inline void setScale( double xScale, uint index = 0 ) { scales[index] = xScale; }
        inline void addScale( double newScale ) { scales.push_back( newScale ); }
        inline void setActivationFunction( FunctionBaseSP xActivationFunction ) { activationFunction = xActivationFunction; }
        inline void setActivationFunction( FunctionBase::FunctionType activationFunctionType, const std::vector<double>& activationFunctionParams ) { activationFunction = FunctionBaseSP( FunctionBase::createSubobject( activationFunctionType, activationFunctionParams ) ); }

        inline void setBTrainableScale( double xTrainableScale ) { bTrainableScale = xTrainableScale; }
        inline void setValue( double xValue ) { value = xValue; }
        inline void setDone( bool xDone ) { done = xDone; }

    //---API
        bool createBias( std::vector<ArcSP>& arcs ); //creates a ner arc representing the bias and adds it to the NeuralWeb total set of arcs passed by ref. For input layer, does not create the bias and returns false
        double forwardProp(); //forward propagation: the values at input layer are propagated towards the output layer and all the nodes are given a value
        //generates the same links to parent and child arcs but related to a new set of arcs. Used for updating nodes when copying a NauralWeb
        inline void updateArcPointers( const std::vector<ArcSP>& newArcs, std::vector<Arc*>& updatedChildren, std::vector<Arc*>& updatedParents ) const { for( uint c = 0; c < children.size(); c++ ) updatedChildren.push_back ( newArcs[ children[c]->getId() ].get() ); 
        																																			for( uint p = 0; p < parents.size(); p++ ) updatedParents.push_back( newArcs[ parents[p]->getId() ].get() ); } 
        

    private:
        uint id; //numeric id that matches index in NeuralWeb vector<NodeSP>
        std::string name; //user-friendly name = the protein or process in the graph

        std::vector<Arc*> children;
        std::vector<Arc*> parents; //terms of the weighted sum. In the simplest first order case, each term = arc = single parent node

        std::vector<double> scales; //trainable scales of each group of terms in the weighted sum. In the simplest first order case, there is a simgle scale but it is prepared for scaling to higher order terms and several groups of weights
        FunctionBaseSP activationFunction; //as a pointer to use polymorphism

        bool bTrainableScale; //whether the node scales are trainable or not. Trainable = internal and output layer. Not trainable = input layer
        double value; //current value, obtained in the last forwardProp
        bool done; //whether the calue is already calculated in the current forward pass. Reset at NeuralWeb level before each forward pass
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline double Node::forwardProp()
///currently a simple weighted sum + activation function with a single set of first order weights. But generalized for future scaling to multiple sets opf weights
{
    if( parents.size() == 0 || done ) //if input layer or already calculated, return the value
        return value;

    std::vector<double> valueCalculations( scales.size(), 0.0 ); //initialize to 0 every weighted sum. In the simplest case, there is a single weighted sum per node

    for( uint s = 0; s < valueCalculations.size(); s++ )
    {
        for( uint p = 0; p < parents.size(); p++ ) //weighted sum normalized
            valueCalculations[s] += parents[p]->forwardProp();
 
        valueCalculations[s] *= scales[s]; //multiply by scale
    }
    value = activationFunction->calculate( valueCalculations ); //apply non-linear activation function
    done = true; //this node is already calculated (do not calculate again during this forward pass)
    return value;
}

#endif //NODE_HPP
