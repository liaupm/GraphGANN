#include "NeuralWeb.hpp"
#include "Function.hpp" //convertToFF()

#include <algorithm> //sort in randomizeWeights(), shuffle in swapInputLayer()


//////////////////////////////////////////////////////////////////* NEURAL WEB *///////////////////////////////////////////////////////////////
NeuralWeb::NeuralWeb( const NeuralWeb* const originalNeuralWeb ) //copy constructor
: NeuralWebBase::NeuralWebBase( *originalNeuralWeb ) //copy params and metrics
, bSaved(false)
{
//---copy nodes (without parent and child links yet)
    for( uint n = 0; n < originalNeuralWeb->nodes.size(); n++ )
    	nodes.emplace_back( new Node( originalNeuralWeb->nodes[n].get() ) );
//---copy arcs and assign the new nodes as parents and children
    for( uint a = 0; a < originalNeuralWeb->arcs.size(); a++ )
    {
    	if( originalNeuralWeb->arcs[a]->getParent() == nullptr ) //if arc is bias
    		arcs.push_back( std::make_shared<Arc>( originalNeuralWeb->arcs[a].get(), nullptr, nodes[ originalNeuralWeb->arcs[a]->getChild()->getId() ].get() ) );
    	else
    		arcs.push_back( std::make_shared<Arc>( originalNeuralWeb->arcs[a].get(), nodes[ originalNeuralWeb->arcs[a]->getParent()->getId() ].get(), nodes[ originalNeuralWeb->arcs[a]->getChild()->getId() ].get() ) );		
    }
//---assign the new arcs as parents and children of the new nodes
    for( uint n = 0; n < originalNeuralWeb->nodes.size(); n++ )
    	originalNeuralWeb->nodes[n]->updateArcPointers( arcs, nodes[n]->getChildrenEditable(), nodes[n]->getParentsEditable() );
//---assign input and output nodes
    findLayers(); 
    initReflection();
}


//==================================== GET ============================================
std::vector<std::string> NeuralWeb::getHeader() const
{
	std::vector<std::string> header;
	header.push_back( outputLayer->getName() );
	for( uint i = 0; i < inputLayer.size(); i++ )
		header.push_back( inputLayer[i]->getName() );
	return header;
}

////////////////////////////////////////////////////////////* API *///////////////////////////////////////////////////////////////////////////////

//==================================== STRUCTURE ============================================

void NeuralWeb::transferParams( const NeuralWeb* originalNeuralWeb )
{
	for( uint n = 0; n < nodes.size(); n++ )
		nodes[n]->setScales( originalNeuralWeb->getNodes()[n]->getScales() );
	for( uint a = 0; a < arcs.size(); a++ )
		arcs[a]->setWeight( originalNeuralWeb->getArcs()[a]->getWeight() );
}

void NeuralWeb::findLayers() 
{
	inputLayer.clear();
    for( uint n = 0; n < nodes.size(); n++ )
    {
        if( nodes[n]->getParents().size() == 0 ) //if no parents, input layer
        {
        	inputLayer.push_back( nodes[n] );
        	nodes[n]->setScale( INPUT_NODE_SCALE );
        	nodes[n]->setBTrainableScale( false ); //input nodes have an untrainable scale of INPUT_NODE_SCALE ( typically 1)
        }
        else if( nodes[n]->getChildren().size() == 0 ) //if no children, output layer
            outputLayer = nodes[n];
    }
//---replace the activation function of the input layer by a sigmoid
    outputLayer->setActivationFunction( std::make_shared<Sigmoid>() );
}


//==================================== PARAM RANDOMIZATION ============================================
void NeuralWeb::randomizeWeights( PopulationCreator& popCreator )
{
///weights are randomized in a node-wise fashion. 
///as all the incoming arcs have to add up to 1 in abs, 1 is distributed between the n parent arcs by randomly placing n - 1 separators
///then the the sign is changet to match restrictions

	for( uint n = 0; n < nodes.size(); n++ )
	{
		const std::vector<Arc*>& parents = nodes[n]->getParents();
		std::vector<float> separators;
		int separatorNum = parents.size() - 1;
	//---get separatorNum sorted random numbers in [0, 1 ) as separators
		for( int p = 0; p < separatorNum; p++ )
			separators.push_back( popCreator.sampleIniDistributionWeights( n ) );
		std::sort( separators.begin(), separators.end() );
	//---add the lower and upper bounds
		separators.push_back( 1.0 );
		separators.insert( separators.begin(), 0.0 );

		for( uint p = 0; p < parents.size(); p++ )
		{
			switch( parents[p]->getSign() )
			{
				case Arc::Sign::POS:
					parents[p]->setWeight( separators[ p + 1 ] - separators[p] ); //weight is the difference between consecutive separators 
					break;
				case Arc::Sign::NEG:
					parents[p]->setWeight( separators[p] - separators[ p + 1 ] ); //if negative, the order is inverted
					break;
				default: //if sign not restricted, it is made pos or neg randomly with equal prob
					if( popCreator.sampleIniDistributionSigns( parents[p]->getId() ) >= 0.5 ) 
						parents[p]->setWeight( separators[ p + 1 ] - separators[p] );
					else
						parents[p]->setWeight( separators[p] - separators[ p + 1 ] );
			}
		}
	}
}

void NeuralWeb::normalizeWeights()
{
	for( uint n = 0; n < nodes.size(); n++ )
	{		
		const std::vector<Arc*>& parents = nodes[n]->getParents();
	//---first, match sign restrictions (just in case)
		for( uint p = 0; p < parents.size(); p++ )
			parents[p]->restrictSign();
	//---add up all the incoming weights in abs
		double totalWeightAbs = 0.0;
		for( uint p = 0; p < parents.size(); p++ )
			totalWeightAbs += std::abs( parents[p]->getWeight() );
	//---then divide every weight by the total to make them add up to 1
		for( uint p = 0; p < parents.size(); p++ )
			parents[p]->setWeight( parents[p]->getWeight() / totalWeightAbs );
	}
}


//==================================== ML ============================================
double NeuralWeb::predict( const std::vector<std::vector<double>>& inputs, uint index ) const
{
//---set the inputs in the input layer
    for( uint i = 0; i < inputLayer.size(); i++ )
        inputLayer[i]->setValue( inputs[index][i] );
//---reset all nodes to "not calculated" state and forward pass
    resetNodes();
    return outputLayer->forwardProp(); //return the real value of the output node as the prediction
}


//==================================== MODIFY STRUCTURE ============================================
void NeuralWeb::convertToFF( const std::vector<uint>& nodeNumPerLayer )
{
	std::vector<uint> nodeNumPerLayerEdited;
	for( uint l = 0; l < nodeNumPerLayer.size(); l++ )
	{
		if( nodeNumPerLayer[l] <= 0 )
			break;
		nodeNumPerLayerEdited.push_back( nodeNumPerLayer[l] );
	}

//---delete all the internal nodes ( have both parents and children )
	nodes = inputLayer;

//---delete all arcs
	arcs.clear();
	for( uint n = 0; n < nodes.size(); n++ )
	{
		nodes[n]->setParents( {} );
		nodes[n]->setChildren( {} );
	}
	outputLayer->setParents( {} );
	outputLayer->setChildren( {} );

//---create nodes
	std::vector<uint> layersStartIndex = { 0, static_cast<uint>( inputLayer.size() ) };

	for( uint l = 0; l < nodeNumPerLayerEdited.size(); l++ )
	{
		for( uint n = 0; n < nodeNumPerLayerEdited[l]; n++ )
		{
			NodeSP newNode = std::make_shared<Node>( nodes.size(), "hidden" + std::to_string( l ) + "_" + std::to_string( n ), FunctionBase::FunctionType::SAT_EXPONENTIAL );
			nodes.push_back( newNode );
		}
		layersStartIndex.push_back( nodes.size() );
	}
	nodes.push_back( outputLayer );
	layersStartIndex.push_back( nodes.size() );

//---connect the nodes in one layer to next layer
	for( uint l = 0; l < nodeNumPerLayerEdited.size() + 1; l++ )
	{
		for( uint n = layersStartIndex[l]; n < layersStartIndex[l + 1]; n++ )
		{
			for( uint n2 = layersStartIndex[l +1]; n2 < layersStartIndex[l +2]; n2++ )
			{
				ArcSP newArc = std::make_shared<Arc>( arcs.size(), Arc::Sign::ANY, nodes[n].get(), nodes[n2].get(), false, false );
				arcs.push_back( newArc );
				nodes[n]->addChild( newArc.get() );
				nodes[n2]->addParent( newArc.get() );
			}
		}
	}

//---correct the ids
	for( uint n = 0; n < nodes.size(); n++ )
		nodes[n]->setId( n );
	for( uint a = 0; a < arcs.size(); a++ )
		arcs[a]->setId( a );

//---create bias
	for( uint n = 0; n < nodes.size(); n++ )
		nodes[n]->createBias( arcs );
}

void NeuralWeb::swapInputLayer( RandomEngine& randomEngine )
{
//---make a bak with the parents of every input node
	std::vector<std::vector<Arc*>> childrenTemp;
	for( uint i = 0; i < inputLayer.size(); i++ )
		childrenTemp.push_back( inputLayer[i]->getChildren() );

//---make a random vector of indexes
	std::vector<uint> indexVector( inputLayer.size() );
	std::iota( indexVector.begin(), indexVector.end(), 0 );
	std::shuffle( indexVector.begin(), indexVector.end(), randomEngine );

//---reassign the parents to input nodes according to the indexes vector
	for( uint i = 0; i < inputLayer.size(); i++ )
	{
		std::vector<Arc*> currentChildren = childrenTemp[ indexVector[i] ];
		for( uint c = 0; c < currentChildren.size(); c++ )
			currentChildren[c]->setParent( inputLayer[i].get() );

		inputLayer[i]->setChildren( currentChildren ); 
	}
}