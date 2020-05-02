
#include "GeneticAlgorithm.hpp"
#include <algorithm> //std::min, std::max for clamping scales and weights in mmxCrossScales() and mmxCrossWeights()

GeneticAlgorithm::GeneticAlgorithm( const std::vector<NeuralWebSP>& iniPopulation, const Parser& parser, RandomnessHandler& randomnessHandler )
: mmxParams( parser )
, gaParams( parser )
, scaleMin( parser.getRealParam( "minActivation" ) )
, scaleMax( parser.getRealParam( "maxActivation" ) )

, currentPopulation(iniPopulation)
, rouletteCumulatedProbs( iniPopulation.size(), 0.f )
, selectedParents( parser.getIntParam( "parentNum" ), nullptr )
, children( parser.getIntParam( "crossNum" ) * parser.getIntParam( "outspringNum" ), nullptr )

, totalMetrics(0.0)
{
//---create distributions for crossover and mutation
    for( uint d = 0; d < INDEX_GA_DISTRIBUTION_NUM; d++ )
    {
        switch( d )
        {
        	case INDEX_GA_DISTRIBUTION_MUT_WEIGHT_AMOUNT: //fixed bounds between 0 and amount
                distributions.emplace_back( DISTRIBUTIONTYPE_UNIFORM, std::vector<double>( { 0.0, gaParams.mutationAmountWeights } ), randomnessHandler.getValidSeed( INDEX_RANDOMNESS_MAINRE_RUN ) );
                break;
            case INDEX_GA_DISTRIBUTION_MUT_SCALE_AMOUNT: //fixed bounds between -amount and amount
                distributions.emplace_back( DISTRIBUTIONTYPE_UNIFORM, std::vector<double>( { - gaParams.mutationAmountScales, gaParams.mutationAmountScales } ), randomnessHandler.getValidSeed( INDEX_RANDOMNESS_MAINRE_RUN ) );
                break;
            default: //if variable bounds, scaling the [0,1] value when sampling
                distributions.emplace_back( DISTRIBUTIONTYPE_UNIFORM, std::vector<double>( { 0.0, 1.0 } ), randomnessHandler.getValidSeed( INDEX_RANDOMNESS_MAINRE_RUN ) );
        }   
    }
}


//==================================== *GET SET* ============================================
NeuralWebSP GeneticAlgorithm::getBestNet()
{
	NeuralWebSP bestNet = currentPopulation[0];
	double bestFitness = currentPopulation[0]->getTrainMetrics().fitness;

	for( uint n = 1; n < currentPopulation.size(); n++ )
	{
		//if( currentPopulation[n]->getMetrics( INDEX_SET_TRAIN )->fitness < bestFitness )
		if( currentPopulation[n]->getTrainMetrics().fitness < bestFitness )
		{
			bestNet = currentPopulation[n];
			bestFitness = currentPopulation[n]->getTrainMetrics().fitness;
			//bestFitness = currentPopulation[n]->getMetrics( INDEX_SET_TRAIN )->fitness;
		}
	}
	return bestNet;
}

NeuralWebSP GeneticAlgorithm::popNet( uint netIndex )
{
	NeuralWebSP net = currentPopulation[netIndex];
	currentPopulation[netIndex] = currentPopulation.back();
	currentPopulation.pop_back();
	return net;
}
//==================================== *end of GET SET* ============================================




// ======================================================================================================= *API* =======================================================================================================
void GeneticAlgorithm::train( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights, uint generationNum, bool evaluateAll )
{
	if( evaluateAll == true )
		evaluateWholePopulation( inputs, outputs, instanceWeights );

	//after first evaluation of whole population, only children are evaluated
	for( uint g = 0; g < generationNum; g++ )
	{
		selectParents();
		mmxCross( inputs, outputs, instanceWeights ); //crossover + mutation
		death(); //as children have been added to the population in mmxCross, they are elligible for death if the worst (no replacement)

		totalMetrics.accumulatePopulationFitness( currentPopulation ); //add up the fitness of the whole population for calculating selection probs
	}
}

void GeneticAlgorithm::evaluateWholePopulation( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights )
{ 
	totalMetrics.reset( 0.0 );
	for( uint n = 0; n < currentPopulation.size(); n++ ) 
		currentPopulation[n]->evaluateWeighted( inputs, outputs, instanceWeights ); 
	totalMetrics.accumulatePopulationFitness( currentPopulation );
}
// ======================================================================================================= *end of API*  =======================================================================================================






//============================================================================================================= *PRIVATE GA STEPS* =======================================================================================================
void GeneticAlgorithm::selectParents()
{
//---create prob roulette
	std::fill( rouletteCumulatedProbs.begin(), rouletteCumulatedProbs.end() - 1, 0.0f );
	rouletteCumulatedProbs.back() = 1.0f;
	
	for( uint p = rouletteCumulatedProbs.size() -1; p > 0; p-- )
		rouletteCumulatedProbs[p -1] = rouletteCumulatedProbs[p] - ( totalMetrics.fitness - currentPopulation[p]->getTrainMetrics().fitness ) / totalMetrics.fitness; //as fitness = loss = higher worse, reverse it by simetry and normalize to [0, 1]
		//rouletteCumulatedProbs[p -1] = rouletteCumulatedProbs[p] - ( totalMetrics.fitness - currentPopulation[p]->getMetrics( INDEX_SET_TRAIN )->fitness ) / totalMetrics.fitness;
//---select parents randomly
	for( uint p = 0; p < gaParams.parentNum; p++ )
	{
		float rnd = distributions[ INDEX_GA_DISTRIBUTION_PARENTS_SELECTOR ].sample(); //sample in [0, 1]
		for( uint r = 0; r < rouletteCumulatedProbs.size(); r++ )
		{
			if( rnd <= rouletteCumulatedProbs[r] )
			{
				selectedParents[p] = currentPopulation[r];
				break;
			}
		}
	}
}

void GeneticAlgorithm::mmxCross( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights )
{
//---create copies of any of the parents for the children
	for( uint c = 0; c < gaParams.outspringNum; c++ )
		children[c] = std::make_shared<NeuralWeb>( selectedParents[0].get() ); 

//---change their scales and weights by crossover and mutation
	mmxCrossScales();
	mmxCrossWeights();

//evaluate children and add them to the population
	for( uint c = 0; c < children.size(); c++ )
	{
		children[c]->evaluateWeighted( inputs, outputs, instanceWeights, INDEX_SET_TRAIN );
		currentPopulation.push_back( children[c] );
	}
}

void GeneticAlgorithm::mmxCrossScales()
{
	uint nodeNum = selectedParents[0]->getNodes().size(); //all parents have the same nodes, so any parent is ok
	for( uint n = 0; n < nodeNum; n++ )
	{
		if( selectedParents[0]->getNodes()[n]->getBTrainableScale() == false ) //input nodes have untrainable scale. All parents have the same nodes, so any parent is ok
			continue;

	//---find min and max
		double minValue = selectedParents[0]->getNodes()[n]->getScales()[0];
		double maxValue = selectedParents[0]->getNodes()[n]->getScales()[0];

		for( uint p = 1; p < selectedParents.size(); p++)
		{
			if( selectedParents[p]->getNodes()[n]->getScales()[0] < minValue )
				minValue = selectedParents[p]->getNodes()[n]->getScales()[0];

			else if( selectedParents[p]->getNodes()[n]->getScales()[0] > maxValue )
				maxValue = selectedParents[p]->getNodes()[n]->getScales()[0];
		}

	//---scale the values to [0, 1] (required by the exploration-exploitation function calculation)
		double scalingSize = scaleMax - scaleMin;
		double scalingStart = scaleMin;

		minValue = ( minValue - scalingStart ) / scalingSize;
		maxValue = ( maxValue - scalingStart ) / scalingSize;

	//---calculate exploration-exploitation function
		double diversity = maxValue - minValue;
		double exploration = diversity < mmxParams.c ? mmxParams.a + diversity * mmxParams.tempCalculation1 : ( diversity - mmxParams.c ) * mmxParams.tempCalculation2; //cross interval narrowing amount
	
	//---calculate crrosover intervals
		double lBound = std::max( 0.0,  minValue + exploration );
		double uBound = std::min( 1.0,  maxValue - exploration );

	//---get children and reescale to the original interval
		double firstChild = distributions[INDEX_GA_DISTRIBUTION_CROSS_SCALE].sampleScaled( lBound, uBound ); //scale the [0, 1] sampled value to the variable [lBound, uBound]
		children[0]->getNodes()[n]->setScale( firstChild * scalingSize + scalingStart ); //reescale
		children[1]->getNodes()[n]->setScale( ( lBound + uBound - firstChild ) * scalingSize + scalingStart ); //children are simetrical to each other

	//---mutation
		if( gaParams.mutationProbActivation > 0.0 )
		{
			for( uint c = 0; c < gaParams.outspringNum; c++ )
			{
				float rnd = distributions[ INDEX_GA_DISTRIBUTION_MUT_SCALE_OCCURENCE ].sample(); //get random uniform in [0,1]
				if( rnd < gaParams.mutationProbActivation ) //decide if mutation occur
				{
					double newValue = children[c]->getNodes()[n]->getScales()[0] + distributions[ INDEX_GA_DISTRIBUTION_MUT_SCALE_AMOUNT ].sample(); //new scale = old scale + random value in [-amoun, amount]
					children[c]->getNodes()[n]->setScale( std::min( std::max( newValue, scaleMin ), scaleMax ) ); //clamp and replace
				}
			}
		}
	}
}

void GeneticAlgorithm::mmxCrossWeights()
{
	const std::vector<NodeSP>& nodes = selectedParents[0]->getNodes(); //all parents have the same nodes, so any parent is ok
	for( uint n = 0; n < nodes.size(); n++ ) //it is done in a node-wise fashion because this is how mutation works
	{
		std::vector<Arc*>& parentArcs = nodes[n]->getParentsEditable();
		if( parentArcs.size() <= 0 ) //there are not incoming are i.e no weights (input layer)
			continue;

		for( uint a = 0; a < parentArcs.size(); a++ ) //iterate the incoming arcs of the node
		{
		//---find min and max
			double minValue = selectedParents[0]->getNodes()[n]->getParents()[a]->getWeight();
			double maxValue = selectedParents[0]->getNodes()[n]->getParents()[a]->getWeight();
			
			for( uint p = 1; p < selectedParents.size(); p++ )
			{
				if( selectedParents[p]->getNodes()[n]->getParents()[a]->getWeight() < minValue )
					minValue = selectedParents[p]->getNodes()[n]->getParents()[a]->getWeight();

				else if( selectedParents[p]->getNodes()[n]->getParents()[a]->getWeight() > maxValue )
					maxValue = selectedParents[p]->getNodes()[n]->getParents()[a]->getWeight();
			}

		//---scale the values to [0, 1] (required by the exploration-exploitation function calculation)
			//if all weights add up to 1 in abs, their max value is 1 and min value is -1.0 (when sign = ANY)
			double scalingSize = 1.0;
			double scalingStart = 0.0; //if sign = POS, min value = 0

			switch( parentArcs[a]->getSign() )
			{
				case Arc::Sign::POS:
					break;
				case Arc::Sign::NEG:
					scalingStart = -1.0;
					break;
				default:
					scalingSize = 2.0;
					scalingStart = -1.0;
					break;
			}
			minValue = ( minValue - scalingStart ) / scalingSize;
			maxValue = ( maxValue - scalingStart ) / scalingSize;

		//---calculate exploration-exploitation function
			double diversity = maxValue - minValue;
			double exploration = diversity < mmxParams.c ? mmxParams.a + diversity * mmxParams.tempCalculation1 : ( diversity - mmxParams.c ) * mmxParams.tempCalculation2; //cross interval narrowing amount
		
		//---calculate crrosover intervals
			double lBound = std::max( minValue + exploration, 0.0 );
			double uBound = std::min( maxValue - exploration, 1.0 );

		//---get children and reescale to the original interval
			double firstChild = distributions[ INDEX_GA_DISTRIBUTION_CROSS_WEIGHT ].sampleScaled( lBound, uBound ); //scale the [0, 1] sampled value to the variable [lBound, uBound]
			children[0]->setWeight( firstChild * scalingSize + scalingStart, n, a ); //reescale
			children[1]->setWeight( ( lBound + uBound - firstChild ) * scalingSize + scalingStart, n, a ); //children are simetrical to each other
		}

	//---mutation
		if( gaParams.mutationProbWeights > 0.0 )
		{
			for( uint c = 0; c < gaParams.outspringNum; c++ )
			{
				float rnd = distributions[ INDEX_GA_DISTRIBUTION_MUT_WEIGHT_OCCURENCE ].sample(); //get random uniform in [0,1]
				if( rnd < gaParams.mutationProbWeights ) //decide if mutation
				{
					uint firstArcIndex = distributions[ INDEX_GA_DISTRIBUTION_MUT_WEIGHT_ARC ].sample() * parentArcs.size(); //randomly select first incoming arc
					uint secondArcIndex = distributions[ INDEX_GA_DISTRIBUTION_MUT_WEIGHT_ARC ].sample() * ( parentArcs.size() - 1 ); //randomly select first incoming arc
					if( firstArcIndex == secondArcIndex ) 
						secondArcIndex = parentArcs.size() - 1; //this way first and second arcs are always different while all the arcs having the same prob

					float change = distributions[ INDEX_GA_DISTRIBUTION_MUT_WEIGHT_AMOUNT ].sample(); //sample change amount and apply with different sign to the arcs
					children[c]->getNodes()[n]->getParents()[firstArcIndex]->changeWeight( change );
					children[c]->getNodes()[n]->getParents()[secondArcIndex]->changeWeight( -change );
				}
			}
		}
	}
//---normalize weights
	for( uint c = 0; c < children.size(); c++ )
		children[c]->normalizeWeights();
}

void GeneticAlgorithm::death()
{
	for( uint i = 0; i < gaParams.deathNum; i++ )
	{
	//---find the net with higher fitness = worst
		uint worstIndex = 0; 
		double worstFitness = currentPopulation[0]->getTrainMetrics().fitness;
		//double worstFitness = currentPopulation[0]->getMetrics( INDEX_SET_TRAIN )->fitness;

		for( uint n = 1; n < currentPopulation.size(); n++ )
		{
			//if( currentPopulation[n]->getMetrics( INDEX_SET_TRAIN )->fitness > worstFitness )
			if( currentPopulation[n]->getTrainMetrics().fitness > worstFitness )
			{
				worstIndex = n;
				//worstFitness = currentPopulation[n]->getMetrics( INDEX_SET_TRAIN )->fitness;
				worstFitness = currentPopulation[n]->getTrainMetrics().fitness;
			}
		}
	//---remove the worst net
		popNet( worstIndex ); //pop the net from current population
	}
}
//============================================================================================================= *end of PRIVATE GA STEPS* =======================================================================================================