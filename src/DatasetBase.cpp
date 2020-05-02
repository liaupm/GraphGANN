#include "DatasetBase.hpp"
#include <math.h> //pow in generateBoolInputs()
#include <algorithm> //shuffle in shuffle()


////////////////////////////////////////////////////////////////////////////* STATIC */////////////////////////////////////////////////////////////////////////////

//=================================================================== MISC ========================================================================================
std::vector<std::vector<double>> DatasetBase::sparseData( const std::vector<std::vector<int>>& indexes, uint elementNum, bool bInverted )
{
    std::vector<std::vector<double>> inputCombinations; //sparse representation
    for( uint d = 0; d < indexes.size(); d++ )
    {
        std::vector<double> temp( elementNum, bInverted ? 1.0 : 0.0 ); //if inverted, default is 1 and indexes mean 0s
        
        for( uint i = 0; i < indexes[d].size(); i++ )
            temp[ indexes[d][i] ] = bInverted ? 0.0 : 1.0; //switch values at the given indexes
        
        inputCombinations.push_back( temp );
    }
    return inputCombinations;
}

std::vector<std::vector<double>> DatasetBase::makeAllCombinations( uint n, uint k, bool bInverted )
{
    std::vector<std::vector<double>> combinations; 
    std::vector<double> tempCombi( k, bInverted ? 0.0 : 1.0 ); //if inverted, k = 0s
    tempCombi.resize( n, bInverted ? 1.0 : 0.0 );  //complete with the other class until size n 
    do 
    {
        combinations.push_back( tempCombi ); //store the combination
    } 
    while( std::next_permutation( tempCombi.begin(), tempCombi.end() ) ); //permute
    return combinations;
}


//=================================================================== WEIGHTING BY SIMILARITY ========================================================================================
double DatasetBase::calculatePairSimilarity( const std::vector<std::vector<double>>& inputs1, const std::vector<std::vector<double>>& inputs2, double output1, double output2, uint index1, uint index2 )
{
    double similarity = 0.0;
//---count the number of equal inputs
    for( uint i = 0; i < inputs1[index1].size(); i++ )
    {
        if( inputs1[index1][i] == inputs2[index2][i] )
            similarity += 1.0;
    }
//---similarity = fraction of equal inputs if same output or fraction of different inputs if different output
    if( output1 == output2 )
        return similarity / inputs1[index1].size();
    return ( inputs1[index1].size() - similarity ) / inputs1[index1].size();
}

std::vector<std::vector<double>> DatasetBase::makeSimilarityMatrix( const std::vector<std::vector<double>>& inputs1, const std::vector<std::vector<double>>& inputs2, const std::vector<double>& outputs1, const std::vector<double>& outputs2 )
{
    std::vector<std::vector<double>> similarityMatrix;

    for( uint c1 = 0; c1 < inputs1.size(); c1++ ) //main dataset
    {
        std::vector<double> tempRow( inputs2.size() );
        for( uint c2 = 0; c2 < inputs2.size(); c2++ ) //reference dataset
            tempRow[c2] = calculatePairSimilarity( inputs1, inputs2, outputs1[c1], outputs2[c2], c1, c2 ); //fill matrix with pair similarities

        similarityMatrix.push_back( tempRow );
    }
    return similarityMatrix;
}

std::vector<double> DatasetBase::sumSimilarityMatrix( const std::vector<std::vector<double>>& similarityMatrix )
{
    std::vector<double> totalSimilarities( similarityMatrix.size(), 0.0 );

    for( uint c1 = 0; c1 < similarityMatrix.size(); c1++ )
    {
        for( uint c2 = 0; c2 < similarityMatrix[c1].size(); c2++ )
            totalSimilarities[c1] += similarityMatrix[c1][c2]; //add up similarities for each case of the main dataset

        totalSimilarities[c1] = 1.0 - totalSimilarities[c1] / similarityMatrix[c1].size(); //normalize to [0,1] and reverse it to make dissimilarity score instead = uniqueness of the case 
    }
    return totalSimilarities;
}



////////////////////////////////////////////////////////////////////////////* DATASET */////////////////////////////////////////////////////////////////////////////

//======================================================================= GENERATE =============================================================================
void DatasetBase::generateOutputs( const NeuralWebBase* net )
{
	outputs.clear();
	for( uint c = 0; c < inputs.size(); c++ )
		outputs.push_back( net->predict( inputs, c ) ); //predict outputs for every set of inputs
	makeBoolOutputs();
}

void DatasetBase::filterInstancesEqual(const  DatasetBase* filter )
{
    const std::vector<std::vector<double>> filterInputs = filter->getInputs();

    uint c1 = 0;
    while( c1 < inputs.size() ) //no for loop because erasing is performed inside
    {
        bool found = false; //whether current instance found in the filter dataset
        for( uint c2 = 0; c2 < filterInputs.size(); c2++ ) //iterate filter dataset
        {
            if( DatasetBase::calculatePairSimilarity( inputs, filterInputs, 0, 0, c1, c2 ) >= DEFAULT_DATASET_FILTER_EQUAL_THRESHOLD ) //if similarity (inputs only) greater than threshold, they are the same
            {
                //remove the repeated instance
                inputs.erase( inputs.begin() + c1 );
                if( outputs.size() > c1 )
                    outputs.erase( outputs.begin() + c1 );
                if( instanceWeights.size() > c1 )
                    instanceWeights.erase( instanceWeights.begin() + c1 );
                found = true; //update control var
                break;
            }
        }
        if( ! found ) //only advance in the iteration if no erase() 
            c1++;
    }
}

void DatasetBase::filterInstancesSuperset( const DatasetBase* filter, uint inputValue, uint classValue )
{
    const std::vector<std::vector<double>> filterInputs = filter->getInputs();
    const std::vector<double> filterOutputs = filter->getOutputs();

    uint c1 = 0;
    while( c1 < inputs.size() ) //no for loop because erasing is performed inside
    {
        bool found = false; //whether current instance found in the filter dataset
        for( uint c2 = 0; c2 < filterInputs.size(); c2++ ) //iterate filter dataset
        {
            if( ( filterOutputs[c2] >= classThreshold && classValue == 0 ) || ( filterOutputs[c2] < classThreshold && classValue == 1 ) ) //if the class does not match the one of interest, skip instance
                continue;

            bool bDifferent = false; //whether a difference between c1 instance and c2 filter instance found
            uint filterCounter = 0; //counter of inputs different form the one of interest in the filter c2 instance
            for( uint i = 0; i < inputs[c1].size(); i++ ) //iterate inputs
            {
                if( ( filterInputs[c2][i] >= 0.5 && inputValue == 0 ) || ( filterInputs[c2][i] < 0.5 && inputValue == 1 ) ) //count inputs not of interest in the filter instance
                    filterCounter ++;

                if( ( inputs[c1][i] >= 0.5 && filterInputs[c2][i] < 0.5 && inputValue == 0 ) || ( inputs[c1][i] < 0.5 && filterInputs[c2][i] >= 0.5 && inputValue == 1 ) ) //if the input makes c1 instance no superset of of c2 filter instance
                {
                    bDifferent = true;
                    break;
                }
            }
            if( ! bDifferent && filterCounter < inputs[c1].size() ) //if c1 is superset of c2 with output = classValue and c2 is not a trivial instance that any one would be a superset of
            {
                //remove the repeated instance
                inputs.erase( inputs.begin() + c1 );
                if( outputs.size() > c1 )
                    outputs.erase( outputs.begin() + c1 );
                if( instanceWeights.size() > c1 )
                    instanceWeights.erase( instanceWeights.begin() + c1 );
                found = true;
                break;
            }
        }
        if( ! found ) //only advance in the iteration if no erase() 
            c1++;
    }
}
//======================================================================= end of GENERATE =============================================================================





//======================================================================= INSTANCE WEIGHTING =============================================================================
void DatasetBase::weightInstances( double instanceWeightByOutput, double instanceWeightByInput, bool bSimilarityAsWeights )
{
	instanceWeights.clear();

    if( ! bSimilarityAsWeights )
    {
        for( uint d = 0; d < inputs.size(); d++ )
        {
    //---1-by output(class)
            double weight = outputs[d] < classThreshold ? instanceWeightByOutput : 1 - instanceWeightByOutput;

    //---2-by input(number of 0s)
            //count 0s
            uint counter0 = 0;
            for( uint i = 0; i < inputs[d].size(); i++ )
            {
                if( inputs[d][i] < 0.5 )
                    counter0++;
            }
            weight *= std::exp( -instanceWeightByInput * counter0 );
            instanceWeights.push_back( weight );
        }  
    }
    else
    {
   //---3 by similarity
        makeSimilarityMatrix( nullptr );
        dissimilarityAsInstanceWeights();     
    }
//---4-normalize
	normalizeInstanceWeights();
}

void DatasetBase::normalizeInstanceWeights()
{
//---add up instance weights
	double totalInstanceWeight = 0.0;
	for( uint d = 0; d < instanceWeights.size(); d++ )
		totalInstanceWeight += instanceWeights[d];
//---divide instance weights by total to keep them in [0,1]
	for( uint d = 0; d < instanceWeights.size(); d++ )
		instanceWeights[d] /= totalInstanceWeight;
}

void DatasetBase::makeSimilarityMatrix( const DatasetBase* referenceDataset )
{
    if( referenceDataset == nullptr ) //use self as reference dataset. Typically for training dataset
    {
        similarityMatrix = makeSimilarityMatrix( inputs, inputs, outputs, outputs );
        dissimilaritySum = sumSimilarityMatrix( similarityMatrix );
    }
    else //use the provided reference dataset. Typically for val and test datasets
    {
        similarityMatrixRelative = makeSimilarityMatrix( inputs, referenceDataset->getInputs(), outputs, referenceDataset->getBoolOutputs() );
        dissimilaritySumRelative = sumSimilarityMatrix( similarityMatrixRelative );
    }
}
//======================================================================= end of INSTANCE WEIGHTING =============================================================================




//=======================================================================* DATA SPLITS *=============================================================================
void DatasetBase::shuffle( RandomEngine& randomEngine )
{
//---create a vector of indexes and shuffle it
	std::vector<uint> indexVector( inputs.size() );
	std::iota( indexVector.begin(), indexVector.end(), 0 );
	std::shuffle( indexVector.begin(), indexVector.end(), randomEngine );
//---sort inputs, outputs and instance weights according to it
	sortByIndexVector( indexVector );
}

void DatasetBase::sortByIndexVector( const std::vector<uint>& indexVector )
{
	std::vector<std::vector<double>> newInputs;
	std::vector<double> newOutputs;
	std::vector<double> newInstanceWeights;
//---create new inputs, outputs and instance weights vectors following the order in index vector
	for( uint i = 0; i < indexVector.size(); i++ )
	{
		newInputs.push_back( inputs[ indexVector[i] ] );
		newOutputs.push_back( outputs[ indexVector[i] ] );
		newInstanceWeights.push_back( instanceWeights[ indexVector[i] ] );
	}
//---replace the old vectors
	inputs.swap( newInputs );
	outputs.swap( newOutputs );
	instanceWeights.swap( newInstanceWeights );

//---recalculate bool outputs. Similarity weighting structures cannot be automatically recalculated because they may need a reference dataset, so must be explicitly done if necessary
	makeBoolOutputs();
}
//=======================================================================* end of DATA SPLITS *=============================================================================






//=======================================================================* PRIVATE MISC *=============================================================================
void DatasetBase::makeBoolOutputs()
{
	boolOutputs.clear();
    for( uint o = 0; o < outputs.size(); o++ )
    {
        if( outputs[o] >= classThreshold )
            boolOutputs.push_back(1.0);
        else
            boolOutputs.push_back(0.0);
    }
}
//=======================================================================* end of PRIVATE MISC *=============================================================================