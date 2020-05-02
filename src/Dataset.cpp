#include "Dataset.hpp"
#include <algorithm> //shuffle in makeKFold(), makeStratifiedKFold(), makeSingleFold(), next_permutation in makeAllCombinations()


////////////////////////////////////////////////////////////////////////////* DATASET */////////////////////////////////////////////////////////////////////////////

//======================================================================= INSTANCE WEIGHTING =============================================================================
void Dataset::dissimilaritySubsets()
{
	for( uint s = 0; s < foldsTraining.size(); s++ )
	{
		foldsTraining[s]->makeSimilarityMatrix(); //for training folds, similarity relative to self
		foldsTraining[s]->dissimilarityAsInstanceWeights();

		foldsTest[s]->makeSimilarityMatrix( foldsTraining[s].get() ); //for test folds, similarity relative to corresponding train fold
		foldsTest[s]->relativeDissimilarityAsInstanceWeights();
	}
}
//======================================================================= end of INSTANCE WEIGHTING =============================================================================




//======================================================================= DATA SPLITS =============================================================================
void Dataset::leaveOneOut()
{	
	foldsTraining.clear();
	foldsTest.clear();

	for( uint c = 0; c < inputs.size(); c++ )
	{
	//---test split is a single instance
		std::vector<std::vector<double>> testInputs = { inputs[c] };
		std::vector<double> testOutputs = { outputs[c] };
		std::vector<double> testInstanceWeights = { instanceWeights[c] };

	//---train split is all the instances but the test instance
		std::vector<std::vector<double>> trainInputs = inputs;
		std::vector<double> trainOutputs = outputs;
		std::vector<double> trainInstanceWeights = instanceWeights;

		trainInputs.erase( trainInputs.begin() + c );
		trainOutputs.erase( trainOutputs.begin() + c );
		trainInstanceWeights.erase( trainInstanceWeights.begin() + c );

	//---create the child Datasets
		foldsTraining.push_back( std::make_shared<Dataset>( trainInputs, trainOutputs, trainInstanceWeights, classThreshold ) );
		foldsTest.push_back( std::make_shared<Dataset>( testInputs, testOutputs, testInstanceWeights, classThreshold ) );
		foldsTraining.back()->makeBoolOutputs();
		foldsTraining.back()->normalizeInstanceWeights();
		foldsTest.back()->makeBoolOutputs();
		foldsTest.back()->normalizeInstanceWeights();
	}
	initKFold( foldsTraining.size() );
}

void Dataset::makeSingleFold( RandomEngine& randomEngine, uint instanceNum )
{
	foldsTraining.clear();
	foldsTest.clear();

//---split all the cases in 2 vectors based on their output
	std::vector<uint> indexVector0;
	std::vector<uint> indexVector1;

	for( uint o = 0; o < outputs.size(); o++ )
	{
		if( outputs[o] >= classThreshold )
			indexVector1.push_back(o);
		else
			indexVector0.push_back(o);
	}

//---calculate the number of instances to pick from each of the 2 vectors, to keep the proportion
	uint instanceNum0 = static_cast<uint>( (float)indexVector0.size() / inputs.size() * instanceNum + 0.5 );
	uint instanceNum1 = instanceNum - instanceNum0;

//---randomize order
	std::shuffle( indexVector0.begin(), indexVector0.end(), randomEngine );
	std::shuffle( indexVector1.begin(), indexVector1.end(), randomEngine );

//---pop the last cases indexes and put them in the test index vector
	std::vector<uint> indexVectorTest; 
	for( uint i = 0; i < instanceNum0; i++ )
	{
		indexVectorTest.push_back( indexVector0.back() );
		indexVector0.pop_back();
	}
	for( uint i = 0; i < instanceNum1; i++ )
	{
		indexVectorTest.push_back( indexVector1.back() );
		indexVector1.pop_back();
	}

//---concat the remaining indexes (train), shuffle and, if size of test < instanceNum, add some more cases
	indexVector0.insert( indexVector0.end(), indexVector1.begin(), indexVector1.end() );
	std::shuffle( indexVector0.begin(), indexVector0.end(), randomEngine );

	while( indexVectorTest.size() < instanceNum )
	{
		indexVectorTest.push_back( indexVector0.back() );
		indexVector0.pop_back();
	}

	//for printing the test case indexes 
	/*std::cout << "single fold test indexes:";
	for( uint c = 0; c < indexVectorTest.size(); c++ )
		std::cout << indexVectorTest[c] << ",";
	std::cout << "\n ";*/

//---split inputs, outptus and instance weights into train and test according to the two index vectors
	std::vector<std::vector<double>> inputsTraining;
	std::vector<double> outputsTraining;
	std::vector<double> instanceWeightsTraining;

	std::vector<std::vector<double>> inputsTest;
	std::vector<double> outputsTest;
	std::vector<double> instanceWeightsTest;

	for( uint d = 0; d < inputs.size(); d++ )
	{
		if( std::find( indexVectorTest.begin(), indexVectorTest.end(), d ) != indexVectorTest.end() )
		{
			inputsTest.push_back( inputs[d] );
			outputsTest.push_back( outputs[d] );
			instanceWeightsTest.push_back( instanceWeights[d] );
		}
		else
		{
			inputsTraining.push_back( inputs[d] );
			outputsTraining.push_back( outputs[d] );
			instanceWeightsTraining.push_back( instanceWeights[d] );
		}
	}

//---create child train and test Dataset
	foldsTraining.push_back( std::make_shared<Dataset>( inputsTraining, outputsTraining, instanceWeightsTraining, classThreshold ) );
	foldsTraining[0]->makeBoolOutputs();
	foldsTraining[0]->normalizeInstanceWeights();
	foldsTest.push_back( std::make_shared<Dataset>( inputsTest, outputsTest, instanceWeightsTest, classThreshold ) );
	foldsTest[0]->makeBoolOutputs();
	foldsTest[0]->normalizeInstanceWeights();

//---if previously weighted by similarity, weight the subsets by similarity
	if( similarityMatrix.size() > 0 )
		dissimilaritySubsets();
}

void Dataset::makeStratifiedKFold( RandomEngine& randomEngine, uint k )
{
	//may be unified with makeSingleFold() because they have some code in common

	foldsTraining.clear();
	foldsTest.clear();
	initKFold( k );

//---split all the cases in 2 vectors based on their output
	std::vector<uint> indexVector0;
	std::vector<uint> indexVector1;

	for( uint o = 0; o < outputs.size(); o++ )
	{
		if( outputs[o] >= classThreshold )
			indexVector1.push_back(o);
		else
			indexVector0.push_back(o);
	}

//---calculate the number of instances to pick from each of the 2 vectors, to keep the proportion
	uint instancesPerFold0 = indexVector0.size() / k;
	uint instancesPerFold1 = indexVector1.size() / k;

//---randomize order
	std::shuffle( indexVector0.begin(), indexVector0.end(), randomEngine );
	std::shuffle( indexVector1.begin(), indexVector1.end(), randomEngine );

//---for each fold, pop the last cases indexes for the test index vector
	std::vector<std::vector<uint>> indexVectorFolds = std::vector<std::vector<uint>>( k, std::vector<uint>(0) ); 
	for( uint f = 0; f < k; f++ )
	{
		for( uint i = 0; i < instancesPerFold0; i++ )
		{
			indexVectorFolds[f].push_back( indexVector0.back() );
			indexVector0.pop_back();
		}
		for( uint i = 0; i < instancesPerFold1; i++ )
		{
			indexVectorFolds[f].push_back( indexVector1.back() );
			indexVector1.pop_back();
		}
	}

//---concat the remaining indexes, shuffle and distribute them between the folds while the remaining number is greater than the number of folds. If total instance num not divisible by k, some instances are not used
	indexVector0.insert( indexVector0.end(), indexVector1.begin(), indexVector1.end() );
	std::shuffle( indexVector0.begin(), indexVector0.end(), randomEngine );

	while( indexVector0.size() >= k )
	{
		for( uint f = 0; f < k; f++ )
		{
			indexVectorFolds[f].push_back( indexVector0.back() );
			indexVector0.pop_back();
		}
	}
	indexVectorFolds.back().insert( indexVectorFolds.back().end(), indexVector0.begin(), indexVector0.end() ); //add the remaining instances to the end

//---sort the whole dataset inputs, outputs and instance weights by the index vectors
	std::vector<std::vector<double>> newInputs;
	std::vector<double> newOutputs;
	std::vector<double> newInstanceWeights;

	for( uint f = 0; f < indexVectorFolds.size(); f++ )
	{
		//std::cout << "fold " << f << ": ";
		for( uint i = 0; i < indexVectorFolds[f].size(); i++ )
		{
			//std::cout << indexVectorFolds[f][i] << ",";
			newInputs.push_back( inputs[ indexVectorFolds[f][i] ] );
			newOutputs.push_back( outputs[ indexVectorFolds[f][i] ] );
			newInstanceWeights.push_back( instanceWeights[ indexVectorFolds[f][i] ] );
		}
		//std::cout << "\n";
	}
	inputs = newInputs;
	outputs = newOutputs;
	instanceWeights = newInstanceWeights;
	makeBoolOutputs();


	for( uint f = 0; f < k; f++ ) //for each fold
	{
	//---split inputs, outptus and instance weights into train and test according to each fold's index range
		std::vector<std::vector<double>> inputsTest;
		std::vector<double> outputsTest;
		std::vector<double> instanceWeightsTest;

		std::vector<std::vector<double>> inputsTraining;
		std::vector<double> outputsTraining;
		std::vector<double> instanceWeightsTraining;

		uint startIntex = f * foldSize;

		for( uint d = 0; d < inputs.size(); d++ )
		{
			if( d < startIntex || d >= ( startIntex + foldSize ) )
			{
				inputsTraining.push_back( inputs[d] );
				outputsTraining.push_back( outputs[d] );
				instanceWeightsTraining.push_back( instanceWeights[d] );
			}
			else
			{
				inputsTest.push_back( inputs[d] );
				outputsTest.push_back( outputs[d] );
				instanceWeightsTest.push_back( instanceWeights[d] );
			}
		}
	//---create child train and test Dataset
		foldsTraining.push_back( std::make_shared<Dataset>( inputsTraining, outputsTraining, instanceWeightsTraining, classThreshold ) );
		foldsTraining.back()->makeBoolOutputs();
		foldsTraining.back()->normalizeInstanceWeights();
		foldsTest.push_back( std::make_shared<Dataset>( inputsTest, outputsTest, instanceWeightsTest, classThreshold ) );
		foldsTest.back()->makeBoolOutputs();
		foldsTest.back()->normalizeInstanceWeights();
	}

//---if previously weighted by similarity, weight the subsets by similarity
	if( similarityMatrix.size() > 0 )
		dissimilaritySubsets();
}
//======================================================================= end of DATA SPLITS =============================================================================