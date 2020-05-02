#include "Emitter.hpp"
#include <memory>

//////////////////////////////////////////////////////////////////////////* STATIC *///////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> Emitter::metricNames = { "loss", "lossW", "lossOutW", "accuracy", "accuracyW", "accuracyOutW" }; //not safe: must match order in Metrics and defines.hpp indexes
std::vector<std::string> Emitter::setNames = { "train", "val", "fair" }; //not safe

std::string Emitter::resizeStr( const std::string& originalStr, uint targetSize )
{
	std::string resultStr( originalStr );
	while( resultStr.size() < targetSize )
		resultStr.insert( resultStr.begin(), ' ' );
	return resultStr;
}

bool Emitter::printNetwork( const NeuralWeb* net, uint64_t options, const std::string& netFileName, const std::string& activationFileName, const std::string& metricsFileName )
{
	const std::vector<NodeSP>& netNodes = net->getNodes();
	const std::vector<ArcSP>& netArcs = net->getArcs();

//---main structure and weights file
	std::ofstream netFile ( netFileName );
	if( ! netFile.is_open() )
		return false;

	if( GET_FLAG( options, FLAG_NET_TRAINED ) ) //trained net: biases included
	{
		for( uint a = 0; a < netArcs.size(); a++ )
		{
			if( netArcs[a]->getParent() != nullptr ) //is actual arc (not bias)
				netFile << netArcs[a]->getParent()->getName() << EMITTER_NET_SEPARATOR << netArcs[a]->getWeight() << EMITTER_NET_SEPARATOR << netArcs[a]->getChild()->getName();
			else //bias
				netFile << "bias" << EMITTER_NET_SEPARATOR << netArcs[a]->getWeight() << EMITTER_NET_SEPARATOR << netArcs[a]->getChild()->getName();

			if( a < netArcs.size() - 1 )
				netFile << "\n";
		}
	}
	else //untrained net: biased not saved
	{
		for( uint a = 0; a < netArcs.size(); a++ )
		{
			std::string signStr = "";
			if( netArcs[a]->getSign() == Arc::Sign::NEG )
				signStr = PARSER_NET_ARCS_SIGN_NEG;
			else if( netArcs[a]->getSign() == Arc::Sign::ANY )
				signStr = PARSER_NET_ARCS_SIGN_ANY;

			if( netArcs[a]->getParent() != nullptr )
			{
				if( a >= 1 )
					netFile << "\n";
				netFile << netArcs[a]->getParent()->getName() << EMITTER_NET_SEPARATOR << signStr << EMITTER_NET_SEPARATOR << netArcs[a]->getChild()->getName();
			}
		}
	}
	netFile.close();

	if( GET_FLAG( options, FLAG_NET_TRAINED ) ) //only trained nets have scales and metrics files
	{
	//---scales or activation file
		std::ofstream netFile ( activationFileName );
		if( ! netFile.is_open() )
			return false;

		for( uint n = 0; n < netNodes.size(); n++ )
		{
			if( netNodes[n]->getBTrainableScale() ) //scale of untrainable nodes = input layer is always 1 -> don't print
				netFile << netNodes[n]->getName() << EMITTER_NET_SEPARATOR << netNodes[n]->getScales()[0];

			if( n < netNodes.size() - 1 )
				netFile << "\n";
		}
		netFile.close();
	
	//---metrics file
		std::ofstream metricsFile ( metricsFileName );
		if( ! metricsFile.is_open() )
			return false;
		//train
		const Metrics& netTrainMetrics = net->getTrainMetrics();
		for( uint m = 0; m < metricNames.size(); m++ )
			metricsFile << setNames[INDEX_SET_TRAIN] + "_" + metricNames[m] << PARSER_EQUAL << netTrainMetrics.getMember( m ) << "\n";
		//val
		const Metrics& netValMetrics = net->getTestMetrics();
		for( uint m = 0; m < metricNames.size(); m++ )
		{
			metricsFile << setNames[INDEX_SET_VAL] + "_"  + metricNames[m] << PARSER_EQUAL << netValMetrics.getMember( m );
			if( m < metricNames.size() - 1 )
				metricsFile << "\n";
		}
		metricsFile.close();
	}
    return true;
}

bool Emitter::printHistorical( const HistoricalTrack& historicalTrack, const std::string& fileName )
{
   std::ofstream historicalFile ( fileName );
    if( ! historicalFile.is_open() )
        return false;

//---header
    historicalFile << "generation";

    for( uint s = INDEX_SET_TRAIN; s <= INDEX_SET_VAL; s++ )
    {
    	for( uint m = 0; m < metricNames.size(); m++ )
    		historicalFile << PARSER_DATA_SEPARATOR << setNames[s] + "_" + metricNames[m]; //colum names = set + metric
    }

//---values
    const std::vector<HistoricalTrack::Record>& records = historicalTrack.getRecords();

    for( uint r = 0; r < records.size(); r++ )
    {
		historicalFile << "\n" << r; //print generation num
    	for( uint s = INDEX_SET_TRAIN; s <= INDEX_SET_VAL; s++ ) //sets
    	{
    		const Metrics& metrics = records[r].metrics[s];
    		for( uint m = 0; m < metricNames.size(); m++ ) //metrics
	    	{
	    		historicalFile << PARSER_DATA_SEPARATOR << metrics.getMember( m );
	    	}
    	}
    }
    historicalFile.close();
    return true;
}

//===================================================================== OUT FILES ===========================================================================================
bool Emitter::printDataset( const Dataset& correctDataset, const Dataset& predictedDataset, uint64_t options, const std::string& fileName, double outputLBound, double outputUBound, double classThreshold )
{
	std::ofstream dataFile ( fileName );
	if( ! dataFile.is_open() )
		return false;

//---header
	dataFile << header[0]; //"output"

	if( GET_FLAG( options, FLAG_DATA_PRED ) ) //output predictions
	{
		dataFile << "," << header[0] << "_predictedBool";
		dataFile << "," << header[0] << "_predictedReal";
	}
	if( GET_FLAG( options, FLAG_DATA_WEIGHT ) ) //instance weights
		dataFile << ",weights";

	if( GET_FLAG( options, FLAG_DATA_COUNT ) ) //number of 0 inputs
		dataFile << ",zeros_num";

	for( uint h = 1; h < header.size(); h++ ) //inputs
		dataFile << "," << header[h];
	dataFile << "\n";

//---data
	for( uint d = 0; d < correctDataset.getInputs().size(); d++ )
	{
	//---filter by predicted real ouput
		if( GET_FLAG( options, FLAG_DATA_FILTER ) && ( predictedDataset.getOutputs()[d] < outputLBound || predictedDataset.getOutputs()[d] > outputUBound ) )
			continue;

	//---output
		dataFile << correctDataset.getOutputs()[d];

		if( GET_FLAG( options, FLAG_DATA_PRED ) ) //predictions
		{
			dataFile << PARSER_DATA_SEPARATOR << ( predictedDataset.getOutputs()[d] >= classThreshold ? 1 : 0 ); //binarized prediction
			dataFile << PARSER_DATA_SEPARATOR << predictedDataset.getOutputs()[d]; //real prediction
		}
		if( GET_FLAG( options, FLAG_DATA_WEIGHT ) ) //weights
			dataFile << PARSER_DATA_SEPARATOR << correctDataset.getInstanceWeights()[d];

	//---inputs
		if( GET_FLAG( options, FLAG_DATA_COUNT ) ) //number of 0 in the inputs
		{
			uint counter0 = 0;
			for( uint i = 0; i < correctDataset.getInputs()[d].size(); i++ )
			{
				if( correctDataset.getInputs()[d][i] < 0.5 )
					counter0++;
			}
			dataFile << PARSER_DATA_SEPARATOR << counter0;
		}
		for( uint i = 0; i < correctDataset.getInputs()[d].size(); i++ ) //input values
			dataFile << PARSER_DATA_SEPARATOR << correctDataset.getInputs()[d][i];

		if( d < correctDataset.getInputs().size() - 1 )
			dataFile << "\n";
	}
	dataFile.close();
	return true;
}

bool Emitter::printExternalMetrics( const Metrics* metrics, const std::string& prefix )
{
//---construct the message
    std::string message = "";
    for( uint m = 0; m < metricNames.size(); m++ )
    {
        message = message + prefix + metricNames[m] + ": " + std::to_string( metrics->getMember( m ) ) + "  |   ";
        //message = message + prefix + metricNames[m] + ": " + std::to_string( metrics.getMember( m ) ) + ( m < metricNames.size() - 1  ?  "  |   " : "\n" );
    }
    message = message + "\n";
//---cout
    std::cout << message;
//---result file
    resultFile->open( OUTFILE_RESULT, std::ios_base::out | std::ios_base::app );
    if( ! resultFile->is_open() )
        return false;
    (*resultFile) << message; 
    resultFile->close();
    return true;
}

bool Emitter::printMeanKfoldValues( uint k )
{
//---construct the message
    std::string messageTrain = "";
    std::string messageVal = "";
    std::string messageFair = "";
    for( uint m = 0; m < metricNames.size(); m++ )
    {
        messageTrain = messageTrain + resizeStr( setNames[INDEX_SET_TRAIN], EMITTER_SET_NAME_FIXED_SIZE ) + " " + metricNames[m] + ": " + std::to_string( totalMetrics[INDEX_SET_TRAIN].getMember( m ) / k ) + ( m < metricNames.size() - 1  ?  "  |   " : "\n" );
        messageVal = messageVal + resizeStr( setNames[INDEX_SET_VAL], EMITTER_SET_NAME_FIXED_SIZE ) + " " + metricNames[m] + ": " + std::to_string( totalMetrics[INDEX_SET_VAL].getMember( m ) / k ) + ( m < metricNames.size() - 1  ?  "  |   " : "\n" );
        messageFair = messageFair + resizeStr( setNames[INDEX_SET_TEST], EMITTER_SET_NAME_FIXED_SIZE ) + " " + metricNames[m] + ": " + std::to_string( totalMetrics[INDEX_SET_TEST].getMember( m ) / k ) + ( m < metricNames.size() - 1  ?  "  |   " : "\n" );
    }
//---cout
    std::cout << "\n *FINAL AVERAGED RESULT*\n" << messageTrain << messageVal << messageFair;
   
//---result file
    resultFile->open( OUTFILE_RESULT, std::ios_base::app );
    if( ! resultFile->is_open() )
        return false;
    (*resultFile) << "\n\n *FINAL AVERAGED RESULT*\n" << messageTrain << messageVal << messageFair;
    resultFile->close();
    return true;
}

void Emitter::printAll( NeuralWebBase* currentNet, const Dataset* dataset, int setIndex, uint currentFold, bool bSaveNet, bool bSavePredictions, bool bEnsemble, const std::string& sufix )
{
	if( bEnsemble )
		bSaveNet = false;

    int correctedSetIndex = std::min( setIndex, 1 ); //correction needed because dataset has only 2 sets: train and test (val or fair depending on dataset)
    if( setIndex == INDEX_SET_TRAIN )
    	printMessage( "---net " + std::to_string( currentFold ) );

    printExternalMetrics( &currentNet->getReflectedMetrics( correctedSetIndex ), ( bEnsemble ? "ensemble " : "" ) + resizeStr( setNames[setIndex], EMITTER_SET_NAME_FIXED_SIZE ) + " " );
    addMetrics( &currentNet->getReflectedMetrics( correctedSetIndex ), setIndex );

    if( bSaveNet )
        Emitter::printNetwork( static_cast<NeuralWeb*>(currentNet), FLAG_NET_TRAINED, MAKE_FILENAME( OUTFILE_NET_W + sufix, currentFold ), MAKE_FILENAME( OUTFILE_NET_ACTI + sufix, currentFold ), MAKE_FILENAME( OUTFILE_NET_METRICS + sufix, currentFold ) );

    if( bSavePredictions )
    {
        Dataset datasetPred = Dataset( dataset->getReflectedFold( correctedSetIndex )->getInputs(), {}, {} );
        datasetPred.generateOutputs( currentNet );
        printDataset( dataset->getReflectedFold( correctedSetIndex ).get(), datasetPred, FLAG_DATA_ALL, MAKE_FILENAME( OUTFILE_DATAPRED +  ( bEnsemble ? std::string( "_ensemble " ) : std::string( "" ) ) + ( "_" + setNames[setIndex] ) + sufix, currentFold ) );  
    }
}