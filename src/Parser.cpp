#include "Parser.hpp"

#include <algorithm> //std::count for arcs separator
#include <sstream> //parsing lines
#include <fstream> //input files

//static
std::map<std::string, FunctionBase::FunctionType> Parser::functionTypeNM = { { "satExponential", FunctionBase::FunctionType::SAT_EXPONENTIAL }, { "sigmoid", FunctionBase::FunctionType::SIGMOID } };
std::map<std::string, int> Parser::metricNM ={ { "none", NO_METRIC }, { "loss", INDEX_METRIC_LOSS }, { "lossW", INDEX_METRIC_LOSS_W }, { "lossOutW", INDEX_METRIC_LOSS_OUTW }, { "acc", INDEX_METRIC_ACC }, { "accW", INDEX_METRIC_ACC_W }, { "accOutW", INDEX_METRIC_ACC_OUTW } };


bool Parser::parseOptions( const std::string& fileName )
{
	std::ifstream optionsFile ( fileName );
	if( ! optionsFile.is_open() )
		return false;

	std::string line; 
	std::string paramName;
	std::string value;

	while( std::getline( optionsFile, line ) )
	{
		if( line.find( PARSER_EQUAL ) == std::string::npos || line.at(0) == PARSER_COMMENT ) //skip lines that are not param assigments or are all coment
			continue;

		std::stringstream lineStream( line ); //convert line to stream for further parsing

	//---get param name and value and trim spaces
		std::getline( lineStream, paramName, PARSER_EQUAL ); //name comes before assigment char
		paramName.erase( std::remove( paramName.begin(), paramName.end(), ' ' ), paramName.end() );
		std::getline( lineStream, value, PARSER_COMMENT ); //value comes before comment char or eol
		value.erase( std::remove( value.begin(), value.end(), ' ' ), value.end() );

	//---store the param in the right params map according to its type
		if( intParams.find( paramName ) != intParams.end() )
			intParams[ paramName ] = std::stoi( value );
		else if( realParams.find( paramName ) != realParams.end() )
			realParams[ paramName ] = std::stod( value );
		else if( strParams.find( paramName ) != strParams.end() )
			strParams[ paramName ] = value;
		else
			std::cout << "!!!  wrong param \"" << paramName << "\" !!!\n"; //error msg if the param is not in any of the param maps
	}
	optionsFile.close();

//---convert str params into int with name map //TODO
	intParams["ensembleCriterion"] = metricNM.find( strParams["ensembleCriterion"] )->second;
	intParams["bestNetCriterion"] = metricNM.find( strParams["bestNetCriterion"] )->second;
	return true;
}

bool Parser::parseNetwork( uint64_t options, const std::string& fileNameNet, const std::string& fileNameActi, const std::string& fileNameMetrics )
{
	nodes.clear();
	arcs.clear();
	metrics.clear();
	//header.clear(); //do not do this (all the parsed nets are supposed to have the same structure and order )

//---parse  weights file
	std::ifstream netFile ( fileNameNet );
	if( ! netFile.is_open() )
		return false;

	std::string line; 
	while( std::getline( netFile, line ) )
	{
		std::string parentName;
		std::string childName;

		std::string signStr;
		Arc::Sign sign = Arc::Sign::POS;
		std::string weightStr;
		double weight = INI_ARC_WEIGHT;
		
	//---get the info in each sentence: parent and child node and sign (untrained)/weight (trained)
		std::stringstream lineStream( line );
		std::getline( lineStream, parentName, PARSER_NET_ARCS_SEPARATOR );

		if( GET_FLAG( options, FLAG_NET_TRAINED ) ) //if trained, parse weight
		{
			std::getline( lineStream, weightStr, PARSER_NET_ARCS_SEPARATOR );
			weight = std::stod( weightStr );
		}
		else //if not trained, parse sign
		{
			if( std::count( line.begin(), line.end(), PARSER_NET_ARCS_SEPARATOR ) >= 2 ) //if single separator, positive sign, do nothing
			{
				std::getline( lineStream, signStr, PARSER_NET_ARCS_SEPARATOR );

				if( signStr == PARSER_NET_ARCS_SIGN_NEG )
					sign = Arc::Sign::NEG;
				else if( signStr == PARSER_NET_ARCS_SIGN_ANY )
					sign = Arc::Sign::ANY;
			}
		}
		std::getline( lineStream, childName, PARSER_NET_ARCS_SEPARATOR );

	//---create nodes if not created yet
		bool foundParent = false;
		bool foundChild = false;
		NodeSP parent = nullptr;
		NodeSP child = nullptr;

		//search for the nodes in the previously created ones
		for( uint n = 0; n < nodes.size() && ( foundParent == false || foundChild == false ); n++ )
		{
			if( nodes[n]->getName() == parentName )
			{
				foundParent = true;
				parent = nodes[n];
			}
			if( nodes[n]->getName() == childName )
			{
				foundChild = true;
				child = nodes[n];
			}
		}
		//if not found, add them
		if( foundParent == false )
		{
			parent = std::make_shared<Node>( nodes.size(), parentName, functionTypeNM.find( strParams.find( "functionType" )->second )->second );
			nodes.push_back( parent );
		}
		if( foundChild == false )
		{
			child = std::make_shared<Node>( nodes.size(), childName, functionTypeNM.find( strParams.find( "functionType" )->second )->second );
			nodes.push_back( child );
		}

	//---create arc
		arcs.push_back( std::make_shared<Arc>( arcs.size(), sign, parent.get(), child.get(), foundParent, foundChild ) ); //problem: trained nets will be given positive sign for all arcs. Always parse both trained and untrained and transfer param values
		if( GET_FLAG( options, FLAG_NET_TRAINED ) )
			arcs.back()->setWeight( weight );

		parent->addChild( arcs.back().get() );
		child->addParent( arcs.back().get() );
	}

	if( ! GET_FLAG( options, FLAG_NET_TRAINED ) ) //create all the bias at the end so they are the last arcs
	{
		for( uint n = 0; n < nodes.size(); n++ )
			nodes[n]->createBias( arcs );
	}
	netFile.close();

	if( ! GET_FLAG( options, FLAG_NET_TRAINED ) ) //untrained nets have no sacales and metrics files, so it's done
    	return true;


//---parse scale or activation file
	std::ifstream actiFile ( fileNameActi );
	if( ! actiFile.is_open() )
		return false;

	while( std::getline( actiFile, line ) )
	{
		std::string nodeName;
		std::string actiStr;
	//---get the info in each line: node name and scale value
		std::stringstream lineStream( line );
		std::getline( lineStream, nodeName, PARSER_NET_SCALES_SEPARATOR );
		std::getline( lineStream, actiStr, PARSER_NET_SCALES_SEPARATOR );
	//---find the node and set its scale
		for( uint n = 0; n < nodes.size(); n++ )
		{
			if( nodes[n]->getName() == nodeName )
			{
				nodes[n]->setScale( std::stod( actiStr) );
				break;
			}
		}
	}
	actiFile.close();

//---parse metrics file
	std::ifstream metricsFile ( fileNameMetrics );
	if( ! metricsFile.is_open() )
		return false;

	std::string metricName;
	std::string metricValue;
	while( std::getline( metricsFile, line ) )
	{
	//---get metric values and store them in the metrics vector
		std::stringstream lineStream( line );
		std::getline( lineStream, metricName, PARSER_EQUAL );
		std::getline( lineStream, metricValue, PARSER_COMMENT );
		metricValue.erase( std::remove( metricValue.begin(), metricValue.end(), ' ' ), metricValue.end() );
		metrics.push_back( std::stod( metricValue ) );
	}
	metricsFile.close();
    return true;
}

bool Parser::parseDataset( uint64_t options, const std::string& fileName )
///should be always called after parsing net and setting the header
{
	inputs.clear();
	outputs.clear();
	instanceWeights.clear();
	originalDataHeader.clear();

	std::ifstream dataFile ( fileName );
	if( ! dataFile.is_open() )
		return false;
	
	std::string line; 

//---load the header
	//different input order than input layer in the net. Requires reordering
	std::getline( dataFile, line );
	std::stringstream lineStream( line );
	std::string element;

	while( std::getline( lineStream, element, PARSER_DATA_SEPARATOR ) )
		originalDataHeader.push_back( element );

//---change input order to match the input layer of the network
	std::vector<uint> inputOrder;
	for( uint dh = 1; dh < originalDataHeader.size(); dh++ )
	{
		bool found = false;
		for( uint h = 1; h < header.size(); h++ )
		{
			if( originalDataHeader[dh] == header[h] )
			{
				inputOrder.push_back( h -1 );
				found = true;
				break;
			}
		}
		if( found == false )
		{
			std::cout << "\"" << originalDataHeader[dh] << "\" not found in the net inputs\n"; //error msg if an input from the dataset is not in the net's input layer
			return false;
		}
	}

//---load the values
	while( std::getline( dataFile, line ) )
	{
		std::stringstream lineStream( line );
		std::string element;

	//---load output
		std::getline( lineStream, element, PARSER_DATA_SEPARATOR );
		outputs.push_back( std::stod( element ) );

	 //---if weighted flag, load the weight
		if( GET_FLAG( options, FLAG_DATA_WEIGHT ) )
		{
			std::getline( lineStream, element, PARSER_DATA_SEPARATOR );
			instanceWeights.push_back( std::stod( element ) );
		}

	//---load inputs, following the inputOrder
		std::vector<double> temp( inputOrder.size(), DEFAULT_PARSER_DATA_INPUTVALUE );
		uint counter = 0;
		while( std::getline( lineStream, element, PARSER_DATA_SEPARATOR ) )
		{
			temp[ inputOrder[counter] ] = std::stod( element );
			counter++;
		}
		inputs.push_back( temp );
	}
	dataFile.close();
	std::cout << "dataset size: " << inputs.size() << "\n";

//---equal instance weights created when no weights in the dataset
	if( ! GET_FLAG( options, FLAG_DATA_WEIGHT ) )
	{
		double equalWeight = 1.0 / inputs.size();
		for( uint d = 0; d < inputs.size(); d++ )
			instanceWeights.push_back(  equalWeight );
	}
	return true;
}