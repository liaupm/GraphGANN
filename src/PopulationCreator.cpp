#include "PopulationCreator.hpp"
#include "NeuralWeb.hpp" //newPopulation, NeuralWeb::randomize()

void PopulationCreator::init( NeuralWebSP xBaseNet, RandomnessHandler& randomnessHandler, float xScaleMax, float xScaleMin )
{
	baseNet = xBaseNet;

	iniDistributionWeights.clear();
	iniDistributionSigns.clear();
	iniDistributionScales.clear();

	for( uint n = 0; n < baseNet->getNodes().size(); n++ ) //create distributions for nodes
	{
		iniDistributionWeights.emplace_back( DISTRIBUTIONTYPE_UNIFORM, std::vector<double>( { 0.0f, 1.0f } ), randomnessHandler.getValidSeed( INDEX_RANDOMNESS_MAINRE_INI ) );
		iniDistributionScales.emplace_back( DISTRIBUTIONTYPE_UNIFORM, std::vector<double>( { xScaleMin, xScaleMax } ), randomnessHandler.getValidSeed( INDEX_RANDOMNESS_MAINRE_INI ) );
	}
	for( uint a = 0; a < baseNet->getArcs().size(); a++ ) //create distributions for arcs (for those with restrictec sign, they are unnecessary)
		iniDistributionSigns.emplace_back( DISTRIBUTIONTYPE_UNIFORM, std::vector<double>( { 0.0f, 1.0f } ), randomnessHandler.getValidSeed( INDEX_RANDOMNESS_MAINRE_INI ) );
}

void PopulationCreator::reseed( RandomnessHandler& randomnessHandler )
{
	for( uint n = 0; n < baseNet->getNodes().size(); n++ )
	{
		iniDistributionWeights[n].setSeed( randomnessHandler.getValidSeed( INDEX_RANDOMNESS_MAINRE_INI ) );
		iniDistributionScales[n].setSeed( randomnessHandler.getValidSeed( INDEX_RANDOMNESS_MAINRE_INI ) );
	}
	for( uint a = 0; a < baseNet->getArcs().size(); a++ )
		iniDistributionSigns[a].setSeed( randomnessHandler.getValidSeed( INDEX_RANDOMNESS_MAINRE_INI ) );
}

std::vector<NeuralWebSP> PopulationCreator::createPopulation( uint populationSize )
{
    std::vector<NeuralWebSP> netPopulation;
    for( uint n = 0; n < populationSize; n++ )
    {
        netPopulation.push_back( std::make_shared<NeuralWeb>( baseNet.get() ) ); //deep copy reference net
        netPopulation.back()->randomize( *this ); //randomize the last net using the distributions from this
    }
    return netPopulation;
}
