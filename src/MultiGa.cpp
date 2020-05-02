#include "MultiGa.hpp"
#include <algorithm> //shuffle in mix()

MultiGa::MultiGa( std::vector<GeneticAlgorithmSP> gas, const Parser& parser, RandomnessHandler& randomnessHandler )
: params( parser )
, gas(gas)
{
//---create distributions for migration (extraction of nets and reassigment )
    for( uint d = 0; d < INDEX_MGA_DISTRIBUTION_NUM; d++ )
        distributions.emplace_back( DISTRIBUTIONTYPE_UNIFORM, std::vector<double>( { 0.0, 1.0 } ), randomnessHandler.getValidSeed( INDEX_RANDOMNESS_MAINRE_RUN ) );
;}


//==================================== *GET SET* ============================================
NeuralWebSP MultiGa::getBestNet()
{
    //compare the best nets of every population 
    NeuralWebSP bestNet = gas[0]->getBestNet();
    //double bestLoss = bestNet->getMetrics( INDEX_SET_TRAIN )->lossW;
    double bestLoss = bestNet->getTrainMetrics().getMember( INDEX_METRIC_LOSS_W );

    for( uint g = 1; g < gas.size(); g++ )
    {
       NeuralWebSP tempBestNet = gas[g]->getBestNet();
       //if( tempBestNet->getMetrics( INDEX_SET_TRAIN )->lossW < bestLoss )
       if( tempBestNet->getTrainMetrics().getMember( INDEX_METRIC_LOSS_W ) < bestLoss )
       {
            bestNet = tempBestNet;
            //bestLoss = tempBestNet->getMetrics( INDEX_SET_TRAIN )->lossW;
            bestLoss = tempBestNet->getTrainMetrics().getMember( INDEX_METRIC_LOSS_W );
       }
    }
    return bestNet;
}
//==================================== *end of GET SET* ============================================




// ======================================================================================================= *API* =======================================================================================================
void MultiGa::train( const Dataset* dataset, uint generationsPerMix, uint mixNum )
{
    for( uint m = 0; m < mixNum; m++ ) //for every mix round
    {
        for( uint g = 0; g < gas.size(); g++ ) //for every population
            gas[g]->train( dataset->getTrainingFold()->getInputs(), dataset->getTrainingFold()->getOutputs(), dataset->getTrainingFold()->getInstanceWeights(), generationsPerMix, true ); //train the given number of generations between migration events
        if( m < mixNum - 1 ) //no mix in the last round
            mix(); //migration
    }
}

void MultiGa::trainAndTrack( const Dataset* dataset, uint generationsPerMix, uint mixNum )
{
    for( uint m = 0; m < mixNum; m++ ) //for every mix round
    {
        for( uint ge = 0; ge < generationsPerMix; ge++ ) //training generation by generation in order to track historical info
        {
            for( uint g = 0; g < gas.size(); g++ ) //for every population
                gas[g]->train( dataset->getTrainingFold()->getInputs(), dataset->getTrainingFold()->getOutputs(), dataset->getTrainingFold()->getInstanceWeights(), 1, ge <= 0 ); //train a single generation and only evaluate all if it is the first generation (may be better optimized)
            
            historicalTrack.addRecord( getBestNet(), *dataset ); //save historical record
        }
        if( m < mixNum - 1) //no mix in the last round
            mix(); //migration
    }
}
// ======================================================================================================= *end of API*  =======================================================================================================




//============================================================================================================= *PRIVATE MGA STEPS* =======================================================================================================
void MultiGa::mix()
{
    std::vector<NeuralWebSP> tempNets;
//---1 get mixNets nets from each ga pop without replacement
    for( uint g = 0; g < gas.size(); g++ )
    {
        for( uint n = 0; n < params.mixNets; n++ )
        {
            uint rndNetIndex = distributions[ INDEX_MGA_DISTRIBUTION_MIXOUT ].sample() * gas[g]->getPopSize(); //scale to variable pop size from [0,1] random uniform
            tempNets.push_back( gas[g]->popNet( rndNetIndex ) );
        }
    }
//---2 assign the selected nets to new ga
    std::shuffle( tempNets.begin(), tempNets.end(), *distributions[ INDEX_MGA_DISTRIBUTION_MIXIN_SHUFFLE ].getRandomEngine() ); //use the random engine from distribution to shuffle
    for( uint g = 0; g < gas.size(); g++ )
    {
        for( uint n = 0; n < params.mixNets; n++ )
            gas[g]->addNet( tempNets[ g * params.mixNets + n ] ); //reassign the emigrated nets to populations
    }
}
//============================================================================================================= *end of PRIVATE MGA STEPS* =======================================================================================================