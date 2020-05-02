#include "NeuralWebBase.hpp"


double NeuralWebBase::evaluateWeighted( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights, uint setIndex )
{
    double equalW = 1.0 / inputs.size(); //weight for unweighted metrics i.e. all the cases have the same weight
    Metrics& currentMetrics = setIndex == INDEX_SET_TRAIN ? trainMetrics : testMetrics;
    currentMetrics.reset( 0.0 );

    for ( uint d = 0; d < inputs.size(); d++ )
    {
    	double predicted = predict( inputs, d );
        
    //calculate loss
        double baseLoss = lossFunction->evaluate( predicted, outputs[d] );
        currentMetrics.changeMember( equalW * baseLoss, INDEX_METRIC_LOSS );
        currentMetrics.changeMember( instanceWeights[d] * baseLoss, INDEX_METRIC_LOSS_W );

    //calculate accuracy
        if( ( predicted >= params.classThreshold && outputs[d] >= params.classThreshold ) || ( predicted < params.classThreshold && outputs[d] < params.classThreshold ) )
        {
            currentMetrics.changeMember( equalW, INDEX_METRIC_ACC );
            currentMetrics.changeMember( instanceWeights[d], INDEX_METRIC_ACC_W );
        }
    }
    return currentMetrics.getMember( INDEX_METRIC_LOSS_W ); 
}

