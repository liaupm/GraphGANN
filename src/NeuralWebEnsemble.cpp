#include "NeuralWebEnsemble.hpp"

double NeuralWebEnsemble::predict( const std::vector<std::vector<double>>& inputs, uint index ) const
{
	double totalPrediction = 0.0;
	double totalWeight = 0.0;
	for( uint n = 0; n < memberNets.size(); n++ )
	{
		if( ensembleParams.qualityCriterion == NO_METRIC ) //no criterion = all nets included and no weighting
		{
			totalWeight += 1.0;
			totalPrediction += memberNets[n]->predict( inputs, index );
		}
		else
		{
			double qualityValue = memberNets[n]->getSavedMetrics().getMember( ensembleParams.qualityCriterion );

			if( ensembleParams.qualityCriterion < METRIC_LOSS_NUM ) //criterion is loss = higher is worse. Reverse weighting
			{
				if(  qualityValue <= ensembleParams.qualityThreshold )
				{
					totalWeight += ensembleParams.bWeighted ? ( ensembleParams.qualityThreshold - qualityValue ) : 1.0;
					totalPrediction += ( ensembleParams.bWeighted ? ( ensembleParams.qualityThreshold - qualityValue ) : 1.0  ) * memberNets[n]->predict( inputs, index );
				}
			}
			else //criterion is accuracy = higher is better. Direct weighting (fitness is not considered as a valid criterion)
			{
				if( qualityValue >= ensembleParams.qualityThreshold )
				{
					totalWeight += ensembleParams.bWeighted ? qualityValue : 1.0;
					totalPrediction += ( ensembleParams.bWeighted ? qualityValue : 1.0 ) * memberNets[n]->predict( inputs, index );
				}
			}
		}
	}
	return totalWeight > 0.0 ? totalPrediction / totalWeight : -1.0; //return average or -1 if no member fulfilled the criterion (avoids division by 0)
}

Metrics NeuralWebEnsemble::averageMetrics( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights, uint setIndex )
{
//--evaluate all the members and add up their metrics
	Metrics resultMetrics(0.0);
	for( uint n = 0; n < memberNets.size(); n++ )
	{
		memberNets[n]->initReflection();
		memberNets[n]->evaluateWeighted( inputs, outputs, instanceWeights, setIndex );
		resultMetrics.add( &memberNets[n]->getTestMetrics() );
	}
//---divide the total metrics by the number of member nets
	resultMetrics.scale( 1.0 / memberNets.size() );
	return resultMetrics;
}
