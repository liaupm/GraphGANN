#include "HistoricalTrack.hpp"


void HistoricalTrack::addRecord( NeuralWebSP net, const Dataset& dataset )
{
    net->setBSaved( true ); //prevents the saved net from being deleted by the GA
    Record newRecord( net );

    newRecord.metrics.push_back( Metrics( net->getTrainMetrics() ) ); //save train metrics
    net->evaluateWeighted( dataset.getTestFold()->getInputs(), dataset.getTestFold()->getBoolOutputs(), dataset.getTestFold()->getInstanceWeights(), INDEX_SET_VAL ); //evaluate with val set
    newRecord.metrics.push_back( Metrics( net->getTestMetrics() ) ); //save val metrics

    records.push_back( newRecord );
}

NeuralWebSP HistoricalTrack::getHistoricalBestNet( uint& bestGeneration, uint minGeneration, uint setIndex, uint metricIndex )
{
    bestGeneration = minGeneration;
    for( uint r = minGeneration + 1; r < records.size(); r++ )
    {
    	double candidateMetricValue = records[r].metrics[INDEX_SET_VAL].getMember( metricIndex );
    	double trainMetricValue = records[r].metrics[INDEX_SET_TRAIN].getMember( metricIndex );

    	if( metricIndex < METRIC_LOSS_NUM ) //criterion is loss = higher is worse
    	{
    		//additional conditions for avoiding 1) good metric in val by chance and 2) overfitting
	        if( candidateMetricValue <= records[bestGeneration].metrics[INDEX_SET_VAL].getMember( metricIndex ) && candidateMetricValue > trainMetricValue && candidateMetricValue / trainMetricValue <= HTRACK_MAX_METRIC_RATIO )
	            bestGeneration = r;
    	}
    	else //criterion is accuracy = higher is better (fitness is not considered as a valid criterion)
    	{
			if( candidateMetricValue >= records[bestGeneration].metrics[INDEX_SET_VAL].getMember( metricIndex ) && candidateMetricValue < trainMetricValue )
				bestGeneration = r;
    	}
    }
    return records[bestGeneration].bestNet;
}