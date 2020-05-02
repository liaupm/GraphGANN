#ifndef HISTORICAL_TRACK_HPP
#define HISTORICAL_TRACK_HPP

#include "defines.hpp"
#include "Metrics.hpp" //std::vector<Metrics> metrics
#include "NeuralWeb.hpp" //NeuralWeb* bestNet in Record, addRecord(), getHistoricalBestNet()
#include "Dataset.hpp" //addRecord()

#include <vector> // std::vector<NeuralWeb::Metrics> metrics in Record,  std::vector<Record> records
#include <memory> //Record::NeuralWebSP bestNet

///registry of the historical evolution of the population (best net per generation and metrics )
class HistoricalTrack
{
    public:
        ///saved information about a given generation
        struct Record
        {
            std::vector<Metrics> metrics; //sets of metrics. Typically 2: train and val (test metrics are not evaluated during training)
            NeuralWebSP bestNet; //best net in the generation (in training set)

            Record( NeuralWebSP bestNet = nullptr ) : bestNet(bestNet) {;}
        };

        HistoricalTrack() {;}
        virtual ~HistoricalTrack() { /*for( uint r = 0; r < records.size(); r++ ) delete records[r].bestNet;*/ }

    //---get
        const std::vector<Record>& getRecords() const { return records; }

    //---API
        void addRecord( NeuralWebSP net, const Dataset& dataset ); //create and add a new record given the best net of the generation and a dataset with single split or k-fold
        //return the best historical net in the given metric for the given set. Minimum generation num can be given. Also returns the generation. 
        NeuralWebSP getHistoricalBestNet( uint& bestGeneration, uint minGeneration = HTRACK_MIN_GENERATION, uint setIndex = INDEX_SET_VAL, uint metricIndex = INDEX_METRIC_LOSS_W ); 


    private:
        std::vector<Record> records; //index match generation num
};

#endif //HISTORICAL_TRACK_HPP
