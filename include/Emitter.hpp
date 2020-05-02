#ifndef EMITTER_HPP
#define EMITTER_HPP

#include "defines.hpp"
#include "NeuralWeb.hpp" //std::vector<Metrics> totalMetrics
#include "Dataset.hpp" //printDataset()
#include "HistoricalTrack.hpp" //printHistorical()

#include <vector> //std::vector<std::string> metricNames, std::vector<std::string> setNames, std::vector<std::string> header, std::vector<Metrics> totalMetrics, many methods args
#include <string> //std::vector<std::string> metricNames, std::vector<std::string> setNames, std::vector<std::string> header, many methods args
#include <memory> //std::shared_pointer<std::ofstream> resultFile
#include <fstream> //output files, std::ofstream* resultFile


class Emitter
{
    public:
    //---static
        static std::vector<std::string> metricNames;
        static std::vector<std::string> setNames;
        //out files
        //save a net to a file with given options. If untrained, a single main file; if trained, additional scales and metrics files
        static bool printNetwork( const NeuralWeb* net, uint64_t options = DEFAULT_EMITTER_FLAG_NET, const std::string& netFileName = MAKE_FILENAME( OUTFILE_NET_W, 0 ), const std::string& activationFileName = MAKE_FILENAME( OUTFILE_NET_ACTI, 0 ), const std::string& metricsFileName = MAKE_FILENAME( OUTFILE_NET_METRICS, 0 ) );
        static bool printHistorical( const HistoricalTrack& historicalTrack, const std::string& fileName = MAKE_FILENAME( OUTFILE_HISTORICAL, 0 ) );
        static std::string resizeStr( const std::string& originalStr, uint targetSize ); //add spaces to str until target size. Used for equal-size-fields aligned output
 
        Emitter() : totalMetrics( SET_NUM, Metrics( 0.0, nullptr ) ), resultFile( std::make_shared<std::ofstream>( OUTFILE_RESULT ) ) { resultFile->close(); }
        virtual ~Emitter() { resultFile->close(); }

    //---get 
        inline std::ofstream& getResultFile() { return *resultFile; } 
        const Metrics& getTotalMetrics( uint setIndex ) const { return totalMetrics[setIndex]; }

    //---set
        inline void setHeader( const std::vector<std::string>& xHeader ) { header = xHeader; }

    //---API
        //metrics
        void addMetrics( const Metrics* metricsToAdd, uint setIndex ) { totalMetrics[setIndex].add( metricsToAdd ); } //add metrics to total metrics
        void resetTotalMetrics() { for( uint m = 0; m < totalMetrics.size(); m++ ) totalMetrics[m].reset( 0.0 ); }
        //out files
        //print dataset with given options (binarize, include predictions, count 0 inputs... ) filtered by output range of interest to keep file small. Not static because requires header
        bool printDataset( const Dataset& correctDataset, const Dataset& predictedDataset, uint64_t options = DEFAULT_EMITTER_FLAG_DATA, const std::string& fileName = MAKE_FILENAME( OUTFILE_DATAPRED, 0 ), double outputLBound = DEFAULT_EMITTER_DATA_LBOUND, double outputUBound = DEFAULT_EMITTER_DATA_UBOUND, double classThreshold = 0.5 );
        bool printExternalMetrics( const Metrics* metrics, const std::string& prefix = DEFAULT_EMITTER_METRICS_PREFIX ); //print metrics passed as arg (intead of it own total metrics) to the resultFile
        bool printMeanKfoldValues( uint k ); //make the average of metrics by dividing total metrics by k and print them
        inline bool printMessage( const std::string& message ) { (*resultFile).open( OUTFILE_RESULT, std::ios_base::app ); if( ! (*resultFile).is_open() ) return false; (*resultFile) << message << "\n"; (*resultFile).close(); return true; } //print given msg to resultFile
        void printAll( NeuralWebBase* currentNet, const Dataset* dataset, int setIndex, uint currentFold, bool bSaveNet, bool bSavePredictions, bool bEnsemble = false, const std::string& sufix = "" ); //evaluate and print metrics of best net (to resultFile and console) and (optional) save net and predictions

    private:
        std::vector<Metrics> totalMetrics; //sum of metrics over the folds or rounds for calculating the average. Not the best place for this
        std::vector<std::string> header; //names of the input nodes in the same order that appear in the nets' input layer. Must be set from a net before saving datasets in order to include the header in the file. Must match the parser's header
        std::shared_ptr<std::ofstream> resultFile; //file where everything that is not a net or a dataset is printed. Typically, fold metrics and final avg metrics. Matches the console output
};

#endif //EMITTER_HPP
