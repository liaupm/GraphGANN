#ifndef PARSER_HPP
#define PARSER_HPP

#include "defines.hpp"
#include "Node.hpp" //nodes
#include "Arc.hpp" //arcs

#include <vector> //nodes, arcs, inputs, outputs, metrics
#include <map> //params
#include <string> //param names,std::vector<std::string> header, std::vector<std::string> originalDataHeader, std::map<std::string, std::string> strParams
#include <memory> //std::vector<NodeSP> nodes, std::vector<ArcSP> arcs


class Parser
{
    public:
    //---static
        static std::map<std::string, FunctionBase::FunctionType> functionTypeNM; //name map for str param "activation function type" to FunctionBase::FunctionType
        static std::map<std::string, int> metricNM; //name map for str params metric and quality criterion to metric index
        
        inline Parser();
        virtual ~Parser() = default;

    //---get
        //net
        inline const std::vector<NodeSP>& getNodes() const { return nodes; }
        inline const std::vector<ArcSP>& getArcs() const { return arcs; }
        inline const std::vector<double>& getMetrics() const { return metrics; }
        inline const std::vector<std::string>& getHeader() const { return header; }
        //data
        inline const std::vector<std::vector<double>>& getInputs() const { return inputs; }
        inline const std::vector<double>& getOutputs() const { return outputs; }
        inline const std::vector<double>& getInstanceWeights() const { return instanceWeights; }
        inline const std::vector<std::string>& getOriginalDataHeader() const { return originalDataHeader; }
        //options params
        inline const std::map<std::string, int>& getIntParams() const { return intParams; }
        inline const std::map<std::string, double>& getRealParams() const { return realParams; }
        inline int getIntParam( const std::string& paramName ) const { return intParams.find( paramName )->second; }
        inline uint getUintParam( const std::string& paramName ) const { return static_cast<uint>( intParams.find( paramName )->second ); }
        inline double getRealParam( const std::string& paramName ) const { return realParams.find( paramName )->second; }

    //---set
        inline void setHeader( const std::vector<std::string>& xHeader ) { header = xHeader; }
        inline void setIntParam( const std::string& paramName, int value ) { intParams[paramName] = value; }
        inline void setRealParam( const std::string& paramName, double value ) { realParams[paramName] = value; }

    //---API
        bool parseOptions( const std::string& fileName = DEFAULT_PARSER_INFILE_OPTIONS ); //parse the options and params file
        bool parseNetwork( uint64_t options = DEFAULT_PARSER_FLAG_NET, const std::string& fileName = DEFAULT_PARSER_INFILE_NET_W, const std::string& fileNameActi = DEFAULT_PARSER_INFILE_NET_ACTI, const std::string& fileNameMetrics = DEFAULT_PARSER_INFILE_NET_METRICS );
        bool parseDataset( uint64_t options = DEFAULT_PARSER_FLAG_DATA, const std::string& fileName = DEFAULT_PARSER_INFILE_DATA );


    private:
    //net
        std::vector<NodeSP> nodes; //nodes parsed from the main net file. Used for creating NeuralWeb object
        std::vector<ArcSP> arcs; //arcs parsed from the main net file. Used for creating NeuralWeb object
        std::vector<double> metrics; //train and val metrics concatenated. Parsed from the metrics file of a trained net. Used for seting a net's metrics via reflection
        std::vector<std::string> header; //names of the input nodes in the same order that appear in the nets' input layer. Must be set from a net before parsing datasets in order to make the inputs order match
    //dataset
        std::vector<std::vector<double>> inputs; //dataset inputs, n per case. Dim 0 = case, dim 1 = input 
        std::vector<double> outputs; //dataset outputs, 1 per case. Dim 0 = case
        std::vector<double> instanceWeights; //weights of the cases. Dim 0 = case. Parsed from de dataset file if it includes weights
        std::vector<std::string> originalDataHeader; //input node names in the order they appear in the dataset file. The first one is the output name
    //options params
        std::map<std::string, int> intParams; //int, uint and bool params
        std::map<std::string, double> realParams; //real params
        std::map<std::string, std::string> strParams; //str params that required conversion to specific types via name maps
};


inline Parser::Parser()
{
///parameters default values
//---randomness
    //three different numbers for the three random seeds (do not repeat)
    intParams["seedData"] = 1; //random seed for datasets (making val and test splits and generating random combinations of inputs) 
    intParams["seedIni"] = 2; //random seed for generating the initial population of nets
    intParams["seedRun"] = 3; //random seed for running the GA

    //---neural net
    realParams["minActivation"] = 0.0; //min value for node scales
    realParams["maxActivation"] = 20.0; //max value for node scales

    strParams["functionType"] = "satExponential"; //activation function for hidden nodes: {sigmoid, satExponential}. Output node is always sigmoid
    realParams["classThreshold"] = 0.5; //threshold for binarizing real output

//---ensembles
    strParams["ensembleCriterion"] = "lossW"; //quality metric used for selecting and weighting ensemble members: {none, loss, lossW, lossOutW, acc, accW, accOutW }
    //intParams["ensembleCriterion"] = INDEX_METRIC_LOSS_W;
    realParams["ensembleThreshold"] = 0.5; //quality threshold (ensembleCriterion) for including a net in the ensemble
    intParams["ensembleWeighted"] = 1; //whether to weight member predictions by quality (ensembleCriterion) (1) or not (0)
    
    intParams["netIndex"] = 0; //index of the first net (trained and saved or loaded depending on the program)
    intParams["netNum"] = 1; //number of nets (trained and saved or loaded depending on the program)

//---genetic algoritm
    //MMX crossover params
    realParams["a"] = -0.001; //-0.01  //negative and close to 0
    realParams["b"] = -0.133; //-0.05 //negative and bigger than a in abs
    realParams["c"] = 0.54;  //0.1 //0.2 //positive
    realParams["d"] = 0.226;  //0.4 //positive and smaller than 0.5

    intParams["crossNum"] = 1; //number of crosses per generation. Do not change
    intParams["parentNum"] = 7; //number of parents per cross. Do not change
    intParams["outspringNum"] = 2; //number of children per cross. Do not change
    
    realParams["mutationProbWeights"] = 0.0; //prob of weight mutation per cross 
    realParams["mutationAmountWeights"] = 0.1; //max amount of weight change due to mutation per cross. Min = 0.0
    realParams["mutationProbActivation"] = 0.0; //prob of scale mutation per cross 
    realParams["mutationAmountScales"] = 0.1; //max amount of scale change due to mutation per cross. Min = 0.0

    realParams["mixFraction"] = 1.0; //fraction of population exchanged in multi population migration events

//---amount of training
    intParams["popSize"] = 100; //size of each net population
    intParams["gaNum"] = 1; //number of populations in the multiGA
    intParams["generationNum"] = 1000; //number of GA generations between migration events
    intParams["mixNum"] = 1; //number of migration events. Total generations = mixNum * generationNum

//---instance weighting
    realParams["instanceWeightByOutput"] = 0.5; //weight given to cases with output = 0. Cases with output = 1 are given 1 - instanceWeightByOutput. Must be in [0,1]. For cost-sensitive classification or class balancing
    realParams["instanceWeightByInput"] = 0.0; //exponent of exponential decay of instance weigth with number of 0 inputs (for cases where instances with less 0 inputs are more informative)
    intParams["simWeight"] = 0; //whether to weight the instances by similarity (train set: between them, val set: related to the train set). For offsetting very similar instances in the dataset


//---k-fold and evaluation
    intParams["k"] = 1; //number of cross-validation folds (for making fair test splits or val splits when adjusting hyperparameters)
    intParams["validationInstanceNum"] = 15; //number of instances for validation set (early stopping)

    strParams["bestNetCriterion"] = "lossW"; //metric used for selecting the best net (in val set)
    //intParams["bestNetCriterion"] = INDEX_METRIC_LOSS_W; //metric used for selecting the best net (in val set)
    realParams["requiredValQuality"] = 0.5; //threshold (bestNetCriterion) required for accepting a net as ok after training
    intParams["generationsRequired"] = 500; //min number of training generations. The selected net (best in val set) have to belong to a later generation to ensure true learning and not by chance
    intParams["valMaxTrials"] = 5; //max trials when training a net (point: some trainings may lead to bad results and must be discarded)

//---prediction
    intParams["zerosNum"] = 1; //number of 0 inputs in input combinations for prediction
    realParams["predictionPrintThresholdL"] = 0.0; //lower predicted output threshold for printing a case in the predictions results. Filter to avoid too heavy results files
    realParams["predictionPrintThresholdU"] = 1.0; //upper predicted output threshold for printing a case in the predictions results. Filter to avoid too heavy results files
    intParams["combisFilterMode"] = 0; //filter to apply when making all input combinations for prediction (related to the main loaded dataset). 0 = no filter, 1 = remove repeated, 2 = remove supersetsof instances with given output
    intParams["combisFilterInput"] = 0; //if filter mode = 2, instance value to use for determining that one instance is a superset of another one
    intParams["combisFilterClass"] = 0; //if filter mode = 2, output value to remove
    
//---crazy
    intParams["crazy"] = 0; //whether to perform a random swap of input nodes to test the impact of net structure 
    intParams["crazySeed"] = 1; //random seed for random swap of input nodes
    intParams["make_ff"] = 0; //whether to convert the hidden part of the net into fully-connected feed-forward to test the impact of net structure. Up to 3 hidden layers
    intParams["ff_hidden0"] = 2; //number of nodes(neurons) in hidden layer 0 if make_ff
    intParams["ff_hidden1"] = 0; //number of nodes(neurons) in hidden layer 1 if make_ff
    intParams["ff_hidden2"] = 0; //number of nodes(neurons) in hidden layer 2 if make_ff
    intParams["ff_hidden3"] = 0; //number of nodes(neurons) in hidden layer 3 if make_ff

//---options
    intParams["saveHistorical"] = 0; //whether to save the historical change in train and val metrics during training
    intParams["saveBestNet"] = 0; //whether to save the structure, param value and metrics of the best nets
    intParams["savePredictions"] = 0; //whether to save (val and test) predictions of the best nets

    intParams["datasetIndex"] = -1; //in the case of having performed a dataset split before, index of the fold to use. -1 = use whole dataset
    intParams["program"] = 0; //program to run
}

#endif //PARSER_HPP
