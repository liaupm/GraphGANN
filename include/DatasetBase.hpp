#ifndef DATASET_BASE_HPP
#define DATASET_BASE_HPP

#include "defines.hpp"
#include "NeuralWebBase.hpp" //generateOutputs()

#include <vector> //inputs, outputs, instanceWeights


class NeuralWeb;

///Abstrat base class for Dataset. Set of cases with n inputs, 1 output and instance weights. Input generation and weighting capabilities
class DatasetBase
{
    public:
    //---static
        static std::vector<std::vector<double>> sparseData( const std::vector<std::vector<int>>& indexes, uint elementNum, bool bInverted = DEFAULT_DATASET_SPARSE_INVERTED ); //converts compact representation of inputs into a sparse one
        static std::vector<std::vector<double>> makeAllCombinations( uint n, uint k, bool bInverted = DEFAULT_DATASET_COMBI_INVERTED ); //make all posible combinations of n elements taken k at a time. In sparse representation. For making predicted datasets
        //instance weighting by similarity
         //similarity score between two cases as the fraction of equal inputs if same output or fraction of different inputs if different output
        static double calculatePairSimilarity( const std::vector<std::vector<double>>& inputs1, const std::vector<std::vector<double>>& inputs2, double output1, double output2, uint index1, uint index2 );
        //create the similarity matrix of a series of cases (1) relative to another series of cases (2)
        static std::vector<std::vector<double>> makeSimilarityMatrix( const std::vector<std::vector<double>>& inputs1, const std::vector<std::vector<double>>& inputs2, const std::vector<double>& outputs1, const std::vector<double>& outputs2 ); 
        static std::vector<double> sumSimilarityMatrix( const std::vector<std::vector<double>>& similarityMatrix ); //add up the similarity score for each case and reverse it (higher = more unique)


        DatasetBase( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights = {}, double classThreshold = DEFAULT_DATASET_CLASS_THRESHOLD ) //creation constructor
        : inputs(inputs), outputs(outputs), instanceWeights(instanceWeights), dataNum( inputs.size() ), classThreshold(classThreshold)
        {  makeBoolOutputs(); }

        DatasetBase() : dataNum(0), classThreshold(DEFAULT_DATASET_CLASS_THRESHOLD) {} //null constructor. Allows for not initializing Dataset member vars in constructor
        
        virtual ~DatasetBase() = default;

    //----get
        inline const std::vector<std::vector<double>>& getInputs() const { return inputs; }
        inline const std::vector<double>& getOutputs() const { return outputs; }
        inline const std::vector<double>& getBoolOutputs() const { return boolOutputs; }
        inline const std::vector<double>& getInstanceWeights() const { return instanceWeights; }
        inline uint getDataNum() const { return dataNum; }

    //---set
        inline void setInputs( const std::vector<std::vector<double>>& xInputs ) { inputs = xInputs; }
        inline void setOutputs( const std::vector<double>& xOutputs ) { outputs = xOutputs; makeBoolOutputs(); }
        
        inline void addInput( const std::vector<double>& newInput ) { inputs.push_back( newInput ); }
        inline void addOutput( double newOutput ) { outputs.push_back( newOutput ); }

    //---API
        //generate
        //make all posible input combinations of n inputs with k 1s ( bInverted = false ) or k 0s ( bInverted = true )
        inline void makeInputCombinations( uint n, uint k, bool bInverted = DEFAULT_DATASET_COMBI_INVERTED ) { inputs = makeAllCombinations( n, k, bInverted ); outputs = std::vector<double>( inputs.size(), 1.0 ); makeBoolOutputs(); }
        void generateOutputs( const NeuralWebBase* net ); //generate output predictions for the current inputs by using either a single NeuralWeb or an ensemble
        void filterInstancesEqual( const DatasetBase* filter ); //remove the instances that are equal (input only) to any other in the digen dataset
        void filterInstancesSuperset( const DatasetBase* filter, uint inputValue = DEFAULT_DATASET_FILTER_INPUT, uint classValue = DEFAULT_DATASET_FILTER_CLASS ); //remove the instances that are supersets of any other with the given classValue in the provided dataset
        //instance weighting
        void weightInstances( double instanceWeightByOutput, double instanceWeightByInput, bool bSimilarityAsWeights = false ); //make instance weights by inputs, output or similarity
        void normalizeInstanceWeights(); //make all instance weights add up to 1. Must be called after creating or modifying the weights and after spliting the data

        //instance weighting by similarity
        void makeSimilarityMatrix( const DatasetBase* referenceDataset = nullptr ); //create the similarity matrix relative to a given dataset and the total dissimilarity score. If null dataset, relative to self
        inline void dissimilarityAsInstanceWeights() { instanceWeights = dissimilaritySum; normalizeInstanceWeights(); } //use the total dissimilarity score relative to self as instance weights
        inline void relativeDissimilarityAsInstanceWeights() { instanceWeights = dissimilaritySumRelative; normalizeInstanceWeights(); } //use the total dissimilarity score relative to another dataset as instance weights

        //data splits
        void shuffle( RandomEngine& randomEngine ); //shuffle inputs, outputs and instance weights together. similarity data structures are not shuffled so have to be recalculated after shuffling. Used for randomly spliting dataset
        void sortByIndexVector( const std::vector<uint>& indexVector ); //sort inputs, outputs and instance weights together according to an index vector. similarity data structures are not sorted so have to be recalculated after sorting. Used by shuffle()


    protected:
    //data
        std::vector<std::vector<double>> inputs;
        std::vector<double> outputs; //real output
        std::vector<double> boolOutputs; //binarized output. Typically matches outputs
        std::vector<double> instanceWeights;
        uint dataNum; //number of cases or instances in the whole dataset
        double classThreshold; //threshold using for binarizing real output. Tipically 0.5

    //instance weighting by similarity
        std::vector<std::vector<double>> similarityMatrix; //pair-wise similarity between cases in this dataset
        std::vector<std::vector<double>> similarityMatrixRelative; //pair-wise similarity of the cases in this dataset relative to another dataset
        std::vector<double> dissimilaritySum; //total dissimilarity score of each instance relative to all the other instances in this dataset. Can be used as instance weight for train splits
        std::vector<double> dissimilaritySumRelative; //total dissimilarity score of each instance relative to all the instances in another dataset. Can be used as instance weight for val or test splits (related to training set)

    //misc
        void makeBoolOutputs(); //binarize real outputs
};

#endif //DATASET_BASE_HPP
