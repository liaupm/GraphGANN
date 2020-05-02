#ifndef DATASET_HPP
#define DATASET_HPP

#include "defines.hpp"
#include "DatasetBase.hpp" //parent class

#include <vector> //folds
#include <memory> //std::vector<DatasetSP> foldsTraining, std::vector<DatasetSP> foldsTest

///extends DatasetBase with data spliting functionality: LOO, stratified single split and stratified k-fold. Holds pointers to the resulting child datasets to iterate them
class Dataset : public DatasetBase
{
    public:
        Dataset( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights = {}, double classThreshold = DEFAULT_DATASET_CLASS_THRESHOLD ) //creation constructor
        : DatasetBase::DatasetBase( inputs, outputs, instanceWeights, classThreshold )
        {  initKFold( 1 ); initReflection(); }

        Dataset() : DatasetBase::DatasetBase(), foldNum(0), currentTestFold(0), foldSize(0) { initReflection(); } //null constructor. Allows for not initializing Dataset member vars in constructor
        
        Dataset( const Dataset* originalDataset ) //fake deep copy constructor: rely on DatasetBase default copy constructor for all DatasetBase vars. Make copies of the child Datasets
        : DatasetBase::DatasetBase( *originalDataset )
        { 
            foldsTraining.clear(); foldsTest.clear();
            for( uint f = 0; f < originalDataset->foldsTraining.size(); f++ )
                foldsTraining.push_back( std::make_shared<Dataset>( originalDataset->foldsTraining[f].get() ) );

            for( uint f = 0; f < originalDataset->foldsTest.size(); f++ )
                foldsTest.push_back( std::make_shared<Dataset>( originalDataset->foldsTest[f].get() ) );

            initKFold( originalDataset->foldNum ); initReflection(); 
        } 

        virtual ~Dataset() {}

    //----get
        //data splits
        inline DatasetSP getTrainingFold() const { return foldsTraining[currentTestFold]; } //get current train fold
        inline DatasetSP getTestFold() const { return foldsTest[currentTestFold]; } //get current val or fair test fold
        inline DatasetSP getReflectedFold( uint index ) const { return ( *foldsReflection[index] )[ currentTestFold ]; } //get current fold, selecting either train or test by index

    //---API
        //instance weighting by similarity
        void dissimilaritySubsets(); //create the similarity matrix, dissimilarity score and make them the instance weights for all the contained data splits. Training splits: relative to self. Test splits: relative to the corresponding training split

        //data splits
        inline void makeAllTraining() { foldsTest.clear(); foldsTraining = { std::make_shared<Dataset>( inputs, outputs, instanceWeights, classThreshold ) }; initKFold( 1 ); } //make a single training split with all the cases
        inline void makeAllTest() { foldsTraining.clear(); foldsTest = { std::make_shared<Dataset>( inputs, outputs, instanceWeights, classThreshold ) }; initKFold( 1 );  } //make a single test split with all the cases
        void makeSingleFold( RandomEngine& randomEngine, uint instanceNum ); //separate a test split of instanceNum cases and a training split of total - instanceNum cases. Stratified
        void leaveOneOut(); //make as many train-test split pairs as cases, train with total - 1 cases and test with 1 case
        void makeStratifiedKFold( RandomEngine& randomEngine, uint k ); //make k train-test split pairs, train with total - total/k cases and test with total/k cases
        
        inline void initReflection() { foldsReflection = { &foldsTraining, &foldsTest }; } //reflection of train ans test sets of splits to access them by index
        inline void initKFold( uint k ) { foldNum = k; foldSize = inputs.size() / foldNum; currentTestFold = 0; } //initialize previously splitted k-fold to start iterating the folds from the first one
        inline void nextFold() { currentTestFold = (currentTestFold + 1) % foldNum; } //circular iteration of the folds
        

    private:
    //data splits
        std::vector<DatasetSP> foldsTraining; //splits for training. Used for single split, LOO and k-fold
        std::vector<DatasetSP> foldsTest; //splits for val or fair test (depending on the dataset usage). Used for single split, LOO and k-fold
        std::vector<std::vector<DatasetSP>*> foldsReflection; //reflection access to the 2 sets of splits by index
        uint foldNum; //number of test splits = number of training splits = k. Used for single split, LOO and k-fold
        uint currentTestFold; //current split. Used for single split, LOO and k-fold
        uint foldSize; //number of cases per split in the case of k-fold
};

#endif //DATASET_HPP
