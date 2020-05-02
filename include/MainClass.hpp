#ifndef MAIN_CLASS_HPP
#define MAIN_CLASS_HPP

#include "defines.hpp"
#include "RandomnessHandler.hpp"

#include "NeuralWeb.hpp" //const NeuralWeb* net, NeuralWeb* bestNet
#include "Dataset.hpp" //Dataset dataset, std::vector<Dataset*> partialDatasets, std::vector<Dataset*> generatedDatasets
#include "PopulationCreator.hpp" //PopulationCreator popCreator
#include "MultiGa.hpp" //MultiGa* multiGa
#include "Parser.hpp" //Parser parser, constructor
#include "Emitter.hpp" //Emitter emitter

#include <memory> //MultiGaSP multiGa, NeuralWebSP net, NeuralWebSP bestNet, std::vector<DatasetSP> partialDatasets, std::vector<DatasetSP> generatedDatasets


class MainClass;
typedef void( MainClass::*ProgramPointer )(); //pointer to program

class NeuralWeb;
class Dataset;
class PopulationCreator;
class MultiGa;
class Parser;

///main class that uses all the other classes for running different training, evaluating and dataset management programs. Provides the API for simply using the app as a library
class MainClass
{
    public:
        //main params
        struct Params
        {
            uint k; //total number of cross-validation folds

            Params( const Parser& parser ) : k( parser.getIntParam( "k" ) ) {;}
        };

    //---static
        static std::vector<ProgramPointer> programs; //available programs for running by id

        MainClass( const Parser& parser ) : parser(parser), popCreator()
        , params(parser), currentFold(0), currentNetIndex(0), multiGa(nullptr)
        , net(nullptr), bestNet(nullptr)
        , randomnessHandler( { parser.getUintParam( "seedData" ), parser.getUintParam( "seedIni" ), parser.getUintParam( "seedRun" ) } ) {;}

        virtual ~MainClass() {}


    //---API
        //programs
        inline void runProgram( uint programIndex ) { if( programIndex < programs.size() ) { ( this->*programs[ programIndex ] )(); std::cout << "program finished ok\n"; } else std::cout << "Error: unknown program\n"; } //run a program by id 
        void progTrainOnly(); //trains one net usign a validation fraction for early termination and the rest of the dataset for training
        void progKFold(); //makes stratified k-fold with the whole dataset using the test part for validation (early termination)
        void progKFoldFair(); //same as progKFold but previously separating a fraction for fair test. This fraction is used for nothing but printing its metrics
        void progKFoldFairEnsemble(); //train nets and evaluate avg individual vs ensemble performance in fair test set k times by performing a k-fold
        void progTrainAndSaveNets(); //train n nets by progTrainOnly and save them for future ensemble
        
        void progEvaluateEnsemble(); //load previously trained net and evaluate avg individual vs ensemble performance in fair test set
        void progPredictOutputsEnsemble(); //uses nets saved in progTrainAndSaveNets in a weighted ensemble and the dataset generated with progMakeInputCombinations to predict the dataset outputs
        
        void progSplitDataset(); //save train and test dataset splits following a k-fold
        void progMakeInputCombinations(); //save a dataset with all the posible input combinations and same output. For prediction in a future run
        
      
        //basic
        void init(); //initialize: load reference net, set headers, load base dataset, weight the instances and initialize members
        inline void run() { init(); runProgram( parser.getIntParam( "program" ) ); } //initialize + run program
        NeuralWeb* loadTrainedNet( uint netIndex ); //load a trained net in a safe way: transferring the trained scales and weights to a copy of the reference net
        void trainNet( uint datasetIndex = DEFAULT_MAINC_DATASET, bool bMakeValSplit = DEFAULT_DATASET_TRAIN_VALSPLIT ); //trains a net with multiGA with the given dataset. It can make several trials while the resulting nets do not fulfil the quality requirements
        void printMetrics( uint datasetIndex = DEFAULT_MAINC_DATASET, uint fairDatasetIndex = DEFAULT_MAINC_DATASET_FAIR, uint netIndex = 0, bool bEnsemble = false, const std::string& sufix = "" ); //save metrics and (optional) net and predictions for train, val and (depending on the program) test sets

        

    private:
    //handler objects
        Parser parser; //for parsing files
        Emitter emitter; //for saving files
        PopulationCreator popCreator; //for creating initial net populations
    //fixed params
        Params params;
    //state
        uint currentFold; //also used as current training index when training multiple times
        uint currentNetIndex; //net index for file names when training and saving multiple
        MultiGaSP multiGa; //multiGA that generated the bestNet. Temp saved for printing historical info
    //nets
        NeuralWebSP net; //base untrained net
        NeuralWebSP bestNet; //current best net
    //datasets
        Dataset dataset; //whole base dataset
        std::vector<DatasetSP> partialDatasets; //datasets created from the base dataset for making splits 
        std::vector<DatasetSP> generatedDatasets; //predicted datasets
    //randomness
        RandomnessHandler randomnessHandler; //provides appopriate random seeds for dataset splitting, population initialization and training, based on user-provided seeds
};

#endif //MAIN_CLASS_HPP
