#ifndef DEFINES_HPP
#define DEFINES_HPP

#include <string>
#include <random> // RandomEngine
#include <cstdint> //uint64_t flag
#include <memory>
#include <iostream>



/////////////////////////////////////////////////////////////////////////////////// TYPEDEF //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint; 
typedef std::minstd_rand RandomEngine;
typedef std::mt19937 RandomEngine2;

typedef std::shared_ptr<RandomEngine> RandomEngineSP;
typedef std::shared_ptr<RandomEngine2> RandomEngine2SP;


//============================================================ SMART POINTERS ====================================================
class FunctionBase;
typedef std::shared_ptr<FunctionBase> FunctionBaseSP;
class LossFunctionBase;
typedef std::shared_ptr<LossFunctionBase> LossFunctionBaseSP;
class DistributionInterface;
typedef std::shared_ptr<DistributionInterface> DistributionInterfaceSP;
class Metrics;
typedef std::shared_ptr<Metrics> MetricsSP;

class Node;
typedef std::shared_ptr<Node> NodeSP;
class Arc;
typedef std::shared_ptr<Arc> ArcSP;

class NeuralWebBase;
typedef std::shared_ptr<NeuralWebBase> NeuralWebBaseSP;
class NeuralWeb;
typedef std::shared_ptr<NeuralWeb> NeuralWebSP;
class NeuralWebEnsemble;
typedef std::shared_ptr<NeuralWebEnsemble> NeuralWebEnsembleSP;

class Dataset;
typedef std::shared_ptr<Dataset> DatasetSP;
class PopulationCreator;
typedef std::shared_ptr<PopulationCreator> PopulationCreatorSP;
class GeneticAlgorithm;
typedef std::shared_ptr<GeneticAlgorithm> GeneticAlgorithmSP;
class MultiGa;
typedef std::shared_ptr<MultiGa> MultiGaSP;





/////////////////////////////////////////////////////////////////////////////////// MACROS //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//================================================================ FILES
#define MAKE_FILENAME( prefix, index ) \
( std::string( prefix ) + "_" + std::to_string( index ) + ".txt" )

#define MAKE_FILENAME2( prefix, index0, index1 ) \
( std::string( prefix ) + "_" + std::to_string( index0 ) + "_" + std::to_string( index1 ) + ".txt" )

#define MAKE_FILENAME3( prefix, index0, index1, index2 ) \
( std::string( prefix ) + "_" + std::to_string( index0 ) + "_" + std::to_string( index1 ) + "_" + std::to_string( index2 ) + ".txt" )


//================================================================ FLAGS
#define FLAG( index ) \
( static_cast<uint64_t>( 1 ) << (index) )

#define GET_FLAG( flagSet, flag ) \
( (flagSet) & (flag) )

#define SETON_FLAG( flagSet, flag ) \
(flagSet) |= (flag)

#define SETOFF_FLAG( flagSet, flag ) \
(flagSet) &= ~(flag)


//================================================================ STL VECTOR
#define SET_OR_PUSH( vector, element, index ) \
if( (index) < (vector).size() ) \
	(vector)[index] = (element); \
else \
	(vector).push_back(element);

#define DELETE( vector, index ) \
delete (vector)[index]; \
(vector)[index] = nullptr;

#define CHEAP_POP( vector, index ) \
(vector)[index] = (vector).back(); \
(vector).pop_back();

#define DELETE_AND_POP( vector, index ) \
delete (vector)[index]; \
CHEAP_POP( vector, index )

#define DELETE_ALL( vector ) \
for( uint v = 0; v < (vector).size(); v++ ) \
	{ DELETE( vector, v ) } \
(vector).clear();








/////////////////////////////////////////////////////////////////////////////////// DEFINES //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////* GENERAL */////////////////////////////////////////////////////////////////////////////////////////
#define DEFAULT_SEED 1 //default random seed

//---sets
#define INDEX_SET_TRAIN 0 //index of training set
#define INDEX_SET_VAL 1 //index of validation set
#define INDEX_SET_TEST 2 //index of fair test set
#define SET_NUM 3 //number of data sets



//============================================================ MAIN CLASS ============================================================
//---datasets
#define DEFAULT_MAINC_DATASET 0 //default index of the training dataset in the MainClass Dataset vector
#define DEFAULT_MAINC_DATASET_FAIR 1 //default index of the fair test dataset in the MainClass Dataset vector


//---programs
//training
#define PROGRAM_TRAIN 0 //trains one net usign a validation fraction for early termination and the rest of the dataset for training
#define PROGRAM_KFOLD 1 //makes stratified k-fold with the whole dataset using the test part for validation (early termination)
#define PROGRAM_KFOLD_FAIR 2 //same as progKFold but previously separating a fraction for fair test. This fraction is used for nothing but printing its metrics
#define PROGRAM_KFOLD_FAIR_ENSEMBLE 3 //train nets and evaluate avg individual vs ensemble performance in fair test set k times by performing a k-fold
#define PROGRAM_TRAIN_MULTIPLE 4 //train n nets by progTrainOnly and save them for future ensemble

//prediction and evaluation without training
#define PROGRAM_EVALUATE_ENSEMBLE 5 //load previously trained net and evaluate avg individual vs ensemble performance in fair test set
#define PROGRAM_PREDICTION_ENSEMBLE 6 //uses nets saved in progTrainAndSaveNets in a weighted ensemble and the dataset generated with progMakeInputCombinations to predict the dataset outputs
//dataset
#define PROGRAM_SPLIT_DATASET 7 //save train and test dataset splits following a k-fold
#define PROGRAM_INPUT_COMBINATIONS 8 //save a dataset with all the posible input combinations and same output. For prediction in a future run



#define INDEX_WHOLE_DATASET -1 //index that meand using the wholse dataset for training instead of a split
#define DEFAULT_MAINC_GENERATIONS_LESS 2 //generations offset applied to prevent the selection of a bad net when searching for the best in val set





///////////////////////////////////////////////////////////////////////////////////* MATH AND RANDOMNESS *///////////////////////////////////////////////////////////////////////////////////

//============================================================ FUNCTION =======================================================================
#define DEFAULT_FUNCTION_TYPE FunctionType::SIGMOID //default activation function


//======================================================== DISTRIBUTION INTERFACE =============================================================
#define DISTRIBUTIONTYPE_UNIFORM 0 //id of each type of distribution
#define DISTRIBUTIONTYPE_NORMAL 1


//======================================================== RANDOMNESS HANDLER =============================================================
#define INDEX_RANDOMNESS_MAINRE_DATA 0 //index of the main randomness generator for datasets
#define INDEX_RANDOMNESS_MAINRE_INI 1 //index of the main randomness generator for population initialization
#define INDEX_RANDOMNESS_MAINRE_RUN 2 //index of the main randomness generator for training






///////////////////////////////////////////////////////////////////////////////////* NET STRUCTURE *///////////////////////////////////////////////////////////////////////////////////

//================================================================== NODE =====================================================================
#define INI_NODE_SCALE 0.0  //initial scale for the nodes. May be replaced when initialization with PopulationCreator
#define INI_NODE_VALUE 0.0  //initial value of the nodes before the first forward pass. Irrelevant for training

#define INPUT_NODE_SCALE 1.0 //untrainable scale of the input layer nodes


//================================================================== ARC =====================================================================
#define INI_ARC_WEIGHT 0.0  //initial weight for the arcs. May be replaced when initialization with PopulationCreator


//============================================================ METRICS ================================================================
#define NO_METRIC -1 //no metric
#define INDEX_METRIC_LOSS 0 //unweighted loss
#define INDEX_METRIC_LOSS_W 1 //weighted loss
#define INDEX_METRIC_LOSS_OUTW 2 //loss weighted only by output
#define INDEX_METRIC_ACC 3 //unweighted accuracy
#define INDEX_METRIC_ACC_W 4 //weighted accuracy
#define INDEX_METRIC_ACC_OUTW 5 //accuracy weighted only by output
#define METRIC_NUM 6 //number of quality metrics
#define METRIC_LOSS_NUM 3 //number of loss metrics


//============================================================ NEURAL WEB BASE ================================================================
//---metrics
#define INI_NET_METRIC -1.0 //initial value for quality metrics

//---loss function
#define NET_LOSS_CLAMP_E 0.0000001 //small number added in the cross-entropy loss function to avoid invalid logs


//=========================================================== NEURAL WEB ENSEMBLE =============================================================
#define DEFAULT_ENSEMBLE_BWEIGHTED true //whether the predictions in the ensemble are wighted by the quality of the nets
#define DEFAULT_ENSEMBLE_QUALITY_CRITERION INDEX_METRIC_LOSS_W
#define DEFAULT_ENSEMBLE_QUALITY_THRESHOLD 0.5 //threshold in the quality metric used for including a net in the ensemble or not





///////////////////////////////////////////////////////////////////////////////////* TRAINING *///////////////////////////////////////////////////////////////////////////////////

//=========================================================== DATASET ===========================================================
#define DEFAULT_DATASET_SPARSE_INVERTED true //if true, a compact list of indexes mean the inputs that are 0; if false, those that are 1. When converting a compact representation to a sparse one 
#define DEFAULT_DATASET_COMBI_INVERTED true //if true, the "k" param of a combination means the number of 0s; if false, the number of 1s. When making all the posible inputs combinations with a given number of 0s (1s)
#define DEFAULT_DATASET_CLASS_THRESHOLD 0.5 //threshold using for binarizing real output. Tipically 0.5
#define DEFAULT_DATASET_TRAIN_VALSPLIT true //whether to make a validation split before start training a net for early stopping

#define DEFAULT_DATASET_FILTER_EQUAL_THRESHOLD 0.9999 //similarity threshold to consider two instances equal when filtering
#define DEFAULT_DATASET_FILTER_INPUT 0 //default instance value to use for determining that one instance is a superset of another one when filterning by supersets
#define DEFAULT_DATASET_FILTER_CLASS 0 //defualt output value to remove when filterning by supersets


//=========================================================== POPULATION CREATOR =================================================
#define DEFAULT_POPCREATOR_SCALE_MIN 0.0 //default minimum value for node scale
#define DEFAULT_POPCREATOR_SCALE_MAX 20.0 //default maximum value for node scale
#define DEFAULT_POPCREATOR_POPSIZE 100 //default number of random initial nets


//===========================================================  GENETIC ALGORITHM =================================================
#define DEFAULT_GA_EVALUATEALL true //whether to start by evaluating the whole population when training. Typically, true if first call
#define DEFAULT_GA_GENERATION_NUM 1 //default number of evolution generations (between mixing events) when calling train()

//---randomness
#define INDEX_GA_DISTRIBUTION_PARENTS_SELECTOR 0 //fix base //index of distribution for roulette selection of parents
#define INDEX_GA_DISTRIBUTION_CROSS_SCALE 1 //var //index of distribution for crossover of node scales
#define INDEX_GA_DISTRIBUTION_CROSS_WEIGHT 2 //var //index of distribution for crossover of arc weights
#define INDEX_GA_DISTRIBUTION_MUT_SCALE_OCCURENCE 3 //fix base //index of distribution for mutation occurence in node scales
#define INDEX_GA_DISTRIBUTION_MUT_SCALE_AMOUNT 4 //fix custom //index of distribution for mutation amount in node scales
#define INDEX_GA_DISTRIBUTION_MUT_WEIGHT_OCCURENCE 5 //fix base //index of distribution for mutation occurence in arc weights
#define INDEX_GA_DISTRIBUTION_MUT_WEIGHT_ARC 6 //var //index of distribution for arc selection in mutation of arc weights
#define INDEX_GA_DISTRIBUTION_MUT_WEIGHT_AMOUNT 7 //fix custom //index of distribution for mutation amount in arc weights
#define INDEX_GA_DISTRIBUTION_NUM 8 


//===========================================================  MULTI GA =================================================
#define DEFAULT_MGA_GENERATIONS_PER_MIX 100 //default number of GA generations between migration events
#define DEFAULT_MGA_MIXNUM 1 //number of migration events. Total generations = mixNum * generationNum

//---randomness
#define INDEX_MGA_DISTRIBUTION_MIXOUT 0 //var //index of distribution for selecting the emigrant nets in mixing event
#define INDEX_MGA_DISTRIBUTION_MIXIN_SHUFFLE 1 //RE only //distribution used for assigning emigrated net to their net populations (via shuffling)
#define INDEX_MGA_DISTRIBUTION_NUM 2 






///////////////////////////////////////////////////////////////////////////////////* IO *///////////////////////////////////////////////////////////////////////////////////

//===========================================================  HISTORICAL TRACK =================================================
#define HTRACK_MIN_GENERATION 0 //minimum generation of the selected best net in val. The selected net (best in val set) have to belong to a later generation to ensure true learning and not by chance
#define HTRACK_MAX_METRIC_RATIO 2.0 //maximum ratio between the val and train loss to consider no too much overfitting
#define DEFAULT_HTRACK_BEST_CRITERION INDEX_METRIC_LOSS_W //metric used to seelect the best net in val set


//===========================================================  PARSER =================================================
//---chars and separators
#define PARSER_DATA_SEPARATOR ',' //separator char for dataset header and values
#define PARSER_EQUAL '=' //char used for assigning params to values in the options file
#define PARSER_COMMENT '/' //char used for started a comment (must be ignored by the parser)
#define PARSER_NET_ARCS_SEPARATOR '	' //separator used in the arc sentences in the net file, between node names and sign/weight value
#define PARSER_NET_SCALES_SEPARATOR '	' //separator used in the scales file between the node name and the scale value
#define PARSER_NET_ARCS_SIGN_NEG "NOT" //keyword used to indicate a negative sign in the arc sentences of untrained net file
#define PARSER_NET_ARCS_SIGN_ANY "?" //keyword used to indicate an unknown sign in the arc sentences of untrained net file
#define DEFAULT_PARSER_DATA_INPUTVALUE 0.0 //TODO ?

//--flags
#define FLAG_NULL static_cast<uint64_t>( 0 )
//flags-data
#define FLAG_DATA_PRED FLAG(0) //whether the dataset file includes predictions
#define FLAG_DATA_COUNT FLAG(1) //whether the dataset file includes count of 0 inputs
#define FLAG_DATA_ROUND FLAG(2) //whether the dataset file includes the output rounded = binarized
#define FLAG_DATA_WEIGHT FLAG(3) //whether the dataset file includes instance weights
#define FLAG_DATA_FILTER FLAG(4) //whether the dataset instances are filtered by output (only those in a range are kept)
#define FLAG_DATA_ALL FLAG_DATA_PRED | FLAG_DATA_COUNT | FLAG_DATA_ROUND //all the flags expected when parsing the dataset file (weighting is performed every run after parsing the dataset)
#define FLAG_DATA_ALL_FILTER FLAG_DATA_ALL | FLAG_DATA_FILTER //all the flags expected when parsing the dataset file + filter
#define DEFAULT_PARSER_FLAG_DATA FLAG_NULL //flags applied by default when parsing a dataset
//flags-net 
#define FLAG_NET_TRAINED FLAG(0) //whether the net is trained. If so, the main net file includes weight values instead of arc signs and there must be 2 additional files: node scales and metrics
#define DEFAULT_PARSER_FLAG_NET FLAG_NULL //flags applied by default when parsing a net file


//===========================================================  EMITTER =================================================
#define DEFAULT_EMITTER_TRAINED true //whether the printed nets are, by default, trained (true) or not (false)
#define DEFAULT_EMITTER_DATA_LBOUND -1.0 //only cases with a predicted real value greater than this value are printed for reducing file size. Smaller than 0.0 = no threshold
#define DEFAULT_EMITTER_DATA_UBOUND 2.0 //only cases with a predicted real value lower than this value are printed for reducing file size. Greater than 1.0 = no threshold
#define DEFAULT_EMITTER_METRICS_PREFIX "train "

//---chars and separators
#define EMITTER_NET_SEPARATOR "\t" //separator char used by the Emitter when saving net files
#define EMITTER_SET_NAME_FIXED_SIZE 5 //fixed field size used by the Emitter when printing metrics to the summary file or to the console to keep it aligned and readable

//---flags
#define DEFAULT_EMITTER_FLAG_NET FLAG_NULL //flags applied by default when saving a net file
#define DEFAULT_EMITTER_FLAG_DATA FLAG_NULL	//flags applied by default when saving a dataset file



//=========================================================== FILES =================================================

//---folders
#define FOLDER_NAME_DATA "data"
#define FOLDER_NAME_RESULTS "results"
#define FOLDER_SEPARATOR "\\"

#define FOLDER_MAIN ( std::string("") )
#define FOLDER_DATA ( std::string( FOLDER_NAME_DATA ) + FOLDER_SEPARATOR )
#define FOLDER_RESULTS ( std::string( FOLDER_NAME_RESULTS ) + FOLDER_SEPARATOR )
#define FOLDER_DATA_SPLITS ( std::string( FOLDER_NAME_DATA ) + FOLDER_SEPARATOR +  "splits" + FOLDER_SEPARATOR )
#define FOLDER_DATA_COMBIS ( std::string( FOLDER_NAME_DATA ) + FOLDER_SEPARATOR + "combi" + FOLDER_SEPARATOR )
#define FOLDER_RESULTS_NETS ( std::string( FOLDER_NAME_RESULTS ) + FOLDER_SEPARATOR + "nets" + FOLDER_SEPARATOR )
#define FOLDER_RESULTS_PREDICTIONS ( std::string( FOLDER_NAME_RESULTS ) + FOLDER_SEPARATOR + "predictions" + FOLDER_SEPARATOR )
#define FOLDER_RESULTS_HISTORICAL ( std::string( FOLDER_NAME_RESULTS ) + FOLDER_SEPARATOR + "historical" + FOLDER_SEPARATOR )

//---out file names
#define DEFAULT_FILE_EXT ".txt"
//net
#define FILE_NAME_NET "net" //reference untrained net
#define FILE_NAME_NET_W "net_trained" //main trained net file with weight values
#define FILE_NAME_NET_ACTI "net_trained_acti" //trained net file with node scales
#define FILE_NAME_NET_METRICS "net_trained_metrics" //trained net file with validation metrics
#define FILE_NAME_NET_FF "netFF" //fully-connected feed-forward net untrained
#define FILE_NAME_NET_CRAZY "netCrazy" //untrained net with input layer randomly swaped
//dataset
#define FILE_NAME_DATASPLIT_TRAIN "dataset_train" //dataset training + validation split
#define FILE_NAME_DATASPLIT_TEST "dataset_test" //dataset test split
#define FILE_NAME_INPUTCOMBIS "input_combinations" //dataset all input combinations for prediction
#define FILE_NAME_DATASET "dataset" //dataset with predictions. Predictions made in validation or test datasets while training or evaluating
#define FILE_NAME_DATASET_W "dataset_w" //dataset with predictions. Predictions made in validation or test datasets while training or evaluating
#define FILE_NAME_DATAPRED "dataset_pred" //dataset with predictions. Predictions made in validation or test datasets while training or evaluating
#define FILE_NAME_DATAPRED_FINAL "dataset_pred_final" //dataset with predictions. Final predictions for input combinations
//misc
#define FILE_NAME_OPTIONS "options" //file with metrics summary depending on the program
#define FILE_NAME_RESULT "summary" //file with metrics summary depending on the program
#define FILE_NAME_HISTORICAL "historical" //file with the historical evolution of quality metrics //TODO


//---complete input files-parser
#define DEFAULT_PARSER_INFILE_OPTIONS ( FOLDER_MAIN + FILE_NAME_OPTIONS + DEFAULT_FILE_EXT  ) //default file with param values and options
#define DEFAULT_PARSER_INFILE_NET_W ( FOLDER_DATA + FILE_NAME_NET + DEFAULT_FILE_EXT ) //default file with the net structure and arc signs (untrained)/weight values (trained)
#define DEFAULT_PARSER_INFILE_NET_ACTI ( FOLDER_DATA + FILE_NAME_NET_ACTI + DEFAULT_FILE_EXT ) //default file with the node scale values of a trained net
#define DEFAULT_PARSER_INFILE_NET_METRICS ( FOLDER_DATA + FILE_NAME_NET_METRICS + DEFAULT_FILE_EXT ) //default file with the validation set metrics of a trained net
#define DEFAULT_PARSER_INFILE_DATA ( FOLDER_DATA + FILE_NAME_DATASET + DEFAULT_FILE_EXT ) //default file with unweighted dataset
#define DEFAULT_PARSER_INFILE_DATA_W ( FOLDER_DATA + FILE_NAME_DATASET_W + DEFAULT_FILE_EXT ) //default file with weighted dataset 


//---complete outout files paths. Base names to add indexes and extension to (most have no default extension added here to allow for that )
//net
#define OUTFILE_NET_W ( FOLDER_RESULTS_NETS + FILE_NAME_NET_W  ) //main trained net file with weight values
#define OUTFILE_NET_ACTI ( FOLDER_RESULTS_NETS + FILE_NAME_NET_ACTI ) //trained net file with node scales
#define OUTFILE_NET_METRICS ( FOLDER_RESULTS_NETS + FILE_NAME_NET_METRICS ) //trained net file with validation metrics

//dataset
#define OUTFILE_DATASPLIT_TRAIN ( FOLDER_DATA_SPLITS + FILE_NAME_DATASPLIT_TRAIN ) //dataset training + validation split
#define OUTFILE_DATASPLIT_TEST ( FOLDER_DATA_SPLITS + FILE_NAME_DATASPLIT_TEST )  //dataset test split
#define OUTFILE_INPUTCOMBIS ( FOLDER_DATA_COMBIS + FILE_NAME_INPUTCOMBIS )  //dataset all input combinations for prediction
#define OUTFILE_DATAPRED ( FOLDER_RESULTS_PREDICTIONS + FILE_NAME_DATAPRED  ) //dataset with predictions. Predictions made in validation or test datasets while training or evaluating
#define OUTFILE_DATAPRED_FINAL ( FOLDER_RESULTS_PREDICTIONS + FILE_NAME_DATAPRED_FINAL  ) //dataset with predictions. Final predictions for input combinations

//misc
#define OUTFILE_RESULT ( FOLDER_RESULTS + FILE_NAME_RESULT + DEFAULT_FILE_EXT  ) //file with metrics summary depending on the program
#define OUTFILE_HISTORICAL ( FOLDER_RESULTS_HISTORICAL + FILE_NAME_HISTORICAL  ) //file with the historical evolution of quality metrics //TODO
#define FILE_NET_FF ( FOLDER_DATA + FILE_NAME_NET_FF + DEFAULT_FILE_EXT )  //fully-connected feed-forward net untrained
#define FILE_NET_CRAZY ( FOLDER_DATA + FILE_NAME_NET_CRAZY + DEFAULT_FILE_EXT ) //untrained net with input layer randomly swaped


#endif //DEFINES_HPP
