#ifndef NEURAL_WEB_BASE_HPP
#define NEURAL_WEB_BASE_HPP

#include "defines.hpp"
#include "LossFunction.hpp" //LossFunctionBase* lossFunction;
#include "Metrics.hpp"
#include "Parser.hpp" //constructor and Params' constructor

#include <vector> //std::vector<Metrics*> metricsReflection, std::vector<double*> membersReflection in Metrics, args of many methods
#include <memory> //LossFunctionBaseSP lossFunction


///abstract base class for both trainable NeuralWeb and ensemble of NeuralWeb. Provides common interface for prediction and evaluation
class NeuralWebBase
{
    public:
        ///params that are common to both NeuralWeb and ensemble
        struct Params
        {
            double classThreshold; //threshold for converting real output into 0 or 1 class. Typically 0.5

            Params( const Parser& parser ) : classThreshold( parser.getRealParam( "classThreshold" ) ) {;}
        };


    //================================
        NeuralWebBase( const Parser& parser )
        : params(parser), lossFunction( std::make_shared<CrossEntropy>() )
        , trainMetrics( INI_NET_METRIC ), testMetrics( INI_NET_METRIC )
        { trainMetrics.setNet(this); testMetrics.setNet(this); initReflection(); }

        virtual ~NeuralWebBase() {;}

    //---get
        //fixed
        inline const Params& getParams() const { return params; }
        //state
        inline const Metrics& getTrainMetrics() const { return trainMetrics; }
        inline const Metrics& getTestMetrics() const { return testMetrics; }
        inline const Metrics& getSavedMetrics() const { return savedMetrics; }

        inline Metrics& getTrainMetricsEditable() { return trainMetrics; }
        inline Metrics& getTestMetricsEditable() { return testMetrics; }


        inline const Metrics& getReflectedMetrics( uint index ) const { return *metricsReflection[index]; }
        inline Metrics& getReflectedMetricsEditable( uint index ) { return *metricsReflection[index]; }

    //---set
        inline void setTrainMetrics( const std::vector<double>& metricsVector, uint uIndex = METRIC_NUM, uint lIndex = 0 ) { trainMetrics.setMembers( metricsVector, uIndex, lIndex ); }
        inline void setTestMetrics( const std::vector<double>& metricsVector, uint uIndex = METRIC_NUM * 2, uint lIndex = METRIC_NUM ) { testMetrics.setMembers( metricsVector, uIndex, lIndex ); }
        inline void saveTestMetrics() { savedMetrics = testMetrics; }

    //---API
        virtual double predict( const std::vector<std::vector<double>>& inputs, uint index ) const = 0; //predict output given the inputs for case number index. Pure virtual
        double evaluateWeighted( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights, uint setIndex = INDEX_SET_TRAIN ); //update testMetrics by evaluationg with the given weighted instances and return loss
        inline double calculateFitness() { return trainMetrics.calculateFitness(); } //calculate training fitness by using the trainMetrics
        inline void initReflection() { metricsReflection = { &trainMetrics, &testMetrics }; } //start  std::vector<Metrics*> metricsReflection

    protected:
    //fixed
        Params params; //params that are common to both NeuralWeb and ensemble
        LossFunctionBaseSP lossFunction; //pointer to use polymorphism. Shared to allow for shallow copy
    //state
        Metrics trainMetrics; //metrics used during training. Where fitness is calculated
        Metrics testMetrics; //metrics used for evaluation (either validation or fair test sets ) that are not taken into account during training
        Metrics savedMetrics; //evaluation metrics used for selecting and weighting nets by quality. Having a separate var allows for further testing without overwritting
        std::vector<Metrics*> metricsReflection; //access to train and test metrics via index
};

#endif //NEURAL_WEB_BASE_HPP
