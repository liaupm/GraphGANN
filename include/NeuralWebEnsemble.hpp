#ifndef NEURAL_WEB_ENSEMBLE_HPP
#define NEURAL_WEB_ENSEMBLE_HPP

#include "defines.hpp"
#include "Parser.hpp" //constructor
#include "NeuralWebBase.hpp" //parent class
#include "NeuralWeb.hpp" //memberNets

#include <vector> //memberNets
#include <memory> //std::vector<NeuralWebSP> memberNets


///ensemble of NeuralWeb with prediction and evaluation functionality
class NeuralWebEnsemble : public NeuralWebBase
{
    public:
        ///params that are exclusive of ensembles
        struct EnsembleParams
        {
            bool bWeighted; //whether the prediction must be weighted based on the quality of each member (true) or just averaged (false)
            int qualityCriterion; //id of the metric used as criterion for selecting nets and (optionally) weighting the prediction
            double qualityThreshold; //

            EnsembleParams( const Parser& parser) : bWeighted( parser.getIntParam("ensembleWeighted") ), qualityCriterion(parser.getIntParam( "ensembleCriterion" ) ), qualityThreshold( parser.getRealParam( "ensembleThreshold" ) ) {;}
        };


    //================================
        NeuralWebEnsemble( const Parser& parser ) : NeuralWebBase::NeuralWebBase(parser), ensembleParams(parser) {;}
        virtual ~NeuralWebEnsemble() {}

    //---get
        inline const std::vector<NeuralWebSP>& getMemberNets() const { return memberNets; } 

    //---set
        inline void setMemberNets( const std::vector<NeuralWebSP>& xMemberNets ) { memberNets = xMemberNets; }
        inline void addMemberNet( NeuralWebSP newMemberNet ) { newMemberNet->saveTestMetrics(); memberNets.push_back( newMemberNet ); }
 
    //---API
        double predict( const std::vector<std::vector<double>>& inputs, uint index ) const override; //predict output given the inputs for case number index
        Metrics averageMetrics( const std::vector<std::vector<double>>& inputs, const std::vector<double>& outputs, const std::vector<double>& instanceWeights, uint setIndex = INDEX_SET_TRAIN ); //calculate average metrics
        
    private:
        EnsembleParams ensembleParams; //params that are exclusive of ensembles
        std::vector<NeuralWebSP> memberNets;
};

#endif //NEURAL_WEB_ENSEMBLE_HPP
