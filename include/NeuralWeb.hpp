#ifndef NEURAL_WEB_HPP
#define NEURAL_WEB_HPP

#include "defines.hpp"
#include "Node.hpp" //nodes, inputLayer, outputLayer
#include "Arc.hpp" //arcs
#include "PopulationCreator.hpp" //randomizeScales(), randomizeWeights()
#include "Parser.hpp" //constructor
#include "NeuralWebBase.hpp" //parent class

#include <vector> //nodes, arcs, header, metrics
#include <string> //header
#include <memory> //std::vector<NodeSP> nodes, std::vector<ArcSP> arcs, std::vector<NodeSP> inputLayer, NodeSP outputLayer


class FunctionBase;
class Node;
class Arc;
class PopulationCreator;

class NeuralWeb : public NeuralWebBase
{
    public:
        NeuralWeb( const Parser& parser ) //creation constructor
        : NeuralWebBase::NeuralWebBase( parser )
        , nodes( parser.getNodes() ), arcs( parser.getArcs() )
        , bSaved(false)
        { findLayers(); }

        NeuralWeb( const NeuralWeb* const originalNeuralWeb ); //fake deep copy constructor
        virtual ~NeuralWeb() {}
    
    //---get
        //structure
        inline const std::vector<NodeSP>& getNodes() const { return nodes; }
        inline NodeSP getNode( uint index ) { return nodes[index]; }
        inline const std::vector<ArcSP>& getArcs() const { return arcs; }
        inline const std::vector<NodeSP>& getInputLayer() const { return inputLayer; }
        inline NodeSP getOutputLayer() const { return outputLayer; }
        std::vector<std::string> getHeader() const; //returns the names of input layer nodes in order. First element = output node name. Used for sorting inputs in the same order in the datasets
        //state
        inline bool getBSaved() const {return bSaved; }

    //---set
        //structure
        inline void setWeight( double value, uint arcIndex ) { arcs[arcIndex]->setWeight( value ); }
        inline void setWeight( double value, uint nodeIndex, uint parentIndex ) { nodes[nodeIndex]->getParents()[parentIndex]->setWeight( value ); }
        //state
        inline void setFitness( double xFitness ) { trainMetrics.fitness = xFitness; } //metrics is member var of NeuralWebBase
        inline void setBSaved( bool xSaved ) { bSaved = xSaved; }

    //---API
        //structure
        void transferParams( const NeuralWeb* originalNeuralWeb ); //transfer the values of weights and scales from a net with identical structure. Used for copying from trained to untrained
        void findLayers(); //finds the input and output layer. Must be always called after adding all the arcs and nodes to a new net
        inline void resetNodes() const { for( uint n = 0; n < nodes.size(); n++ ) nodes[n]->setDone( false ); } //return all the nodes to the "no value yet" state. Must be called before every forward pass
        //params randomization
        inline void randomize( PopulationCreator& popCreator ) { randomizeWeights( popCreator ); normalizeWeights(); randomizeScales( popCreator ); } //randomize both arc weights and node scales
        void randomizeWeights( PopulationCreator& popCreator ); //set all the arc weight to random values. Used for population initialization
        void normalizeWeights(); //make all the weights of a node's parent arcs add up to 1 in abs. Must be called after any change in weights: randomization, crossover or mutation
        inline void randomizeScales( PopulationCreator& popCreator ) { for( uint n = 0; n < nodes.size(); n++ ) nodes[n]->setScale( popCreator.sampleIniDistributionScales( n ) ); } //set all the node scales to random values. Used for population initialization
        //ml
        double predict( const std::vector<std::vector<double>>& inputs, uint index ) const override; //predict output given the inputs for case number index
        //modify structure
        void convertToFF( const std::vector<uint>& nodeNumPerLayer ); //converts the hidden part of the net into a fully-connected feed-forward one with the given number of layers and neurons (nodes) per layer. Input and output layers are kept.
        void swapInputLayer( RandomEngine& randomEngine ); //randomly swaps the nodes in the input layer. For checking the suitability of the chosen structure
        

    private:
        //struct
        std::vector<NodeSP> nodes; //all the nodes. Index matches their id. Strong pointers because nodes belong to the net
        std::vector<ArcSP> arcs; //all the arcs, including biases at the tail. Index matches their id. Strong pointers because arcs belong to the net
        std::vector<NodeSP> inputLayer;
        NodeSP outputLayer; //single output

        //state
        bool bSaved; //if the net is saved in historical (because it is the best of any generation), this flag = true to avoid deteling it by death operator leaving invalid pointers. Not required when using smart shared pointers
};

#endif //NEURAL_WEB_HPP
