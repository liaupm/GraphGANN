#ifndef ARC_HPP
#define ARC_HPP

#include "defines.hpp"

#include <vector> //parents


class Node; //parents, child
class Arc
{
    public:
        enum Sign //sign restriction for the weight
        {
            POS, NEG, ANY, NONE
        };

        inline Arc( uint id, Sign sign, Node* parent, Node* child, bool parentExists = true, bool childExists = true ) //creation constructor
        : id(id), sign(sign)
		, parents( {parent} ), child(child), parentExists(parentExists), childExists(childExists)
		, exponents( {1.0} ), weight(INI_ARC_WEIGHT), bTrainable(true) {;}

        inline Arc( const Arc* originalArc, Node* parent = nullptr, Node* child = nullptr ) //fake deep copy constructor
        : id(originalArc->id), sign(originalArc->sign)
		, parents( {parent} ), child(child), parentExists(originalArc->parentExists), childExists(originalArc->childExists) //parent and child nodes are not copied from original arc but set to the new copies of the nodes
		, exponents( {1.0} ), weight(originalArc->weight), bTrainable(originalArc->bTrainable) {;}

        virtual ~Arc() {};

    //---get
        inline uint getId() const { return id; }
        inline Sign getSign() const { return sign; }

        inline const std::vector<Node*>& getParents() const { return parents; }
        inline Node* getParent( uint index = 0 ) { return parents[index]; }
        inline uint getOrder() const { return parents.size(); }
        inline Node* getChild() { return child; }
        inline bool getParentExists() const { return parentExists; }
        inline bool getChildExists() const { return childExists; }

        inline double getWeight() const { return weight; }
        inline bool getBTrainable() const { return bTrainable; }

    //---set
        inline void setId( uint xId ) { id = xId; }
        inline void setParent( Node* xParent ) { parents[0] = xParent; }
        inline void setParents( const std::vector<Node*>& xParents ) { parents = xParents; }
        inline void setParentExists( bool xParentExists ) { parentExists = xParentExists; }
        inline void setChildExists( bool xChildExists ) { childExists = xChildExists; }
        
        inline void setWeight( double xWeight ) { weight = xWeight; }
        inline void setBTrainable( bool xTrainable ) { bTrainable = xTrainable; }

    //---API
        //change the weight by an amount. It must increase in abs so change is added to positive weights and substracted from negative ones. 0 is considered as positive in Sign::ANY case. Used for mutation in GA
        inline void changeWeight( double change ) { ( weight >= 0 && sign != Sign::NEG ) ? weight += change : weight -= change; restrictSign(); } 
        inline void restrictSign() { if( ( sign == Sign::POS && weight < 0.0 ) || ( sign == Sign::NEG && weight > 0.0 ) ) weight = 0.0; } //fit the sign restrictions. Used after crossover of mutation in GA
        double forwardProp(); //forward pass. Called from the same method in Node


    private:
        uint id; //numeric id that matches index in NeuralWeb vector<ArcSP>
        Sign sign; //restricted sign of the interaction

        std::vector<Node*> parents; //parent nodes multiplied (an Arc is a term in the weighted sum that can contain more than one parent node). In the simplest first order case, there is a single parent per arc
        Node* child; 
        bool parentExists; //whether this arc is the first one where the parent node appears (false) or not (true). For parsing and copying purposes
        bool childExists; //whether this arc is the first one where the child node appears (false) or not (true). For parsing and copying purposes

        std::vector<double> exponents; //exponent of each parent, indexes match those in std::vector<Node*> parents. A single 1.0 if first order case
        double weight; //trainable weight
        bool bTrainable; //whether the term is trainable or (temp) fixed. In the simplest case, all the arcs have trainable weight so this var is not required
};

#endif //ARC_HPP
