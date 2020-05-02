#ifndef LOSS_FUNCTION_HPP
#define LOSS_FUNCTION_HPP

#include "defines.hpp"

#include <math.h> //std::log (cross entropy)


///abstract base class for deriving custom loss functions
class LossFunctionBase
{
    public:
        LossFunctionBase() {;}
        virtual ~LossFunctionBase() {}

        virtual double evaluate( double predictedOutput, double actualOutput ) = 0;
};

///binary cross-entropy
class CrossEntropy : public LossFunctionBase
{   
    public:
        CrossEntropy() {;}
        virtual ~CrossEntropy() {}

        double evaluate( double predictedOutput, double actualOutput ) override //safe version that avoids invalid logs by clamping the predicted output to [ NET_LOSS_CLAMP_E, 1 - NET_LOSS_CLAMP_E ]
        { 
            if( predictedOutput == actualOutput ) return 0.0;
            double clampedPredictedOutput = std::max( std::min( predictedOutput, 1.0 - NET_LOSS_CLAMP_E ), NET_LOSS_CLAMP_E );
            return - ( actualOutput * std::log( clampedPredictedOutput ) + ( 1.0 - actualOutput ) * std::log( 1.0 - clampedPredictedOutput ) ); 
        }
};

#endif //LOSS_FUNCTION_HPP
