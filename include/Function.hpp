#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include "defines.hpp"

#include <vector> //std::vector<double> params, const std::vector<double>& input in calculate()
#include <math.h> //std::exp in  SatExponential::calculate()


//==================================== *FUNCTION BASE* =======================================
///abstract base class for activation function. Can be extended with new derived classes
class FunctionBase
{
    public:
        enum FunctionType //type reflection
        {
            SAT_EXPONENTIAL, SIGMOID
        };

        //static
        static FunctionBase* createSubobject( FunctionType functionType = DEFAULT_FUNCTION_TYPE, const std::vector<double>& params = {} ); //factory

        inline FunctionBase( const std::vector<double>& params, FunctionType functionType = DEFAULT_FUNCTION_TYPE ) : params(params), functionType(functionType) {;}
        virtual ~FunctionBase() {};
        //get
        inline std::vector<double> getParams() const { return params; }
        inline FunctionType getFunctionType() const { return functionType; }
        //set
        inline void setParams( const std::vector<double>& xParams ) { params = xParams; }
        //API
        virtual double calculate( const std::vector<double>& input ) = 0;

    protected:
        std::vector<double> params; //meaning depends on the specific function. SatExponential and sigmoid have no params
        FunctionType functionType; //type reflection
};


//==================================== *SAT EXPONENTIAL* =======================================
class SatExponential : public FunctionBase //saturating exponential with formula y = 1 - exp(-x) if x > 0; 0 if x <= 0
{
    public:
        inline SatExponential() : FunctionBase( {}, FunctionType::SAT_EXPONENTIAL ) {;}
        virtual ~SatExponential() {};

        double calculate( const std::vector<double>& input ) override { return input[0] > 0.0 ? 1.0 - std::exp( - input[0] ) : 0.0; }
};


//==================================== *SIGMOID* =======================================
class Sigmoid : public FunctionBase
{
    public:
        inline Sigmoid() : FunctionBase( {}, FunctionType::SIGMOID ) {;}
        virtual ~Sigmoid() {};

        double calculate( const std::vector<double>& input ) override { return 1.0 / ( 1.0 + std::exp( - input[0] ) ); }
};

#endif //FUNCTION_HPP
