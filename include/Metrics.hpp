#ifndef METRICS_HPP
#define METRICS_HPP

#include "defines.hpp"


class NeuralWebBase;
class NeuralWeb; //forward declaration of derived class required by Metrics::accumulatePopulationFitness()

///quality metrics for evaluating a net or ensemble
struct Metrics
{
    const NeuralWebBase* net; //bidirectional link to the net that holds the metrics (weak to avoid cycle)
    std::vector<double> members;
    double fitness;

    inline Metrics( double iniValue = INI_NET_METRIC, const NeuralWebBase* net = nullptr ) : net(net), members(METRIC_NUM, iniValue ), fitness(iniValue) { ; } //init all members to the same value
    inline Metrics( const std::vector<double>& values, const NeuralWebBase* net = nullptr ) : net(net), members(METRIC_NUM ), fitness(INI_NET_METRIC) { setMembers( values, values.size(), 0 ); } //init members to different values
    inline Metrics( const Metrics& originalMetrics ) : net( originalMetrics.net), members( originalMetrics.members), fitness(originalMetrics.fitness) { ; } //copy constructor

    virtual ~Metrics() {}

//---get
    inline double getMember( uint index ) const { return members[index]; }

//---set
    inline void setNet( const NeuralWebBase* xNet ) { net = xNet; }
    inline void setMember( double value, uint index ) { members[index] = value; } //set member metric by index
    inline void changeMember( double amount, uint index ) { members[index] += amount; } //set member metric by index
    inline void setMembers( const std::vector<double>& values, uint uIndex, uint lIndex = 0 ) { for( uint v = lIndex; v < uIndex; v++ ) { members[ v - lIndex ] = values[v]; } } //set metrics from range of a vector

//---API
    inline void reset( double iniValue = INI_NET_METRIC ) { for( uint m = 0; m < members.size(); m++ ) members[m] = iniValue; } //set all the metrics to a value (typically 0)
    inline void add( const Metrics* metricsToAdd ) { for( uint m = 0; m < members.size(); m++ ) ( members[m] ) += metricsToAdd->getMember( m ); } //add another Metrics to this Metrics. For making totals and averages (of the population, of a k-fold...)
    inline void add( double value, uint index ) { members[index] += value; } //add quantity to a member metric
    inline void scale( double multiplier ) { for( uint m = 0; m < members.size(); m++ ) members[m] *= multiplier; } //multiply all the member metrics by a given value. For weighting
    
    inline double calculateFitness() { fitness = members[INDEX_METRIC_LOSS_W]; return fitness; } //GA fitness is equal to lossW in this case. Modifiable
    void accumulatePopulationFitness( const std::vector<NeuralWebSP>& population ); //calls calculateFitness() in all the nets in the population and adds it. Only used from Metrics that represent the total, used for normalizing the fitnesses in roulette selection
};

#endif //METRICS_HPP
