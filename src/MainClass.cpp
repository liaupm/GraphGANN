#include "MainClass.hpp"
#include "Metrics.hpp" //progEvaluateEnsemble()
#include "NeuralWebEnsemble.hpp" //ensemble programs

#include <algorithm> //next_permutation in makeAllCombinations()

//static
std::vector<ProgramPointer> MainClass::programs( { MainClass::progTrainOnly, MainClass::progKFold, MainClass::progKFoldFair, MainClass::progKFoldFairEnsemble, MainClass::progTrainAndSaveNets, MainClass::progEvaluateEnsemble, MainClass::progPredictOutputsEnsemble, MainClass::progSplitDataset, MainClass::progMakeInputCombinations } );



///////////////////////////////////////////////////////////////////////////////////////////////// *PROGRAMS* //////////////////////////////////////////////////////////////////////////////////////////

//================================================================ TRAINING PROGRAMS ====================================================================
void MainClass::progTrainOnly()
{
    std::cout << "program = train only\n\n";
//---init
    emitter.resetTotalMetrics();
    params.k = 1;
    currentFold = 0;

//---make a copy of the base dataset for training
    partialDatasets.push_back( std::make_shared<Dataset>( &dataset ) );

//---train one net with multiGA and delete it after having accessed its data
    trainNet( 0, true );
    printMetrics( 0, 0, 0, false );
    partialDatasets.clear();
}

void MainClass::progKFold()
{
    std::cout << "program = weighted k-fold with k = " << parser.getIntParam( "k" ) << "\n\n";
//---init
    emitter.resetTotalMetrics();
    params.k = parser.getIntParam( "k" );
    
//---make a copy of dataset and make folds
    partialDatasets.push_back( std::make_shared<Dataset>( &dataset ) );
    partialDatasets.back()->makeStratifiedKFold( *randomnessHandler.getDataDistributionRE(0), params.k );

//---for each fold, train a net and delete it after having accessed its data
    for( currentFold = 0; currentFold < params.k; currentFold++ )
    {
        std::cout << "fold " << currentFold << "\n";
        trainNet( 0, false );
        printMetrics( 0, 0, currentFold, false );
        partialDatasets.back()->nextFold();
    }
    emitter.printMeanKfoldValues( params.k );
    partialDatasets.clear();
}

void MainClass::progKFoldFair()
{
    std::cout << "program = weighted k-fold with k = " << parser.getIntParam( "k" ) << " FAIR\n\n";
//---init
    emitter.resetTotalMetrics();
    params.k = parser.getIntParam( "k" );
    
//---make a copy of dataset and make folds
    Dataset wholeDataset( &dataset );
    wholeDataset.makeStratifiedKFold( *randomnessHandler.getDataDistributionRE(0), params.k );

//---for each fold
    for( currentFold = 0; currentFold < params.k; currentFold++ )
    {
        std::cout << "fold " << currentFold << "\n";
    //---make separate datasets with the current train and test fold. Then, make the test dataset all test
        partialDatasets.push_back( std::make_shared<Dataset>( wholeDataset.getTrainingFold().get() ) );
        partialDatasets.push_back( std::make_shared<Dataset>( wholeDataset.getTestFold().get() ) );
        partialDatasets[ currentFold * 2 + 1 ]->makeAllTest();

    //---train a net and delete it after having accessed its data
        trainNet( currentFold * 2, true );
        printMetrics( currentFold * 2, currentFold * 2 + 1, currentFold, false );
        wholeDataset.nextFold();
    }
    emitter.printMeanKfoldValues( params.k );
    partialDatasets.clear();
}

void MainClass::progKFoldFairEnsemble()
{
    std::cout << "program = complete k-fold fair with ensemble of " << parser.getIntParam( "netNum" ) << " nets starting at " << parser.getIntParam( "netIndex" )  << "\n\n";
//---init
    emitter.resetTotalMetrics();
    Emitter emitterForEnsemble; 
    emitterForEnsemble.setHeader( net->getHeader() );
    emitterForEnsemble.resetTotalMetrics();

    params.k = parser.getUintParam( "k" );
    uint maxNetIndex =  parser.getUintParam( "netIndex" ) + parser.getUintParam( "netNum");

    Metrics summaryEnsembleFairMetrics( 0.0 ); 
    Metrics summaryAvgFairMetrics( 0.0 ); 


//---make a copy of dataset and make folds
    Dataset wholeDataset( &dataset );
    wholeDataset.makeStratifiedKFold( *randomnessHandler.getDataDistributionRE(0), params.k );

//---for each fold
    for( currentFold = 0; currentFold < params.k; currentFold++ )
    {
        emitter.printMessage( "\n\n================================================fold " + std::to_string( currentFold ) );

    //---save untrained dataset (optional)
        //emitter.printDataset( wholeDataset.getTrainingFold(), wholeDataset.getTrainingFold(), FLAG_NULL, MAKE_FILENAME( OUTFILE_DATASPLIT_TRAIN, currentFold ) );
        //emitter.printDataset( wholeDataset.getTestFold(), wholeDataset.getTestFold(), FLAG_NULL, MAKE_FILENAME( OUTFILE_DATASPLIT_TEST, currentFold ) );

    //---make separate datasets with the current train and test fold. Then, make the test dataset all test
        partialDatasets.push_back( std::make_shared<Dataset>( wholeDataset.getTrainingFold().get() ) );
        partialDatasets.push_back( std::make_shared<Dataset>( wholeDataset.getTestFold().get() ) );
        partialDatasets[ currentFold * 2 + 1 ]->makeAllTest();

    //---train and load nets in ensemble
        NeuralWebEnsemble ensemble( parser );
        for( currentNetIndex = parser.getUintParam( "netIndex" ); currentNetIndex < maxNetIndex; currentNetIndex++ ) //for each net to train
        {
            std::cout << "net " << currentNetIndex << "\n";
            //---train and evaluate net
            trainNet( currentFold * 2, true );
            printMetrics( currentFold  * 2, currentFold * 2 + 1, currentNetIndex, false, "_" + std::to_string( currentFold ) );
            ensemble.addMemberNet( bestNet );
        }
    //---evaluate ensemble
        emitter.printMessage( "\nENSEMBLE METRICS FOR FOLD " + std::to_string( currentFold) );
        //val
        ensemble.evaluateWeighted( partialDatasets[ currentFold * 2 ]->getInputs(), partialDatasets[ currentFold * 2 ]->getOutputs(), partialDatasets[ currentFold * 2 ]->getInstanceWeights(), INDEX_SET_VAL );
        emitter.printAll( &ensemble, partialDatasets[ currentFold * 2 ].get(), INDEX_SET_VAL, currentFold, false, parser.getIntParam( "savePredictions"), true );
        //fair test
        ensemble.evaluateWeighted( partialDatasets[ currentFold * 2 + 1 ]->getInputs(), partialDatasets[ currentFold * 2 + 1 ]->getOutputs(), partialDatasets[ currentFold * 2 + 1 ]->getInstanceWeights(), INDEX_SET_TEST );
        emitter.printAll( &ensemble, partialDatasets[ currentFold * 2 + 1 ].get(), INDEX_SET_TEST, currentFold, false, parser.getIntParam( "savePredictions"), true );

    //evaluate separately each of the ensemble member nets and average
        //val
        Metrics avgValMetrics = ensemble.averageMetrics( partialDatasets[ currentFold * 2 ]->getInputs(), partialDatasets[ currentFold * 2 ]->getOutputs(), partialDatasets[ currentFold * 2 ]->getInstanceWeights(), INDEX_SET_VAL );
        emitter.printExternalMetrics( &avgValMetrics, "avg val  " );
        //fair test
        Metrics avgFairMetrics = ensemble.averageMetrics( partialDatasets[ currentFold * 2 + 1 ]->getInputs(), partialDatasets[ currentFold * 2 + 1 ]->getOutputs(), partialDatasets[ currentFold * 2 + 1 ]->getInstanceWeights(), INDEX_SET_TEST );
        emitter.printExternalMetrics( &avgFairMetrics, "avg fair " );

        summaryEnsembleFairMetrics.add( &ensemble.getTestMetrics() );
        summaryAvgFairMetrics.add( &avgFairMetrics );

        wholeDataset.nextFold();
    }
    summaryEnsembleFairMetrics.scale( 1.0 / params.k );
    summaryAvgFairMetrics.scale( 1.0 / params.k );
    emitter.printMessage( "\n\n//////////////////FINAL AVERAGE OF ALL THE FOLDS/////////////////" );
    emitter.printExternalMetrics( &summaryEnsembleFairMetrics, "final ensemble fair " );
    emitter.printExternalMetrics( &summaryAvgFairMetrics, "final avg fair " );

    partialDatasets.clear();
}

void MainClass::progTrainAndSaveNets()
{
    std::cout << "program = train and save " << parser.getIntParam( "netNum" ) << " nets starting at " << parser.getIntParam( "netIndex" )  << "\n\n";
//---init
    emitter.resetTotalMetrics();
    params.k = parser.getIntParam( "netIndex" ) + parser.getIntParam( "netNum" );
    parser.setIntParam( "saveBestNet", 1 ); //in this program, nets are always saved 

//---for each net to train
    for( currentNetIndex = parser.getIntParam( "netIndex" ); currentNetIndex < params.k; currentNetIndex++ )
    {
        std::cout << "net " << currentNetIndex << "\n";
    //---create new dataset copy for training each net
        partialDatasets.push_back( std::make_shared<Dataset>( &dataset ) );

    //---load and use corresponding fair test for an online fair evaluation of each net while training
        if( parser.getIntParam( "datasetIndex" ) != INDEX_WHOLE_DATASET ) //if there is a test split
        {    
        //---train a net and delete it after having accessed its data
            trainNet( ( currentNetIndex - parser.getIntParam( "netIndex" ) ) * 2, true );    
        //---load fair dataset
            parser.parseDataset( FLAG_NULL, MAKE_FILENAME( OUTFILE_DATASPLIT_TEST, parser.getIntParam( "datasetIndex" ) ) );
            partialDatasets.push_back( std::make_shared<Dataset>( parser.getInputs(), parser.getOutputs(), parser.getInstanceWeights(), parser.getRealParam( "classThreshold" ) ) );
            partialDatasets.back()->weightInstances( parser.getRealParam( "instanceWeightByOutput" ), parser.getRealParam( "instanceWeightByInput" ) );
            partialDatasets.back()->makeAllTest();
        //---evaluation of trained net
            printMetrics( ( currentNetIndex - parser.getIntParam( "netIndex" ) ) * 2, ( currentNetIndex - parser.getIntParam( "netIndex" ) ) * 2 + 1, currentNetIndex, false );
        }
        else //there is not a test split
        {
            trainNet( currentNetIndex - parser.getIntParam( "netIndex" ), true );
            printMetrics( currentNetIndex - parser.getIntParam( "netIndex" ), currentNetIndex - parser.getIntParam( "netIndex" ), currentNetIndex, false );
        }
    }
    partialDatasets.clear();
}
//================================================================ end of TRAINING PROGRAMS ====================================================================






//================================================================ EVALUATION AND PREDICTION PROGRAMS ==============================================================
void MainClass::progEvaluateEnsemble()
{
    std::cout << "program = evaluate ensemble of " << parser.getIntParam( "netNum" ) << " nets starting at " << parser.getIntParam( "netIndex" )  << "\n\n";
//---init
    params.k = parser.getIntParam( "netIndex" ) + parser.getIntParam( "netNum" );

//---load all the nets in the ensemble
    NeuralWebEnsemble ensemble( parser );
    for( uint n = parser.getIntParam( "netIndex" ); n < params.k; n++ )
        ensemble.addMemberNet( NeuralWebSP( loadTrainedNet( n ) ) );
    std::cout << "nets loaded\n";

//---copy train dataset
    partialDatasets.push_back( std::make_shared<Dataset>( &dataset ) );
    partialDatasets.back()->makeAllTraining();

//---parse fair dataset
    parser.parseDataset( FLAG_NULL, MAKE_FILENAME( OUTFILE_DATASPLIT_TEST, parser.getIntParam( "datasetIndex" ) ) );
    partialDatasets.push_back( std::make_shared<Dataset>( parser.getInputs(), parser.getOutputs(), parser.getInstanceWeights(), parser.getRealParam( "classThreshold" ) ) );
    partialDatasets.back()->weightInstances( parser.getRealParam( "instanceWeightByOutput"), parser.getRealParam( "instanceWeightByInput") );
    partialDatasets.back()->makeAllTest();

//---evaluate ensemble
    //train + val (named "train")
    ensemble.evaluateWeighted( partialDatasets[0]->getInputs(), partialDatasets[0]->getOutputs(), partialDatasets[0]->getInstanceWeights(), INDEX_SET_TRAIN );
    emitter.printAll( &ensemble, partialDatasets[0].get(), INDEX_SET_TRAIN, 0, false, parser.getIntParam( "savePredictions"), true );
    //fair test
    ensemble.evaluateWeighted( partialDatasets[1]->getInputs(), partialDatasets[1]->getOutputs(), partialDatasets[1]->getInstanceWeights(), INDEX_SET_TEST );
    emitter.printAll( &ensemble, partialDatasets[1].get(), INDEX_SET_TEST, 0, false, parser.getIntParam( "savePredictions"), true );

//evaluate separately each of the ensemble member nets and average
    //train + val (named "train")
    Metrics avgTrainMetrics = ensemble.averageMetrics( partialDatasets[0]->getInputs(), partialDatasets[0]->getOutputs(), partialDatasets[0]->getInstanceWeights(), INDEX_SET_TRAIN );
    emitter.printExternalMetrics( &avgTrainMetrics, "avg train " );
    //fair test
    Metrics avgFairMetrics = ensemble.averageMetrics( partialDatasets[1]->getInputs(), partialDatasets[1]->getOutputs(), partialDatasets[1]->getInstanceWeights(), INDEX_SET_TEST );
    emitter.printExternalMetrics( &avgFairMetrics, "avg fair " );

//---clean
    partialDatasets.clear();
    generatedDatasets.clear();
}

void MainClass::progPredictOutputsEnsemble()
{
    std::cout << "prediction with ensemble of " << parser.getIntParam( "netNum" ) << " nets starting at " << parser.getIntParam( "netIndex" )  << "\n\n";
//---init
    params.k = parser.getIntParam( "netIndex" ) + parser.getIntParam( "netNum" );

//---load all the nets in the ensemble
    NeuralWebEnsemble ensemble( parser );
    for( uint n = parser.getIntParam( "netIndex" ); n < params.k; n++ )
        ensemble.addMemberNet( NeuralWebSP( loadTrainedNet( n) ) );
    
//---parse dataset (input combinations) 
    parser.parseDataset( FLAG_NULL, MAKE_FILENAME( OUTFILE_INPUTCOMBIS, parser.getIntParam( "zerosNum" ) ) );
    partialDatasets.push_back( std::make_shared<Dataset>( parser.getInputs(), parser.getOutputs(), parser.getInstanceWeights(), parser.getRealParam( "classThreshold" ) ) );

//---generate and save predicted dataset
    generatedDatasets.emplace_back( new Dataset( parser.getInputs(), {}, {}, parser.getRealParam( "classThreshold" ) ) );
    generatedDatasets[0]->generateOutputs( &ensemble );
    emitter.printDataset( partialDatasets[0].get(), generatedDatasets[0].get(), FLAG_DATA_ALL_FILTER, MAKE_FILENAME3( OUTFILE_DATAPRED_FINAL, parser.getIntParam( "zerosNum" ), parser.getIntParam( "netNum" ), parser.getIntParam( "netIndex" ) ), parser.getRealParam( "predictionPrintThresholdL" ), parser.getRealParam( "predictionPrintThresholdU" ) );
    
//---clean
    partialDatasets.clear();
    generatedDatasets.clear();
}
//========================================================== end of EVALUATION AND PREDICTION PROGRAMS =======================================================






//================================================================ DATASET PROGRAMS ==============================================================
void MainClass::progSplitDataset()
{
    std::cout << "program = save k-fold splited dataset with k = " << parser.getIntParam( "k" ) << "\n\n";
//---init
    params.k = parser.getIntParam( "k" );
    
//---make folds
    Dataset wholeDataset( &dataset );

    if( params.k >= partialDatasets.back()->getInputs().size() ) //if k => number of cases, leave one out
         wholeDataset.leaveOneOut();
    else //stratified k-fold otherwise
        wholeDataset.makeStratifiedKFold( *randomnessHandler.getDataDistributionRE(0), params.k );

//---for each fold, save train and test splits
    for( currentFold = 0; currentFold < params.k; currentFold++ )
    {
        std::cout << "making data split " << currentFold << "\n";
        emitter.printDataset( wholeDataset.getTrainingFold().get(), wholeDataset.getTrainingFold().get(), FLAG_NULL, MAKE_FILENAME( OUTFILE_DATASPLIT_TRAIN, currentFold ) );
        emitter.printDataset( wholeDataset.getTestFold().get(), wholeDataset.getTestFold().get(), FLAG_NULL, MAKE_FILENAME( OUTFILE_DATASPLIT_TEST, currentFold ) );
        wholeDataset.nextFold();
    }
}

void MainClass::progMakeInputCombinations()
{
    std::cout << "program = save all posible input combinations with " << parser.getIntParam( "zerosNum" ) << " zeros \n\n";
    Dataset combinationsDataset( {}, {}, {}, parser.getRealParam( "classThreshold") );
    combinationsDataset.makeInputCombinations( dataset.getInputs()[0].size(), parser.getIntParam( "zerosNum" ) );
    
    switch( parser.getUintParam( "combisFilterMode") ) //filter the input combinations
    {
        case 1: //remove equal
            combinationsDataset.filterInstancesEqual( &dataset );
        case 2: //remove supersets
            combinationsDataset.filterInstancesSuperset( &dataset, parser.getUintParam( "combisFilterInput"), parser.getUintParam( "combisFilterClass") );
            break;
        default: //no filter
            break;
    }
    
    emitter.printDataset( combinationsDataset, combinationsDataset, 0, MAKE_FILENAME( OUTFILE_INPUTCOMBIS, parser.getIntParam( "zerosNum" ) ) );
}
//================================================================ end of DATASET PROGRAMS ==============================================================









///////////////////////////////////////////////////////////////////////// *BASIC* ///////////////////////////////////////////////////////////////////////////////////////////////////
void MainClass::init()
{
//---parse untrained net
    parser.parseNetwork( FLAG_NULL );
    net = std::make_shared<NeuralWeb>( parser );

 //---if convert to fully-connected ff option
    if( parser.getIntParam( "make_ff" ) == 1 )
    {
        if( parser.getIntParam( "program" ) <= 3 ) //if training program, create the ff net and save it
        {
            net->convertToFF( { parser.getUintParam( "ff_hidden0" ),  parser.getUintParam( "ff_hidden1" ), parser.getUintParam( "ff_hidden2" ), parser.getUintParam( "ff_hidden3" ), } );
            Emitter::printNetwork( net.get(), FLAG_NULL, FILE_NET_FF );
        }
        else //if evaluating program, load the ff net
        {
            parser.parseNetwork( FLAG_NULL, FILE_NET_FF );
            net = std::make_shared<NeuralWeb>( parser );
        }
    }

//---if swapt input layer option
    else if( parser.getIntParam( "crazy" ) == 1  )
    {
        if( parser.getIntParam( "program" ) <= 3 ) //if training program, create the crazy net and save it
        {
            RandomEngine crazyRE( parser.getIntParam( "crazySeed" ) );
            net->swapInputLayer( crazyRE );
            Emitter::printNetwork( net.get(), FLAG_NULL, FILE_NET_CRAZY );
        }
        else //if evaluating program, load the crazy net
        {
            parser.parseNetwork( FLAG_NULL, FILE_NET_CRAZY );
            net = std::make_shared<NeuralWeb>( parser );
        } 
    }

//---set headers
    parser.setHeader( net->getHeader() );
    emitter.setHeader( net->getHeader() );

//---parse non-weighted dataset and weight it
    //---load base dataset (whole dataset or previously made training split)
    if( parser.getIntParam( "datasetIndex" ) == INDEX_WHOLE_DATASET ) //load wholse dataset
        parser.parseDataset( FLAG_NULL );
    else //load a previously made training split
        parser.parseDataset( FLAG_NULL, MAKE_FILENAME( OUTFILE_DATASPLIT_TRAIN, parser.getIntParam( "datasetIndex" ) ) );

    dataset = Dataset( parser.getInputs(), parser.getOutputs(), parser.getInstanceWeights(), parser.getRealParam( "classThreshold" ) );
    dataset.weightInstances( parser.getRealParam( "instanceWeightByOutput"), parser.getRealParam( "instanceWeightByInput") );

//---save the weighted dataset to file
    emitter.printDataset( dataset, dataset, FLAG_DATA_WEIGHT, DEFAULT_PARSER_INFILE_DATA_W );
    std::cout << "instance weight done\n";


//---init members
    popCreator.init( net, randomnessHandler, parser.getRealParam( "maxActivation" ), parser.getRealParam( "minActivation" ) );

    std::cout << "init done\n";
}

NeuralWeb* MainClass::loadTrainedNet( uint netIndex )
{
//---load the trained net (no arc signs info)
    parser.parseNetwork( FLAG_NET_TRAINED, MAKE_FILENAME( OUTFILE_NET_W, netIndex ), MAKE_FILENAME( OUTFILE_NET_ACTI, netIndex ), MAKE_FILENAME( OUTFILE_NET_METRICS, netIndex ) );
    NeuralWeb* trainedNetBad = new NeuralWeb( parser );

//---transfer the scale and weight values to a copy of the untrained reference net (with arc sign info)
    NeuralWeb* trainedNet = new NeuralWeb( net.get() );
    trainedNet->transferParams( trainedNetBad );

//---load metrics into the net
    trainedNet->setTestMetrics( parser.getMetrics() );
    //trainedNet->setMetrics( parser.getMetrics(), INDEX_SET_VAL );

//---clean
    delete trainedNetBad;
    return trainedNet;
}

void MainClass::trainNet( uint datasetIndex, bool bMakeValSplit )
{
    uint qualityCriterion = parser.getUintParam( "bestNetCriterion" );
    uint trials = 0; //number of training trials done
    double bestMetric = qualityCriterion < METRIC_LOSS_NUM ? 100.0 : 0.0; //arbitrary big or small value (depending on the metric)
    uint bestGenerations = 0; //smallest posible number of generations. Quality criterion
    multiGa = nullptr; //DELETE

    do
    {
//---1-train
    //---separate val set for early stopping
        if( bMakeValSplit )
            partialDatasets[datasetIndex]->makeSingleFold( *randomnessHandler.getDataDistributionRE(0), parser.getIntParam( "validationInstanceNum" ) );

    //---create initial random ga populations
        popCreator.reseed( randomnessHandler ); //optional
        std::vector<GeneticAlgorithmSP> populations; 
        for( int p = 0; p < parser.getIntParam( "gaNum" ); p++ )
        {
            std::vector<NeuralWebSP> pop = popCreator.createPopulation( parser.getIntParam( "popSize" ) );
            populations.push_back( std::make_shared<GeneticAlgorithm>( pop, parser, randomnessHandler ) );
        }

    //---load populations in a multi ga and train
        MultiGaSP multiGaTemp = std::make_shared<MultiGa>( populations, parser, randomnessHandler );
        multiGaTemp->trainAndTrack( partialDatasets[datasetIndex].get(), parser.getUintParam( "generationNum"), parser.getUintParam( "mixNum" ) );

//---2-decide if metrics are good enough
        uint bestGenerationsCandidate;
        NeuralWebSP bestNetCandidate = multiGaTemp->getHistoricalBestNet( bestGenerationsCandidate, parser.getUintParam( "generationsRequired" ) - DEFAULT_MAINC_GENERATIONS_LESS ); 
        double bestMetricCandidate = bestNetCandidate->getTestMetrics().getMember( qualityCriterion );
        //double bestMetricCandidate = bestNetCandidate->getMetrics( INDEX_SET_VAL )->getReflectedMember( qualityCriterion );

        //condition 1: metric better than bestMetric and ( gen >= required gen or gen >= bestGen )
        bool condition1 = ( ( bestMetricCandidate < bestMetric && qualityCriterion < METRIC_LOSS_NUM ) || ( bestMetricCandidate > bestMetric && qualityCriterion >= METRIC_LOSS_NUM ) ) && ( bestGenerationsCandidate >= parser.getUintParam( "generationsRequired" ) || bestGenerationsCandidate >= bestGenerations );
        //condition 2: metric == best metric and gen > bestGen
        bool condition2 = bestMetricCandidate == bestMetric && bestGenerationsCandidate > bestGenerations;
        //condition 3: generations >= required generations and bestGenerations < requiered generations  
        bool condition3 = bestGenerationsCandidate >= parser.getUintParam( "generationsRequired" ) && bestGenerations < parser.getUintParam( "generationsRequired" );

        if( trials == 0 || condition1 || condition2 || condition3 ) //if conditions met, replace best net
        {
            bestMetric = bestMetricCandidate;
            bestNet = bestNetCandidate;
            bestGenerations = bestGenerationsCandidate;
            multiGa = multiGaTemp;
        }

        std::cout << "trial " << trials << " with val acc: " << bestNetCandidate->getTestMetrics().getMember( INDEX_METRIC_ACC_W ) << " and val loss " << bestNetCandidate->getTestMetrics().getMember( INDEX_METRIC_LOSS_W ) << " in generation " << bestGenerationsCandidate << "\n";
        trials++;
    }
    while( trials < parser.getUintParam("valMaxTrials") && ( bestGenerations < parser.getUintParam( "generationsRequired" ) || ( bestMetric > parser.getRealParam("requiredValQuality") && qualityCriterion < METRIC_LOSS_NUM )  || ( bestMetric < parser.getRealParam("requiredValQuality") && qualityCriterion >= METRIC_LOSS_NUM ) ) );
    //repeat while quality or generations not achieved and not max trials reached
}

void MainClass::printMetrics( uint datasetIndex, uint fairDatasetIndex, uint netIndex, bool bEnsemble, const std::string& sufix )
{
    //assumed that bestNet contains the best historical net in val set before call
//---train
    if( ! bEnsemble )
        emitter.printAll( bestNet.get(), partialDatasets[datasetIndex].get(), INDEX_SET_TRAIN, netIndex, false, false, bEnsemble, sufix ); //train predictions not saved. Evaluation done during training
//---val   
    emitter.printAll( bestNet.get(), partialDatasets[datasetIndex].get(), INDEX_SET_VAL, netIndex, parser.getIntParam( "saveBestNet" ), parser.getIntParam( "savePredictions" ), bEnsemble, sufix ); //saving the net once is enough. Evaluation done in historical track
//---fair
    if( datasetIndex != fairDatasetIndex )
    {
        bestNet->evaluateWeighted( partialDatasets[fairDatasetIndex]->getInputs(), partialDatasets[fairDatasetIndex]->getOutputs(), partialDatasets[fairDatasetIndex]->getInstanceWeights(), INDEX_SET_TEST ); //evaluation required
        emitter.printAll( bestNet.get(), partialDatasets[fairDatasetIndex].get(), INDEX_SET_TEST, netIndex, false, parser.getIntParam( "savePredictions" ), bEnsemble, sufix );
    }
}
//=============================== *end of BASIC* =========================================