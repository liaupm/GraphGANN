#Select OS
ifeq ($(OS),Windows_NT)
    OS_NAME=windows
    COMPILER=g++
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        OS_NAME=linux
        COMPILER=g++
    endif
    ifeq ($(UNAME_S),Darwin)
        OS_NAME=osx
        COMPILER=clang++
    endif
endif

MODE_FLAGS=-O3
INCLUDE=-Iinclude
TEMP=temp
BUILD=.

OBJECTS=$(TEMP)/Function.o $(TEMP)/LossFunction.o $(TEMP)/DistributionInterface.o $(TEMP)/DistributionCombi.o $(TEMP)/RandomnessHandler.o $(TEMP)/Metrics.o $(TEMP)/Node.o $(TEMP)/Arc.o $(TEMP)/NeuralWebBase.o $(TEMP)/NeuralWeb.o $(TEMP)/NeuralWebEnsemble.o $(TEMP)/HistoricalTrack.o $(TEMP)/DatasetBase.o $(TEMP)/Dataset.o $(TEMP)/Parser.o $(TEMP)/Emitter.o $(TEMP)/PopulationCreator.o $(TEMP)/GeneticAlgorithm.o $(TEMP)/MultiGa.o $(TEMP)/MainClass.o $(TEMP)/main.o

CPP=$(COMPILER) -std=c++11 -Wall -c $(MODE_FLAGS) $(INCLUDE) -o
CPP_L=g++ -std=c++11 $(MODE_FLAGS) -o

all:
	$(CPP) $(TEMP)/Function.o src/Function.cpp
	$(CPP) $(TEMP)/LossFunction.o src/LossFunction.cpp
	$(CPP) $(TEMP)/DistributionInterface.o src/DistributionInterface.cpp
	$(CPP) $(TEMP)/DistributionCombi.o src/DistributionCombi.cpp
	$(CPP) $(TEMP)/RandomnessHandler.o src/RandomnessHandler.cpp
	$(CPP) $(TEMP)/Metrics.o src/Metrics.cpp
	$(CPP) $(TEMP)/Node.o src/Node.cpp
	$(CPP) $(TEMP)/Arc.o src/Arc.cpp
	$(CPP) $(TEMP)/NeuralWebBase.o src/NeuralWebBase.cpp
	$(CPP) $(TEMP)/NeuralWeb.o src/NeuralWeb.cpp
	$(CPP) $(TEMP)/NeuralWebEnsemble.o src/NeuralWebEnsemble.cpp
	$(CPP) $(TEMP)/HistoricalTrack.o src/HistoricalTrack.cpp
	$(CPP) $(TEMP)/DatasetBase.o src/DatasetBase.cpp
	$(CPP) $(TEMP)/Dataset.o src/Dataset.cpp
	$(CPP) $(TEMP)/Parser.o src/Parser.cpp
	$(CPP) $(TEMP)/Emitter.o src/Emitter.cpp
	$(CPP) $(TEMP)/PopulationCreator.o src/PopulationCreator.cpp
	$(CPP) $(TEMP)/GeneticAlgorithm.o src/GeneticAlgorithm.cpp
	$(CPP) $(TEMP)/MultiGa.o src/MultiGa.cpp
	$(CPP) $(TEMP)/MainClass.o src/MainClass.cpp
	$(CPP) $(TEMP)/main.o src/main.cpp

	$(CPP_L) GraphANN $(OBJECTS)
