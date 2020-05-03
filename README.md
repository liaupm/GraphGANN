# GraphGANN #

Artificial neural networks with a meaningful graph as structure trained by a genetic algorithm.

Developed by Elena Núñez Berrueco during the PhD at LIA-UPM.

Under GNU GPL v3.0 license


## Description ##

Software for training, using for prediction and evaluating artificial neural networks (ANN). The training is performed by means of a real-coded steady-state multipopulation genetic algoritm. The current version uses roulette selection of parents, MMX crossover, stochastic mutation and deterministic replacement of the worst networks.

Even though fully-connected networks can be trained, the software is aimed at custom structures not restricted to layers (they could be understood as very sparse nets with many skip connections). If meaningful graphs (known relationships) are used as the structure, the nets turn into 'white box' classifiers. At the same time, deep nets with very few connections can be obtained by this knowledge-directed sparsity. This approach can be useful for training with very small datasets. 


## Current state ##

The current version (1.0) is a prototype under-development that has been designed for a specific dataset of the Biology field. Consequently, it does not provide a big range of options and functionalities but the ones that were required. However, it can be easily extended.

The expected usage is for research in the topic. It provides a starting point for developing non-conventional deep learning and not an alternative to the existing deep learning libraries. The code is prepared for extension to several sets of weights per neuron and polynomials of order n.

It is not currently optimized nor parallelized.


## How to use ##

The main dir contains an .exe file for Windows 10. It also contains a makefile for compiling for other operative systems.

This is an standalone console application with no command line args. The "options.txt" file contains all the values for the parameters and options. 8 different programs can be run, including training nets, using trained nets for prediction, cross-validation, dataset management...(more info in the "options.txt" file).

As input files, it requires a "net.tex" file with the custom structure of the network in the shape of propositions and a "dataset.txt" fiel with the data in csv format. Currently, only binary classification problems with a single output are allowed. The inputs can be either binary or numeric while some functionalities may not work for numeric ones. Both files must by in the "data" folder, where two example files can be found. The provided "net.txt" belongs to David Ruano Gallego. The provided "dataset.txt" belongs to David Ruano Gallego, Massiel Cepeda Molero and Gad Frankel from Centre for Molecular Microbiology and Infection, Imperial College.













## Contact ##
elena.nunez@upm.es
