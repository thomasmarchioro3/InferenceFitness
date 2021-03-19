# InferenceFitness

This repository contains the code to run inference experiments on smartband data.

In order to run the code, first install the necessary libraries: the easiest way is to import the Anaconda environment `environment.yml`

Runnable code:
- `main_lstm.py`: trains a network consisting of a single LSTM layer. The employed data are time series of 7 days with 2 features (steps and calories). 
The goal of the model is to distinguish people over 30 from under 30.
