# DeepFloodPrediction
This project aims at using Deep Learning methods to replace Shallow Water Equation(SWE) solvers in flood prediction tasks to boost the prediction speed.

## Components
The codes here are made up of three main parts: Simulation of flood with SWE solver, training of models, model evaluation.
Training data and test data are stored in (https://drive.google.com/drive/folders/1YoUBOwVOBgkOKDQSRtwJjAlvZYn99jT6?usp=sharing).

## Prequisites
The simulation of flood depends on the Shallow Water Equation Solver-Anuga Hydro(https://github.com/GeoscienceAustralia/anuga_core). As ANUGA is developed in Python 2 environment, the simulation part is written in Python 2.7.
The Deep Learning architecture and training is precessed with Pytorch. Thus pytorch is needed to run the codes.

