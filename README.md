# DeepFloodPrediction
This project aims at using Deep Learning methods to replace Shallow Water Equation(SWE) solvers in flood prediction tasks to boost the prediction speed.

## Components
The codes here are made up of three main parts: Simulation of flood with SWE solver, training of models, model evaluation.
Training data and test data are stored in (https://drive.google.com/drive/folders/1YoUBOwVOBgkOKDQSRtwJjAlvZYn99jT6?usp=sharing).

## Prequisites
The simulation of flood depends on the Shallow Water Equation Solver-Anuga Hydro(https://github.com/GeoscienceAustralia/anuga_core). As ANUGA is developed in Python 2 environment, the simulation part is written in Python 2.7.
The Deep Learning architecture and training is precessed with Pytorch. Thus pytorch is needed to run the codes.

## Overview of Codes
In `floodsimulation` file, you can find the code used for the simulation of floods. In `models`, the architectures of the models mentioned in the paper are presented. In `trainingcode`, peole can find the training details for the architectures.Trained models, training data and test data and the covariance matrix used for the posterior adjustment are stored in Google Drive(https://drive.google.com/drive/folders/1YoUBOwVOBgkOKDQSRtwJjAlvZYn99jT6?usp=sharing). To evaluate the results, `model_test.ipynb` amd `model_posterior_update.ipynb` can be applied.

## Citation
If you find this helpful, please kindly cite the paper:


## License
MIT
