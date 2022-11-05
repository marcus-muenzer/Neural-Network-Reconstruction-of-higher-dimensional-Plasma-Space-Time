###  INFORMATION FOR TRACKING/TRAINING_DATA  ###

This directory is intended for saving the training data/trajectories.
It can also be used as artifact URI for tracking them with mlflow.

Naming convention for pytorch models:
mlflow-runid_training_data.h5

Every data file contains two datasets:
    - st: spacetimes of trajectories
    - U: MHD states of trajetories
