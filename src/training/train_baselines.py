"""
NAME
    train_baselines

DESCRIPTION
    This module provides functions to train baseline models only on plain data.
"""

import torch
from utils.mlflow import start_mlflow_run, track_model, track_training_data
from evaluation.evaluate_predictions import get_metrics
from evaluation.intermediate_eval import check_for_eval
from models.reconstructors.MLP import MLP
from models.reconstructors.kNN import kNNRegressor
from globals import constants as const


# Loss function
loss = torch.nn.MSELoss()
if const.cuda:
    loss.cuda()


def train_kNN(config):
    """
    Trains a kNN baseline model only using plain data.

    Parameters:
        config: dictionary containing values for the kNN training search space

    Returns:
         best measured metric of type const.eval_metric of the trained kNN model on the evaluation data
    """

    # Instantiate model from search space dictionary
    knn = kNNRegressor(config['n_neighbors'], config['leaf_size'], config['weights'])

    # Train model
    knn.train(const.st_train, const.U_train)

    # Evaluate model
    metrics = get_metrics(const.U_eval, knn.forward(const.st_eval))

    # Start mlflow run
    start_mlflow_run()

    # Track parameters
    const.mlflow.log_param("data_augmented", const.data_augmented)
    for param in config:
        const.mlflow.log_param(param, config[param])

    # Save and track training data
    track_training_data()

    # Track metrics
    for metric in metrics:
        const.mlflow.log_metric(metric, metrics[metric])

    # Save and track model
    track_model(knn, "knn", pytorch=False)

    # Stop mlflow run
    const.mlflow.end_run()

    return metrics[const.eval_metric]


def train_MLP(config):
    """
    Trains a MLP baseline model only using plain data.

    Parameter:
        config: dictionary containing values for the MLP training search space

    Returns:
         best measured metric of type const.eval_metric of the trained MLP model on the evaluation data
    """

    # ==============================
    # Initializations
    # ==============================

    # Instantiate model
    mlp = MLP(config['layers'], config['act_func'])

    # Best evaluation metric
    best_metric = float('inf')

    # Optimizer
    optimizer = torch.optim.Adam(mlp.parameters(), config['lr'])

    # Start mlflow run
    start_mlflow_run()

    # Track parameters
    const.mlflow.log_param("data_augmented", const.data_augmented)
    for param in config:
        const.mlflow.log_param(param, config[param])

    # Save and track training data
    track_training_data()

    # ==============================
    # Training procedure
    # ==============================

    trial = config['trial']
    n_epochs = config['n_epochs']

    for epoch in range(n_epochs):
        for st_train, U_train in const.dataloader_train:

            optimizer.zero_grad()

            # Predict
            predictions = mlp(st_train)

            # Loss of predictions
            mlp_loss = loss(predictions, U_train)

            # Optimize model
            mlp_loss.backward()
            optimizer.step()

        # Intermediate or final evaluation
        metric = check_for_eval(epoch, n_epochs, mlp, trial)
        if metric and metric < best_metric:
            best_metric = metric
            track_model(mlp, "mlp")

    # Stop mlflow run
    const.mlflow.end_run()

    return best_metric
