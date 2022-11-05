"""
NAME
    hpo_baselines

DESCRIPTION
    This module provides optuna-based hyperparameter optimization methods for the baseline architectures.
    It can optimize kNN and MLP models.
"""

import optuna
import globals.constants as const
from training.train_baselines import train_kNN, train_MLP


def objective_kNN(trial):
    """ Objective function for tuning the kNN in the optuna-framework. """

    # Amount of neighbors
    n_neighbors = trial.suggest_int('n_neighbors', 1, 100)

    # Leaf size
    leaf_size = trial.suggest_int('leaf_size', 10, 30)

    # Weighting criteria for predictions
    weights = trial.suggest_categorical('weights', ['distance', 'uniform'])

    config = {
        'n_neighbors': n_neighbors,
        'leaf_size': leaf_size,
        'weights': weights
    }

    try:
        metric = train_kNN(config)
    except (RuntimeError, ValueError):
        # Stop mlflow run
        const.mlflow.end_run()

        # Return really bad metric to avoid similar trials in the future
        metric = 1000 if const.direction == "minimize" else 0
        return metric

    return metric


def objective_MLP(trial):
    """ Objective function for tuning the MLP in the optuna-framework. """

    # Prediction layers
    prediction_layers = []
    n_prediction_layers = trial.suggest_int('n_prediction_layers', 3, 6)
    decrease_size = False
    minimal_size = 124
    maximal_size = 320
    current_size = minimal_size
    for n in range(n_prediction_layers):
        if decrease_size:
            layer_size = trial.suggest_int('prediction_neurons_{}'.format(n), minimal_size, current_size)
        else:
            layer_size = trial.suggest_int('prediction_neurons_{}'.format(n), minimal_size, maximal_size)

        # Check if layer size is decreasing
        # After this point only smaller layers are allowed
        if layer_size < current_size:
            decrease_size = True
        # Update current layer size
        current_size = layer_size

        # Add layer
        prediction_layers.append(layer_size)

    # Activation function
    act_func = trial.suggest_categorical('act_func', ['leakyReLu', 'tanh'])

    # Number of epochs
    n_epochs = trial.suggest_int('n_epochs', 8000, 15000)

    # Learning rate
    lr = trial.suggest_float('lr', 1e-6, 1e-3)

    config = {
        'trial': trial,
        'layers': prediction_layers,
        'act_func': act_func,
        'n_epochs': n_epochs,
        'lr': lr,
    }

    try:
        metric = train_MLP(config)
    except (RuntimeError, ValueError):
        # Stop mlflow run
        const.mlflow.end_run()

        # Return really bad metric to avoid similar trials in the future
        metric = 1000 if const.direction == "minimize" else 0
        return metric

    return metric


def hpo_kNN(n_trials=10):
    """
    Starts HPO for kNN baseline.

    Parameters:
        n_trials: number of trials

    Returns:
        study: optuna.study.Study object
               contains important information about the optimization
    """

    # Set study and storage name
    # Allows resuming studies
    study_name = const.study_name

    # Use multivariate TPE sampler
    sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)

    # Create or resume study
    study = optuna.create_study(study_name=study_name, storage=const.storage, load_if_exists=True,
                                direction=const.direction, sampler=sampler)

    # Optimize
    study.optimize(objective_kNN, n_trials=n_trials, gc_after_trial=True)

    return study


def hpo_MLP(n_trials=10):
    """
    Starts HPO for MLP baseline.

    Parameters:
        n_trials: number of trials

    Returns:
        study: optuna.study.Study object
               contains important information about the optimization
    """

    # Set study and storage name
    # Allows resuming studies
    study_name = const.study_name

    # Use multivariate TPE sampler
    sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)

    # Create or resume study
    study = optuna.create_study(study_name=study_name, storage=const.storage, load_if_exists=True,
                                direction=const.direction, sampler=sampler)

    # Optimize
    study.optimize(objective_MLP, n_trials=n_trials, gc_after_trial=True)

    return study
