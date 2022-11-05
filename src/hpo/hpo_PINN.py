"""
NAME
    hpo_PINN

DESCRIPTION
    This module provides optuna-based hyperparameter optimization methods for the PINN architecture.
    It can optimize the following reconstruction models: MLP, STB_MLP, Transformer

"""

import optuna
import globals.constants as const
from training.train_PINN import train_PINN


def objective_PINN(trial):
    """ Objective function for tuning the PINN in the optuna-framework. """

    # Model types
    model_type = trial.suggest_categorical('model_type', ['MLP', 'STB_MLP', 'Transformer'])

    # Embedding layers
    # Only for STB_MLP
    embedding_layers = []
    if model_type == 'STB_MLP':
        n_embedding_layers = trial.suggest_int('n_embedding_layers', 1, 3)
        layer_size = 16
        for n in range(n_embedding_layers):
            layer_size = trial.suggest_int('embedding_neurons_{}'.format(n), layer_size, 32)
            embedding_layers.append(layer_size)

    # Prediction layers
    prediction_layers = []
    minimal_size = 124
    maximal_size = 320
    n_prediction_layers = trial.suggest_int('n_prediction_layers', 3, 6)
    # Transformer
    transformer_neurons = None
    if model_type == 'Transformer':
        transformer_neurons = trial.suggest_int('transformer_neurons', minimal_size, maximal_size)
        prediction_layers = [transformer_neurons] * n_prediction_layers
    else:
        # No transformer
        decrease_size = False
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

    # Number of epochs
    n_epochs = trial.suggest_int('n_epochs', 25000, 25000)

    # Number of warm-up epochs
    n_warm_up_epochs = trial.suggest_int('n_warm_up_epochs', 50, 200)

    # Curriculum training
    curr_method = const.curr_method
    if curr_method == 'hpo':
        curr_method = trial.suggest_categorical('curr_method', [None, 'colloc_inc_points', 'colloc_cuboid', 'colloc_cylinder', 'phys', 'trade_off', 'coeff', 'num_diff'])
    curr_axis = const.curr_axis
    if curr_method == 'colloc_cuboid' and curr_axis == 'hpo':
        curr_axis = trial.suggest_categorical('curr_axis', ['x', 'y', 't'])

    # Viscosity
    # Not relevant if curriculum method 'coeff' is activited -> which schedules viscosity itself
    visc_nu = trial.suggest_float('visc_nu', 0, .004) if curr_method != 'coeff' else None

    # Resistivity
    # Not relevant if curriculum method 'coeff' is activited -> which schedules resistivity itself
    resis_eta = trial.suggest_float('resis_eta', .001, .007) if curr_method != 'coeff' else None

    # Ratio of specific heats (pressure equation) for calculation of physical loss
    gamma = trial.suggest_float('gamma', 1.3, 1.7)

    # Loss type for reduction of physical loss
    loss_type = trial.suggest_categorical('loss_type', ['mse', 'mlogcosh'])

    # Share for PINN to weight physical loss
    # Possible shares are in ~(.5, .9)
    # Center of share distribution = .7
    # Not relevant if curriculum method 'trade_off' is activited -> which schedules share itself
    share_phys = trial.suggest_float('share_phys', .51, .9) if curr_method != 'trade_off' else None

    # Learning rate
    lr = trial.suggest_float('lr', 1e-5, 1e-3)

    config = {
        'trial': trial,
        'model_type': model_type,
        'embedding_layers': embedding_layers,
        'layers': prediction_layers,
        'transformer_neurons': transformer_neurons,
        'n_epochs': n_epochs,
        'n_warm_up_epochs': n_warm_up_epochs,
        'curr_method': curr_method,
        'curr_axis': curr_axis,
        'visc_nu': visc_nu,
        'resis_eta': resis_eta,
        'gamma': gamma,
        'loss_type': loss_type,
        'share_phys': share_phys,
        'lr': lr,
    }

    try:
        metric = train_PINN(config)
    except (RuntimeError, ValueError):
        # Stop mlflow run
        const.mlflow.end_run()

        # Return really bad metric to avoid similar trials in the future
        metric = 1000 if const.direction == "minimize" else 0
        return metric

    return metric


def hpo_PINN(n_trials=10):
    """
    Starts HPO for PINN architecture.

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
    study.optimize(objective_PINN, n_trials=n_trials, gc_after_trial=True)

    return study
