"""
NAME
    hpo_cGAN

DESCRIPTION
    This module provides an optuna-based hyperparameter optimization method for the cGAN architecture.
    It can optimize the following models within the cGAN architecture:
        Generators: MLP, STB_MLP, Transformer
        Discriminators: MLP, MHDB_MLP, Transformer
"""

import optuna
from training.train_cGAN import train_cGAN
import globals.constants as const


def objective_cGAN(trial):
    """ Objective function for tuning the cGAN in the optuna-framework. """

    # Model types
    generator_type = trial.suggest_categorical('generator_type', ['MLP', 'STB_MLP', 'Transformer'])
    discriminator_type = trial.suggest_categorical('discriminator_type', ['MLP', 'MHDB_MLP', 'Transformer'])

    # Embedding layers
    # Only for STB_MLP and MHDB_MLP
    embedding_layers = []
    if generator_type == 'STB_MLP' or discriminator_type == 'MHDB_MLP':
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
    if generator_type == 'Transformer' or discriminator_type == 'Transformer':
        transformer_neurons = trial.suggest_int('transformer_neurons', minimal_size, maximal_size)
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

    # Lambda factor inside exponential decay method
    # Used for physical residuals as discriminator input
    lambda_decay = trial.suggest_float('lambda_decay', .05, .15)

    # Share for PINN to weight physical loss
    # Possible shares are in ~(.5, .9)
    # Center of share distribution = .7
    # Not relevant if curriculum method 'trade_off' is activited -> which schedules share itself
    share_phys = trial.suggest_float('share_phys', .51, .9) if curr_method != 'trade_off' else None

    # Learning rate
    lr = trial.suggest_float('lr', 1e-6, 1e-3)

    config = {
        'trial': trial,
        'generator_type': generator_type,
        'discriminator_type': discriminator_type,
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
        'lambda_decay': lambda_decay,
        'share_phys': share_phys,
        'lr': lr,
    }

    try:
        metric = train_cGAN(config)
    except (RuntimeError, ValueError):
        # Stop mlflow run
        const.mlflow.end_run()

        # Return really bad metric to avoid similar trials in the future
        metric = 1000 if const.direction == "minimize" else 0
        return metric

    return metric


def hpo_cGAN(n_trials=10):
    """
    Starts HPO for cGAN architecture.

    Parameters:
        n_trials: number of trials

    Returns:
        study: optuna.study.Study object
               contains important information about the optimization
    """

    # Set study and storage name
    # Allows resuming studies
    study_name = const.study_name

    # Use TPE sampler
    sampler = optuna.samplers.TPESampler(multivariate=False, warn_independent_sampling=False)

    # Create or resume study
    study = optuna.create_study(study_name=study_name, storage=const.storage, load_if_exists=True,
                                direction=const.direction, sampler=sampler)

    # Optimize
    study.optimize(objective_cGAN, n_trials=n_trials, gc_after_trial=True)

    return study
