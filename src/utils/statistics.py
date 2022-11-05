"""
NAME:
    statistics

DESCRIPTION:
    Provides helper functions to retrieve statistical insights.
"""

import optuna
import globals.constants as const


def get_conv_epoch(trial):
    """
    Calculates the epoch in which the trial converged.

    Parameters:
        trial: optuna trial

    Returns:
        epoch in which the trial converged
    """

    invs = trial.intermediate_values
    for key in invs:
        if invs[key] == trial.value: return key
    return 0


def get_avg_conv_epoch(study_name, storage=None):
    """
    Calculates the average amount of epochs until convergence for an optuna study.
    Only respects completed, non-pruned trials.

    Parameters:
        study_name: name of the optuna study
        storage: storage where the optuna study is stored
                 if None: storage of globals module is used

    Returns:
        average amount of epochs until convergence
    """

    if not storage: storage = const.storage
    study = optuna.study.load_study(study_name, storage=storage)

    epochs = 0
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for trial in trials:
        epochs += get_conv_epoch(trial)

    return round(epochs / len(trials), 2)


def get_best_conv_epoch(study_name, storage=None):
    """
    Calculates the convergence epochs for the best trial of an optuna study.

    Parameters:
        study_name: name of the optuna study
        storage: storage where the optuna study is stored
                 if None: storage of globals module is used

    Returns:
        convergence epoch
    """

    if not storage: storage = const.storage
    study = optuna.study.load_study(study_name, storage=storage)

    return get_conv_epoch(study.best_trial)


def get_training_time(trial):
    """
    Calculates the full training time of a trial

    Parameters:
        trial: optuna trial

    Returns:
        timespan in minutes of the trial training
    """

    training_time = trial.datetime_complete - trial.datetime_start
    training_seconds = training_time.total_seconds()

    return round(training_seconds / 60, 2)


def get_avg_training_time(study_name, storage=None):
    """
    Calculates the average training time for an optuna study.
    Only respects completed, non-pruned trials.

    Parameters:
        study_name: name of the optuna study
        storage: storage where the optuna study is stored
                 if None: storage of globals module is used

    Returns:
        average training time in minutes
    """

    if not storage: storage = const.storage
    study = optuna.study.load_study(study_name, storage=storage)

    timespan = 0
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for trial in trials:
        timespan += get_training_time(trial)

    return round(timespan / len(trials), 2)


def get_best_training_time(study_name, storage=None):
    """
    Calculates the training time for the best trial of an optuna study.

    Parameters:
        study_name: name of the optuna study
        storage: storage where the optuna study is stored
                 if None: storage of globals module is used

    Returns:
        training time in minutes of the best trial
    """

    if not storage: storage = const.storage
    study = optuna.study.load_study(study_name, storage=storage)

    return get_training_time(study.best_trial)


def get_capacity(trial):
    """
    Calculates the capacity (total prediction neurons) of a trial

    Parameters:
        trial: optuna trial

    Returns:
        capacity in neurons
    """

    capacity = 0
    for i in range(trial.params['n_prediction_layers']):
        capacity += trial.params['prediction_neurons_' + str(i)]

    return capacity


def get_avg_capacity(study_name, storage=None):
    """
    Calculates the average capacity (total prediction neurons) for an optuna study.
    Only respects completed, non-pruned trials.

    Parameters:
        study_name: name of the optuna study
        storage: storage where the optuna study is stored
                 if None: storage of globals module is used

    Returns:
        average capacity in neurons
    """

    if not storage: storage = const.storage
    study = optuna.study.load_study(study_name, storage=storage)

    capacity = 0
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for trial in trials:
        capacity += get_capacity(trial)

    return round(capacity / len(trials), 2)


def get_best_capacity(study_name, storage=None):
    """
    Calculates the capacity (total prediction neurons) for the best trial of an optuna study.

    Parameters:
        study_name: name of the optuna study
        storage: storage where the optuna study is stored
                 if None: storage of globals module is used

    Returns:
        capacity in neurons of the best trial
    """

    if not storage: storage = const.storage
    study = optuna.study.load_study(study_name, storage=storage)

    return get_capacity(study.best_trial)


def get_avg_performance(study_name, storage=None):
    """
    Calculates the average performance for an optuna study.
    Only respects completed, non-pruned trials.

    Parameters:
        study_name: name of the optuna study
        storage: storage where the optuna study is stored
                 if None: storage of globals module is used

    Returns:
        average performance
    """

    if not storage: storage = const.storage
    study = optuna.study.load_study(study_name, storage=storage)

    metric = 0
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for trial in trials:
        metric += trial.value

    return round(metric / len(trials), 6)
