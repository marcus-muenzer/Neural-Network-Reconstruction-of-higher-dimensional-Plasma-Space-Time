"""
NAME
    intermediate_eval

DESCRIPTION
    Ckecks if an evaluation is planned. If so, it evaluates, tracks and report everything that is necessary.
"""

import optuna
import math
from evaluation.evaluate_predictions import get_metrics
import globals.constants as const


def check_for_eval(epoch, n_epochs, model, trial):
    """
    Checks if there is an evaluation of the model's performance planned in a specific epoch.
    If yes, it evaluates the performance, tracks it, and report it to HPO.

    Parameters:
        epoch: current epoch (int)
        n_epochs: number of epochs overall
                  if epoch is the last epoch overall => evaluate
        model: pytorch model
        trial: optuna.trial.Trial to report eventual results to HPO

    Returns:
        None: if no evaluation is planned at this epoch
        eval_metric: evaluation metric that is calculated on the evaluation data
    """

    if epoch % const.intermediate_steps == 0 or epoch == n_epochs - 1:
        metrics = {'mse': 0, 'mae': 0, 'pc': 0}
        for st, U in const.dataloader_eval:
            metrics_batch = get_metrics(U, model(st), {'pc', 'mse', 'mae'})
            metrics['mse'] += metrics_batch['mse']
            metrics['mae'] += metrics_batch['mae']
            metrics['pc'] += metrics_batch['pc']

        metrics['pc'] /= len(const.dataloader_eval)
        print('[Epoch ' + str(epoch) + '/' + str(n_epochs) + ']', 'PC: ', metrics['pc'], '   MSE:', metrics['mse'], '   MAE:', metrics['mae'])

        # Track metrics
        for metric in metrics:
            const.mlflow.log_metric(metric, metrics[metric])

        eval_metric = metrics[const.eval_metric]

        # Manually prune bad trial
        # Avoids optuna storage internal error
        if math.isinf(eval_metric) or math.isnan(eval_metric): raise optuna.exceptions.TrialPruned()

        # Report result to optuna
        trial.report(eval_metric, epoch)

        # Check if trail should be pruned
        if trial.should_prune():
            # Stop mlflow run
            const.mlflow.end_run()
            # Early stopping
            raise optuna.TrialPruned()

        return eval_metric

    return None
