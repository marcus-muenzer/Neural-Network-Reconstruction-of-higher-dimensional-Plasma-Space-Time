"""
NAME:
    evaluate_predictions

DESCRIPTION:
    Provides analysis functionality for evaluating predictions along different metrics.
"""

import torch
from audtorch.metrics import PearsonR


def get_metrics(y_true, y_pred, metrics={'mse', 'mae', 'pc', 'mare'}):
    """
    Evaluates the given prediction against the ground truth.
    It can compute the following measurements:
        Mean Squared Error
        Mean Absolute Error
        Pearson Correlation
        Mean Absolute Relative Error

    Parameters:
        y_true: Ground truth (correct) target values.
        y_pred: Estimated target values.
        metrics: Dictionary that tells which metrics should be computed.
                 Possible keys are: 'mse', 'mae', 'pc', 'mare'
    Returns:
        Dictionary that contains all specified metrics.
        It can happen that there are nan-values in the list.
    """

    # Ensure data is a pytorch tensor
    if type(y_true) != torch.Tensor: y_true = torch.tensor(y_true)
    if type(y_pred) != torch.Tensor: y_pred = torch.tensor(y_pred)

    ret = {}

    # MSE
    if 'mse' in metrics:
        ret['mse'] = torch.nn.MSELoss()(y_true, y_pred).item()

    # MAE
    if 'mae' in metrics:
        ret['mae'] = torch.nn.L1Loss()(y_true, y_pred).item()

    # PC
    if 'pc' in metrics:
        ret['pc'] = PearsonR()(y_true, y_pred).item()

    # Mean Absolute Relative Error
    if 'mare' in metrics:
        # Replace zeros with very small value to avoid division by zero
        y_true[y_true == 0] = 1e-10
        abs_rel_err = torch.abs(y_pred - y_true) / torch.abs(y_true)
        ret['mare'] = torch.mean(abs_rel_err).item()

    return ret
