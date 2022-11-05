"""
NAME
    train_PINN

DESCRIPTION
    This module provides functions to train models within the PINN architecture.
"""

import torch
from torch.optim.lr_scheduler import StepLR
from training.warm_up import warm_up
from utils.mlflow import start_mlflow_run, track_model, track_training_data
import utils.curriculum_learning as cl
from utils.collocation_points import sample_coll_points
from utils.model_initialization import init_recon_model
from evaluation.intermediate_eval import check_for_eval
from loss.physical_loss import calc_physical_losses
from loss.loss_weighting import get_lambda_from_share
from globals import constants as const


# Loss function
loss = torch.nn.MSELoss()
if const.cuda:
    loss.cuda()


def train_PINN(config):
    """
    Trains the all models inside the PINN architecture
    using plain data and additional physical information of collocation points.

    Parameters:
        config: dictionary containing values for the PINN training search space

    Returns:
        best measured metric of type const.eval_metric of the trained model on the evaluation data
    """

    # ==============================
    # Initializations
    # ==============================

    # Model
    model = init_recon_model(config['model_type'], config['embedding_layers'],
                             config['layers'], config['transformer_neurons'], 'tanh')

    # Best evaluation metric
    best_metric = float('inf')

    # Optimizer
    lr = config['lr']
    adam_optim = True
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=100, gamma=0.98)

    # Physical parameters
    visc_nu = config['visc_nu']
    resis_eta = config['resis_eta']
    gamma = config['gamma']
    loss_type = config['loss_type']
    # Error weighting
    lambda_phys, _ = get_lambda_from_share(config['share_phys'])

    # HPO trial
    trial = config['trial']

    # Number of epochs
    n_epochs = config['n_epochs']

    # Initialize curriculum parameters
    curr_method = config['curr_method']
    curr_step = 0
    curr_max_epoch, curr_epochs_per_step, n_steps, curr_dx, curr_dy, curr_dt = \
        cl.init_params(curr_method, config['curr_axis'], n_epochs, const.curr_steps)

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

    # ----------
    # Warm up
    # ----------
    model = warm_up(model, lr, config['n_warm_up_epochs'])

    # Start training loop
    for epoch in range(n_epochs):

        # Update curr_step
        last_curr_step = curr_step
        curr_step = cl.update_curr_step(epoch, curr_max_epoch, curr_step, curr_epochs_per_step)

        optimizer.zero_grad()

        # Predict
        predictions = model(const.st_train.data)

        # Data loss
        data_loss = loss(predictions, const.U_train.data)

        # Sample collocation points
        st_coll = sample_coll_points(curr_method, n_steps, curr_step, curr_max_epoch, epoch, curr_dx, curr_dy, curr_dt)

        # End of curriculum training
        if curr_max_epoch <= epoch: curr_method = None

        # Predict MHD states and calculate physical loss
        phys_curr_step = 2
        if curr_method == 'phys': phys_curr_step = curr_step
        if curr_method == 'coeff': visc_nu, resis_eta = cl.schedule_viscosity(curr_step), cl.schedule_resistivity(curr_step)
        if curr_method == 'num_diff' and last_curr_step < curr_step: cl.schedule_numerical_diff(curr_step)
        U_coll, _, phys_loss = calc_physical_losses(st=st_coll, model=model, visc_nu=visc_nu, resis_eta=resis_eta, gamma=gamma, loss_type=loss_type, curr_step=phys_curr_step)

        # Clamp physical error
        torch.clamp_(phys_loss, min=data_loss / 10, max=data_loss * 10)

        # Loss for model
        # Weighted combination of data & physical error
        if curr_method == 'trade_off': lambda_phys = cl.get_lambda_phys(curr_step)
        pinn_loss = (data_loss + lambda_phys * phys_loss) / (1 + lambda_phys)
        pinn_loss.backward()

        # Clip gradients by norm
        # Gradient scaling
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

        # Optimize model
        def closure():
            return pinn_loss

        optimizer.step(closure)

        # Intermediate or final evaluation
        metric = check_for_eval(epoch, n_epochs, model, trial)
        if metric and metric < best_metric:
            best_metric = metric
            track_model(model, "PINN")

        # Forward step learning rate scheduler after curriculum training
        if epoch > curr_max_epoch and adam_optim:
            scheduler.step()

        # Transition to LBFGS optimizer for last epochs
        if adam_optim and epoch > (1 - const.fraction_lbfgs) * n_epochs:
            adam_optim = False
            optimizer = torch.optim.LBFGS(model.parameters())

        if epoch % 50 == 0: print('[' + str(epoch) + '/' + str(n_epochs) + ']' + " Pinn Loss", pinn_loss.item(), "Data loss: ", data_loss.item(), "Phys Loss: ", phys_loss.item())

    # Stop mlflow run
    const.mlflow.end_run()

    return best_metric
