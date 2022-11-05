"""
NAME
    train_cGAN

DESCRIPTION
    This module provides functions to train models inside the cGAN architecture
    using plain data, collocation points, and physical informations.
"""

import torch
from torch.optim.lr_scheduler import StepLR
from utils.mlflow import start_mlflow_run, track_model, track_training_data
import utils.curriculum_learning as cl
from utils.collocation_points import sample_coll_points
from utils.model_initialization import init_recon_model, init_disc_model
from evaluation.intermediate_eval import check_for_eval
from loss.physical_loss import calc_physical_losses
from loss.loss_weighting import get_lambda_from_share
from training.warm_up import warm_up
import globals.constants as const


# Adversarial loss
adversarial_loss = torch.nn.BCELoss()
if const.cuda:
    adversarial_loss.cuda()


def train_cGAN(config):
    """
    Trains the all models inside the cGAN architecture
    using plain data and additional physical information of collocation points.

    Parameters:
        config: dictionary containing values for the cGAN training search space

    Returns:
        best measured metric of type const.eval_metric of the trained generator model on the evaluation data
    """

    # ==============================
    # Initializations
    # ==============================

    # Models
    embedding_layers = config['embedding_layers']
    layers = config['layers']
    transformer_neurons = config['transformer_neurons']
    generator = init_recon_model(config['generator_type'], embedding_layers, layers, transformer_neurons, 'tanh')
    discriminator = init_disc_model(config['discriminator_type'], embedding_layers, layers, transformer_neurons, 'tanh')

    # Best evaluation metric
    best_metric = float('inf')

    # Adversarial ground truths
    batch_size = const.st_train.shape[0]
    valid = torch.autograd.Variable(torch.full((batch_size * const.curr_factor_total_points, const.dims_mhd_state), 1.0), requires_grad=False)
    fake = torch.autograd.Variable(torch.full((batch_size * const.curr_factor_total_points, const.dims_mhd_state), 0.0), requires_grad=False)

    # Optimizers
    lr = config['lr']
    adam_optim = True
    optimizer_G = torch.optim.Adam(generator.parameters(), lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr)

    # Learning rate scheduler
    scheduler_G = StepLR(optimizer_G, step_size=100, gamma=0.98)
    scheduler_D = StepLR(optimizer_D, step_size=100, gamma=0.98)

    # Physical parameters
    visc_nu = config['visc_nu']
    resis_eta = config['resis_eta']
    gamma = config['gamma']
    loss_type = config['loss_type']
    lambda_decay = config['lambda_decay']
    # Error weighting for generator
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
    generator = warm_up(generator, lr, config['n_warm_up_epochs'])

    # Start training loop
    for epoch in range(n_epochs):

        # Update curr_step
        last_curr_step = curr_step
        curr_step = cl.update_curr_step(epoch, curr_max_epoch, curr_step, curr_epochs_per_step)

        # -----------------
        # Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample collocation points
        st_coll = sample_coll_points(curr_method, n_steps, curr_step, curr_max_epoch, epoch, curr_dx, curr_dy, curr_dt)
        batch_size_coll = st_coll.shape[0]

        # Generate a batch of fake MHD states
        # Calculate physical loss
        phys_curr_step = 2
        if curr_method == 'phys': phys_curr_step = curr_step
        if curr_method == 'coeff': visc_nu, resis_eta = cl.schedule_viscosity(curr_step), cl.schedule_resistivity(curr_step)
        if curr_method == 'num_diff' and last_curr_step < curr_step: cl.schedule_numerical_diff(curr_step)
        U_coll, phys_residuals, phys_loss = calc_physical_losses(st=st_coll, model=generator, visc_nu=visc_nu, resis_eta=resis_eta, gamma=gamma, loss_type=loss_type, curr_step=phys_curr_step)

        # Transform residuals to correct dimensionality
        # shape: torch.size([batch_size, 1])
        phys_residuals = phys_residuals.unsqueeze(1)

        # Exponentially decay physical residuals
        phys_residuals = torch.exp(-lambda_decay * phys_residuals)

        # Adversial generator loss
        if curr_method == 'trade_off': lambda_phys = cl.get_lambda_phys(curr_step)
        validity_coll = discriminator(st_coll, U_coll, phys_residuals)
        g_loss = (adversarial_loss(validity_coll, valid[:batch_size_coll]) + lambda_phys * phys_loss) / (1 + lambda_phys)

        # Optimize generator
        g_loss.backward()

        # Clip gradients by norm
        # Gradient scaling
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=2.0, norm_type=2)

        # Optimize generator
        def closure_G():
            return g_loss

        optimizer_G.step(closure_G)

        # --------------------------------------
        # Train physics-informed Discriminator
        # --------------------------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real MHD states
        validity_real = discriminator(const.st_train.data, const.U_train.data, torch.ones([batch_size, 1]))
        d_real_loss = adversarial_loss(validity_real, valid[:batch_size])

        # Measure discriminator's ability to classify fake MHD states
        validity_fake = discriminator(st_coll.detach(), U_coll.detach(), phys_residuals.detach())
        d_fake_loss = adversarial_loss(validity_fake, fake[:batch_size_coll])

        # Adversial discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Optimize discriminator
        d_loss.backward()

        # Clip gradients by norm
        # Gradient scaling
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=2.0, norm_type=2)

        # Optimize generator
        def closure_D():
            return d_loss

        optimizer_D.step(closure_D)

        # Intermediate or final evaluation
        metric = check_for_eval(epoch, n_epochs, generator, trial)
        if metric and metric < best_metric:
            best_metric = metric
            track_model(generator, config['generator_type'] + "_generator")
            track_model(discriminator, config['discriminator_type'] + "_discriminator")

        # Forward step learning rate scheduler after curriculum training
        if epoch > curr_max_epoch and adam_optim:
            scheduler_G.step()
            scheduler_D.step()

        # Transition to LBFGS optimizers for last epochs
        if adam_optim and epoch > (1 - const.fraction_lbfgs) * n_epochs:
            adam_optim = False
            optimizer_G = torch.optim.LBFGS(generator.parameters())
            optimizer_D = torch.optim.LBFGS(discriminator.parameters())

        if epoch % 50 == 0: print('[' + str(epoch) + '/' + str(n_epochs) + ']' + " D Loss", d_loss.item(), "G loss: ", g_loss.item(), "Phys Loss: ", phys_loss.item())

    # Stop mlflow run
    const.mlflow.end_run()

    return best_metric
