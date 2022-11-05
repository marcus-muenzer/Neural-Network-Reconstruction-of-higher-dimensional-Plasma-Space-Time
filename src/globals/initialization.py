"""
NAME:
    initialization

DESCRIPTION:
    This file contains a function to set constants of the globals module.
    It should only be used once in the very beginning of the program.
"""

import torch
import os
from data.data_loading import load_data, get_plane_training_data, get_traj_training_data
from data.data_processing import reduce_resolution, ravel_data, add_noise
from data.dataset import PlasmaDataset
from utils.mlflow import init_mlflow
import globals.constants as const


def set_constants(problem='../data/problem.h5', curr_method=None, curr_steps=30, curr_fraction_of_total_epochs=0, curr_factor_total_points=1, curr_axis='hpo', ssh_alias='master', tracking_uri='http://localhost:5000', artifact_uri=None, storage='mysql+pymysql://root:password@ip_address:3306/database', exp_name='Default', study_name='Default',
                  eval_metric='mse', intermediate_steps=1000, fraction=.2, fraction_lbfgs=0,
                  plane=False, x_bounds=None, y_bounds=None, t_bounds=None, n_points=100, random_trajs=True, space_usage_non_random_traj=.5, n_trajs=4, noise=False, dtype=torch.float32, no_cuda=False,
                  dx=.001, dy=.001, dt=.001):
    """
    Sets important global constants.
    Uses default parameters of data loading modules.
    Correct usage is only calling it once in the very beginning
    of any program using any of these global constants.

    Parameters:
        test: determines if small test data should be used
        problem: MHD problem/benchmark that should be reconstructed
                 path to 2D MHD-datafile (.h5)
        curr_method: method of the curriculum learning
                     options: None, 'colloc_inc_points', 'colloc_cuboid', 'colloc_cylinder', 'phys', 'coeff', 'num_diff'
                     None: no curriculum learning
                     'colloc_inc_points': number of sampled collocation points increases over time
                     'colloc_cuboid': stepwise shift or expansion of the spacetimes for the collocation point sampling
                                      along either x, y, or t axis
                     'colloc_cylinder': stepwise extension of the spacetimes for the collocation point sampling
                                        in concentric circles around one or more spacecraft trajectories
                     'phys': stepwise addition of MHD equations
                     'trade_off': schedules the trade-off parameter weighting the physical loss
                     'coeff': schedules the viscosity and resistivity coefficients
                     'num_diff': schedules the deltas dx, dy, dt for calculating the derivatives
                     'hpo': HPO decides which method to use
        curr_steps: number of curriculum steps
        curr_fraction_of_total_epochs: percentage of the overall epochs that is used for curriculum learning
        curr_factor_total_points: only relevant if curr_method == 'colloc_inc_points'
                                  multiplier determining how many collocation points are sampled in the last curriculum step
                                  basis: number of points in the training data
        curr_axis: only relevant if curr_method == 'colloc_cuboid'
                   axis along the spacetimes for the collocation point sampling will be shifted or expanded
                   options: 'x', 'y', 't', 'hpo' (HPO decides which axis to use)
        ssh_alias: ssh alias
                   connection must be authorized!
        tracking_uri: tracking uri for mlflow experiments and runs
        artifact_uri: uri to track artifact/models in mlflow
        storage: database URL
                 if storage = None, in-memory storage is used, and the study will not be persistent
                 for a very lightweight storage one can use SQLite (NOT RECOMMENDED!):
                     advantage: no need for an additionally installed backend database
                     disadvantage: could cause blocking errors when working with multiple processes or distributed machines in general
                     example value: "sqlite:///{}.db".format(studies/hpo)
        exp_name: name of mlflow experiment
        study_name: name of optuna study (HPO)
        eval_metric: metric to decide which model is the best
                     used for HPO
                     options: 'mse', 'mae', 'pc', 'sc', 'r2', 'avg_rel_err'
        intermediate_steps: number of epochs after which an intermediate evaluation of pytorch models is executed
        fraction: percentage of the data that should be kept
                  used for reducing the resolution of the evaluation/validation data
                  range of values: ]0; 1]
        fraction_lbfgs: fraction of epochs for which the LBFGS optimizer will be used (in the end of the training process)
                        range of values: ]0; 1]
        plane: determines if the training data lies in a predefined plane
        x_bounds: only used if parameter "plane" == True
                  list of the two boundaries for the plane (training data) along x axis
                  length: 2
                  x_bounds[0]: start value (float). Will be set to the closest value in x if x_bounds[0] not in x.
                  x_bounds[1]: end value (float). Will be set to the closest value in x if x_bounds[1] not in x.
                  if x_bounds = None: boundaries will include whole x domain
        y_bounds: only used if parameter "plane" == True
                  list of the two boundaries for the plane (training data) along y axis
                  length: 2
                  y_bounds[0]: start value (float). Will be set to the closest value in y if y_bounds[0] not in y.
                  y_bounds[1]: end value (float). Will be set to the closest value in y if y_bounds[1] not in y.
                  if y_bounds = None: boundaries will include whole y domain
        t_bounds: only used if parameter "plane" == True
                  list of the two boundaries for the plane (training data) along t axis
                  length: 2
                  t_bounds[0]: start value (float). Will be set to the closest value in t if t_bounds[0] not in t.
                  t_bounds[1]: end value (float). Will be set to the closest value in t if t_bounds[1] not in t.
                  if t_bounds = None: boundaries will include whole t domain
        n_points: if parameter "plane" == True: number of points in the training data: n_points^2
                    if parameter "plane" == False: number of points in the training data: n_points * n_trajs
        random_trajs: determines sampling strategy for trajectories for training data
                      if True: trajectories are randomly sampled
                      if False: trajectories include whole x, y, t domains
                                every trajectory will then be the same
                                recommendation: use only one trajectory
        space_usage_non_random_traj: percentage of the space domain from which a non-random trajectory will be sampled from
                                     only used if random_trajs = False
        n_trajs: only used if parameter "plane" == False
                 number of trajectories to sample (each trajectory consists of n_points many points)
        noise: determines if Gaussian noise should be added to the training data
        no_cuda: do not use cuda even if one is available
        dx: delta along x axis used for the numerical differentiation to calculate the physical loss
        dy: delta along y axis used for the numerical differentiation to calculate the physical loss
        dt: delta along t axis used for the numerical differentiation to calculate the physical loss

    Returns:
        None
    """

    # Set dtype
    const.dtype = dtype

    # Check for GPU speedup
    const.cuda = torch.cuda.is_available() and not no_cuda
    const.device = 'cuda' if const.cuda else 'cpu'
    if const.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # LBFGS fraction
    # Determines amount of epochs for which the LBFGS optimizer is used
    const.fraction_lbfgs = fraction_lbfgs

    # Curriculum training
    const.curr_method = curr_method
    const.curr_steps = curr_steps
    const.curr_fraction_of_total_epochs = curr_fraction_of_total_epochs
    const.curr_factor_total_points = curr_factor_total_points if curr_method == 'colloc_inc_points' else 1
    const.curr_axis = curr_axis

    # Initialize mlflow for tracking
    const.ssh_alias = ssh_alias
    const.artifact_uri = artifact_uri if artifact_uri else 'file://' + os.getcwd() + '/mlruns/'
    const.mlflow, const.exp_id = init_mlflow(tracking_uri, exp_name, const.artifact_uri)

    # HPO study name
    const.storage = storage
    const.study_name = study_name

    # Set evaluation metric
    const.eval_metric = eval_metric
    if const.eval_metric in ['mse', 'mae', 'avg_rel_err']:
        const.direction = 'minimize'
        const.error_value = 1000
    else:
        const.direction = 'maximize'
        const.error_value = 0

    # Steps between metric measurements
    const.intermediate_steps = intermediate_steps

    # Load data
    # Full domain, full resolution
    x, y, t, U = load_data(problem)

    # Full domain, (eventually) reduced resolution
    const.x_red, const.y_red, const.t_red, const.U_red = reduce_resolution(x, y, t, U, fraction)

    # Number of trajectories in the training data
    const.n_trajs = n_trajs

    # Minimal and maximal values in x, y, t domain
    const.x_min = x.min()
    const.x_max = x.max()
    const.y_min = y.min()
    const.y_max = y.max()
    const.t_min = t.min()
    const.t_max = t.max()

    # Training data (raveled)
    # Training data is a plane
    if plane:
        const.st_train, const.U_train = get_plane_training_data(x, y, t, U, x_bounds, y_bounds, t_bounds, n_points)
    # Training data is random trajectories
    else:
        const.st_train, const.U_train = get_traj_training_data(x, y, t, U, random_trajs, space_usage_non_random_traj, n_trajs, n_points)

    # Eventually add Gaussian noise
    if noise:
        const.U_train = add_noise(const.U_train, .1)

    const.data_augmented = False

    # Evaluation data (raveled trajectories)
    # The evaluation data will be sampled from the full domain, reduced resolution data
    const.st_eval, const.U_eval = ravel_data(const.x_red, const.y_red, const.t_red, const.U_red)

    # Datasets
    ds_train = PlasmaDataset(const.st_train, const.U_train)
    ds_eval = PlasmaDataset(const.st_eval, const.U_eval)

    # Dataloader
    const.dataloader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=256,
        shuffle=True,
        generator=torch.Generator(const.device)
    )
    const.dataloader_eval = torch.utils.data.DataLoader(
        ds_eval,
        batch_size=1000000,
        shuffle=False,
        generator=torch.Generator(const.device)
    )

    # Derive metadata
    const.dims_st = const.st_train.shape[1]
    const.dims_mhd_state = const.U_train.shape[1]

    # Set constants for numerical differentiation
    # Used for calculation of physical loss
    const.dx = dx
    const.dy = dy
    const.dt = dt

    d_tensor_length = const.st_train.shape[0] * const.curr_factor_total_points
    d_tensor = torch.zeros(d_tensor_length, const.dims_st)

    dx_tmp = torch.full([d_tensor_length], dx)
    const.dx_tensor = d_tensor.clone()
    const.dx_tensor[:, 0] = dx_tmp

    dy_tmp = torch.full([d_tensor_length], dy)
    const.dy_tensor = d_tensor.clone()
    const.dy_tensor[:, 1] = dy_tmp

    dt_tmp = torch.full([d_tensor_length], dt)
    const.dt_tensor = d_tensor.clone()
    const.dt_tensor[:, 2] = dt_tmp

    # Free GPU cache
    del x, y, t, U
    torch.cuda.empty_cache()
