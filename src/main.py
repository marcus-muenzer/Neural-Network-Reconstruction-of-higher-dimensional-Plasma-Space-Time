"""
NAME:
    main

DESCRIPTION:
    Everything can be started from here.
"""
import argparse
import configparser
from contextlib import suppress
import os.path
import errno
import random
import torch
import ast
from utils.value_check import CheckFraction, CheckDelta
from hpo.hpo_baselines import hpo_kNN, hpo_MLP
from hpo.hpo_PINN import hpo_PINN
from hpo.hpo_cGAN import hpo_cGAN
from hpo.hpo_cWassersteinGAN_GP import hpo_cWassersteinGAN_GP
from data.data_augmentation import augment_data
from globals.initialization import set_constants


# Version number
version = '1.0'

# Helper strings
help_config = 'path to config file'
help_problem = '''MHD problem/benchmarks that should be reconstructed\npath to 2D MHD-datafile (.h5)'''
help_model = '''model/architecture to run the HPO for\noptions: 'cgan', 'cwgangp', 'knn', 'mlp', 'pinn' '''
help_augmentation_model = '''path to augmentation model\nmust be stored as .pth (pytorch model) or .joblib (no-pytorch model) file\nno-pytorch models must have a method "forward" to map spacetime -> MHD state\nif augmentation-model = None, training data will not be augmented'''
help_curr_method = '''method of the curriculum learning.\noptions: None, 'colloc_inc_points', 'colloc_cuboid', 'colloc_cylinder', 'phys', 'trade_off', 'coeff', 'num_diff', 'hpo'\n'None': no curriculum learning will be applied\n'colloc_inc_points': stepwise increase of the number of sampled collocation points\n'colloc_cuboid': stepwise shift or expansion of the spacetimes for the collocation point sampling along either x, y, or t axis\n'colloc_cylinder': stepwise extension of the spacetimes for the collocation point sampling in concentric circles around one or more spacecraft trajectories\n'phys': curriculum learning via stepwise addition of MHD equations\n'trade_off': schedules the trade-off parameter weighting the physical loss\n'coeff': schedules the viscosity and resistivity coefficients\n'num_diff': schedules the deltas dx, dy, dt for calculating the derivatives\n'hpo': HPO decides which method to use'''
help_curr_fraction_of_total_epochs = 'percentage of the overall epochs that is used for curriculum learning'
help_curr_steps = 'number of curriculum steps'
help_curr_factor_total_points = '''only relevant if curr-method = 'colloc_inc_points'\nmultiplier determining how many collocation points are sampled in the last curriculum step\nbasis: number of points in the training data'''
help_curr_axis = '''only relevant if curr-method = 'colloc_cuboid'\naxis along which the spacetimes for the collocation point sampling will be shifted or expanded\noptions: 'x', 'y', 't' 'hpo' (HPO decides which axis to use)'''
help_ssh_alias = '''ssh alias\nconnection must be authorized!'''
help_tracking_uri = 'tracking uri for mlflow experiments and runs'
help_artifact_uri = 'uri to track artifact/models in mlflow'
help_storage = '''storage: database URL\nif storage = None, in-memory storage is used, and the study will not be persistent\nfor a very lightweight storage one can use SQLite (NOT RECOMMENDED!):\n    advantage: no need for an additionally installed backend database\n    disadvantage: could cause blocking errors when working with multiple processes or distributed machines in general\n    example value: "sqlite:///{}.db".format(studies/hpo)'''
help_exp_name = 'name of mlflow experiment'
help_study_name = 'name of optuna study (HPO)'
help_eval_metric = '''metric to decide which model is the best\nused for HPO\noptions: 'mse', 'mae', 'pc', 'sc', 'r2', 'avg_rel_err' '''
help_intermediate_steps = 'number of epochs after which an intermediate evaluation of pytorch models is executed'
help_fraction = '''percentage of the data that should be kept\nused for reducing the resolution of the evaluation/validation data\nrange of values: ]0; 1]'''
help_fraction_lbfgs = '''fraction of epochs for which the LBFGS optimizer will be used (in the end of the training process)\nrange of values: [0; 1]'''
help_plane = 'determines if the training data should be a 2D plane'
help_bounds = '''only used if argument "plane" is set\ntwo boundaries for the plane (training data) along {dim} axis\nfirst parameter: start value (float)\nsecond parameter: end value (float)\nboth parameters will be set to the closest value in {dim} if they are not in {dim}'''
help_n_points = '''if parameter "plane" is set: number of points in the training data: n_points^2\nif parameter "plane" is not set: number of points in the training data: n_points * n_trajs'''
help_random_trajs = '''determines sampling strategy for trajectories for training data\nif parameter "random-trajs" is set: trajectories are randomly sampled\notherwise: trajectories include whole x, y, t domains\n           every trajectory will then be the same\n           recommendation: use only one trajectory'''
help_space_usage_non_random_traj = '''percentage of the space domain from which a non-random trajectory will be sampled from\nonly used if random_trajs = False'''
help_n_trajs = '''only used if parameter "plane" is not set.\nnumber of trajectories to sample (each trajectory consists of n_points many points)'''
help_noise = 'determines if Gaussian noise should be added to the training data'
help_no_cuda = 'do not use cuda even if one is available'
help_deltas = 'delta along {dim} axis used for the numerical differentiation to calculate the physical loss'
help_seed = 'reproducibility seed for numpy and torch modules'
help_n_trials = 'number of HPO trials'

# Create argument parser
parser = argparse.ArgumentParser(description='These are the options on how the reconstruction task can be parametrized:', formatter_class=argparse.RawTextHelpFormatter)

# Add command line arguments
parser.add_argument("-v", "--version", action='version', version='Version ' + version)
parser.add_argument("-c", "--config", metavar='', type=str, help=help_config)
parser.add_argument("-p", "--problem", metavar='', type=str, default='../data/problem.h5', help=help_problem)
parser.add_argument("-m", "--model", choices=['cgan', 'cwgangp', 'knn', 'mlp', 'pinn'], metavar='', type=str, default='pinn', help=help_model)
parser.add_argument("-am", "--augmentation-model", metavar='', type=str, help=help_augmentation_model)
parser.add_argument("--curr-method", choices=[None, 'colloc_inc_points', 'colloc_cuboid', 'colloc_cylinder', 'phys', 'trade_off', 'coeff', 'num_diff', 'hpo'], metavar='', type=str, default=None, help=help_curr_method)
parser.add_argument("--curr-steps", metavar='', type=int, default=30, help=help_curr_steps)
parser.add_argument("--curr-fraction-of-total-epochs", metavar='', type=float, default=0, help=help_curr_fraction_of_total_epochs)
parser.add_argument("--curr-factor-total-points", metavar='', type=int, default=1, help=help_curr_factor_total_points)
parser.add_argument("--curr-axis", choices=['x', 'y', 't', 'hpo'], metavar='', type=str, default='hpo', help=help_curr_axis)
parser.add_argument("-ssh-alias", metavar='', type=str, default='master', help=help_ssh_alias)
parser.add_argument("--tracking-uri", metavar='', type=str, default='http://localhost:5000', help=help_tracking_uri)
parser.add_argument("--artifact-uri", metavar='', type=str, help=help_artifact_uri)
parser.add_argument("--storage", metavar='', type=str, default='mysql+pymysql://root:password@ip_address:3306/database', help=help_storage)
parser.add_argument("--exp-name", metavar='', type=str, default='Default', help=help_exp_name)
parser.add_argument("--study-name", metavar='', type=str, default='Default', help=help_study_name)
parser.add_argument("--eval-metric", metavar='', choices=['mse', 'mae', 'pc', 'mare'], type=str, default='mse', help=help_eval_metric)
parser.add_argument("--intermediate-steps", metavar='', type=int, default=1000, help=help_intermediate_steps)
parser.add_argument("-f", "--fraction", metavar='', type=float, default=.2, action=CheckFraction, help=help_fraction)
parser.add_argument("--fraction_lbfgs", metavar='', type=float, default=0, action=CheckFraction, help=help_fraction_lbfgs)
parser.add_argument("--plane", action='store_true', help=help_plane)
parser.add_argument("--x-bounds", metavar='', type=int, nargs=2, help=help_bounds.format(dim='x'))
parser.add_argument("--y-bounds", metavar='', type=int, nargs=2, help=help_bounds.format(dim='y'))
parser.add_argument("--t-bounds", metavar='', type=int, nargs=2, help=help_bounds.format(dim='t'))
parser.add_argument("--n-points", metavar='', type=int, default=100, help=help_n_points)
parser.add_argument("--random-trajs", action='store_true', help=help_random_trajs)
parser.add_argument("--space_usage_non_random_traj", metavar='', type=float, default=.5, help=help_space_usage_non_random_traj)
parser.add_argument("--n-trajs", metavar='', type=int, default=4, help=help_n_trajs)
parser.add_argument("--noise", action='store_true', help=help_noise)
parser.add_argument("--no-cuda", action='store_true', help=help_no_cuda)
parser.add_argument("--dx", metavar='', type=float, default=.001, action=CheckDelta, help=help_deltas.format(dim='x'))
parser.add_argument("--dy", metavar='', type=float, default=.001, action=CheckDelta, help=help_deltas.format(dim='y'))
parser.add_argument("--dt", metavar='', type=float, default=.001, action=CheckDelta, help=help_deltas.format(dim='t'))
parser.add_argument("-s", "--seed", metavar='', type=int, default=0, help=help_seed)
parser.add_argument("--n-trials", metavar='', type=int, default=10, help=help_n_trials)

# Parse command line arguments
args = parser.parse_args()
args = vars(args)

# ================================================================================
# If a config file is provided all parameters in the config file will
# overwrite the related parameters that are passed manually via the command line
# ================================================================================

# Check for config file
if args['config']:

    # Check if file exists
    if not os.path.isfile(args['config']):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args['config'])

    # Create config parser
    config = configparser.ConfigParser()

    # Read config file
    config.read(args['config'])

    # Iterate through sections
    for s in config.sections():
        # Iterate through options in section
        for o in config.options(s):
            # Check if option is needed
            if o in args:
                # Convert to correct type
                value = config[s][o]
                with suppress(ValueError): value = float(value)
                with suppress(ValueError): value = int(value) if isinstance(value, float) and value % 1 == 0 else value
                value = None if value == 'None' else value
                value = True if value == 'True' else value
                value = False if value == 'False' else value
                value = ast.literal_eval(value) if isinstance(value, str) and value[0] == '[' else value

                # Set argument
                args[o] = value
            else:
                # Unrecognized argument
                err_msg = "\'" + o + "\' is no valid argument"
                raise KeyError(err_msg)

# -----------------------
# All arguments are set
# -----------------------

# Set seed for reproducibility
if args['seed'] is not None:
    seed = args['seed']
    random.seed(seed)
    torch.manual_seed(seed)

# Initialize constants
# No cuda for sklearn model
model = args['model']
if model == 'knn': args['no_cuda'] = True

set_constants(problem=args['problem'], curr_method=args['curr_method'], curr_steps=args['curr_steps'], curr_fraction_of_total_epochs=args['curr_fraction_of_total_epochs'], curr_factor_total_points=args['curr_factor_total_points'], curr_axis=args['curr_axis'], ssh_alias=args['ssh_alias'], tracking_uri=args['tracking_uri'], artifact_uri=args['artifact_uri'], storage=args['storage'], exp_name=args['exp_name'], study_name=args['study_name'],
              eval_metric=args['eval_metric'], intermediate_steps=args['intermediate_steps'], fraction=args['fraction'], fraction_lbfgs=args['fraction_lbfgs'],
              plane=args['plane'], x_bounds=args['x_bounds'], y_bounds=args['y_bounds'], t_bounds=args['t_bounds'], n_points=args['n_points'], random_trajs=args['random_trajs'],
              space_usage_non_random_traj=args['space_usage_non_random_traj'], n_trajs=args['n_trajs'], noise=args['noise'], no_cuda=args['no_cuda'],
              dx=args['dx'], dy=args['dy'], dt=args['dt'])

# Augment data
if args['augmentation_model']:
    augment_data(args['augmentation_model'])

# Start HPO
n_trials = args['n_trials']

if model == 'knn':
    hpo_kNN(n_trials)
elif model == 'mlp':
    hpo_MLP(n_trials)
elif model == 'cgan':
    hpo_cGAN(n_trials)
elif model == 'cwgangp':
    hpo_cWassersteinGAN_GP(n_trials)
elif model == 'pinn':
    hpo_PINN(n_trials)
