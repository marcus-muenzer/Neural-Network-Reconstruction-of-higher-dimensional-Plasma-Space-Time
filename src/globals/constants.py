"""
NAME:
    constants

DESCRIPTION:
    This file contains all important global constants.
"""

# GPU available
cuda = None
device = None

# LBFGS fraction
fraction_lbfgs = None

# Curriculum training
curr_method = None
curr_steps = None
curr_fraction_of_total_epochs = None
curr_factor_total_points = None
curr_axis = None

# Tracking
ssh_alias = None
artifact_uri = None
mlflow = None
exp_id = None

# HPO study name
storage = None
study_name = None

# Evaluation metric
eval_metric = None
direction = None
error_value = None

# Steps between metric measurements
intermediate_steps = None

# Data
# Metadata
dtype = None
dims_st = None
dims_mhd_state = None

# Number of trajectories in the training data
n_trajs = None

# Minimal and maximal values in x, y, t domain
x_min = None
x_max = None
y_min = None
y_max = None
t_min = None
t_max = None

# Full domain data, (eventually) reduced resolution
x_red = None
y_red = None
t_red = None
U_red = None

# Training data (raveled trajectories)
# The training data will be sampled from the (eventually) reduced resolution data
st_train = None
U_train = None
data_augmented = None

# Evaluation data
# The evaluation data is a flattened form
# of the full domain, reduced resolution data
st_eval = None
U_eval = None

# Dataloader
dataloader_train = None
dataloader_eval = None

# Constants for numerical differentiation
dx = None
dy = None
dt = None

dx_tensor = None
dy_tensor = None
dt_tensor = None
