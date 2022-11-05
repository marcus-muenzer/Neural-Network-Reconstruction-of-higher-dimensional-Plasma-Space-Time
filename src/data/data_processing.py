"""
NAME
    data_processing

DESCRIPTION
    This module provides functions to process and shape data.
"""

import torch
import numpy as np
import math
from utils.data import index_to_phys_unit


def ravel_data(x, y, t, U):
    """
    Returns raveled data of the given spacetime and MHD domain.
    The data directly be passed to the models or evaluation methods.

    Parameters:
    Condition for parameters: len(x) = len(y) = len(t)
        x: tensor of x space
        y: tensor of y space
        t: tensor of times
        U: tensor of MHD states
           shape: torch.Size([MHD states, len(x), len(y), len(t)])

    Returns:
        st: 2D tensor of spacetimes
            shape: torch.Size([number of points in spacetime domain (=len(x)), 3])
        U:  2D tensor of MHD states
            shape: torch.Size([number of points in spacetime domain (=len(x)), MHD states])
    """

    # Create 2D raveled space-time vector
    x_mg, y_mg, t_mg = torch.meshgrid((x, y, t), indexing='ij')
    st = torch.stack((x_mg.ravel(), y_mg.ravel(), t_mg.ravel()), dim=1)

    # Create 2D raveled MHD vector
    U_tmp = []
    for i in range(U.shape[0]):
        U_tmp.append(U[i].ravel())

    U = torch.stack(U_tmp, dim=1)

    return st, U


def reshape_raveled_data(data, shape):
    """
    Reshapes (MHD) data into a specific shape (3D).

    Parameters:
        data: raveled data
              shape: torch.Size([number of predictions/points, MHD states])
        shape: shape to convert the data to
               Usually 3D - e.g. torch.Size([1280, 384, 201])

    Returns:
        reshaped_data: shape (3D MHD example): torch.Size([MHD states, 1280, 384, 201])
    """

    # Convert type to torch.Tensor
    # Necessary for working with predictions of sklearn framework
    if type(data) == np.ndarray:
        data = torch.from_numpy(data)

    # Extraxt data seperately for every physical unit
    data_physical_units = []
    for u in range(len(data[0, :])):
        data_tmp = data[:, u]
        data_tmp = torch.reshape(data_tmp, shape)
        data_physical_units.append(data_tmp)

    # Stack single units together to fit original datagrid
    reshaped_data = torch.stack(data_physical_units)

    return reshaped_data


def extract_single_trajectories_from_data(data, n_trajs):
    """
    Extracts single x, y, t trajectories out of one big spacetime trajectory
    which summarizes multiple smaller trajectories by concatenating and raveling them (like they are used in the training process).

    Parameters:
        data: tensor of spacetimes
              shape: torch.Size([points per trajectory * n_trajs, const.dims_mhd_state])
        n_trajs: number of trajectories in data

    Returns:
        trajectories: list of extracted trajectories
                      every single trajectory is represented by a tupel of tensors (x, y, t)
    """

    # Determine points per single trajectory
    points_per_traj = data.shape[0] / n_trajs

    # Extract trajectories
    trajectories = []
    for n in range(n_trajs):
        # Get indices for one trajectory inside the data
        start_index = int(n * points_per_traj)
        end_index = int((n + 1) * points_per_traj)

        # Extract trajectory by indices
        # and split it in x, y, t tensors
        x_traj = data[start_index:end_index, 0]
        y_traj = data[start_index:end_index, 1]
        t_traj = data[start_index:end_index, 2]

        # Append trajector to list
        trajectories.append((x_traj, y_traj, t_traj))

    return trajectories


def reduce_resolution(x, y, t, U, fraction=0.3):
    """
    Reduces the resolution of x, y, t, and U according to the fraction.
    E.g. fraction=0.3 => data will be reduced by 70% and only keep 30%.

    Parameters:
        x: tensor of x space
        y: tensor of y space
        t: tensor of times
        U: tensor of MHD states
        fraction: percentage of the data that should be kept
                  range of values: ]0; 1]

    Returns:
        x: lower resolutioned tensor of x space
        y: lower resolutioned tensor of y space
        t: lower resolutioned tensor of times
        U: lower resolutioned tensor of MHD states
    """

    # Calculate number of points per dimension
    points_x = math.ceil(x.shape[0] * fraction)
    points_y = math.ceil(y.shape[0] * fraction)
    points_t = math.ceil(t.shape[0] * fraction)

    # Calculate indices that define the lower resolution
    indices_x = torch.linspace(0, x.size(dim=0) - 1, points_x, dtype=torch.int32)
    indices_y = torch.linspace(0, y.size(dim=0) - 1, points_y, dtype=torch.int32)
    indices_t = torch.linspace(0, t.size(dim=0) - 1, points_t, dtype=torch.int32)

    # Reduce resolution of spacetimes
    x = torch.index_select(x, 0, indices_x)
    y = torch.index_select(y, 0, indices_y)
    t = torch.index_select(t, 0, indices_t)

    # Reduce resolution of MHD matrix
    U = torch.index_select(U, 1, indices_x)
    U = torch.index_select(U, 2, indices_y)
    U = torch.index_select(U, 3, indices_t)

    return x, y, t, U


def add_noise(data, std=[0.01]):
    """
    Adds random Gaussion noise to data.

    Parameters:
        data: data tensor to add noise to
        std: standard deviation

    Returns:
        noisy_data: noise-added data
    """

    # Generate noise
    mean = torch.zeros(data.shape)
    noise = torch.normal(mean, std)

    # Add noise to data
    noisy_data = data + noise

    # Enforce positive density/pressure
    for index in range(data.shape[1]):
        if index_to_phys_unit(index) == "Density" or index_to_phys_unit(index) == "P":
            torch.clamp_(noisy_data[:, index], min=0)

    return noisy_data
