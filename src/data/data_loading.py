"""
NAME
    data_loading

DESCRIPTION
    Provides functions to load and provide the data in multiple different shapes
    that are applicable for different scenarios.
"""

import h5py
import torch
import numpy as np
from random import uniform
import globals.constants as const


def get_trajectory_indices(x_bounds=[0, float('inf')], y_bounds=[0, float('inf')], t_bounds=[0, float('inf')], n_points=100):
    """
    Returns homogeneous sampled indices of the x, y and t dimensions for a linear trajectory through the whole domain of the given space-times.

    Parameters:
        x_bounds: list of boundaries for trajectory indices along x axis
                  length: 2
                  x_bounds[0]: start index
                  x_bounds[1]: end index
        y_bounds: list of boundaries for trajectory indices along y axis
                  length: 2
                  y_bounds[0]: start index
                  y_bounds[1]: end index
        t_bounds: list of boundaries for trajectory indices along t axis
                  length: 2
                  t_bounds[0]: start index
                  t_bounds[1]: end index
        n_points: number of points in trajectory

    Returns:
        indices: indices to sample trajectory data from
                 shape: torch.Size([3, n_points]) => "3" for each dimension: x, y, t
    """

    # Calculate x values
    x_ind = torch.linspace(x_bounds[0], x_bounds[1], n_points)
    x_ind = torch.round(x_ind).int()

    # Calculate y values
    y_ind = torch.linspace(y_bounds[0], y_bounds[1], n_points)
    y_ind = torch.round(y_ind).int()

    # Calculate t values
    t_ind = torch.linspace(t_bounds[0], t_bounds[1], n_points)
    t_ind = torch.round(t_ind).int()

    # Concatenate values to trajectory
    indices = torch.stack((x_ind, y_ind, t_ind))

    return indices


def sample_st_trajectory_from_indices(x, y, t, indices):
    """
    Samples a trajectory according to the given indices.

    Parameters:
        x: tensor of x space
        y: tensor of y space
        t: tensor of times
        indices: indices for all 3 dimensions which determine the data that should be sampled

    Returns:
        st_trajectory: 3D trajectory containing real numbers from the datagrid (x, y, t).
    """

    # Sample x trajectory
    x_traj = torch.index_select(x, 0, indices[0])

    # Sample y trajectory
    y_traj = torch.index_select(y, 0, indices[1])

    # Sample t trajectory
    t_traj = torch.index_select(t, 0, indices[2])

    # Concatenate 1D trajectories to 3D trajectory
    st_trajectory = torch.stack((x_traj, y_traj, t_traj), dim=1)

    return st_trajectory


def sample_mhd_trajectory_from_indices(U, indices):
    """
    Samples the mhd states according to the given trajectory indices.

    Parameters:
        U: Tensor of full-domain MHD states
        indices: indices for all 3 dimensions which determine the data that should be sampled

    Returns:
        U_traj: mhd states of trajectory defined by indices containing real numbers from the datagrid.
    """

    U_tmps = []
    for i in range(len(indices[0])):
        x_ind = indices[0, i]
        y_ind = indices[1, i]
        t_ind = indices[2, i]
        U_tmps.append(U[:, x_ind, y_ind, t_ind])

    U_traj = torch.stack(U_tmps)

    return U_traj


def extract_tajectory_data_from_indices(x, y, t, U, indices):
    """
    Returns trajectory data (spacetimes and MHD states) usable as training data.
    The data is raveled and can therefore directly passed to the models.

    Parameters:
        x: tensor of x space
        y: tensor of y space
        t: tensor of times
        U: tensor of MHD states
        indices: tensors of indices of trajectory for sampling the data

    Returns:
        st_traj: 2D tensor of spacetimes from training trajectory
                 shape: torch.Size([points in trajectory, 3])
        U_traj:  2D tensor of MHD states from training trajectory
                 shape: torch.Size([points in trajectory, mhd states])
    """

    st_traj = sample_st_trajectory_from_indices(x, y, t, indices)
    U_traj = sample_mhd_trajectory_from_indices(U, indices)

    return st_traj, U_traj


def offset_index_bounds(tensor, bounds, offset):
    """
    Offsets index boundaries. Used for spacing between trajectories.

    Parameters:
        tensor: tensor to which the boundaries belong
        bounds: list of boundaries for trajectory indices
                length: 2
                bounds[0]: start index.
                bounds[1]: end index
        offset: integer that defines the offset (in indices)

    Returns:
        bounds_offset: List of start and end index of offset boudaries.
                       length: 2
                       Contained indices are in-bounds of tensor size and can be directly used without further error checking.
    """

    # Offset bounds
    bounds_offset = [bounds[0] + offset, bounds[1] + offset]

    # Correct out-of-bounds indices
    # Start and end index must be > 0
    if bounds_offset[0] < 0:
        bounds_offset[0] = 0
    if bounds_offset[1] < 0:
        bounds_offset[1] = 0

    # Start and end index must be < tensor.size(dim=0)
    if bounds_offset[0] > tensor.size(dim=0) - 1:
        bounds_offset[0] = tensor.size(dim=0) - 1
    if bounds_offset[1] > tensor.size(dim=0) - 1:
        bounds_offset[1] = tensor.size(dim=0) - 1

    return bounds_offset


def get_index_for_value(tensor, value):
    """
    Determines the index of a tensor containing the value closest to "value".

    Parameters:
        tensor: 1D tensor with distinct values
        value: float to search for in the tensor

    Returns:
        index: index of closest value in tensor
    """

    index_tensor = torch.abs(tensor - value).argmin()
    index = index_tensor.item()

    return index


def get_traj_training_data(x, y, t, U, random_trajs=False, space_usage_non_random_traj=.3, n_trajs=1, n_points=1000):
    """
    Returns spacetime and MHD state data usable as training data.
    The spacetimes are either located in one ore more independent and randomly sampled linear 1D trajectories,
    or are located in one trajectory that includes the whole x, y, t domains.
    The data is raveled and can therefore directly passed to the models.

    Parameters:
        x: tensor of x space
        y: tensor of y space
        t: tensor of times
        U: tensor of MHD states
        random_trajs: determines sampling strategy for trajectories along the t axis for training data
                      if True: trajectories are randomly sampled concerning the space domain
                      if False: every trajectory will then be the same lying approximately in the center of the space
                                recommendation: use only one trajectory
        space_usage_non_random_traj: percentage of the space domain from which a non-random trajectory will be sampled from
                                     only used if random_trajs = False
        n_trajs: number of trajectories to sample
        n_points: number of points in one trajectory

    Returns:
        st_trajs: 2D tensor of spacetimes of the randomly sampled trajectories
                  shape: torch.Size([n_trajs*n_points, dims_st])
        U_trajs: 2D tensor of MHD states of the randomly sampled trajectories
                 shape: torch.Size([n_trajs*n_points, mhd states])
    """

    # Set boundaries to whole x, y, t domains
    x_bounds = [None, None]
    y_bounds = [None, None]
    t_bounds = [None, None]
    x_bounds[0] = x[0].item()
    y_bounds[0] = y[0].item()
    t_bounds[0] = t[0].item()
    x_bounds[1] = x[-1].item()
    y_bounds[1] = y[-1].item()
    t_bounds[1] = t[-1].item()
    x_center = np.mean(x_bounds)
    y_center = np.mean(y_bounds)
    # t_center = np.mean(t_bounds)

    # Get trajectories
    # Define temporary boundaries
    x_bounds_n = [None, None]
    y_bounds_n = [None, None]
    t_bounds_n = [None, None]

    st_trajs = []
    U_trajs = []
    for n_traj in range(n_trajs):
        if random_trajs:
            if n_traj == 0:
                # Sample middle trajectory
                st_traj, U_traj = get_traj_training_data(x, y, t, U, False, .001, 1, n_points)
                st_trajs.append(st_traj)
                U_trajs.append(U_traj)
                continue

            # Sample random boundary values for trajectory
            x_bounds_n[0] = uniform(x_bounds[0], x_bounds[1])
            x_bounds_n[1] = uniform(x_bounds[0], x_bounds[1])
            y_bounds_n[0] = uniform(y_bounds[0], y_bounds[1])
            y_bounds_n[1] = uniform(y_bounds[0], y_bounds[1])
            t_bounds_n = t_bounds.copy()
        else:
            # Use "space_usage_non_random_traj" percent of space domains for tajectory sampling
            x_range = abs(x_bounds[0] - x_bounds[1])
            y_range = abs(y_bounds[0] - y_bounds[1])
            # t_range = abs(t_bounds[0] - t_bounds[1])

            # Determine center of space data
            x_center = x_bounds[0] + x_range / 2
            y_center = y_bounds[0] + y_range / 2
            # t_center = t_bounds[0] + t_range / 2

            # Set x, y, t boundaries
            x_bounds_n[0] = x_center - x_range / 2 * space_usage_non_random_traj
            x_bounds_n[1] = x_center + x_range / 2 * space_usage_non_random_traj
            y_bounds_n[0] = y_center - y_range / 2 * space_usage_non_random_traj
            y_bounds_n[1] = y_center + y_range / 2 * space_usage_non_random_traj
            t_bounds_n = t_bounds.copy()

        # Transform real-valued x, y, t boundaries into index boundaries
        x_bounds_n[0] = get_index_for_value(x, x_bounds_n[0])
        x_bounds_n[1] = get_index_for_value(x, x_bounds_n[1])
        y_bounds_n[0] = get_index_for_value(y, y_bounds_n[0])
        y_bounds_n[1] = get_index_for_value(y, y_bounds_n[1])
        t_bounds_n[0] = get_index_for_value(t, t_bounds_n[0])
        t_bounds_n[1] = get_index_for_value(t, t_bounds_n[1])

        # Get x, y, t indices for trajectory
        traj_indices = get_trajectory_indices(x_bounds_n, y_bounds_n, t_bounds_n, n_points)

        # Retrieve real valued trajectory data
        st_traj, U_traj = extract_tajectory_data_from_indices(x, y, t, U, traj_indices)

        # Save trajectory data in the corresponding lists
        st_trajs.append(st_traj)
        U_trajs.append(U_traj)

    # Concatenate single trajectories
    st_trajs = torch.cat(st_trajs)
    U_trajs = torch.cat(U_trajs)

    return st_trajs, U_trajs


def get_plane_training_data(x, y, t, U, x_bounds=None, y_bounds=None, t_bounds=None, n_points=100):
    """
    Returns uniformly sampled spacetime and MHD state data (located in one plane) usable as training data.
    The data is raveled and can therefore directly passed to the models.
    The returned data consists of n_points^2 points

    Parameters:
        x: tensor of x space
        y: tensor of y space
        t: tensor of times
        U: tensor of MHD states
        x_bounds: list of the two boundaries for the plane along x axis
                  length: 2
                  x_bounds[0]: start value (float). Will be set to closest value in x if x_bounds[0] not in x.
                  x_bounds[1]: end value (float). Will be set to closest value in x if x_bounds[1] not in x.
                  if x_bounds = None: boundaries will include whole x domain
        y_bounds: list of the two boundaries for the plane along y axis
                  length: 2
                  y_bounds[0]: start value (float). Will be set to closest value in y if y_bounds[0] not in y.
                  y_bounds[1]: end value (float). Will be set to closest value in y if y_bounds[1] not in y.
                  if y_bounds = None: boundaries will include whole y domain
        t_bounds: list of the two boundaries for the plane along t axis
                  length: 2
                  t_bounds[0]: start value (float). Will be set to closest value in t if t_bounds[0] not in t.
                  t_bounds[1]: end value (float). Will be set to closest value in t if t_bounds[1] not in t.
                  if t_bounds = None: boundaries will include whole t domain
        n_points: number of points that the plane has along one dimension (=> total points in plane: n_points^2)
                    number of lines that mimicks the plane using discrete samples
                    number of point in one line

    Returns:
        st_plane: 2D plane tensor of spacetimes
                  shape: torch.Size([n_points^2, dims_st])
        U_plane: 2D plane tensor of MHD states
                 shape: torch.Size([n_points^2, mhd states])
    """

    # Set boundaries to whole x, y, t domains if they are not passed via parameter
    if not x_bounds:
        x_bounds = [None, None]
        x_bounds[0] = x[0].item()
        x_bounds[1] = x[-1].item()
    if not y_bounds:
        y_bounds = [None, None]
        y_bounds[0] = y[0].item()
        y_bounds[1] = y[-1].item()
    if not t_bounds:
        t_bounds = [None, None]
        t_bounds[0] = t[0].item()
        t_bounds[1] = t[-1].item()

    # Transform real-valued x, y, t boundaries into index boundaries
    x_bounds[0] = get_index_for_value(x, x_bounds[0])
    x_bounds[1] = get_index_for_value(x, x_bounds[1])
    y_bounds[0] = get_index_for_value(y, y_bounds[0])
    y_bounds[1] = get_index_for_value(y, y_bounds[1])
    t_bounds[0] = get_index_for_value(t, t_bounds[0])
    t_bounds[1] = get_index_for_value(t, t_bounds[1])

    # Extract 2D plane of spacetime and MHD values out of 3D spacetime
    num_lines = n_points
    st_lines = []
    U_lines = []

    # Plane has "num_lines" many lines each consisting of "n_points (=num_lines)" many points
    for num_line in range(num_lines):
        # Sample indices
        dim = 0
        if y_bounds[0] == y_bounds[1]:
            dim = 1
        if t_bounds[0] == t_bounds[1]:
            dim = 2

        # Calculate all available x, y, t indices
        x_ind, y_ind, t_ind = get_trajectory_indices(x_bounds, y_bounds, t_bounds, n_points)

        # Shrink indices to 1D line
        if dim == 0:
            y_ind = torch.linspace(y_ind[num_line], y_ind[num_line], n_points, dtype=torch.int32)
        if dim == 1:
            t_ind = torch.linspace(t_ind[num_line], t_ind[num_line], n_points, dtype=torch.int32)
        if dim == 2:
            x_ind = torch.linspace(x_ind[num_line], x_ind[num_line], n_points, dtype=torch.int32)

        # Concatenate values to line
        indices = torch.stack((x_ind, y_ind, t_ind))

        # Retrieve real valued line (=1D trajectory) data
        st_line, U_line = extract_tajectory_data_from_indices(x, y, t, U, indices)

        # Save line data in the corresponding lists
        st_lines.append(st_line)
        U_lines.append(U_line)

    # Concatenate single lines/trajectories together to receive plane data
    st_plane = torch.cat(st_lines)
    U_plane = torch.cat(U_lines)

    return st_plane, U_plane


def load_data(problem='../data/problem.h5'):
    """
    Returns the full spacetime and MHD domain data.
    The data is not processed and therefore in the original shape that the h5-files provides.

    Parameters:
        problem: MHD problem/benchmark that should be reconstructed
                 path to 2D MHD-datafile (.h5)

    Returns:
        x: tensor of x space
        y: tensor of y space
        t: tensor of times
        U: tensor of MHD states
    """

    # Retrieve fully sampled data
    data = h5py.File(problem, "r")

    # Get x coordinates
    x = torch.tensor(data['x'][:]).type(const.dtype)

    # Get y coordinates
    y = torch.tensor(data['y'][:]).type(const.dtype)

    # Get times
    t = torch.tensor(data['t'][:]).type(const.dtype)

    # Get MHD states
    U = torch.tensor(data['U'][:]).type(const.dtype)

    data.close()

    return x, y, t, U
