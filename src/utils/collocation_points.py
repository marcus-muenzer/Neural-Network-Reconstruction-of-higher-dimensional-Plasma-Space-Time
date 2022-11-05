"""
NAME
    collocation_points

DESCRIPTION
    Provides functions for sampling collocation points.
"""

import torch
from botorch.utils.sampling import sample_hypersphere
import utils.curriculum_learning as cl
import globals.constants as const


def sample_coll_points(curr_method, n_steps, curr_step, max_epoch, epoch,
                       dx=None, dy=None, dt=None):
    """
    Samples collocation points according to the specified curriculum method.

    Parameters:
        curr_method: method of the curriculum learning
                options: 'None', 'colloc_inc_points', 'colloc_cuboid', 'colloc_cylinder', 'phys', 'trade_off', 'coeff', 'num_diff'
                generally, the collocation points are sampled from the whole spacetime cuboid
                'colloc_inc_points': number of sampled collocation points increases over time
                'colloc_cuboid': stepwise shift or expansion of the spacetimes for the collocation point sampling
                                 starting with a small initial cuboid
                'colloc_cylinder': collocation points are located in a cylinder around the spacecraft trajectory
        n_steps: overall amount of steps of the curriculum training
        curr_step: current step of the curriculum training
        max_epoch: epoch after which the curriculum annealing stops
                              normal training starts at this epoch
        epoch: number of now starting epoch
        dx: delta of time between curriculum steps -> if set: dy and dt should be None
        dy: delta of time between curriculum steps -> if set: dx and dt should be None
        dt: delta of time between curriculum steps -> if set: dx and dy should be None
    Returns:
        collocation points
        shape: torch.Size([n_points, const.dims_st])
    """

    if curr_method == 'colloc_cylinder':
        return sample_coll_points_cylinder(n_steps, curr_step)

    # Determine the amount of collocation points that is to be sampled
    factor = 1
    # Increase points by the factor of const.curr_factor_total_points in total
    if const.curr_method == 'colloc_inc_points':
        factor += (const.curr_factor_total_points - 1) * curr_step / (n_steps - 1)
    n_points = int(const.st_train.shape[0] * factor)

    x_min, x_max, y_min, y_max, t_min, t_max = \
        cl.get_colloc_bounds_cuboid(curr_method, epoch, max_epoch, curr_step, dx, dy, dt)
    return sample_coll_points_cuboid(x_min, x_max, y_min, y_max, t_min, t_max, n_points)


def sample_coll_points_cuboid(x_min=None, x_max=None, y_min=None, y_max=None, t_min=None, t_max=None, n_points=None):
    """
    Samples collocation points from a cuboid.

    Parameters:
        x_min: minimal value of x for sampling collocation points
               if None -> minimal value of the full x domain will be used
        x_max: maximal value of x for sampling collocation points
               if None -> maximal value of the full x domain will be used
        y_min: minimal value of y for sampling collocation points
               if None -> minimal value of the full y domain will be used
        y_max: maximal value of y for sampling collocation points
               if None -> maximal value of the full y domain will be used
        t_min: minimal value of t for sampling collocation points
               if None -> minimal value of the full t domain will be used
        t_max: maximal value of t for sampling collocation points
               if None -> maximal value of the full t domain will be used
        n_points: number of collocation points that are to be sampled

    Returns:
        st_coll: collocation points
                 shape: torch.Size([n_points, const.dims_st])
    """

    if not n_points: n_points = const.st_train.shape[0]

    x_min = const.x_min if not x_min else x_min
    x_max = const.x_max if not x_max else x_max
    y_min = const.y_min if not y_min else y_min
    y_max = const.y_max if not y_max else y_max
    t_min = const.t_min if not t_min else t_min
    t_max = const.t_max if not t_max else t_max

    # Sample (semi-) random spacetimes
    x_coll = torch.FloatTensor(n_points).uniform_(x_min, x_max).to(const.device)
    y_coll = torch.FloatTensor(n_points).uniform_(y_min, y_max).to(const.device)
    t_coll = torch.FloatTensor(n_points).uniform_(t_min, t_max).to(const.device)
    st_coll = torch.stack((x_coll, y_coll, t_coll), dim=1).type(const.dtype)

    return st_coll


def sample_coll_points_cylinder(n_steps, curr_step):
    """
    Samples collocation points that are located in a cylinder around the spacecraft trajectory.

    Parameters:
        n_steps: overall amount of steps of the curriculum training
        curr_step: current step of the curriculum training

    Returns:
        st_coll: collocation points
                 shape: torch.Size([const.st_train.shape[0], const.dims_st])
    """

    n_points = const.st_train.shape[0]

    # Assumption: training trajectories are approximately uniformely distributed in the space domain
    # => divide space domain by const.n_trajs + 1
    x_rad = (const.x_max - const.x_min) / (const.n_trajs + 1) * (curr_step + 1) / n_steps
    y_rad = (const.y_max - const.y_min) / (const.n_trajs + 1) * (curr_step + 1) / n_steps
    t_rad = (const.t_max - const.t_min) / (const.n_trajs + 1) * (curr_step + 1) / n_steps

    # Sample hypershperes and add them to the trajectory
    cyl = sample_hypersphere(const.dims_st, n_points)
    cyl[:, 0] *= x_rad
    cyl[:, 1] *= y_rad
    cyl[:, 2] *= t_rad

    st_coll = const.st_train + cyl

    # Clamp values into domains
    torch.clamp_(st_coll[:, 0], min=const.x_min, max=const.x_max)
    torch.clamp_(st_coll[:, 1], min=const.y_min, max=const.y_max)
    torch.clamp_(st_coll[:, 2], min=const.t_min, max=const.t_max)

    return st_coll
