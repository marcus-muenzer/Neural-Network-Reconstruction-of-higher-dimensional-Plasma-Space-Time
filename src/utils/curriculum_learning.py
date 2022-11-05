"""
NAME
    curriculum_learning

DESCRIPTION
    Provides helper functions for the curriculum training.
"""

import torch
import globals.constants as const


def init_params(method, axis, n_epochs, n_steps):
    """
    Calculates constant parameters for the curriculum training.

    Parameters:
        method: method of the curriculum learning
                options: None, 'colloc_inc_points', 'colloc_cuboid', 'colloc_cylinder', 'phys', 'trade_off', 'coeff', 'num_diff'
                None: no curriculum learning
                'colloc_inc_points': stepwise increase of the number of sampled collocation points
                'colloc_cuboid': stepwise shift/extension/scan of the spacetimes for the collocation point sampling
                                 along either x, y, or t axis
                'colloc_cylinder': stepwise extension of the spacetimes for the collocation point sampling
                                   in concentric circles around one or more spacecraft trajectories
                'phys': stepwise addition of MHD equations
                'trade_off': schedules the trade-off parameter weighting the physical loss
                'coeff': schedules the viscosity and resistivity coefficients
                'num_diff': schedules the deltas dx, dy, dt for calculating the derivatives
        axis: only relevant if curr-method = 'colloc_cuboid'
              axis along which the spacetimes for the collocation point sampling will be shifted/extended/scanned
              options: 'x', 'y', 't'
        n_epochs: number of total epochs in the whole training loop
        n_steps: overall amount of steps of the curriculum training

    Returns:
        max_epoch: epoch before which the curriculum annealing stops
                   normal training starts at this epoch
        epochs_per_step: amount of epochs within one curriculum step
        n_steps: overall amount of steps of curriculum learning
        dx: used if method = 'colloc_cuboid'
            delta along x dimension between curriculum steps
        dy: used if method = 'colloc_cuboid'
            delta along y dimension between curriculum steps
        dt: used if method = 'colloc_cuboid'
            delta along t dimension between curriculum steps
    """

    # No curriculum training
    if not method: return 0, None, None, None, None, None

    # Determine maximal epoch for stopping curriculum learning
    # After this epoch collocation points will be sampled from the whole time domain
    max_epoch = const.curr_fraction_of_total_epochs * n_epochs

    # Determine amount of epochs after which the time boundaries
    # for the collocation point sampling will be shifted
    # Each curriculum step has its own exclusive and limited time set
    # All curriculum steps together cover the whole time domain
    epochs_per_step = max_epoch / n_steps

    # Determine deltas for curriculum training
    dx, dy, dt = None, None, None
    if method == 'colloc_cuboid':
        if axis == 'x':
            dx = (const.x_max - const.x_min) / n_steps
        elif axis == 'y':
            dy = (const.y_max - const.y_min) / n_steps
        elif axis == 't':
            dt = (const.t_max - const.t_min) / n_steps

    return max_epoch, epochs_per_step, n_steps, dx, dy, dt


def get_colloc_bounds_cuboid(method, epoch, max_epoch, curr_step, dx=None, dy=None, dt=None):
    """
    Calculates minimal and maximal time values which restrict the spacetime domain for sampling collocation points.

    Parameters:
        method: method of the curriculum learning
                generally, the collocation points are sampled from the whole spacetime cuboid
                options: 'None', 'colloc_inc_points', 'colloc_cuboid', 'colloc_cylinder', 'phys', 'trade_off', 'coeff', 'num_diff'
                'colloc_cuboid': stepwise shift or expansion the spacetimes for the collocation point sampling
        axis: only relevant if method == 'colloc_cuboid'
              axis along the spacetimes for the collocation point sampling will be shifted
                   options: 'x', 'y', 't'
        epoch: number of now starting epoch
        max_epoch: epoch after which the curriculum annealing stops
                   normal training starts at this epoch
        curr_step: current step of the curriculum training
        dx: delta of time between curriculum steps -> if set: dy and dt should be None
        dy: delta of time between curriculum steps -> if set: dx and dt should be None
        dt: delta of time between curriculum steps -> if set: dx and dy should be None

    Returns:
        x_min: minimal x value of collocation points in this epoch
        x_max: maximal x value of collocation points in this epoch
        y_min: minimal y value of collocation points in this epoch
        y_max: maximal y value of collocation points in this epoch
        t_min: minimal t value of collocation points in this epoch
        t_max: maximal t value of collocation points in this epoch
    """

    # Default: use whole spacetimes
    x_min, x_max, y_min, y_max, t_min, t_max = const.x_min, const.x_max, const.y_min, const.y_max, const.t_min, const.t_max

    # Restrict according to curriculum progress
    if method == 'colloc_cuboid' and epoch < max_epoch:
        # Restrict along one dimension
        if dx:
            # Set x boundaries
            # x_min = const.x_min + curr_step * dx  # Shift cuboid
            x_max = const.x_min + (curr_step + 1) * dx
        elif dy:
            # Set y boundaries
            # y_min = const.y_min + curr_step * dy  # Shift cuboid
            y_max = const.y_min + (curr_step + 1) * dy
        elif dt:
            # Set t boundaries
            # t_min = const.t_min + curr_step * dt  # Shift cuboid
            t_max = const.t_min + (curr_step + 1) * dt

    return x_min, x_max, y_min, y_max, t_min, t_max


def get_lambda_phys(curr_step):
    """
    Calculates a weight for the physical loss term
    according to a predefined schedule.

    Parameters:
        curr_step: current step of the curriculum training

    Returns:
        weight for physical loss
    """

    return .1 + .1 * curr_step


def schedule_viscosity(curr_step):
    """
    Calculates the viscosity factor nu for the viscosity term
    according to a predefined schedule.

    Parameters:
        curr_step: current step of the curriculum training

    Returns:
        viscosity
    """

    return .0001 * curr_step


def schedule_resistivity(curr_step):
    """
    Calculates the resistivity factor eta for the resistivity term
    according to a predefined schedule.

    Parameters:
        curr_step: current step of the curriculum training

    Returns:
        resistivity
    """

    return .001 + .00015 * curr_step


def schedule_numerical_diff(curr_step):
    """
    Calculates the delta vectors for the numerical differentiation
    according to a predefined schedule.

    Parameters:
        curr_step: current step of the curriculum training

    Returns:
        None
    """

    decay_factor = .9

    const.dx = const.dx * decay_factor
    const.dy = const.dy * decay_factor
    const.dt = const.dt * decay_factor

    const.dx_tensor = torch.mul(const.dx_tensor, decay_factor)
    const.dy_tensor = torch.mul(const.dy_tensor, decay_factor)
    const.dt_tensor = torch.mul(const.dt_tensor, decay_factor)


def update_curr_step(epoch, max_epoch, curr_step, epochs_per_step):
    """
    Calculates minimal and maximal time values which restrict the spacetime domain for sampling collocation points.

    Parameters:
        epoch: number of now starting epoch
        max_epoch: epoch after which the curriculum annealing stops
                   normal training starts at this epoch
        curr_step: current step of the curriculum training
        epochs_per_step: amount of epochs within one curriculum step

    Returns:
        curr_step: current step of the curriculum training
    """

    if epoch < max_epoch:
        # Increase curriculum step every "epochs_per_step" epochs
        if epoch > (curr_step + 1) * epochs_per_step:
            curr_step += 1

    return curr_step
