"""
NAME:
    physical_loss

DESCRIPTION:
    Provides the functionality to calculate the physical loss using the full set of ideal MHD equations.
    Computes model predictions in parallel (reason: runtime improvement, will be computed anyway).
"""

import torch
import globals.constants as const


def ms_reduction(residuals):
    """
    Calculates the mean of the squared residuals.
    Can be used as a pseudo-loss function. When passing real prediction residuals
    it returns the mean squared error of the residual generating predictions.

    Parameters:
        residuals: tensor of residuals

    Returns:
        tensor of mean squared residuals
    """

    return torch.mean(torch.square(residuals))


def mlogcosh_reduction(residuals):
    """
    Calculates the mean of the logarithm of the hyperbolic cosine of the residuals.
    Can be used as a pseudo-loss function. When passing real prediction residuals
    it returns the logarithmic hyperbolic consine error of the residual generating predictions.

    Parameters:
        residuals: tensor of residuals

    Returns:
        tensor of the logarithm of the hyperbolic cosine of the residuals
    """

    return torch.mean(torch.log(torch.cosh(residuals)))


def calc_physical_losses(st, model, visc_nu=0, resis_eta=0, delta=0.001, gamma=2, loss_type='mse', curr_step=2):
    """
    Computes models predictions for given points in spacetime.
    Also calculates the physical error for these model predictions.

    Parameters:
        st: spacetime-vector
        model: pytorch model, which is used for reconstruction of the plasma environment
        visc: value for weighting viscosity term
              0 means no viscosity term
        gamma: ratio of specific heats (pressure equation)
        loss_type: type of "pseudo-loss" function for reduction of errors
                   Options: 'mse', 'mlogcosh'

    Returns:
        Tupel:
            1. tensor of model predictions
            2. residuals: tensor of summed up (not reduced) residuals
            3. phys_loss: scalar: mean of single losses from MHD equations
    """

    U_model, rho, vx, vy, vz, p, Bx, By, Bz, drho, dvx, dvy, dvz, dp, dBx, dBy, dBz, d2vx, d2vy, d2vz, d2Bx, d2By, d2Bz = derivs(st, model)

    # Define reduction/pseudo-loss function
    if loss_type == 'mse':
        reduce = ms_reduction
    elif loss_type == 'mlogcosh':
        reduce = mlogcosh_reduction
    else:
        raise SystemError("Undefined loss_type: " + str(loss_type) + "!")

    # ===================================
    # 2D MHD equations
    # ===================================

    # Continuity equation:
    # drho/dt + vx*drho/dx + rho*dvx/dx + vy*drho/dy + rho*dvy/dy = 0
    time_deriv = drho[:, 2]
    space_deriv_x = vx * drho[:, 0] + rho * dvx[:, 0]
    space_deriv_y = vy * drho[:, 1] + rho * dvy[:, 1]

    r1 = time_deriv + space_deriv_x + space_deriv_y
    l1 = reduce(r1)

    # vx equation:
    # rho*dvx/dt + rho*vx*dvx/dx + dp/dx + By*dBy/dx+ Bz*dBz/dx + rho*vy*dvx/dy - By*dBx/dy = 0
    time_deriv = rho * dvx[:, 2]
    space_deriv_x = rho * vx * dvx[:, 0] + dp[:, 0] + By * dBy[:, 0] + Bz * dBz[:, 0]
    space_deriv_y = rho * vy * dvx[:, 1] - By * dBx[:, 1]
    viscosity = -rho * visc_nu * (d2vx[:, 0] + d2vx[:, 1])

    r2 = time_deriv + space_deriv_x + space_deriv_y + viscosity
    l2 = reduce(r2)

    # vy equation:
    # rho*dvy/dt + rho*vx*dvy/dx - Bx*dBy/dx + rho*vy*dvy/dy + dp/dy + Bx*dBx/dy + Bz*dBz/dy = 0
    time_deriv = rho * dvy[:, 2]
    space_deriv_x = rho * vx * dvy[:, 0] - Bx * dBy[:, 0]
    space_deriv_y = rho * vy * dvy[:, 1] + dp[:, 1] + Bx * dBx[:, 1] + Bz * dBz[:, 1]
    viscosity = -rho * visc_nu * (d2vy[:, 0] + d2vy[:, 1])

    r3 = time_deriv + space_deriv_x + space_deriv_y + viscosity
    l3 = reduce(r3)

    # vz equation:
    # rho*dvz/dt + rho*vx*dvz/dx - Bx*dBz/dx + rho*vy*dvz/dy - By*dBzdy = 0
    time_deriv = rho * dvz[:, 2]
    space_deriv_x = rho * vx * dvz[:, 0] - Bx * dBz[:, 0]
    space_deriv_y = rho * vy * dvz[:, 1] - By * dBz[:, 1]
    viscosity = -rho * visc_nu * (d2vz[:, 0] + d2vz[:, 1])

    r4 = time_deriv + space_deriv_x + space_deriv_y + viscosity
    l4 = reduce(r4)

    # Pressure equation:
    # dp/dt + vx*dp/dx + gamma*p*dvx/dx + vy*dp/dy + gamma*p*dvy/dy
    time_deriv = dp[:, 2]
    space_deriv_x = vx * dp[:, 0] + gamma * p * dvx[:, 0]
    space_deriv_y = vy * dp[:, 1] + gamma * p * dvy[:, 1]

    r5 = time_deriv + space_deriv_x + space_deriv_y
    l5 = reduce(r5)

    # Bx equation:
    # dBx/dt - vx*dBy/dy - By*dvx/dy + vy*dBx/dy + Bx*dvy/dy = 0
    time_deriv = dBx[:, 2]
    space_deriv_y = - vx * dBy[:, 1] - By * dvx[:, 1] + vy * dBx[:, 1] + Bx * dvy[:, 1]
    resistivity = -resis_eta * (d2Bx[:, 0] + d2Bx[:, 1])

    r6 = time_deriv + space_deriv_y + resistivity
    l6 = reduce(r6)

    # By equation:
    # dBy/dt + vx*dBy/dx + By*dvx/dx -vy*dBx/dx - Bx*dvy/dx = 0
    time_deriv = dBy[:, 2]
    space_deriv_x = vx * dBy[:, 0] + By * dvx[:, 0] - vy * dBx[:, 0] - Bx * dvy[:, 0]
    resistivity = -resis_eta * (d2By[:, 0] + d2By[:, 1])

    r7 = time_deriv + space_deriv_x + resistivity
    l7 = reduce(r7)

    # Bz equation:
    # dBz/dt + vx*dBz/dx + Bz*dvx/dx - vz*dBx/dx - Bx*dvz/dx + vy*dBz/dy + Bz*dvy/dy - vz*dBy/dy - By*dvz/dy = 0
    time_deriv = dBz[:, 2]
    space_deriv_x = vx * dBz[:, 0] + Bz * dvx[:, 0] - vz * dBx[:, 0] - Bx * dvz[:, 0]
    space_deriv_y = vy * dBz[:, 1] + Bz * dvy[:, 1] - vz * dBy[:, 1] - Bz * dvz[:, 1]
    resistivity = -resis_eta * (d2Bz[:, 0] + d2Bz[:, 1])

    r8 = time_deriv + space_deriv_x + space_deriv_y + resistivity
    l8 = reduce(r8)

    # Magnetic field divergence equation:
    # dBx/dx + dBy/dy = 0
    space_deriv_x = dBx[:, 0]
    space_deriv_y = dBy[:, 1]

    r9 = space_deriv_x + space_deriv_y
    l9 = reduce(r9)

    # Sum up (positive) residuals
    residuals = [r1, r2, r3, r4]
    if curr_step == 1: residuals += [r6, r7, r8, r9]
    if curr_step == 2: residuals += [r5, r6, r7, r8, r9]
    residuals = torch.abs(torch.stack(residuals))
    residuals = torch.sum(residuals, dim=0)

    # Calculate overall physical loss
    # Defined as mean of single losses from MHD equations
    losses = [l1, l2, l3, l4]
    if curr_step == 1: losses += [l5]
    if curr_step == 2: losses += [l5, l6, l7, l8, l9]
    phys_loss = torch.stack(losses)
    phys_loss = torch.mean(phys_loss)

    return U_model, residuals, phys_loss


def derivs(st, model):
    """
    Computes derivativs and model predictions.

    Parameters:
        st: tensor of spacetimes
        model: pytorch model, which is used for reconstruction of the plasma environment

    Returns:
        Tupel:
            1. tensor of model predictions
            2..n-1: tensor of derivativs for MHD equations
                    shape: torch.Size([st.shape[0], 3])
                    form of one derivative: (d/dx, d/dy, d/dt)
            n: tensor of viscosity term
    """

    dx = 2 * const.dx
    dy = 2 * const.dy
    dt = 2 * const.dt

    # Make predictions
    U_model = model(st)

    # Make predictions along x for by const.dx shifted spacetimes
    U_model_px = model(st + const.dx_tensor[:st.shape[0]])
    U_model_mx = model(st - const.dx_tensor[:st.shape[0]])

    # Make predictions along y for by const.dy shifted spacetimes
    U_model_py = model(st + const.dy_tensor[:st.shape[0]])
    U_model_my = model(st - const.dy_tensor[:st.shape[0]])

    # Make predictions along t for by const.dt shifted spacetimes
    U_model_pt = model(st + const.dt_tensor[:st.shape[0]])
    U_model_mt = model(st - const.dt_tensor[:st.shape[0]])

    # Extract one dimensional predictions for every physical unit
    rho = U_model[:, 0]
    vx = U_model[:, 1]
    vy = U_model[:, 2]
    vz = U_model[:, 3]
    p = U_model[:, 4]
    Bx = U_model[:, 5]
    By = U_model[:, 6]
    Bz = U_model[:, 7]

    # Calculate derivatives
    # Every derivative is a tensor of shape torch.Size([st.shape[0], 3])
    # Form of every element in a derivative: (d/dx, d/dy, d/dt)
    drho = torch.stack([(U_model_px[:, 0] - U_model_mx[:, 0]) / dx, (U_model_py[:, 0] - U_model_my[:, 0]) / dy, (U_model_pt[:, 0] - U_model_mt[:, 0]) / dt], axis=1)
    dvx = torch.stack([(U_model_px[:, 1] - U_model_mx[:, 1]) / dx, (U_model_py[:, 1] - U_model_my[:, 1]) / dy, (U_model_pt[:, 1] - U_model_mt[:, 1]) / dt], axis=1)
    dvy = torch.stack([(U_model_px[:, 2] - U_model_mx[:, 2]) / dx, (U_model_py[:, 2] - U_model_my[:, 2]) / dy, (U_model_pt[:, 2] - U_model_mt[:, 2]) / dt], axis=1)
    dvz = torch.stack([(U_model_px[:, 3] - U_model_mx[:, 3]) / dx, (U_model_py[:, 3] - U_model_my[:, 3]) / dy, (U_model_pt[:, 3] - U_model_mt[:, 3]) / dt], axis=1)
    dp = torch.stack([(U_model_px[:, 4] - U_model_mx[:, 4]) / dx, (U_model_py[:, 4] - U_model_my[:, 4]) / dy, (U_model_pt[:, 4] - U_model_mt[:, 4]) / dt], axis=1)
    dBx = torch.stack([(U_model_px[:, 5] - U_model_mx[:, 5]) / dx, (U_model_py[:, 5] - U_model_my[:, 5]) / dy, (U_model_pt[:, 5] - U_model_mt[:, 5]) / dt], axis=1)
    dBy = torch.stack([(U_model_px[:, 6] - U_model_mx[:, 6]) / dx, (U_model_py[:, 6] - U_model_my[:, 6]) / dy, (U_model_pt[:, 6] - U_model_mt[:, 6]) / dt], axis=1)
    dBz = torch.stack([(U_model_px[:, 7] - U_model_mx[:, 7]) / dx, (U_model_py[:, 7] - U_model_my[:, 7]) / dy, (U_model_pt[:, 7] - U_model_mt[:, 7]) / dt], axis=1)

    # Derivatives for viscosity
    # Take d/dz = 0 since 2D in space
    dx2 = const.dx * const.dx
    dy2 = const.dy * const.dy

    d2vx = torch.stack([(U_model_px[:, 1] - 2 * vx + U_model_mx[:, 1]) / dx2, (U_model_py[:, 1] - 2 * vx + U_model_my[:, 1]) / dy2], axis=1)

    d2vy = torch.stack([(U_model_px[:, 2] - 2 * vy + U_model_mx[:, 2]) / dx2, (U_model_py[:, 2] - 2 * vy + U_model_my[:, 2]) / dy2], axis=1)

    d2vz = torch.stack([(U_model_px[:, 3] - 2 * vz + U_model_mx[:, 3]) / dx2, (U_model_py[:, 3] - 2 * vz + U_model_my[:, 3]) / dy2], axis=1)

    # Derivatives for resistivity
    # Take d/dz = 0 since 2D in space
    d2Bx = torch.stack([(U_model_px[:, 5] - 2 * Bx + U_model_mx[:, 5]) / dx2, (U_model_py[:, 5] - 2 * Bx + U_model_my[:, 5]) / dy2], axis=1)

    d2By = torch.stack([(U_model_px[:, 6] - 2 * By + U_model_mx[:, 6]) / dx2, (U_model_py[:, 6] - 2 * By + U_model_my[:, 6]) / dy2], axis=1)

    d2Bz = torch.stack([(U_model_px[:, 7] - 2 * Bz + U_model_mx[:, 7]) / dx2, (U_model_py[:, 7] - 2 * Bz + U_model_my[:, 7]) / dy2], axis=1)

    return U_model, rho, vx, vy, vz, p, Bx, By, Bz, drho, dvx, dvy, dvz, dp, dBx, dBy, dBz, d2vx, d2vy, d2vz, d2Bx, d2By, d2Bz
