"""
NAME:
    jmag

DESCRIPTION:
    Calculates the current density.
"""

import torch
import numpy as np
from utils.data import convert_tensor_to_numpy


def get_jx(x, y, t, B):
    jx = np.gradient(B[2], y, axis=1)  # dBt/dy
    jx -= np.gradient(B[1], t, axis=2)  # dBy/dt
    return jx


def get_jy(x, y, t, B):
    jy = np.gradient(B[0], t, axis=2)  # dBx/dt
    jy -= np.gradient(B[2], x, axis=0)  # dBt/dx
    return jy


def get_jt(x, y, t, B):
    jt = np.gradient(B[1], x, axis=0)  # dBy/dx
    jt -= np.gradient(B[0], y, axis=1)  # dBx/dy
    return jt


def get_jmag(x, y, t, B):
    """
    Calculates components of curl of B

    Parameters:
        x: tensor of x space of the magnetic fields
        y: tensor of y space of the magnetic fields
        t: tensor of t space of the magnetic fields
        B: magnetic fields
           shape: torch.Size([3, x, y, t])

    Returns:
        jmag
    """

    # Convert B, x, y, and magnetic fields to numpy
    x = convert_tensor_to_numpy(x)
    y = convert_tensor_to_numpy(y)
    t = convert_tensor_to_numpy(t)
    B = convert_tensor_to_numpy(B)

    jx = get_jx(x, y, t, B)
    jy = get_jy(x, y, t, B)
    jt = get_jt(x, y, t, B)

    return torch.from_numpy(np.sqrt(jx**2 + jy**2 + jt**2))
