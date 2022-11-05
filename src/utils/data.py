"""
NAME:
    data

DESCRIPTION:
    Provides a helper function for getting human understandable physical units for data indices.
"""

import torch


def index_to_phys_unit(index):
    """
    Determines physical unit in words for an index of the MHD data.

    Parameters:
        index: index of the MHD data (along first dimension)

    Returns:
        phys_unit: physical unit as readable string
    """

    phys_unit = ""

    if index == 0:
        phys_unit = "Density"
    elif index == 1:
        phys_unit = "Vx"
    elif index == 2:
        phys_unit = "Vy"
    elif index == 3:
        phys_unit = "Vz"
    elif index == 4:
        phys_unit = "P"
    elif index == 5:
        phys_unit = "Bx"
    elif index == 6:
        phys_unit = "By"
    elif index == 7:
        phys_unit = "Bz"

    return phys_unit


def phys_unit_to_index(phys_unit):
    """
    Determines the index of the MHD data for a physical unit in words.

    Parameters:
        phys_unit: physical unit as readable string

    Returns:
        index: index of the MHD data (along first dimension)
    """

    index = ""

    if phys_unit == "Density":
        index = 0
    elif phys_unit == "Vx":
        index = 1
    elif phys_unit == "Vy":
        index = 2
    elif phys_unit == "Vz":
        index = 3
    elif phys_unit == "P":
        index = 4
    elif phys_unit == "Bx":
        index = 5
    elif phys_unit == "By":
        index = 6
    elif phys_unit == "Bz":
        index = 7
    # No real physical unit but if calculated, jmag data only has index 8
    elif phys_unit == "Jmag":
        index = 8

    return index


def convert_tensor_to_numpy(tensor):
    """
    Return a numpy array of the given tensor.

    Parameters:
        tensor: pytorch tensor or numpy.ndarray

    Returns:
        tensor: numpy.ndarray
    """

    if type(tensor) == torch.Tensor:
        tensor = tensor.cpu().detach().numpy()

    return tensor
