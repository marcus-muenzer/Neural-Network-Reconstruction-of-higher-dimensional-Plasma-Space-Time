"""
NAME:
    color_scale

DESCRIPTION:
    Provides a helper function for a colorbar range.
"""

import globals.constants as const


def get_color_scale(data, scaling_type):
    """
    Determines minimal and maximal values for mapping values to colors.

    Parameters:
        scaling_type: determine minimal and maximal values for mapping values to colors
                      options: 'predefined' -> use predefined default values
                               'data' -> use minimal and maximal values of the data
                               tupel (vmin, vmax) -> use given values

    Returns:
        vmin: minimal value
        vmax: maximal value
    """

    if scaling_type == 'data':
        vmin = data.min()
        vmax = data.max()
    elif scaling_type == 'predefined':
        if const.problem == "gem":
            vmin = 0
            vmax = 4
        else:
            vmin = 0
            vmax = 1.7
    elif type(scaling_type) == tuple and len(scaling_type) == 2:
        vmin, vmax = scaling_type
    else:
        err_msd = "unknown scaling_type: " + scaling_type
        raise ValueError(err_msd)

    return vmin, vmax
