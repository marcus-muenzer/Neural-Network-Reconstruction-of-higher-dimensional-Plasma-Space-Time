"""
NAME
    MLP

DESCRIPTION
    This module provides a MLP class usable as a reconstructor.
"""

import torch.nn as nn
import torch
import globals.constants as const


class MLP(nn.Module):
    def __init__(self, layers: [int], act_func='tanh'):
        """
        Initializes the straight forward MLP for the reconstruction.

        Parameters:
            layer[i]: size of ith layer
            act_func: activation function. Options: 'leakyReLu', 'tanh'

        Returns:
            None
        """

        super(MLP, self).__init__()

        # Activation function
        if act_func == 'leakyReLu':
            self.act_func = torch.nn.LeakyReLU()
        else:
            self.act_func = torch.tanh

        self.beta = nn.Parameter(torch.randn(1))

        # Create layers
        self.layers = torch.nn.ModuleList()
        for idx, layer_size in enumerate(layers):
            # First Layer
            if idx == 0:
                linear_layer = torch.nn.Linear(const.dims_st, layer_size)

            # Intermediate Layer(s)
            else:
                linear_layer = torch.nn.Linear(layers[idx - 1], layer_size)

            self.layers.append(linear_layer)

        # Additional last Layer
        # Transforms to mhd dimensions
        layer = torch.nn.Linear(layers[-1], const.dims_mhd_state)
        self.layers.append(layer)

    def forward(self, z):
        """ Makes predictions """

        for layer in self.layers[:-1]:
            z = layer(z)
            z = self.act_func(self.beta * z)
        z = self.layers[-1](z)
        return z
