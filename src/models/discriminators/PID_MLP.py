"""
NAME
    PID_MLP

DESCRIPTION
    This module provides a MLP class usable as a physics-informed discriminator inside the cGAN architecture.
"""

import torch.nn as nn
import torch
import globals.constants as const


class PID_MLP(nn.Module):
    def __init__(self, layers: [int], act_func='tanh', apply_sig=True):
        """
        Initializes the PID_MLP.
        It receives additional knowledge about the physical residuals in an extra input feature.
        It uses a straight forward MLP for the classification.

        Parameters:
            layer[i]: size of ith layer
            act_func: activation function. Options: 'leakyReLu', 'tanh'
            apply_sig: apply sigmoid function on the output

        Returns:
            None
        """

        super(PID_MLP, self).__init__()

        # Activation functions
        if act_func == 'leakyReLu':
            self.act_func = torch.nn.LeakyReLU()
        else:
            self.act_func = torch.tanh

        self.beta = nn.Parameter(torch.randn(1))

        self.apply_sig = apply_sig
        self.sigmoid = torch.nn.Sigmoid()

        # Create layers
        self.layers = torch.nn.ModuleList()
        for idx, layer_size in enumerate(layers):
            # First Layer
            if idx == 0:
                linear_layer = torch.nn.Linear(const.dims_st + const.dims_mhd_state + 1, layer_size)

            # Intermediate Layer(s)
            else:
                linear_layer = torch.nn.Linear(layers[idx - 1], layer_size)

            self.layers.append(linear_layer)

        # Additional last Layer
        # Transforms to mhd dimensions
        layer = torch.nn.Linear(layers[-1], const.dims_mhd_state)
        self.layers.append(layer)

    def forward(self, st, U, residual):
        """ Makes predictions """

        # Concatenate input
        z = torch.cat((st, U, residual), 1)

        for layer in self.layers[:-1]:
            z = layer(z)
            z = self.act_func(self.beta * z)
        z = self.layers[-1](z)
        if self.apply_sig: z = self.sigmoid(z)

        return z
