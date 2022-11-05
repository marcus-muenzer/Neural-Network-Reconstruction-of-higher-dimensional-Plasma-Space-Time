"""
NAME
    Transformer

DESCRIPTION
    This module provides a Transformer class usable as a reconstructor.
"""

import torch.nn as nn
import torch
import globals.constants as const


class Transformer(nn.Module):
    def __init__(self, n_layers, layer_size, act_func='tanh'):
        """
        Initializes the Transformer.
        It implements a special Transfomer that ist used by C. Bard and J. C. Dorelli
        as physics-informed Neural Network for a 1D Plasma reconstruction task.
        Refer: https://www.frontiersin.org/articles/10.3389/fspas.2021.732275/full

        Parameters:
            n_layers: number of Z-layers
            layer_size: dimensions of layers
            act_func: activation function. Options: 'leakyReLu', 'tanh'

        Returns:
            None
        """

        super(Transformer, self).__init__()

        # Activation function
        if act_func == 'leakyReLu':
            self.act_func = torch.nn.LeakyReLU()
        else:
            self.act_func = torch.tanh

        self.beta = nn.Parameter(torch.randn(1))

        # Create layers
        self.U_layer = torch.nn.Linear(const.dims_st, layer_size)

        self.V_layer = torch.nn.Linear(const.dims_st, layer_size)

        self.H_0 = torch.nn.Linear(const.dims_st, layer_size)

        self.f_theta = torch.nn.Linear(layer_size, const.dims_mhd_state)

        self.Z_layers = torch.nn.ModuleList()
        for k in range(n_layers):
            linear_layer = torch.nn.Linear(layer_size, layer_size)

            self.Z_layers.append(linear_layer)

    def forward(self, z):
        """ Makes predictions """

        u = self.U_layer(z)
        v = self.V_layer(z)
        h = self.H_0(z)

        for z_layer in self.Z_layers:
            z = z_layer(h)
            z = self.act_func(self.beta * z)
            h = (1 - z) * u + z * v
        output = self.f_theta(h)
        return output
